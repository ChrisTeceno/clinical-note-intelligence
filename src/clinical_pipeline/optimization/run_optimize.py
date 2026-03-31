"""CLI entry point for prompt optimization.

Usage:
    python -m clinical_pipeline.optimization.run_optimize \
        --mimic-path /path/to/mimic \
        --n-iterations 5 \
        --target diagnoses_f1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from clinical_pipeline.config import Settings, get_settings
from clinical_pipeline.evaluation.ground_truth import load_ground_truth, select_admissions
from clinical_pipeline.evaluation.run_tracker import RunRecord, RunTracker
from clinical_pipeline.evaluation.scorer import (
    aggregate_results,
    evaluate_extraction,
)
from clinical_pipeline.evaluation.synthetic_notes import generate_batch
from clinical_pipeline.extraction.cost_tracker import CostTracker
from clinical_pipeline.extraction.extractor import DEFAULT_MODEL, ClinicalExtractor, ExtractionError
from clinical_pipeline.extraction.prompts import SYSTEM_PROMPT
from clinical_pipeline.feedback.feedback_store import FeedbackStore
from clinical_pipeline.optimization.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


def _build_eval_runner(
    ground_truths: dict,
    notes: dict,
    settings: Settings,
    model: str,
    use_rag: bool,
) -> callable:
    """Build an evaluation function that runs extraction with a given system prompt.

    Returns a callable that takes a prompt string and returns aggregated metrics.
    """

    def eval_runner(system_prompt: str) -> dict:
        cost_tracker = CostTracker(is_batch=False)

        retriever = None
        if use_rag:
            from clinical_pipeline.extraction.icd10_rag import ICD10Retriever

            code_table_path = settings.reference_dir / "icd10cm_codes_2025.txt"
            retriever = ICD10Retriever(code_table_path)

        extractor = ClinicalExtractor(
            api_key=settings.anthropic_api_key,
            model=model,
            cost_tracker=cost_tracker,
            retriever=retriever,
        )

        # Monkey-patch the system prompt for this eval run
        import clinical_pipeline.extraction.prompts as prompts_module

        original_prompt = prompts_module.SYSTEM_PROMPT
        prompts_module.SYSTEM_PROMPT = system_prompt

        extractions = {}
        for hadm_id, note_text in notes.items():
            try:
                extraction = extractor.extract(
                    note_id=str(hadm_id),
                    transcription=note_text,
                )
                extractions[hadm_id] = extraction
            except ExtractionError:
                logger.warning("Extraction failed for hadm_id=%d during optimization", hadm_id)

        # Restore original prompt
        prompts_module.SYSTEM_PROMPT = original_prompt

        # Score
        all_results = {}
        for hadm_id, extraction in extractions.items():
            gt = ground_truths[hadm_id]
            results = evaluate_extraction(extraction, gt)
            all_results[hadm_id] = results

        return aggregate_results(all_results)

    return eval_runner


def run_optimization(
    mimic_path: Path,
    n_iterations: int = 5,
    target_metric: str = "diagnoses_f1",
    n_admissions: int = 10,
    model: str | None = None,
    use_rag: bool = False,
    settings: Settings | None = None,
) -> dict:
    """Run the prompt optimization loop.

    Uses a small eval set (default 10 admissions) for fast iteration.
    """
    settings = settings or get_settings()
    cache_dir = settings.data_dir / "optimization"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = model or DEFAULT_MODEL

    # Select admissions (use a separate cache from the full eval)
    hadm_ids_path = cache_dir / "opt_hadm_ids.json"
    if hadm_ids_path.exists():
        hadm_ids = json.loads(hadm_ids_path.read_text())
        logger.info("Loaded %d cached optimization admission IDs", len(hadm_ids))
    else:
        hadm_ids = select_admissions(mimic_path, n=n_admissions)
        hadm_ids_path.write_text(json.dumps(hadm_ids, indent=2))

    ground_truths = load_ground_truth(mimic_path, hadm_ids)

    # Generate synthetic notes
    logger.info("Generating synthetic notes for optimization eval set...")
    notes = generate_batch(
        ground_truths=ground_truths,
        api_key=settings.anthropic_api_key,
        cache_dir=cache_dir,
    )

    # Build the eval runner
    eval_runner = _build_eval_runner(
        ground_truths=ground_truths,
        notes=notes,
        settings=settings,
        model=model,
        use_rag=use_rag,
    )

    # Load feedback store
    feedback_path = settings.data_dir / "feedback" / "corrections.json"
    feedback_store = FeedbackStore(feedback_path)

    # Run optimizer
    optimizer = PromptOptimizer(
        eval_runner=eval_runner,
        feedback_store=feedback_store,
        prompt_history_path=cache_dir / "prompt_history.json",
        baseline_prompt=SYSTEM_PROMPT,
    )

    result = optimizer.optimize(
        n_iterations=n_iterations,
        target_metric=target_metric,
    )

    # Record in experiment tracker
    tracker = RunTracker(settings.data_dir / "evaluation" / "run_history.json")
    best = result.get("best_metrics", {})
    run_record = RunRecord(
        model=model,
        n_admissions=len(notes),
        description=f"optimization run ({n_iterations} iterations, target={target_metric})",
        diagnoses=best.get("diagnoses", {}),
        procedures=best.get("procedures", {}),
        medications=best.get("medications", {}),
        duration_seconds=result.get("total_duration_seconds", 0.0),
    )
    tracker.record(run_record)

    logger.info("Optimization complete. Kept %d/%d mutations.", result["n_kept"], n_iterations)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run prompt optimization loop")
    parser.add_argument(
        "--mimic-path",
        type=Path,
        required=True,
        help="Path to MIMIC-IV demo hosp/ directory",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="Number of optimization iterations (default: 5)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="diagnoses_f1",
        help="Target metric to optimize (default: diagnoses_f1)",
    )
    parser.add_argument(
        "--n-admissions",
        type=int,
        default=10,
        help="Number of admissions in the eval set (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for extraction",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable RAG with ICD-10 codebook",
    )
    args = parser.parse_args()

    run_optimization(
        mimic_path=args.mimic_path,
        n_iterations=args.n_iterations,
        target_metric=args.target,
        n_admissions=args.n_admissions,
        model=args.model,
        use_rag=args.use_rag,
    )
