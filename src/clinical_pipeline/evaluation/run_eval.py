"""Main evaluation entry point: generate synthetic notes, extract, and score."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from clinical_pipeline.config import Settings, get_settings
from clinical_pipeline.evaluation.batch_api import (
    ESTIMATED_EXTRACTION_TOKENS,
    ESTIMATED_NOTE_TOKENS,
    BatchError,
    estimate_batch_cost,
    run_batch,
)
from clinical_pipeline.evaluation.ground_truth import (
    load_ground_truth,
    select_admissions,
)
from clinical_pipeline.evaluation.run_tracker import RunRecord, RunTracker
from clinical_pipeline.evaluation.scorer import (
    AggregatedResult,
    EvaluationResult,
    aggregate_results,
    evaluate_extraction,
)
from clinical_pipeline.evaluation.synthetic_notes import (
    GENERATION_MODEL,
    generate_batch,
    generate_batch_api,
)
from clinical_pipeline.extraction.cost_tracker import CostTracker
from clinical_pipeline.extraction.extractor import DEFAULT_MODEL
from clinical_pipeline.extraction.extractor import ClinicalExtractor, ExtractionError
from clinical_pipeline.extraction.models import ClinicalExtraction
from clinical_pipeline.extraction.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationSummary:
    """Full evaluation output."""

    n_admissions: int
    overall: AggregatedResult
    per_admission: dict[int, list[dict]]  # hadm_id -> list of result dicts
    failed_extractions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_admissions": self.n_admissions,
            "overall": self.overall,
            "per_admission": {
                str(k): v for k, v in self.per_admission.items()
            },
            "failed_extractions": self.failed_extractions,
        }


def _load_cached_extraction(cache_dir: Path, hadm_id: int) -> ClinicalExtraction | None:
    """Load a cached extraction result if available."""
    path = cache_dir / "extractions" / f"{hadm_id}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return ClinicalExtraction.model_validate(data)
    return None


def _save_extraction(cache_dir: Path, hadm_id: int, extraction: ClinicalExtraction) -> None:
    """Cache an extraction result."""
    out_dir = cache_dir / "extractions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{hadm_id}.json"
    path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")


DEFAULT_MAX_COST = 3.00


def _check_batch_budget(
    cache_dir: Path, hadm_ids: list[int], extraction_model: str, max_cost: float
) -> None:
    """Log the estimated batch cost for uncached work and abort if it exceeds max_cost."""
    notes_dir = cache_dir / "synthetic_notes"
    n_notes = sum(1 for h in hadm_ids if not (notes_dir / f"{h}.txt").exists())
    n_extractions = sum(
        1 for h in hadm_ids if not (cache_dir / "extractions" / f"{h}.json").exists()
    )
    note_cost = estimate_batch_cost(GENERATION_MODEL, n_notes, ESTIMATED_NOTE_TOKENS)
    extraction_cost = estimate_batch_cost(
        extraction_model, n_extractions, ESTIMATED_EXTRACTION_TOKENS
    )
    estimate = note_cost + extraction_cost
    logger.info(
        "Estimated batch cost: $%.2f (%d notes $%.2f, %d extractions $%.2f), limit $%.2f",
        estimate,
        n_notes,
        note_cost,
        n_extractions,
        extraction_cost,
        max_cost,
    )
    if estimate > max_cost:
        raise BatchError(
            f"Estimated cost ${estimate:.2f} exceeds --max-cost ${max_cost:.2f}; nothing submitted"
        )


def run_evaluation(
    mimic_path: Path | None = None,
    n_admissions: int = 30,
    settings: Settings | None = None,
    description: str = "",
    model: str | None = None,
    use_rag: bool = False,
    batch: bool = False,
    prompt_file: Path | None = None,
    max_cost: float = DEFAULT_MAX_COST,
) -> EvaluationSummary:
    """Run the full evaluation pipeline.

    Steps:
    1. Select and load MIMIC ground truth for n admissions
    2. Generate synthetic notes (or load cached)
    3. Run extraction pipeline on each note
    4. Score against ground truth
    5. Return summary with overall and per-admission metrics

    ``mimic_path`` defaults to the active source's ``hosp/`` directory.
    With ``batch`` set, note generation and extraction go through the Message
    Batches API (50% discount); ``prompt_file`` replaces the extraction system prompt.
    Results are cached in ``settings.evaluation_dir`` to avoid re-generating on repeated runs.
    """
    start_time = time.monotonic()
    settings = settings or get_settings()
    mimic_path = mimic_path or settings.mimic_hosp_dir
    cache_dir = settings.evaluation_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Select admissions and load ground truth
    hadm_ids_path = cache_dir / "selected_hadm_ids.json"
    if hadm_ids_path.exists():
        hadm_ids = json.loads(hadm_ids_path.read_text())
        logger.info("Loaded %d cached admission IDs", len(hadm_ids))
    else:
        hadm_ids = select_admissions(mimic_path, n=n_admissions)
        hadm_ids_path.write_text(json.dumps(hadm_ids, indent=2))

    ground_truths = load_ground_truth(mimic_path, hadm_ids)

    # Cache ground truth as JSON for the Streamlit viewer
    gt_cache_dir = cache_dir / "ground_truth"
    gt_cache_dir.mkdir(parents=True, exist_ok=True)
    for hadm_id, gt in ground_truths.items():
        gt_path = gt_cache_dir / f"{hadm_id}.json"
        if not gt_path.exists():
            gt_dict = {
                "hadm_id": gt.hadm_id,
                "diagnoses": [
                    {"icd_code": d.icd_code, "description": d.description, "seq_num": d.seq_num}
                    for d in gt.diagnoses
                ],
                "procedures": [
                    {"icd_code": p.icd_code, "description": p.description}
                    for p in gt.procedures
                ],
                "medications": [
                    {"drug": m.drug, "dose_val": m.dose or "", "dose_unit": "", "route": m.route or ""}
                    for m in gt.medications
                ],
            }
            gt_path.write_text(json.dumps(gt_dict, indent=2), encoding="utf-8")

    system_prompt = SYSTEM_PROMPT
    if prompt_file is not None:
        system_prompt = prompt_file.read_text(encoding="utf-8")
        logger.info("Using extraction system prompt from %s", prompt_file)

    cost_tracker = CostTracker(is_batch=batch)

    if batch:
        _check_batch_budget(cache_dir, sorted(ground_truths), model or DEFAULT_MODEL, max_cost)

    # Step 2: Generate synthetic notes
    logger.info("Generating synthetic notes...")
    if batch:
        notes = generate_batch_api(
            ground_truths=ground_truths,
            api_key=settings.anthropic_api_key,
            cache_dir=cache_dir,
            cost_tracker=cost_tracker,
        )
    else:
        notes = generate_batch(
            ground_truths=ground_truths,
            api_key=settings.anthropic_api_key,
            cache_dir=cache_dir,
        )

    # Step 3: Run extraction on each note
    logger.info("Running extraction pipeline on %d notes...", len(notes))

    retriever = None
    if use_rag:
        from clinical_pipeline.extraction.icd10_rag import ICD10Retriever

        code_table_path = settings.reference_dir / "icd10cm_codes_2025.txt"
        logger.info("Building ICD-10 retriever from %s...", code_table_path)
        retriever = ICD10Retriever(code_table_path)

    extractor = ClinicalExtractor(
        api_key=settings.anthropic_api_key,
        model=model or DEFAULT_MODEL,
        cost_tracker=cost_tracker,
        retriever=retriever,
        system_prompt=system_prompt,
    )

    extractions: dict[int, ClinicalExtraction] = {}
    failed: list[int] = []
    batch_requests: dict[str, dict] = {}

    for hadm_id, note_text in notes.items():
        # Check cache first
        cached = _load_cached_extraction(cache_dir, hadm_id)
        if cached is not None:
            extractions[hadm_id] = cached
            logger.info("Loaded cached extraction for hadm_id=%d", hadm_id)
            continue

        if batch:
            batch_requests[str(hadm_id)] = extractor.build_request_params(note_text)
            continue

        try:
            extraction = extractor.extract(
                note_id=str(hadm_id),
                transcription=note_text,
            )
            extractions[hadm_id] = extraction
            _save_extraction(cache_dir, hadm_id, extraction)
        except ExtractionError:
            logger.exception("Extraction failed for hadm_id=%d", hadm_id)
            failed.append(hadm_id)

    if batch_requests:
        messages = run_batch(extractor.client, batch_requests, label="extraction")
        for custom_id in batch_requests:
            hadm_id = int(custom_id)
            message = messages.get(custom_id)
            if message is None:
                failed.append(hadm_id)
                continue
            try:
                extraction = extractor._parse_response(message, custom_id)
            except Exception:
                logger.exception("Could not parse batch extraction for hadm_id=%d", hadm_id)
                failed.append(hadm_id)
                continue
            extractions[hadm_id] = extraction
            _save_extraction(cache_dir, hadm_id, extraction)
            cost_tracker.record(
                model=extractor.model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            )

    # Step 4: Score each extraction
    logger.info("Scoring %d extractions...", len(extractions))
    all_results: dict[int, list[EvaluationResult]] = {}

    for hadm_id, extraction in extractions.items():
        gt = ground_truths[hadm_id]
        results = evaluate_extraction(extraction, gt)
        all_results[hadm_id] = results

    # Step 5: Aggregate and save
    overall = aggregate_results(all_results)

    per_admission_dicts: dict[int, list[dict]] = {}
    for hadm_id, results in all_results.items():
        per_admission_dicts[hadm_id] = [
            {
                "entity_type": r.entity_type,
                "true_positives": r.true_positives,
                "false_positives": r.false_positives,
                "false_negatives": r.false_negatives,
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1": round(r.f1, 4),
                "details": [
                    {
                        "ground_truth_value": d.ground_truth_value,
                        "extracted_value": d.extracted_value,
                        "matched": d.matched,
                        "match_method": d.match_method,
                        "score": round(d.score, 2),
                    }
                    for d in r.details
                ],
            }
            for r in results
        ]

    summary = EvaluationSummary(
        n_admissions=len(extractions),
        overall=overall,
        per_admission=per_admission_dicts,
        failed_extractions=failed,
    )

    # Save results
    results_path = cache_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    logger.info("Evaluation results saved to %s", results_path)

    # Save cost report
    cost_tracker.save(cache_dir / "eval_cost_report.json")

    # Record run in history
    duration = time.monotonic() - start_time
    description = description or "evaluation run"
    if batch:
        description += " [batch]"
    if prompt_file is not None:
        description += f" [prompt: {prompt_file.name}]"
    tracker = RunTracker(cache_dir / "run_history.json")
    run_record = RunRecord(
        model=extractor.model,
        n_admissions=len(extractions),
        description=description,
        diagnoses=overall.get("diagnoses", {}),
        procedures=overall.get("procedures", {}),
        medications=overall.get("medications", {}),
        total_cost_usd=cost_tracker.estimated_cost(),
        duration_seconds=round(duration, 2),
    )
    tracker.record(run_record)

    # Log summary
    for entity_type, metrics in overall.items():
        logger.info(
            "%s — P=%.2f R=%.2f F1=%.2f (TP=%d FP=%d FN=%d)",
            entity_type,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["tp"],
            metrics["fp"],
            metrics["fn"],
        )

    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run synthetic note evaluation")
    parser.add_argument(
        "--mimic-path",
        type=Path,
        default=None,
        help="Path to MIMIC-IV hosp/ directory (default: hosp/ of the MIMIC_SOURCE root)",
    )
    parser.add_argument(
        "--n-admissions",
        type=int,
        default=30,
        help="Number of admissions to evaluate (default: 30)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Short note about what changed in this run (e.g. 'baseline with Haiku')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for extraction (e.g. 'claude-sonnet-4-5-20241022')",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable RAG with ICD-10 codebook for grounded code suggestions",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the Message Batches API (50%% discount, results can take minutes to hours)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Text file whose contents replace the extraction system prompt",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=DEFAULT_MAX_COST,
        help="Abort a --batch run if the estimated cost in USD exceeds this (default: 3.00)",
    )
    args = parser.parse_args()

    description = args.description
    if not description and args.use_rag:
        model_name = (args.model or DEFAULT_MODEL).split("-")[1].capitalize()
        description = f"{model_name} + ICD-10 RAG"

    run_evaluation(
        mimic_path=args.mimic_path,
        n_admissions=args.n_admissions,
        description=description,
        model=args.model,
        use_rag=args.use_rag,
        batch=args.batch,
        prompt_file=args.prompt_file,
        max_cost=args.max_cost,
    )
