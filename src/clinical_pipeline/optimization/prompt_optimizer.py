"""Systematically optimize extraction prompts using binary eval (Karpathy autoresearch)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from clinical_pipeline.feedback.feedback_store import FeedbackStore
from clinical_pipeline.optimization.mutations import MUTATIONS

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IterationRecord:
    """Result of a single optimization iteration."""

    iteration: int
    mutation_name: str
    mutation_description: str
    metrics: dict
    target_metric: str
    target_value: float
    best_value: float
    kept: bool
    prompt_snippet: str  # first 200 chars of the mutated prompt
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    duration_seconds: float = 0.0


@dataclass(slots=True)
class OptimizationHistory:
    """Full optimization run history."""

    baseline_metrics: dict
    best_metrics: dict
    best_prompt: str
    iterations: list[dict]
    total_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "baseline_metrics": self.baseline_metrics,
            "best_metrics": self.best_metrics,
            "best_prompt": self.best_prompt,
            "iterations": self.iterations,
            "total_duration_seconds": self.total_duration_seconds,
        }


class PromptOptimizer:
    """Systematically optimize extraction prompts using binary eval."""

    def __init__(
        self,
        eval_runner: Callable[[str], dict],
        feedback_store: FeedbackStore,
        prompt_history_path: Path,
        baseline_prompt: str,
    ) -> None:
        self.eval_runner = eval_runner
        self.feedback_store = feedback_store
        self.prompt_history_path = prompt_history_path
        self.baseline_prompt = baseline_prompt

    def optimize(
        self, n_iterations: int = 10, target_metric: str = "diagnoses_f1"
    ) -> dict:
        """Run the optimization loop.

        For each iteration:
        1. Generate a prompt mutation (informed by feedback data)
        2. Run binary eval on test set
        3. If F1 improves -> keep the mutation, save to history
        4. If F1 doesn't improve -> discard
        5. Log the result either way

        Returns the best prompt configuration found.
        """
        run_start = time.monotonic()
        logger.info("Starting optimization: %d iterations, target=%s", n_iterations, target_metric)

        # Evaluate baseline
        logger.info("Evaluating baseline prompt...")
        baseline_metrics = self._binary_eval(self.baseline_prompt)
        best_metrics = baseline_metrics
        best_prompt = self.baseline_prompt
        current_prompt = self.baseline_prompt

        iterations: list[dict] = []

        for i in range(n_iterations):
            iter_start = time.monotonic()
            logger.info("--- Iteration %d/%d ---", i + 1, n_iterations)

            # Generate mutation
            mutated_prompt, mutation_name, mutation_desc = self._generate_mutation(
                current_prompt, i
            )

            # Skip if mutation didn't change anything
            if mutated_prompt == current_prompt:
                logger.info(
                    "Mutation '%s' produced no change, skipping", mutation_name
                )
                record = IterationRecord(
                    iteration=i + 1,
                    mutation_name=mutation_name,
                    mutation_description=mutation_desc + " (no change)",
                    metrics=best_metrics,
                    target_metric=target_metric,
                    target_value=self._get_metric(best_metrics, target_metric),
                    best_value=self._get_metric(best_metrics, target_metric),
                    kept=False,
                    prompt_snippet=mutated_prompt[:200],
                    duration_seconds=round(time.monotonic() - iter_start, 2),
                )
                iterations.append(asdict(record))
                continue

            # Evaluate mutated prompt
            logger.info("Evaluating mutation: %s", mutation_name)
            new_metrics = self._binary_eval(mutated_prompt)
            iter_duration = round(time.monotonic() - iter_start, 2)

            # Binary decision: did the target metric improve?
            kept = self._is_improvement(new_metrics, best_metrics, target_metric)
            new_val = self._get_metric(new_metrics, target_metric)
            best_val = self._get_metric(best_metrics, target_metric)

            if kept:
                logger.info(
                    "KEPT: %s improved %s from %.4f to %.4f",
                    mutation_name, target_metric, best_val, new_val,
                )
                best_metrics = new_metrics
                best_prompt = mutated_prompt
                current_prompt = mutated_prompt
            else:
                logger.info(
                    "DISCARDED: %s — %s=%.4f (best=%.4f)",
                    mutation_name, target_metric, new_val, best_val,
                )

            record = IterationRecord(
                iteration=i + 1,
                mutation_name=mutation_name,
                mutation_description=mutation_desc,
                metrics=new_metrics,
                target_metric=target_metric,
                target_value=new_val,
                best_value=best_val if not kept else new_val,
                kept=kept,
                prompt_snippet=mutated_prompt[:200],
                duration_seconds=iter_duration,
            )
            iterations.append(asdict(record))

        total_duration = round(time.monotonic() - run_start, 2)

        # Save history
        history = OptimizationHistory(
            baseline_metrics=baseline_metrics,
            best_metrics=best_metrics,
            best_prompt=best_prompt,
            iterations=iterations,
            total_duration_seconds=total_duration,
        )
        self._save_history(history)

        # Save best prompt
        best_prompt_path = self.prompt_history_path.parent / "best_prompt.txt"
        best_prompt_path.write_text(best_prompt, encoding="utf-8")
        logger.info("Best prompt saved to %s", best_prompt_path)

        return {
            "baseline_metrics": baseline_metrics,
            "best_metrics": best_metrics,
            "best_prompt": best_prompt,
            "n_iterations": n_iterations,
            "n_kept": sum(1 for it in iterations if it["kept"]),
            "total_duration_seconds": total_duration,
        }

    def _generate_mutation(
        self, current_prompt: str, iteration: int
    ) -> tuple[str, str, str]:
        """Generate a prompt mutation and a description of what changed.

        Cycles through predefined mutation strategies, informed by HITL feedback.

        Returns (mutated_prompt, mutation_name, mutation_description).
        """
        mutation = MUTATIONS[iteration % len(MUTATIONS)]
        mutated = mutation["apply"](current_prompt, self.feedback_store)
        return mutated, mutation["name"], mutation["description"]

    def _binary_eval(self, prompt: str) -> dict:
        """Run evaluation and return metrics."""
        return self.eval_runner(prompt)

    def _is_improvement(
        self, new_metrics: dict, best_metrics: dict, target: str
    ) -> bool:
        """Binary pass/fail: did the target metric improve?"""
        new_val = self._get_metric(new_metrics, target)
        best_val = self._get_metric(best_metrics, target)
        return new_val > best_val

    @staticmethod
    def _get_metric(metrics: dict, target: str) -> float:
        """Extract a metric value like 'diagnoses_f1' from the metrics dict.

        Supports both flat keys ('diagnoses_f1') and nested keys
        (metrics['diagnoses']['f1']).
        """
        if target in metrics:
            return float(metrics[target])
        # Parse 'diagnoses_f1' -> metrics['diagnoses']['f1']
        parts = target.rsplit("_", 1)
        if len(parts) == 2:
            entity_type, metric_name = parts
            if entity_type in metrics and isinstance(metrics[entity_type], dict):
                return float(metrics[entity_type].get(metric_name, 0.0))
        return 0.0

    def _save_history(self, history: OptimizationHistory) -> None:
        """Save optimization history to JSON."""
        self.prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_history_path.write_text(
            json.dumps(history.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Optimization history saved to %s", self.prompt_history_path)
