"""BatchExtractor: process multiple notes with rate limiting and progress tracking."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from clinical_pipeline.config import Settings, get_settings
from clinical_pipeline.extraction.cost_tracker import CostTracker
from clinical_pipeline.extraction.extractor import ClinicalExtractor, ExtractionError
from clinical_pipeline.extraction.models import ClinicalExtraction

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_DELAY = 0.5  # seconds between requests
DEFAULT_SAMPLE_SIZE = 50


class BatchExtractor:
    """Process multiple clinical notes with rate limiting and result persistence."""

    def __init__(
        self,
        extractor: ClinicalExtractor,
        cost_tracker: CostTracker,
        output_dir: Path,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    ) -> None:
        self.extractor = extractor
        self.cost_tracker = cost_tracker
        self.output_dir = output_dir
        self.rate_limit_delay = rate_limit_delay
        self.results: list[ClinicalExtraction] = []
        self.errors: list[dict[str, str]] = []

    def process_notes(
        self, notes: list[dict[str, str]], *, resume_from: int = 0
    ) -> list[ClinicalExtraction]:
        """Extract clinical data from a list of notes.

        Each note dict must have 'id' and 'transcription' keys.
        """
        total = len(notes)
        logger.info("Processing %d notes (starting from index %d)", total, resume_from)

        for i, note in enumerate(notes[resume_from:], start=resume_from):
            note_id = str(note["id"])
            logger.info("[%d/%d] Processing note %s", i + 1, total, note_id)

            try:
                extraction = self.extractor.extract(
                    note_id=note_id,
                    transcription=note["transcription"],
                )
                self.results.append(extraction)
                self._save_single_result(extraction)
            except ExtractionError as e:
                logger.error("Failed to extract note %s: %s", note_id, e)
                self.errors.append({"note_id": note_id, "error": str(e)})

            if i < total - 1:
                time.sleep(self.rate_limit_delay)

            # Save checkpoint every 25 notes
            if (i + 1) % 25 == 0:
                self._save_checkpoint(i + 1)

        self._save_final_results()
        return self.results

    def _save_single_result(self, extraction: ClinicalExtraction) -> None:
        """Save a single extraction result to its own JSON file."""
        note_dir = self.output_dir / "extractions"
        note_dir.mkdir(parents=True, exist_ok=True)
        path = note_dir / f"{extraction.note_id}.json"
        path.write_text(extraction.model_dump_json(indent=2))

    def _save_checkpoint(self, processed_count: int) -> None:
        """Save a checkpoint with progress info."""
        checkpoint = {
            "processed": processed_count,
            "successful": len(self.results),
            "errors": len(self.errors),
            "cost_summary": self.cost_tracker.summary(),
        }
        path = self.output_dir / "checkpoint.json"
        path.write_text(json.dumps(checkpoint, indent=2))
        logger.info("Checkpoint saved: %d processed, %d errors", processed_count, len(self.errors))

    def _save_final_results(self) -> None:
        """Save combined results, errors, and cost report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_path = self.output_dir / "all_extractions.json"
        results_data = [r.model_dump() for r in self.results]
        results_path.write_text(json.dumps(results_data, indent=2))

        if self.errors:
            errors_path = self.output_dir / "errors.json"
            errors_path.write_text(json.dumps(self.errors, indent=2))

        self.cost_tracker.save(self.output_dir / "cost_report.json")
        logger.info(
            "Batch complete: %d successful, %d errors. Cost: $%.4f",
            len(self.results),
            len(self.errors),
            self.cost_tracker.estimated_cost(),
        )


def load_notes_from_parquet(parquet_path: Path, sample_size: int | None = None) -> list[dict[str, str]]:
    """Load notes from parquet files into a list of dicts."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if sample_size is not None:
        df = df.head(sample_size)
    return df[["id", "transcription"]].dropna(subset=["transcription"]).to_dict("records")


def run_batch(settings: Settings | None = None, sample_size: int = DEFAULT_SAMPLE_SIZE) -> None:
    """Entry point for batch extraction."""
    settings = settings or get_settings()

    cost_tracker = CostTracker(is_batch=False)
    extractor = ClinicalExtractor(
        api_key=settings.anthropic_api_key,
        cost_tracker=cost_tracker,
    )

    parquet_path = settings.processed_dir / "mtsamples.parquet"
    notes = load_notes_from_parquet(parquet_path, sample_size=sample_size)

    output_dir = settings.processed_dir / "extraction_results"
    batch = BatchExtractor(
        extractor=extractor,
        cost_tracker=cost_tracker,
        output_dir=output_dir,
    )
    batch.process_notes(notes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_batch()
