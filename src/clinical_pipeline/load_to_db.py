"""Load extraction results into the database with ICD-10 code matching."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from clinical_pipeline.coding.code_matcher import CodeMatcher
from clinical_pipeline.coding.icd10_loader import load_from_cms_txt
from clinical_pipeline.config import Settings, get_settings
from clinical_pipeline.db.models import Base
from clinical_pipeline.db.repository import save_extraction, save_note
from clinical_pipeline.db.session import get_engine, get_session_factory
from clinical_pipeline.extraction.models import ClinicalExtraction

logger = logging.getLogger(__name__)


def load_results(settings: Settings | None = None) -> None:
    settings = settings or get_settings()

    # Set up database
    engine = get_engine(settings.database_url)
    Base.metadata.create_all(engine)
    SessionFactory = get_session_factory(settings.database_url)

    # Load ICD-10 code table
    icd10_path = settings.reference_dir / "icd10cm_codes_2025.txt"
    code_table = load_from_cms_txt(icd10_path)
    matcher = CodeMatcher(code_table)

    # Load raw notes for context
    notes_df = pd.read_csv(settings.raw_dir / "mtsamples.csv")

    # Load extraction results
    results_dir = settings.processed_dir / "extraction_results" / "extractions"
    if not results_dir.exists():
        logger.error("No extraction results found at %s", results_dir)
        return

    extraction_files = sorted(results_dir.glob("*.json"))
    logger.info("Loading %d extraction results into database", len(extraction_files))

    loaded = 0
    matched_exact = 0
    matched_partial = 0
    matched_none = 0

    session = SessionFactory()
    try:
        for ext_file in extraction_files:
            data = json.loads(ext_file.read_text())
            extraction = ClinicalExtraction.model_validate(data)

            # Match ICD-10 codes and collect match metadata
            match_data = []
            for dx in extraction.diagnoses:
                match = matcher.match_code(dx.icd10_suggestion, dx.name)
                dx.icd10_suggestion = match.matched_code or dx.icd10_suggestion
                match_data.append(match)
                if match.match_type == "exact":
                    matched_exact += 1
                elif match.match_type == "partial":
                    matched_partial += 1
                else:
                    matched_none += 1

            # Find the raw note
            source_id = extraction.note_id
            note_row = notes_df[notes_df.index == int(source_id)]
            if note_row.empty:
                note_row = notes_df[notes_df.iloc[:, 0] == int(source_id)]

            note_data = {
                "id": source_id,
                "description": note_row["description"].iloc[0] if not note_row.empty else None,
                "medical_specialty": note_row["medical_specialty"].iloc[0] if not note_row.empty else None,
                "sample_name": note_row["sample_name"].iloc[0] if not note_row.empty else None,
                "transcription": note_row["transcription"].iloc[0] if not note_row.empty else None,
                "keywords": note_row["keywords"].iloc[0] if not note_row.empty else None,
            }

            db_note = save_note(session, note_data)
            db_ext = save_extraction(session, db_note, extraction)

            # Update diagnosis records with match metadata
            for dx_record, match in zip(db_ext.diagnoses, match_data):
                dx_record.icd10_matched = match.matched_code
                dx_record.match_type = match.match_type
                dx_record.match_score = match.score

            session.flush()
            loaded += 1

        session.commit()
        total_dx = matched_exact + matched_partial + matched_none
        logger.info(
            "Loaded %d extractions. ICD-10 matching: %d exact, %d partial, %d none (of %d total diagnoses)",
            loaded, matched_exact, matched_partial, matched_none, total_dx,
        )
        if total_dx > 0:
            logger.info(
                "Match rates: exact=%.1f%%, partial=%.1f%%, none=%.1f%%",
                100 * matched_exact / total_dx,
                100 * matched_partial / total_dx,
                100 * matched_none / total_dx,
            )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    load_results()
