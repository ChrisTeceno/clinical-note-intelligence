"""CRUD operations for clinical extractions."""

from __future__ import annotations

import uuid
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from clinical_pipeline.db.models import (
    ClinicalNote,
    DiagnosisRecord,
    Extraction,
    MedicationRecord,
    ProcedureRecord,
)
from clinical_pipeline.extraction.models import ClinicalExtraction


def save_note(session: Session, note_data: dict) -> ClinicalNote:
    """Insert or update a clinical note."""
    existing = session.execute(
        select(ClinicalNote).where(ClinicalNote.source_id == str(note_data["id"]))
    ).scalar_one_or_none()

    if existing is not None:
        existing.description = note_data.get("description")
        existing.medical_specialty = note_data.get("medical_specialty")
        existing.sample_name = note_data.get("sample_name")
        existing.transcription = note_data.get("transcription")
        existing.keywords = note_data.get("keywords")
        return existing

    note = ClinicalNote(
        source_id=str(note_data["id"]),
        description=note_data.get("description"),
        medical_specialty=note_data.get("medical_specialty"),
        sample_name=note_data.get("sample_name"),
        transcription=note_data.get("transcription"),
        keywords=note_data.get("keywords"),
    )
    session.add(note)
    session.flush()
    return note


def save_extraction(session: Session, note: ClinicalNote, extraction: ClinicalExtraction) -> Extraction:
    """Save a structured extraction linked to a clinical note.

    Replaces any existing extraction for the same note (idempotent).
    """
    existing = session.execute(
        select(Extraction).where(Extraction.note_id == note.id)
    ).scalar_one_or_none()
    if existing is not None:
        session.delete(existing)
        session.flush()

    ext = Extraction(
        note_id=note.id,
        chief_complaint=extraction.chief_complaint,
        medical_specialty=extraction.medical_specialty,
    )
    session.add(ext)
    session.flush()

    for dx in extraction.diagnoses:
        session.add(
            DiagnosisRecord(
                extraction_id=ext.id,
                name=dx.name,
                icd10_suggested=dx.icd10_suggestion,
                confidence=dx.confidence,
                evidence_span=dx.evidence_span,
            )
        )

    for proc in extraction.procedures:
        session.add(
            ProcedureRecord(
                extraction_id=ext.id,
                name=proc.name,
                cpt_suggestion=proc.cpt_suggestion,
                confidence=proc.confidence,
                evidence_span=proc.evidence_span,
            )
        )

    for med in extraction.medications:
        session.add(
            MedicationRecord(
                extraction_id=ext.id,
                name=med.name,
                dosage=med.dosage,
                frequency=med.frequency,
                route=med.route,
                confidence=med.confidence,
                evidence_span=med.evidence_span,
            )
        )

    session.flush()
    return ext


def get_pending_reviews(session: Session, limit: int = 20) -> list[Extraction]:
    """Get extractions pending human review, ordered by creation date."""
    stmt = (
        select(Extraction)
        .where(Extraction.status == "pending")
        .order_by(Extraction.created_at)
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())


def get_extraction_by_note(session: Session, note_id: uuid.UUID) -> Extraction | None:
    """Get the extraction for a specific note."""
    return session.execute(
        select(Extraction).where(Extraction.note_id == note_id)
    ).scalar_one_or_none()


def update_review_status(
    session: Session,
    extraction_id: uuid.UUID,
    status: Literal["pending", "in_review", "approved", "rejected"],
) -> Extraction | None:
    """Update the review status of an extraction."""
    from datetime import datetime, timezone

    ext = session.get(Extraction, extraction_id)
    if ext is None:
        return None
    ext.status = status
    if status in ("approved", "rejected"):
        ext.reviewed_at = datetime.now(timezone.utc)
    session.flush()
    return ext
