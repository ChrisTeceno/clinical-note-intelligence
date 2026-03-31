"""SQLAlchemy ORM models for clinical note extractions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    medical_specialty: Mapped[str | None] = mapped_column(String(100), index=True)
    sample_name: Mapped[str | None] = mapped_column(String(255))
    transcription: Mapped[str | None] = mapped_column(Text)
    keywords: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    extraction: Mapped[Extraction | None] = relationship(back_populates="note", cascade="all, delete-orphan")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    note_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clinical_notes.id"), unique=True, index=True)
    chief_complaint: Mapped[str | None] = mapped_column(Text)
    medical_specialty: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(
        Enum("pending", "in_review", "approved", "rejected", name="review_status"),
        default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    note: Mapped[ClinicalNote] = relationship(back_populates="extraction")
    diagnoses: Mapped[list[DiagnosisRecord]] = relationship(back_populates="extraction", cascade="all, delete-orphan")
    procedures: Mapped[list[ProcedureRecord]] = relationship(back_populates="extraction", cascade="all, delete-orphan")
    medications: Mapped[list[MedicationRecord]] = relationship(back_populates="extraction", cascade="all, delete-orphan")


class DiagnosisRecord(Base):
    __tablename__ = "diagnoses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    extraction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extractions.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    icd10_suggested: Mapped[str | None] = mapped_column(String(20))
    icd10_matched: Mapped[str | None] = mapped_column(String(20))
    match_type: Mapped[str | None] = mapped_column(String(20))
    match_score: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(10))
    evidence_span: Mapped[str] = mapped_column(Text)

    extraction: Mapped[Extraction] = relationship(back_populates="diagnoses")


class ProcedureRecord(Base):
    __tablename__ = "procedures"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    extraction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extractions.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    cpt_suggestion: Mapped[str | None] = mapped_column(String(20))
    confidence: Mapped[str] = mapped_column(String(10))
    evidence_span: Mapped[str] = mapped_column(Text)

    extraction: Mapped[Extraction] = relationship(back_populates="procedures")


class MedicationRecord(Base):
    __tablename__ = "medications"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    extraction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extractions.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    dosage: Mapped[str | None] = mapped_column(String(100))
    frequency: Mapped[str | None] = mapped_column(String(100))
    route: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[str] = mapped_column(String(10))
    evidence_span: Mapped[str] = mapped_column(Text)

    extraction: Mapped[Extraction] = relationship(back_populates="medications")
