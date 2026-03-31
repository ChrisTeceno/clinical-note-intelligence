"""Pydantic models for structured clinical extraction."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    name: str = Field(description="Name of the diagnosis")
    icd10_suggestion: Optional[str] = Field(
        default=None,
        description="Suggested ICD-10-CM code (e.g. 'I25.10')",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this extraction"
    )
    evidence_span: str = Field(
        description="Verbatim text from the note supporting this diagnosis"
    )


class Procedure(BaseModel):
    name: str = Field(description="Name of the procedure performed or planned")
    cpt_suggestion: Optional[str] = Field(
        default=None,
        description="Suggested CPT code if identifiable",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this extraction"
    )
    evidence_span: str = Field(
        description="Verbatim text from the note supporting this procedure"
    )


class Medication(BaseModel):
    name: str = Field(description="Medication name")
    dosage: Optional[str] = Field(default=None, description="Dosage if mentioned")
    frequency: Optional[str] = Field(default=None, description="Frequency if mentioned")
    route: Optional[str] = Field(default=None, description="Route of administration if mentioned")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this extraction"
    )
    evidence_span: str = Field(
        description="Verbatim text from the note supporting this medication"
    )


class ClinicalExtraction(BaseModel):
    note_id: str = Field(description="Identifier of the source clinical note")
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    chief_complaint: Optional[str] = Field(
        default=None,
        description="Chief complaint or reason for visit",
    )
    medical_specialty: Optional[str] = Field(
        default=None,
        description="Medical specialty of this note",
    )
