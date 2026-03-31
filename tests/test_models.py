"""Tests for Pydantic model validation."""

import pytest
from pydantic import ValidationError

from clinical_pipeline.extraction.models import (
    ClinicalExtraction,
    Diagnosis,
    Medication,
    Procedure,
)


class TestDiagnosis:
    def test_valid_diagnosis(self):
        dx = Diagnosis(
            name="Hypertension",
            icd10_suggestion="I10",
            confidence="high",
            evidence_span="History of hypertension",
        )
        assert dx.name == "Hypertension"
        assert dx.icd10_suggestion == "I10"

    def test_optional_icd10(self):
        dx = Diagnosis(
            name="Unknown condition",
            confidence="low",
            evidence_span="possible unknown etiology",
        )
        assert dx.icd10_suggestion is None

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            Diagnosis(
                name="Test",
                confidence="very high",
                evidence_span="test span",
            )

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Diagnosis(name="Test")  # type: ignore[call-arg]


class TestProcedure:
    def test_valid_procedure(self):
        proc = Procedure(
            name="Knee arthroscopy",
            cpt_suggestion="29881",
            confidence="high",
            evidence_span="arthroscopic examination of the right knee",
        )
        assert proc.cpt_suggestion == "29881"

    def test_optional_cpt(self):
        proc = Procedure(
            name="Some procedure",
            confidence="medium",
            evidence_span="procedure was performed",
        )
        assert proc.cpt_suggestion is None


class TestMedication:
    def test_full_medication(self):
        med = Medication(
            name="Metformin",
            dosage="1000mg",
            frequency="twice daily",
            route="oral",
            confidence="high",
            evidence_span="metformin 1000mg twice daily",
        )
        assert med.dosage == "1000mg"
        assert med.route == "oral"

    def test_minimal_medication(self):
        med = Medication(
            name="Aspirin",
            confidence="medium",
            evidence_span="aspirin was prescribed",
        )
        assert med.dosage is None
        assert med.frequency is None
        assert med.route is None


class TestClinicalExtraction:
    def test_valid_extraction(self):
        ext = ClinicalExtraction(
            note_id="123",
            diagnoses=[
                Diagnosis(
                    name="Hypertension",
                    icd10_suggestion="I10",
                    confidence="high",
                    evidence_span="hypertension noted",
                )
            ],
            procedures=[],
            medications=[],
            chief_complaint="Chest pain",
            medical_specialty="Cardiovascular",
        )
        assert ext.note_id == "123"
        assert len(ext.diagnoses) == 1

    def test_empty_lists_by_default(self):
        ext = ClinicalExtraction(note_id="456")
        assert ext.diagnoses == []
        assert ext.procedures == []
        assert ext.medications == []
        assert ext.chief_complaint is None
        assert ext.medical_specialty is None

    def test_model_dump_roundtrip(self):
        ext = ClinicalExtraction(
            note_id="789",
            diagnoses=[
                Diagnosis(
                    name="Type 2 DM",
                    icd10_suggestion="E11.9",
                    confidence="high",
                    evidence_span="type 2 diabetes mellitus",
                )
            ],
            medications=[
                Medication(
                    name="Metformin",
                    dosage="500mg",
                    confidence="high",
                    evidence_span="metformin 500mg",
                )
            ],
        )
        data = ext.model_dump()
        roundtripped = ClinicalExtraction.model_validate(data)
        assert roundtripped == ext
