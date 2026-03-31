"""Prompt templates and tool schema for clinical extraction."""

SYSTEM_PROMPT = """\
You are a clinical data extraction specialist. Your task is to extract structured \
medical information from clinical notes with high accuracy.

Use the extract_clinical_data tool to return your results.

Rules:
1. Only extract information explicitly stated in the note. Do not infer diagnoses \
that are not mentioned.
2. For each extraction, provide the exact verbatim text span from the note as evidence.
3. Assign confidence levels:
   - "high": explicitly stated and unambiguous
   - "medium": strongly implied or uses clinical shorthand
   - "low": mentioned in differential or uncertain context
4. For ICD-10 codes, suggest the most specific code you can. Use the format "X00.00". \
If unsure, provide the broader category code.
5. If a field has no relevant information in the note, return an empty list or null.

Example: For a note mentioning "Type 2 diabetes mellitus" in the assessment, extract \
a diagnosis with name="Type 2 diabetes mellitus", icd10_suggestion="E11.9", \
confidence="high", and evidence_span set to the verbatim text.\
"""

CONFIDENCE_ENUM = ["high", "medium", "low"]

EXTRACTION_TOOL = {
    "name": "extract_clinical_data",
    "description": (
        "Extract structured clinical data from a clinical note. "
        "Call this tool with all diagnoses, procedures, medications, "
        "chief complaint, and medical specialty found in the note."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnoses": {
                "type": "array",
                "description": "Diagnoses found in the note",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the diagnosis",
                        },
                        "icd10_suggestion": {
                            "type": ["string", "null"],
                            "description": "Suggested ICD-10-CM code (e.g. 'I25.10')",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": CONFIDENCE_ENUM,
                            "description": "Confidence in this extraction",
                        },
                        "evidence_span": {
                            "type": "string",
                            "description": "Verbatim text from the note supporting this diagnosis",
                        },
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "procedures": {
                "type": "array",
                "description": "Procedures performed or planned in the note",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the procedure performed or planned",
                        },
                        "cpt_suggestion": {
                            "type": ["string", "null"],
                            "description": "Suggested CPT code if identifiable",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": CONFIDENCE_ENUM,
                            "description": "Confidence in this extraction",
                        },
                        "evidence_span": {
                            "type": "string",
                            "description": "Verbatim text from the note supporting this procedure",
                        },
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "medications": {
                "type": "array",
                "description": "Medications mentioned in the note",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Medication name",
                        },
                        "dosage": {
                            "type": ["string", "null"],
                            "description": "Dosage if mentioned",
                        },
                        "frequency": {
                            "type": ["string", "null"],
                            "description": "Frequency if mentioned",
                        },
                        "route": {
                            "type": ["string", "null"],
                            "description": "Route of administration if mentioned",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": CONFIDENCE_ENUM,
                            "description": "Confidence in this extraction",
                        },
                        "evidence_span": {
                            "type": "string",
                            "description": "Verbatim text from the note supporting this medication",
                        },
                    },
                    "required": ["name", "confidence", "evidence_span"],
                },
            },
            "chief_complaint": {
                "type": ["string", "null"],
                "description": "Chief complaint or reason for visit, null if not stated",
            },
            "medical_specialty": {
                "type": ["string", "null"],
                "description": "Medical specialty of this note, null if not identifiable",
            },
        },
        "required": [
            "diagnoses",
            "procedures",
            "medications",
            "chief_complaint",
            "medical_specialty",
        ],
    },
}


def build_extraction_prompt(transcription: str) -> str:
    """Build the user message for a single note extraction."""
    return f"Extract structured clinical data from this note:\n\n{transcription}"
