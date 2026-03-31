"""Predefined prompt mutation strategies for the optimization loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clinical_pipeline.feedback.feedback_store import FeedbackStore


def _add_hitl_examples(prompt: str, feedback: FeedbackStore) -> str:
    """Add few-shot examples derived from reviewer corrections."""
    examples = feedback.get_few_shot_examples(n=3)
    if not examples:
        return prompt
    lines = [
        "",
        "Learn from these reviewer corrections to avoid repeating mistakes:",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"  Example {i}:")
        snippet = ex.get("note_snippet", "")
        if snippet:
            lines.append(f"    Note text: \"{snippet[:200]}\"")
        correct = ex.get("correct_extraction", {})
        if correct:
            name = correct.get("name", "")
            code = correct.get("icd10_suggestion", "")
            if name:
                lines.append(f"    Correct extraction: {name}" + (f" ({code})" if code else ""))
        mistake = ex.get("common_mistake", "")
        if mistake:
            lines.append(f"    Common mistake: {mistake}")
    return prompt + "\n" + "\n".join(lines)


def _add_error_warnings(prompt: str, feedback: FeedbackStore) -> str:
    """Add warnings about common extraction errors observed by reviewers."""
    summary = feedback.summary()
    errors = summary.get("common_errors", {})
    if not errors:
        return prompt
    top_errors = list(errors.items())[:5]
    lines = [
        "",
        "Common extraction errors to avoid (based on reviewer feedback):",
    ]
    for error_desc, count in top_errors:
        lines.append(f"  - {error_desc} (seen {count} time{'s' if count != 1 else ''})")
    return prompt + "\n" + "\n".join(lines)


def _specificity_boost(prompt: str, feedback: FeedbackStore) -> str:
    """Add instruction to prefer specific ICD-10 codes over broad categories."""
    addition = (
        "\n\n6. When suggesting ICD-10 codes, always prefer the most specific code "
        "available. Avoid using unspecified codes (ending in .9) when more specific "
        "information is present in the note. For example, prefer E11.65 (Type 2 "
        "diabetes with hyperglycemia) over E11.9 (Type 2 diabetes without "
        "complications) when hyperglycemia is documented."
    )
    return prompt + addition


def _evidence_strictness(prompt: str, feedback: FeedbackStore) -> str:
    """Require longer, more specific evidence spans."""
    addition = (
        "\n\n7. For evidence spans, include enough surrounding context to make the "
        "extraction unambiguous. A good evidence span is typically 10-50 words and "
        "includes the clinical finding plus its immediate context (e.g., the full "
        "sentence containing the diagnosis, not just the diagnosis name)."
    )
    return prompt + addition


def _reduce_hallucination(prompt: str, feedback: FeedbackStore) -> str:
    """Add stronger instruction against extracting entities not in the note."""
    addition = (
        "\n\n8. CRITICAL: Only extract entities that are explicitly documented in "
        "this specific note. Do NOT extract diagnoses from standard review-of-systems "
        "negations (e.g., do not extract 'chest pain' from 'denies chest pain'). "
        "Do NOT infer diagnoses from medications alone (e.g., do not infer "
        "'hypertension' just because lisinopril is listed). Do NOT extract entities "
        "from the plan/recommendations section unless they are confirmed findings."
    )
    return prompt + addition


MUTATIONS: list[dict] = [
    {
        "name": "add_hitl_examples",
        "description": "Add few-shot examples from reviewer corrections",
        "apply": _add_hitl_examples,
    },
    {
        "name": "add_error_warnings",
        "description": "Add warnings about common extraction errors",
        "apply": _add_error_warnings,
    },
    {
        "name": "specificity_boost",
        "description": "Add instruction to prefer specific ICD-10 codes over broad categories",
        "apply": _specificity_boost,
    },
    {
        "name": "evidence_strictness",
        "description": "Require longer, more specific evidence spans",
        "apply": _evidence_strictness,
    },
    {
        "name": "reduce_hallucination",
        "description": "Add stronger instruction against extracting entities not in the note",
        "apply": _reduce_hallucination,
    },
]
