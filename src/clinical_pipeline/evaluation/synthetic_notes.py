"""Generate synthetic discharge summaries from structured MIMIC data using Claude."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anthropic

from clinical_pipeline.evaluation.ground_truth import GroundTruth

logger = logging.getLogger(__name__)

GENERATION_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096
RATE_LIMIT_DELAY = 0.5

NOTE_STYLES = [
    "standard discharge summary",
    "SOAP-format discharge note",
    "H&P-style discharge summary",
    "brief discharge summary",
    "detailed narrative discharge summary",
]

SYSTEM_PROMPT = """\
You are a clinical documentation specialist. Write realistic synthetic \
discharge summaries for training and evaluation purposes. \
Your notes should read like real hospital discharge summaries with natural \
clinical language, appropriate abbreviations, and realistic formatting.

CRITICAL RULES:
1. Mention ALL provided diagnoses naturally in the text — do NOT just list them.
2. Reference ALL procedures in an operative/procedural section.
3. Include ALL medications in a medication reconciliation or discharge medications section.
4. Use realistic clinical language, abbreviations (e.g., "pt", "h/o", "BID", "PO"), \
and paragraph structure.
5. Do NOT include any ICD codes, ICD-10 codes, or billing codes in the note.
6. Vary sentence structure and phrasing — avoid formulaic repetition.
7. Include realistic vital signs, lab values, and clinical course narrative.
8. The note should be 400-800 words depending on complexity.\
"""


def _build_admission_prompt(gt: GroundTruth, style: str) -> str:
    """Build the user prompt for generating a synthetic note from ground truth."""
    dx_lines = []
    for d in gt.diagnoses:
        dx_lines.append(f"  - {d.description} (seq #{d.seq_num})")

    proc_lines = []
    for p in gt.procedures:
        proc_lines.append(f"  - {p.description}")

    med_lines = []
    for m in gt.medications:
        parts = [m.drug]
        if m.dose:
            parts.append(m.dose)
        if m.route:
            parts.append(f"via {m.route}")
        med_lines.append(f"  - {' '.join(parts)}")

    sections = [
        f"Write a {style} for a hospital admission with the following clinical data.",
        "",
        "DIAGNOSES (mention all naturally in the clinical narrative):",
        *(dx_lines or ["  (none)"]),
        "",
        "PROCEDURES PERFORMED:",
        *(proc_lines or ["  (none)"]),
        "",
        "MEDICATIONS:",
        *(med_lines or ["  (none)"]),
        "",
        "Remember: Do NOT include any ICD codes or billing codes. "
        "Write the note as a clinician would, mentioning conditions by name "
        "in a natural clinical narrative.",
    ]
    return "\n".join(sections)


def generate_synthetic_note(
    gt: GroundTruth,
    client: anthropic.Anthropic,
    style: str = "standard discharge summary",
) -> str:
    """Generate a realistic discharge summary from structured MIMIC data.

    Parameters
    ----------
    gt:
        Ground truth data for a single admission.
    client:
        Anthropic API client.
    style:
        Clinical note style (varies for diversity).

    Returns
    -------
    The generated discharge summary text.
    """
    user_message = _build_admission_prompt(gt, style)
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def generate_batch(
    ground_truths: dict[int, GroundTruth],
    api_key: str,
    cache_dir: Path,
    rate_limit_delay: float = RATE_LIMIT_DELAY,
) -> dict[int, str]:
    """Generate synthetic notes for multiple admissions, with caching.

    Notes are cached to ``cache_dir/synthetic_notes/`` so repeated runs
    skip already-generated admissions.

    Returns
    -------
    dict mapping hadm_id -> synthetic note text
    """
    notes_dir = cache_dir / "synthetic_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(api_key=api_key)
    results: dict[int, str] = {}
    hadm_ids = sorted(ground_truths.keys())

    for i, hadm_id in enumerate(hadm_ids):
        cache_path = notes_dir / f"{hadm_id}.txt"

        # Use cached note if available
        if cache_path.exists():
            results[hadm_id] = cache_path.read_text(encoding="utf-8")
            logger.info("[%d/%d] Loaded cached note for hadm_id=%d", i + 1, len(hadm_ids), hadm_id)
            continue

        gt = ground_truths[hadm_id]
        style = NOTE_STYLES[i % len(NOTE_STYLES)]

        logger.info(
            "[%d/%d] Generating %s for hadm_id=%d (%d dx, %d proc, %d rx)",
            i + 1,
            len(hadm_ids),
            style,
            hadm_id,
            len(gt.diagnoses),
            len(gt.procedures),
            len(gt.medications),
        )

        try:
            note = generate_synthetic_note(gt, client, style)
            cache_path.write_text(note, encoding="utf-8")
            results[hadm_id] = note
        except Exception:
            logger.exception("Failed to generate note for hadm_id=%d", hadm_id)
            continue

        if i < len(hadm_ids) - 1:
            time.sleep(rate_limit_delay)

    # Save manifest
    manifest = {
        "total_generated": len(results),
        "hadm_ids": sorted(results.keys()),
    }
    manifest_path = notes_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info("Generated %d/%d synthetic notes", len(results), len(hadm_ids))
    return results
