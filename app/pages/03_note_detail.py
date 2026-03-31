"""Note Detail — single-note deep dive with HITL review controls."""

import html
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from db import execute, query_df
from tooltips import TOOLTIP_CSS, tt

# Inject tooltip CSS once
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load ICD-10 reference table for code descriptions
# ---------------------------------------------------------------------------
ICD10_REF_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "reference" / "icd10cm_codes_2025.txt"
)
_icd10_descriptions: dict[str, str] = {}


@st.cache_data(ttl=3600)
def _load_icd10_descriptions() -> dict[str, str]:
    """Load ICD-10-CM code -> description mapping from CMS reference file."""
    desc_map: dict[str, str] = {}
    if not ICD10_REF_PATH.exists():
        return desc_map
    with open(ICD10_REF_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) < 8:
                continue
            code = line[:7].strip()
            description = line[7:].strip()
            if code and description:
                desc_map[code] = description
    return desc_map


_icd10_descriptions = _load_icd10_descriptions()

st.title("Note Detail")
st.caption(
    "Review a single extraction: view the clinical note, "
    "inspect extracted entities, and approve or reject."
)

# ---------------------------------------------------------------------------
# Note selector — supports query param from Review Queue links
# ---------------------------------------------------------------------------
notes_df = query_df("""
    SELECT cn.id, cn.source_id, cn.medical_specialty, cn.sample_name, e.status
    FROM clinical_notes cn
    JOIN extractions e ON e.note_id = cn.id
    ORDER BY e.created_at DESC
""")

if notes_df.empty:
    st.warning("No notes with extractions found.")
    st.stop()

note_labels = {
    row["id"]: (
        f"#{row['source_id']} — {row['medical_specialty']}"
        f" — {row['sample_name'] or 'Untitled'} [{row['status']}]"
    )
    for _, row in notes_df.iterrows()
}
note_ids = list(note_labels.keys())

# Check for query param from Review Queue link
preselected_index = 0
qp = st.query_params
if "note_id" in qp:
    linked_id = qp["note_id"]
    if linked_id in note_ids:
        preselected_index = note_ids.index(linked_id)

note_id = st.selectbox(
    "Select a note",
    options=note_ids,
    index=preselected_index,
    format_func=lambda x: note_labels[x],
)

if not note_id:
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
note_row = query_df("SELECT * FROM clinical_notes WHERE id = ?", (note_id,))
extraction_row = query_df("SELECT * FROM extractions WHERE note_id = ?", (note_id,))

if note_row.empty or extraction_row.empty:
    st.error("Note or extraction not found.")
    st.stop()

note = note_row.iloc[0]
extraction = extraction_row.iloc[0]
extraction_id = extraction["id"]

dx_df = query_df("SELECT * FROM diagnoses WHERE extraction_id = ?", (extraction_id,))
px_df = query_df("SELECT * FROM procedures WHERE extraction_id = ?", (extraction_id,))
rx_df = query_df("SELECT * FROM medications WHERE extraction_id = ?", (extraction_id,))

# ---------------------------------------------------------------------------
# Status badge + Note ID
# ---------------------------------------------------------------------------
status = extraction["status"]
status_colors = {
    "pending": "orange",
    "in_review": "blue",
    "approved": "green",
    "rejected": "red",
}
badge_color = status_colors.get(status, "gray")

STATUS_TOOLTIPS = {
    "pending": "Awaiting human review — no reviewer has assessed this extraction yet",
    "in_review": "Currently being reviewed by a clinician or coding specialist",
    "approved": "Reviewed and confirmed as accurate by a human reviewer",
    "rejected": (
        "Reviewed and flagged as inaccurate"
        " — needs re-extraction or manual correction"
    ),
}
status_badge = (
    f"<span style='background:{badge_color};color:white;"
    f"padding:2px 10px;border-radius:12px;font-size:0.85em'>"
    f"{status.upper()}</span>"
)
status_html = tt(status_badge, STATUS_TOOLTIPS.get(status, ""))

st.markdown(
    f"**Note ID:** `{note['source_id']}` &nbsp;&nbsp; "
    f"**Status:** {status_html} &nbsp;&nbsp; **Specialty:** {note['medical_specialty']}",
    unsafe_allow_html=True,
)

if extraction["chief_complaint"]:
    st.markdown(f"**Chief Complaint:** {extraction['chief_complaint']}")

st.divider()

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

# Confidence tooltip explanations
CONFIDENCE_TOOLTIPS = {
    "high": (
        "Explicitly stated and unambiguous in the clinical note"
        " — directly named with no uncertainty language."
    ),
    "medium": (
        "Strongly implied or uses clinical shorthand"
        " — e.g., abbreviated terminology, standard-of-care"
        " assumptions, or clinical context."
    ),
    "low": (
        "Mentioned in differential diagnosis, ruled-out context,"
        " or with uncertainty language"
        " (e.g., 'possible', 'rule out', 'cannot exclude')."
    ),
}


def _highlight_spans(text: str, spans: list[str]) -> str:
    """Wrap each evidence span occurrence in the transcription with a highlight."""
    escaped = html.escape(text)
    for span in spans:
        if not span:
            continue
        escaped_span = html.escape(span)
        escaped = escaped.replace(
            escaped_span,
            f"<mark style='background:#fff3cd;padding:1px 2px;border-radius:3px'>"
            f"{escaped_span}</mark>",
        )
    return escaped


with left:
    st.subheader("Clinical Note")

    # Gather all evidence spans for highlighting
    all_spans: list[str] = []
    for df in [dx_df, px_df, rx_df]:
        if not df.empty and "evidence_span" in df.columns:
            all_spans.extend(df["evidence_span"].dropna().tolist())

    # Sort longest first so longer spans get highlighted before substrings
    all_spans = sorted(set(all_spans), key=len, reverse=True)

    transcription = note["transcription"] or ""
    highlighted = _highlight_spans(transcription, all_spans)

    st.markdown(
        f"<div style='max-height:600px;overflow-y:auto;padding:12px;"
        f"border:1px solid #ddd;border-radius:8px;font-size:0.9em;"
        f"line-height:1.6;white-space:pre-wrap'>{highlighted}</div>",
        unsafe_allow_html=True,
    )


def _confidence_badge(conf: str) -> str:
    colors = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}
    bg = colors.get(conf, "#95a5a6")
    tip = CONFIDENCE_TOOLTIPS.get(conf, "")
    badge = (
        f"<span style='background:{bg};color:white;padding:1px 8px;"
        f"border-radius:10px;font-size:0.8em'>{conf}</span>"
    )
    return tt(badge, tip)


def _match_badge(match_type: str) -> str:
    colors = {"exact": "#2ecc71", "partial": "#f39c12", "none": "#e74c3c"}
    tooltips = {
        "exact": (
            "Claude's suggested ICD-10 code exactly matches"
            " a code in the CMS 2025 ICD-10-CM code table."
        ),
        "partial": (
            "Fuzzy matched by description text against CMS"
            " code descriptions (rapidfuzz WRatio score)."
        ),
        "none": (
            "No match found in the CMS ICD-10-CM code table."
            " Requires manual ICD-10 code assignment."
        ),
    }
    bg = colors.get(match_type, "#95a5a6")
    tip = tooltips.get(match_type, "")
    badge = (
        f"<span style='background:{bg};color:white;padding:1px 8px;"
        f"border-radius:10px;font-size:0.8em'>{match_type}</span>"
    )
    return tt(badge, tip)


def _icd10_badge(code: str) -> str:
    """Render an ICD-10 code with its CMS description as a hover tooltip."""
    normalized = code.strip().upper().replace(".", "")
    desc = _icd10_descriptions.get(normalized, "")
    code_html = f"<code>{html.escape(code)}</code>"
    if desc:
        return tt(code_html, f"{normalized}: {desc}")
    return code_html


with right:
    # --- Diagnoses ---
    st.subheader(f"Diagnoses ({len(dx_df)})")
    if not dx_df.empty:
        for _, d in dx_df.iterrows():
            icd = d["icd10_matched"] or d["icd10_suggested"] or "N/A"
            icd_html = _icd10_badge(icd) if icd != "N/A" else "<code>N/A</code>"
            score_text = ""
            if d["match_score"] and d["match_type"] == "partial":
                score_text = f" <small style='color:#888'>({d['match_score']:.0f}%)</small>"
            st.markdown(
                f"**{d['name']}** &mdash; {icd_html} "
                f"{_match_badge(d['match_type'] or 'none')}{score_text} "
                f"{_confidence_badge(d['confidence'])}  \n"
                f"<small style='color:#777'>Evidence: \"{d['evidence_span']}\"</small>",
                unsafe_allow_html=True,
            )
            st.markdown("")
    else:
        st.caption("No diagnoses extracted.")

    st.divider()

    # --- Procedures ---
    st.subheader(f"Procedures ({len(px_df)})")
    if not px_df.empty:
        for _, p in px_df.iterrows():
            cpt = p["cpt_suggestion"] or "N/A"
            st.markdown(
                f"**{p['name']}** &mdash; CPT `{cpt}` "
                f"{_confidence_badge(p['confidence'])}  \n"
                f"<small style='color:#777'>Evidence: \"{p['evidence_span']}\"</small>",
                unsafe_allow_html=True,
            )
            st.markdown("")
    else:
        st.caption("No procedures extracted.")

    st.divider()

    # --- Medications ---
    st.subheader(f"Medications ({len(rx_df)})")
    if not rx_df.empty:
        for _, m in rx_df.iterrows():
            details = " | ".join(
                filter(None, [m["dosage"], m["frequency"], m["route"]])
            )
            st.markdown(
                f"**{m['name']}** {f'({details})' if details else ''} "
                f"{_confidence_badge(m['confidence'])}  \n"
                f"<small style='color:#777'>Evidence: \"{m['evidence_span']}\"</small>",
                unsafe_allow_html=True,
            )
            st.markdown("")
    else:
        st.caption("No medications extracted.")

# ---------------------------------------------------------------------------
# HITL Review controls
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Review Actions")

review_col1, review_col2, review_col3 = st.columns([1, 1, 2])

with review_col1:
    if st.button("Approve", type="primary", use_container_width=True):
        now = datetime.now(UTC).isoformat()
        execute(
            "UPDATE extractions SET status = 'approved', reviewed_at = ? WHERE id = ?",
            (now, extraction_id),
        )
        st.cache_data.clear()
        st.success("Extraction approved.")
        st.rerun()

with review_col2:
    if st.button("Reject", type="secondary", use_container_width=True):
        now = datetime.now(UTC).isoformat()
        execute(
            "UPDATE extractions SET status = 'rejected', reviewed_at = ? WHERE id = ?",
            (now, extraction_id),
        )
        st.cache_data.clear()
        st.success("Extraction rejected.")
        st.rerun()

with review_col3:
    reviewer_notes = st.text_area(
        "Reviewer notes",
        placeholder="Optional notes about this extraction...",
        label_visibility="collapsed",
    )
