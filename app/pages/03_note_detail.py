"""Note Detail — single-note deep dive with HITL review controls."""

import html
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from db import execute, query_df
from tooltips import TOOLTIP_CSS, tt

from clinical_pipeline.feedback.feedback_store import FeedbackItem, FeedbackStore

# Inject tooltip CSS once
st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

# Scroll to top after auto-advance
if st.session_state.get("_scroll_to_top"):
    st.session_state["_scroll_to_top"] = False
    components.html(
        "<script>parent.document.querySelector('section.main').scrollTo(0, 0);</script>",
        height=0,
    )

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

# ---------------------------------------------------------------------------
# Feedback store
# ---------------------------------------------------------------------------
FEEDBACK_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "feedback" / "corrections.json"
)
_feedback_store = FeedbackStore(FEEDBACK_PATH)


def _record_feedback(
    note_id: str,
    entity_type: str,
    action: str,
    original: dict,
    corrected: dict | None,
    snippet: str,
) -> None:
    """Write a feedback item and show confirmation."""
    item = FeedbackItem(
        note_id=str(note_id),
        entity_type=entity_type,
        action=action,
        original_value=original,
        corrected_value=corrected,
        note_snippet=snippet[:500],
        timestamp=datetime.now(UTC).isoformat(),
        reviewer="dashboard_user",
    )
    _feedback_store.add(item)


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
    snippet = (transcription or "")[:500]

    # --- Diagnoses ---
    st.subheader(f"Diagnoses ({len(dx_df)})")
    if not dx_df.empty:
        for idx, d in dx_df.iterrows():
            icd = d["icd10_matched"] or d["icd10_suggested"] or "N/A"
            icd_html = _icd10_badge(icd) if icd != "N/A" else "<code>N/A</code>"
            score_text = ""
            if d["match_score"] and d["match_type"] == "partial":
                score_text = f" <small style='color:#888'>({d['match_score']:.0f}%)</small>"
            # Get the ICD-10 description for inline display
            icd_desc = ""
            if icd != "N/A":
                normalized = icd.strip().upper().replace(".", "")
                desc = _icd10_descriptions.get(normalized, "")
                if desc:
                    icd_desc = f"<br><small style='color:#999;margin-left:4px'>{html.escape(desc)}</small>"
            st.markdown(
                f"**{d['name']}** &mdash; {icd_html} "
                f"{_match_badge(d['match_type'] or 'none')}{score_text} "
                f"{_confidence_badge(d['confidence'])}{icd_desc}  \n"
                f"<small style='color:#777'>Evidence: \"{d['evidence_span']}\"</small>",
                unsafe_allow_html=True,
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                with st.popover("Edit", use_container_width=True):
                    icd_val = icd if icd != "N/A" else ""
                    normalized_current = icd_val.strip().upper().replace(".", "")
                    current_desc = _icd10_descriptions.get(normalized_current, "")

                    # Two columns: ICD code | Description
                    edit_c1, edit_c2 = st.columns([1, 2])

                    with edit_c1:
                        code_input = st.text_input(
                            "ICD-10 Code",
                            value=normalized_current,
                            key=f"dx_code_{idx}",
                            placeholder="e.g. I340",
                        )

                    with edit_c2:
                        desc_input = st.text_input(
                            "Description",
                            value=d["name"],
                            key=f"dx_desc_{idx}",
                            placeholder="e.g. mitral regurgitation",
                        )

                    # Search by code input
                    code_matches = []
                    if code_input and len(code_input) >= 2:
                        cl = code_input.strip().upper().replace(".", "")
                        code_matches = [
                            (c, desc) for c, desc in _icd10_descriptions.items()
                            if c.startswith(cl)
                        ]

                    # Search by description input
                    desc_matches = []
                    if desc_input and len(desc_input) >= 3:
                        words = [w.lower() for w in desc_input.split() if len(w) >= 2]
                        for c, desc in _icd10_descriptions.items():
                            if all(w in desc.lower() for w in words):
                                desc_matches.append((c, desc))
                            if len(desc_matches) >= 100:
                                break

                    # Combine and deduplicate, prefer code matches first
                    seen = set()
                    combined = []
                    for c, desc in code_matches + desc_matches:
                        if c not in seen:
                            combined.append((c, desc))
                            seen.add(c)

                    # Only show results when filtered enough
                    new_name = desc_input
                    new_icd = code_input.strip().upper().replace(".", "") if code_input else normalized_current

                    if combined and len(combined) <= 100:
                        options = [f"{c} — {desc}" for c, desc in combined]
                        selected = st.selectbox(
                            f"{len(combined)} matches",
                            options=options,
                            index=0,
                            key=f"dx_select_{idx}",
                        )
                        new_icd = selected.split(" — ")[0].strip()
                        new_desc = selected.split(" — ", 1)[1] if " — " in selected else ""
                        if new_icd != normalized_current:
                            st.markdown(
                                f"<small>Changing: <s>{normalized_current} ({current_desc})</s> → "
                                f"**{new_icd}** ({new_desc})</small>",
                                unsafe_allow_html=True,
                            )
                    elif combined:
                        st.caption(f"{len(combined)}+ results — keep typing to narrow down.")
                    else:
                        if current_desc:
                            st.caption(f"Current: **{normalized_current}** — {current_desc}")

                    if st.button("Save", key=f"dx_save_{idx}", type="primary"):
                        original = {
                            "name": d["name"],
                            "icd10_suggestion": icd,
                        }
                        corrected = {
                            "name": new_name,
                            "icd10_suggestion": new_icd,
                        }
                        _record_feedback(
                            note_id, "diagnosis", "corrected",
                            original, corrected, snippet,
                        )
                        st.success("Feedback captured")
            with btn_col2:
                if st.button("Remove", key=f"dx_rm_{idx}", use_container_width=True):
                    original = {"name": d["name"], "icd10_suggestion": icd}
                    _record_feedback(note_id, "diagnosis", "removed", original, None, snippet)
                    st.success("Feedback captured")
            st.markdown("")
    else:
        st.caption("No diagnoses extracted.")

    with st.expander("Add Missing Diagnosis"):
        add_dx_name = st.text_input("Diagnosis name", key="add_dx_name")
        add_dx_icd = st.text_input("ICD-10 code", key="add_dx_icd")
        if st.button("Add", key="add_dx_btn"):
            if add_dx_name:
                corrected = {"name": add_dx_name, "icd10_suggestion": add_dx_icd}
                _record_feedback(note_id, "diagnosis", "added", {}, corrected, snippet)
                st.success("Feedback captured")

    st.divider()

    # --- Procedures ---
    st.subheader(f"Procedures ({len(px_df)})")
    if not px_df.empty:
        for idx, p in px_df.iterrows():
            cpt = p["cpt_suggestion"] or "N/A"
            st.markdown(
                f"**{p['name']}** &mdash; CPT `{cpt}` "
                f"{_confidence_badge(p['confidence'])}  \n"
                f"<small style='color:#777'>Evidence: \"{p['evidence_span']}\"</small>",
                unsafe_allow_html=True,
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                with st.popover("Edit", use_container_width=True):
                    new_px_name = st.text_input("Name", value=p["name"], key=f"px_name_{idx}")
                    cpt_val = cpt if cpt != "N/A" else ""
                    new_cpt = st.text_input(
                        "CPT", value=cpt_val, key=f"px_cpt_{idx}"
                    )
                    if st.button("Save", key=f"px_save_{idx}"):
                        original = {
                            "name": p["name"],
                            "cpt_suggestion": cpt,
                        }
                        corrected = {
                            "name": new_px_name,
                            "cpt_suggestion": new_cpt,
                        }
                        _record_feedback(
                            note_id, "procedure", "corrected",
                            original, corrected, snippet,
                        )
                        st.success("Feedback captured")
            with btn_col2:
                if st.button("Remove", key=f"px_rm_{idx}", use_container_width=True):
                    original = {"name": p["name"], "cpt_suggestion": cpt}
                    _record_feedback(note_id, "procedure", "removed", original, None, snippet)
                    st.success("Feedback captured")
            st.markdown("")
    else:
        st.caption("No procedures extracted.")

    with st.expander("Add Missing Procedure"):
        add_px_name = st.text_input("Procedure name", key="add_px_name")
        add_px_cpt = st.text_input("CPT code", key="add_px_cpt")
        if st.button("Add", key="add_px_btn"):
            if add_px_name:
                corrected = {"name": add_px_name, "cpt_suggestion": add_px_cpt}
                _record_feedback(note_id, "procedure", "added", {}, corrected, snippet)
                st.success("Feedback captured")

    st.divider()

    # --- Medications ---
    st.subheader(f"Medications ({len(rx_df)})")
    if not rx_df.empty:
        for idx, m in rx_df.iterrows():
            details = " | ".join(
                filter(None, [m["dosage"], m["frequency"], m["route"]])
            )
            st.markdown(
                f"**{m['name']}** {f'({details})' if details else ''} "
                f"{_confidence_badge(m['confidence'])}  \n"
                f"<small style='color:#777'>Evidence: \"{m['evidence_span']}\"</small>",
                unsafe_allow_html=True,
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                with st.popover("Edit", use_container_width=True):
                    new_rx_name = st.text_input("Name", value=m["name"], key=f"rx_name_{idx}")
                    new_dosage = st.text_input(
                        "Dosage", value=m["dosage"] or "",
                        key=f"rx_dose_{idx}",
                    )
                    new_route = st.text_input(
                        "Route", value=m["route"] or "",
                        key=f"rx_route_{idx}",
                    )
                    if st.button("Save", key=f"rx_save_{idx}"):
                        original = {
                            "name": m["name"],
                            "dosage": m["dosage"],
                            "route": m["route"],
                        }
                        corrected = {
                            "name": new_rx_name,
                            "dosage": new_dosage,
                            "route": new_route,
                        }
                        _record_feedback(
                            note_id, "medication", "corrected",
                            original, corrected, snippet,
                        )
                        st.success("Feedback captured")
            with btn_col2:
                if st.button("Remove", key=f"rx_rm_{idx}", use_container_width=True):
                    original = {"name": m["name"], "dosage": m["dosage"], "route": m["route"]}
                    _record_feedback(note_id, "medication", "removed", original, None, snippet)
                    st.success("Feedback captured")
            st.markdown("")
    else:
        st.caption("No medications extracted.")

    with st.expander("Add Missing Medication"):
        add_rx_name = st.text_input("Medication name", key="add_rx_name")
        add_rx_dose = st.text_input("Dosage", key="add_rx_dose")
        add_rx_route = st.text_input("Route", key="add_rx_route")
        if st.button("Add", key="add_rx_btn"):
            if add_rx_name:
                corrected = {"name": add_rx_name, "dosage": add_rx_dose, "route": add_rx_route}
                _record_feedback(note_id, "medication", "added", {}, corrected, snippet)
                st.success("Feedback captured")

# ---------------------------------------------------------------------------
# HITL Review controls
# ---------------------------------------------------------------------------


def _advance_to_next_pending(current_note_id: str) -> None:
    """Set query params to the next pending note, or stay on current if none."""
    result = query_df(
        """
        SELECT cn.id
        FROM clinical_notes cn
        JOIN extractions e ON e.note_id = cn.id
        WHERE e.status = 'pending' AND cn.id != ?
        ORDER BY e.created_at DESC
        LIMIT 1
        """,
        (current_note_id,),
    )
    if not result.empty:
        st.query_params["note_id"] = result.iloc[0]["id"]
        st.session_state["_scroll_to_top"] = True


st.divider()
st.subheader("Review Actions")

review_col1, review_col2, review_col3 = st.columns([1, 1, 2])

with review_col1:
    if st.button(
        "Approve",
        type="primary",
        use_container_width=True,
        help="Mark this extraction as reviewed and correct. Status changes to 'approved' and the note moves out of the review queue.",
    ):
        now = datetime.now(UTC).isoformat()
        execute(
            "UPDATE extractions SET status = 'approved', reviewed_at = ? WHERE id = ?",
            (now, extraction_id),
        )
        # Record as confirmed feedback
        _record_feedback(
            note_id, "extraction", "confirmed",
            {"status": "pending", "dx_count": len(dx_df), "px_count": len(px_df), "rx_count": len(rx_df)},
            {"status": "approved"},
            (transcription or "")[:500],
        )
        st.cache_data.clear()
        st.toast("Extraction approved", icon="✅")
        _advance_to_next_pending(note_id)
        st.rerun()

with review_col2:
    if st.button(
        "Reject",
        type="secondary",
        use_container_width=True,
        help="Mark this extraction as incorrect. Status changes to 'rejected'. Use the Edit/Remove buttons above to record specific errors first.",
    ):
        now = datetime.now(UTC).isoformat()
        execute(
            "UPDATE extractions SET status = 'rejected', reviewed_at = ? WHERE id = ?",
            (now, extraction_id),
        )
        _record_feedback(
            note_id, "extraction", "removed",
            {"status": "pending", "dx_count": len(dx_df), "px_count": len(px_df), "rx_count": len(rx_df)},
            {"status": "rejected"},
            (transcription or "")[:500],
        )
        st.cache_data.clear()
        st.toast("Extraction rejected", icon="❌")
        _advance_to_next_pending(note_id)
        st.rerun()

with review_col3:
    reviewer_notes = st.text_area(
        "Reviewer notes",
        placeholder="Optional notes about this extraction...",
        label_visibility="collapsed",
    )
