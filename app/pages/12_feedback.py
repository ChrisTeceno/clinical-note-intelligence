"""Feedback Dashboard — view HITL correction statistics and few-shot examples."""

from pathlib import Path

import pandas as pd
import streamlit as st

from clinical_pipeline.feedback.feedback_store import FeedbackStore

st.title("Feedback Dashboard")
st.caption("Review HITL corrections: statistics, error patterns, and generated few-shot examples.")

# ---------------------------------------------------------------------------
# Load feedback store
# ---------------------------------------------------------------------------
FEEDBACK_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "feedback" / "corrections.json"
)
store = FeedbackStore(FEEDBACK_PATH)
summary = store.summary()
total = summary["total"]

if total == 0:
    st.info(
        "No feedback has been captured yet. "
        "Use the Note Detail page to edit, remove, or add entities."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Corrections", total)

by_action = summary.get("by_action", {})
col2.metric("Corrected", by_action.get("corrected", 0))
col3.metric("Added (missed)", by_action.get("added", 0))
col4.metric("Removed (false positive)", by_action.get("removed", 0))

# ---------------------------------------------------------------------------
# Corrections by entity type
# ---------------------------------------------------------------------------
st.subheader("Corrections by Entity Type")
by_type = summary.get("by_entity_type", {})
if by_type:
    type_df = pd.DataFrame(
        [{"Entity Type": k, "Count": v} for k, v in by_type.items()]
    )
    st.bar_chart(type_df.set_index("Entity Type"))
else:
    st.caption("No data yet.")

# ---------------------------------------------------------------------------
# Common error patterns
# ---------------------------------------------------------------------------
st.subheader("Common Error Patterns")
errors = summary.get("common_errors", {})
if errors:
    error_df = pd.DataFrame(
        [{"Error": k, "Count": v} for k, v in errors.items()]
    ).sort_values("Count", ascending=False)
    st.dataframe(error_df, use_container_width=True, hide_index=True)
else:
    st.caption("No error patterns detected yet.")

# ---------------------------------------------------------------------------
# Recent corrections table
# ---------------------------------------------------------------------------
st.subheader("Recent Corrections")
recent = store.get_corrections(limit=50)
if recent:
    rows = []
    for item in recent:
        original_name = item.original_value.get("name", "")
        corrected_name = (item.corrected_value or {}).get("name", "")
        rows.append({
            "Note ID": item.note_id,
            "Type": item.entity_type,
            "Action": item.action,
            "Original": original_name,
            "Corrected": corrected_name,
            "Reviewer": item.reviewer,
            "Timestamp": item.timestamp,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("No corrections recorded.")

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
st.subheader("Few-Shot Prompt Examples")
st.caption(
    "These examples are derived from reviewer corrections and can be injected "
    "into the extraction prompt to improve future accuracy."
)

if st.button("Generate Few-Shot Examples"):
    examples = store.get_few_shot_examples(n=5)
    if examples:
        st.json(examples)
    else:
        st.info("No corrected or added feedback items to generate examples from.")
