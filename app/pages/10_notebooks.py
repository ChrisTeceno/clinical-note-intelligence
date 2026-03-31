"""Notebooks — Databricks pipeline notebooks."""

import time
from pathlib import Path

import streamlit as st

st.title("Databricks Notebooks")
st.caption("Full pipeline notebooks exported from Databricks. PySpark ingestion, Claude API extraction, and ICD-10 validation running end-to-end in the cloud.")

notebooks = {
    "Full Pipeline": {
        "file": "full_pipeline.html",
        "description": "End-to-end pipeline: PySpark ingestion, Claude API extraction (tool_use), and ICD-10 code validation — all running on Databricks serverless compute.",
    },
}

selected = st.selectbox("Select Notebook", list(notebooks.keys()))
info = notebooks[selected]
st.markdown(info["description"])

col1, col2 = st.columns([1, 1])
with col1:
    st.link_button(
        "Open in full page",
        f"https://clinical.christeceno.com/notebooks/{info['file']}",
        use_container_width=True,
    )

st.divider()

# Embed directly — the Databricks HTML renders with its own JS
cache_bust = int(time.time())
st.components.v1.iframe(
    src=f"https://clinical.christeceno.com/notebooks/{info['file']}?v={cache_bust}",
    height=900,
    scrolling=True,
)
