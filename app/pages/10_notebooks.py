"""Notebooks — Databricks pipeline notebooks, auto-synced from workspace."""

import json
import time
from pathlib import Path

import streamlit as st

st.title("Databricks Notebooks")
st.caption(
    "Pipeline notebooks synced from Databricks. "
    "PySpark ingestion, Claude API extraction, and ICD-10 validation."
)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static" / "notebooks"
MANIFEST_PATH = STATIC_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------
if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    notebooks = {nb["name"]: nb for nb in manifest if nb.get("html_file")}
else:
    notebooks = {}

# Fallback: discover HTML files directly
if not notebooks:
    for html_file in sorted(STATIC_DIR.glob("*.html")):
        if html_file.name.startswith("_"):
            continue
        name = html_file.stem
        notebooks[name] = {
            "name": name,
            "html_file": html_file.name,
            "source_file": None,
        }

if not notebooks:
    st.warning(
        "No notebooks found. Run the sync script:\n\n"
        "```\npython scripts/sync_databricks_notebooks.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Notebook selector
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "03_full_pipeline_databricks": "Full Pipeline (Ingestion + Extraction + ICD-10)",
    "02_spark_ingestion_databricks": "PySpark Ingestion Only",
}

nb_keys = list(notebooks.keys())
selected = st.selectbox(
    "Select Notebook",
    options=nb_keys,
    format_func=lambda k: DISPLAY_NAMES.get(k, k),
)

nb_info = notebooks[selected]
html_file = nb_info.get("html_file", f"{selected}.html")

col1, col2 = st.columns([1, 1])
with col1:
    st.link_button(
        "Open in full page",
        f"https://clinical.christeceno.com/notebooks/{html_file}",
        use_container_width=True,
    )
with col2:
    if nb_info.get("path"):
        db_url = f"https://dbc-cfaa4306-63f1.cloud.databricks.com/#notebook/{nb_info.get('path', '')}"
        st.link_button("Open in Databricks", db_url, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Embed the rendered notebook
# ---------------------------------------------------------------------------
cache_bust = int(time.time())
st.components.v1.iframe(
    src=f"https://clinical.christeceno.com/notebooks/{html_file}?v={cache_bust}",
    height=900,
    scrolling=True,
)
