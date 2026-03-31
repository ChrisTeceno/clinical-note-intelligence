"""Clinical Note Intelligence Pipeline — Streamlit Dashboard."""

import sys
from pathlib import Path

# Ensure the app directory is importable for page scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="Clinical Note Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

dashboard = st.Page(
    "pages/01_dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True
)
review_queue = st.Page(
    "pages/02_review_queue.py", title="Review Queue", icon=":material/checklist:"
)
note_detail = st.Page(
    "pages/03_note_detail.py", title="Note Detail", icon=":material/description:"
)
analytics = st.Page(
    "pages/04_analytics.py", title="Analytics", icon=":material/analytics:"
)
failed = st.Page(
    "pages/05_failed_extractions.py", title="Failed Extractions", icon=":material/error:"
)
evaluation = st.Page(
    "pages/06_evaluation.py", title="Evaluation", icon=":material/science:"
)
experiments = st.Page(
    "pages/07_experiments.py", title="Experiments", icon=":material/labs:"
)
documentation = st.Page(
    "pages/08_documentation.py", title="Documentation", icon=":material/menu_book:"
)
imaging = st.Page(
    "pages/09_imaging.py", title="Imaging", icon=":material/image:"
)

pg = st.navigation([
    dashboard, review_queue, note_detail, analytics, failed, evaluation, experiments, documentation, imaging,
])

st.sidebar.markdown("---")
st.sidebar.caption("Clinical Note Intelligence v0.1")

pg.run()
