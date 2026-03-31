"""Documentation — model cards, QA audit, and data governance."""

import json
from pathlib import Path

import streamlit as st

st.title("Documentation")
st.caption("Model cards, QA audit reports, and data governance policies.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------
DOCS = {
    "Model Cards": {
        "Claude Extraction Pipeline": DOCS_DIR / "model_cards" / "claude_extraction.md",
        "ICD-10 Code Matcher": DOCS_DIR / "model_cards" / "icd10_matcher.md",
    },
    "QA Audit": {
        "Quality Assurance Report": DOCS_DIR / "qa_audit" / "qa_report.md",
        "Error Taxonomy": DOCS_DIR / "qa_audit" / "error_taxonomy.md",
    },
    "Data Governance": {
        "Data Lineage": DOCS_DIR / "governance" / "data_lineage.md",
        "PHI / PII Policy": DOCS_DIR / "governance" / "phi_policy.md",
    },
}

# ---------------------------------------------------------------------------
# Category selector
# ---------------------------------------------------------------------------
categories = list(DOCS.keys())
selected_category = st.selectbox("Category", categories)

docs_in_category = DOCS[selected_category]
doc_names = list(docs_in_category.keys())
selected_doc = st.selectbox("Document", doc_names)

st.divider()

# ---------------------------------------------------------------------------
# Render the selected document
# ---------------------------------------------------------------------------
doc_path = docs_in_category[selected_doc]

if doc_path.exists():
    content = doc_path.read_text(encoding="utf-8")
    st.markdown(content, unsafe_allow_html=True)
else:
    st.error(f"Document not found: {doc_path}")

# ---------------------------------------------------------------------------
# Dynamic model performance table from run history (for extraction model card)
# ---------------------------------------------------------------------------
if selected_doc == "Claude Extraction Pipeline":
    run_history_path = EVAL_DIR / "run_history.json"
    if run_history_path.exists():
        with open(run_history_path) as f:
            runs = json.load(f)

        # Filter to runs with actual results
        valid_runs = [r for r in runs if r.get("diagnoses", {}).get("f1", 0) > 0]

        if valid_runs:
            st.divider()
            st.subheader("All Evaluated Models (from Experiment History)")
            st.caption("This table updates automatically when new evaluation runs are recorded.")

            for run in sorted(valid_runs, key=lambda r: r.get("diagnoses", {}).get("f1", 0), reverse=True):
                dx = run.get("diagnoses", {})
                px = run.get("procedures", {})
                rx = run.get("medications", {})
                model = run.get("model", "unknown")
                desc = run.get("description", "")
                timestamp = run.get("timestamp", "")[:10]

                with st.expander(f"{model} — Dx F1: {dx.get('f1', 0):.1%} | {desc}", expanded=False):
                    st.markdown(f"**Run:** {desc}")
                    st.markdown(f"**Date:** {timestamp}")
                    st.markdown(f"**Admissions:** {run.get('n_admissions', '?')}")

                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("**Diagnoses**")
                        st.metric("F1", f"{dx.get('f1', 0):.1%}")
                        st.caption(f"P={dx.get('precision', 0):.1%}  R={dx.get('recall', 0):.1%}")
                        st.caption(f"TP={dx.get('tp', 0)}  FP={dx.get('fp', 0)}  FN={dx.get('fn', 0)}")
                    with cols[1]:
                        st.markdown("**Procedures**")
                        st.metric("F1", f"{px.get('f1', 0):.1%}")
                        st.caption(f"P={px.get('precision', 0):.1%}  R={px.get('recall', 0):.1%}")
                        st.caption(f"TP={px.get('tp', 0)}  FP={px.get('fp', 0)}  FN={px.get('fn', 0)}")
                    with cols[2]:
                        st.markdown("**Medications**")
                        st.metric("F1", f"{rx.get('f1', 0):.1%}")
                        st.caption(f"P={rx.get('precision', 0):.1%}  R={rx.get('recall', 0):.1%}")
                        st.caption(f"TP={rx.get('tp', 0)}  FP={rx.get('fp', 0)}  FN={rx.get('fn', 0)}")
