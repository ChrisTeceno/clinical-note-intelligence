"""Project Overview — landing page for the Clinical Note Intelligence Pipeline."""

import json
from pathlib import Path

import streamlit as st

st.title("Clinical Note Intelligence Pipeline")

# ---------------------------------------------------------------------------
# Plain English Overview
# ---------------------------------------------------------------------------
st.markdown("""
Every time a patient visits a hospital, doctors write detailed notes about what happened —
diagnoses, treatments, medications, and more. These notes are written in plain English,
which makes them easy for doctors to read but very hard for computers to work with.
Hospitals need this information in a structured, coded format (like ICD-10 codes) for
billing, research, and quality tracking. Today, trained medical coders read through these
notes manually and assign the right codes — a process that is slow, expensive, and
error-prone.

This project automates that process. It uses AI to read clinical notes, pull out the
important medical information (diagnoses, procedures, medications), suggest the correct
billing codes, and present everything in a review dashboard where a human expert can
verify the results before they are finalized. The goal is not to replace medical coders,
but to do 80% of the work so they can focus on the hard cases.
""")

st.divider()

# ---------------------------------------------------------------------------
# Technical Architecture
# ---------------------------------------------------------------------------
st.subheader("How It Works")

st.markdown("""
The pipeline has five stages, each handling a different part of the problem:
""")

# SVG Architecture Diagram
st.markdown("""
<svg viewBox="0 0 900 420" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;margin:0 auto;display:block">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64b5f6"/>
    </marker>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#f39c12"/>
    </marker>
  </defs>

  <!-- Stage 1 -->
  <rect x="20" y="30" width="150" height="70" rx="10" fill="#1e1e32" stroke="#3a3a55" stroke-width="1.5"/>
  <text x="95" y="55" text-anchor="middle" fill="#888" font-size="11" font-family="sans-serif">Stage 1</text>
  <text x="95" y="75" text-anchor="middle" fill="#e0e0e8" font-size="13" font-weight="600" font-family="sans-serif">PySpark Ingestion</text>
  <text x="95" y="118" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">5,000 clinical notes</text>
  <text x="95" y="132" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">cleaned &amp; deduplicated</text>

  <!-- Arrow 1-2 -->
  <line x1="170" y1="65" x2="208" y2="65" stroke="#64b5f6" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Stage 2 -->
  <rect x="210" y="30" width="150" height="70" rx="10" fill="#1e1e32" stroke="#64b5f6" stroke-width="1.5"/>
  <text x="285" y="55" text-anchor="middle" fill="#64b5f6" font-size="11" font-family="sans-serif">Stage 2</text>
  <text x="285" y="75" text-anchor="middle" fill="#e0e0e8" font-size="13" font-weight="600" font-family="sans-serif">Claude API Extraction</text>
  <text x="285" y="118" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">Diagnoses, procedures,</text>
  <text x="285" y="132" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">medications + ICD-10 codes</text>

  <!-- Arrow 2-3 -->
  <line x1="360" y1="65" x2="398" y2="65" stroke="#64b5f6" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Stage 3 -->
  <rect x="400" y="30" width="150" height="70" rx="10" fill="#1e1e32" stroke="#64b5f6" stroke-width="1.5"/>
  <text x="475" y="55" text-anchor="middle" fill="#64b5f6" font-size="11" font-family="sans-serif">Stage 3</text>
  <text x="475" y="75" text-anchor="middle" fill="#e0e0e8" font-size="13" font-weight="600" font-family="sans-serif">ICD-10 Validation</text>
  <text x="475" y="118" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">74,260 CMS codes</text>
  <text x="475" y="132" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">exact + fuzzy matching</text>

  <!-- Arrow 3-4 -->
  <line x1="550" y1="65" x2="588" y2="65" stroke="#64b5f6" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Stage 4 -->
  <rect x="590" y="30" width="150" height="70" rx="10" fill="#1e1e32" stroke="#2ecc71" stroke-width="1.5"/>
  <text x="665" y="55" text-anchor="middle" fill="#2ecc71" font-size="11" font-family="sans-serif">Stage 4</text>
  <text x="665" y="75" text-anchor="middle" fill="#e0e0e8" font-size="13" font-weight="600" font-family="sans-serif">HITL Review</text>
  <text x="665" y="118" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">Human reviews + approves</text>
  <text x="665" y="132" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">evidence-highlighted notes</text>

  <!-- Stage 5 - Medical Imaging -->
  <rect x="780" y="30" width="100" height="70" rx="10" fill="#1e1e32" stroke="#9b59b6" stroke-width="1.5"/>
  <text x="830" y="55" text-anchor="middle" fill="#9b59b6" font-size="11" font-family="sans-serif">Stage 5</text>
  <text x="830" y="75" text-anchor="middle" fill="#e0e0e8" font-size="12" font-weight="600" font-family="sans-serif">Imaging</text>
  <text x="830" y="118" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">Chest X-ray</text>
  <text x="830" y="132" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">classification</text>

  <line x1="740" y1="65" x2="778" y2="65" stroke="#9b59b6" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- RAG Enhancement -->
  <rect x="210" y="170" width="150" height="55" rx="10" fill="#1e1e32" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="285" y="193" text-anchor="middle" fill="#e67e22" font-size="11" font-family="sans-serif">RAG Enhancement</text>
  <text x="285" y="210" text-anchor="middle" fill="#888" font-size="10" font-family="sans-serif">TF-IDF retrieval from codebook</text>
  <line x1="285" y1="100" x2="285" y2="168" stroke="#e67e22" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arr2)"/>

  <!-- Evaluation Framework -->
  <rect x="400" y="170" width="190" height="55" rx="10" fill="#1e1e32" stroke="#f39c12" stroke-width="1.5"/>
  <text x="495" y="193" text-anchor="middle" fill="#f39c12" font-size="11" font-family="sans-serif">Evaluation Framework</text>
  <text x="495" y="210" text-anchor="middle" fill="#888" font-size="10" font-family="sans-serif">P/R/F1 vs MIMIC-IV ground truth</text>
  <line x1="475" y1="100" x2="475" y2="168" stroke="#f39c12" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arr2)"/>

  <!-- Experiment Tracking -->
  <rect x="620" y="170" width="140" height="55" rx="10" fill="#1e1e32" stroke="#3a3a55" stroke-width="1.5"/>
  <text x="690" y="193" text-anchor="middle" fill="#888" font-size="11" font-family="sans-serif">Experiment Tracking</text>
  <text x="690" y="210" text-anchor="middle" fill="#666" font-size="10" font-family="sans-serif">Model comparison over time</text>
  <line x1="590" y1="197" x2="618" y2="197" stroke="#3a3a55" stroke-width="1" stroke-dasharray="4,3"/>

  <!-- Metrics boxes -->
  <rect x="120" y="270" width="200" height="65" rx="8" fill="#1e1e32" stroke="#2ecc71" stroke-width="1"/>
  <text x="220" y="293" text-anchor="middle" fill="#2ecc71" font-size="20" font-weight="700" font-family="sans-serif">82.1%</text>
  <text x="220" y="315" text-anchor="middle" fill="#888" font-size="11" font-family="sans-serif">Diagnosis F1 (best run)</text>

  <rect x="350" y="270" width="200" height="65" rx="8" fill="#1e1e32" stroke="#f39c12" stroke-width="1"/>
  <text x="450" y="293" text-anchor="middle" fill="#f39c12" font-size="20" font-weight="700" font-family="sans-serif">71.8%</text>
  <text x="450" y="315" text-anchor="middle" fill="#888" font-size="11" font-family="sans-serif">ICD-10 Exact Match Rate</text>

  <rect x="580" y="270" width="200" height="65" rx="8" fill="#1e1e32" stroke="#9b59b6" stroke-width="1"/>
  <text x="680" y="293" text-anchor="middle" fill="#9b59b6" font-size="20" font-weight="700" font-family="sans-serif">0.711</text>
  <text x="680" y="315" text-anchor="middle" fill="#888" font-size="11" font-family="sans-serif">Best CXR AUC (DenseNet All)</text>

  <!-- Tech stack bar -->
  <rect x="20" y="365" width="860" height="40" rx="8" fill="#1e1e32" stroke="#3a3a55" stroke-width="1"/>
  <text x="55" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">PySpark</text>
  <text x="140" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">Claude API</text>
  <text x="240" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">Databricks</text>
  <text x="340" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">Streamlit</text>
  <text x="430" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">Pydantic</text>
  <text x="515" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">SQLAlchemy</text>
  <text x="615" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">TorchXRayVision</text>
  <text x="740" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">scikit-learn</text>
  <text x="840" y="390" fill="#64b5f6" font-size="11" font-weight="600" font-family="sans-serif">Altair</text>
</svg>
""", unsafe_allow_html=True)

st.markdown("""
1. **PySpark Ingestion** -- Clinical notes are loaded, cleaned (HTML removal, encoding fixes, whitespace normalization), deduplicated, and written to partitioned Delta/Parquet tables. Runs locally or on Databricks.

2. **Claude API Extraction** -- Each note is sent to Claude with a structured `tool_use` schema that guarantees valid JSON output. The model extracts diagnoses (with ICD-10 codes), procedures (with CPT codes), and medications (with dosage/route/frequency), plus evidence spans linking each finding back to the source text.

3. **ICD-10 Code Validation** -- Extracted codes are matched against the CMS 2025 ICD-10-CM code table (74,260 codes). Exact code lookup is tried first; unmatched codes fall back to fuzzy description matching via TF-IDF or rapidfuzz. Partial matches are flagged for human review.

4. **Human-in-the-Loop Review** -- A Streamlit dashboard displays each note alongside its extracted entities with highlighted evidence spans. Reviewers can approve, reject, or edit extractions. An audit trail logs every action.

5. **Medical Imaging** -- Chest X-rays from the NIH ChestX-ray14 dataset are classified using TorchXRayVision (6 DenseNet models) and Claude Vision, with per-pathology AUC evaluation against ground truth labels.
""")

st.divider()

# ---------------------------------------------------------------------------
# Current Status
# ---------------------------------------------------------------------------
st.subheader("Current Status")

status_data = [
    ("PySpark Data Pipeline", "Complete", "2,358 notes ingested, cleaned, deduplicated into partitioned Parquet/Delta"),
    ("Claude API Extraction", "Complete", "50 notes extracted with tool_use structured output, 0% parse failures"),
    ("ICD-10 Code Matching", "Complete", "71.8% exact match, 27.0% partial, 1.2% unmatched against CMS 2025"),
    ("HITL Review Dashboard", "Complete", "10 pages: review queue, note detail with evidence highlighting, analytics"),
    ("Evaluation Framework", "Complete", "P/R/F1 scoring against MIMIC-IV ground truth (30 synthetic admissions)"),
    ("Experiment Tracking", "Complete", "3 runs tracked: Haiku baseline, Sonnet 4, Haiku + RAG"),
    ("RAG Enhancement", "Complete", "TF-IDF retrieval from 74K ICD-10 codes; improved Rx F1 by +3.6%"),
    ("Medical Imaging", "Complete", "7 models compared on 500 NIH chest X-rays (best AUC: 0.711)"),
    ("Databricks Integration", "Complete", "Full pipeline notebook running on Databricks serverless"),
    ("Model Cards & Documentation", "Complete", "Google-format model cards, QA audit, data governance docs"),
]

for feature, status, detail in status_data:
    col1, col2, col3 = st.columns([2, 1, 4])
    col1.markdown(f"**{feature}**")
    if status == "Complete":
        col2.markdown(":green[Complete]")
    elif status == "In Progress":
        col2.markdown(":orange[In Progress]")
    else:
        col2.markdown(f":blue[{status}]")
    col3.markdown(f"<small style='color:#888'>{detail}</small>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Key Results
# ---------------------------------------------------------------------------
st.subheader("Key Results")

col1, col2, col3 = st.columns(3)

col1.markdown("**Clinical Note Extraction**")
col1.metric("Best Diagnosis F1", "82.1%", help="Sonnet 4 on 30 MIMIC-IV admissions")
col1.metric("Diagnosis Precision", "93.5%", help="Sonnet: when it extracts a diagnosis, it's almost always correct")
col1.metric("Medication Precision", "93.4%", help="Haiku + RAG")

col2.markdown("**ICD-10 Coding**")
col2.metric("Exact Match Rate", "71.8%", help="Against CMS 2025 ICD-10-CM code table (74,260 codes)")
col2.metric("Partial Match Rate", "27.0%", help="Fuzzy matched by description, flagged for review")
col2.metric("RAG Rx Improvement", "+3.6%", help="RAG improved medication F1 from 47.5% to 51.1%")

col3.markdown("**Medical Imaging**")
col3.metric("Best CXR AUC", "0.711", help="DenseNet-121 trained on all datasets (500 images, 14 pathologies)")
col3.metric("Best Pathology", "Cardiomegaly 0.819", help="AUC for cardiomegaly detection")
col3.metric("Models Compared", "7", help="6 TorchXRayVision + Claude Vision")

st.divider()

# ---------------------------------------------------------------------------
# Planned Updates
# ---------------------------------------------------------------------------
st.subheader("Planned Updates")

planned = [
    ("Real MIMIC-IV Note Evaluation", "Pending PhysioNet credentialing (CITI training complete, application submitted). Will replace synthetic notes with real discharge summaries for rigorous F1 evaluation."),
    ("MIMIC-CXR Imaging Integration", "Same PhysioNet credentialing unlocks 227K chest X-rays with paired radiology reports for multimodal cross-referencing."),
    ("Full 2,358 Note Extraction", "Scale from 50-note demo to full dataset using Anthropic Batch API for 50% cost reduction."),
    ("Prompt Optimization", "Systematic prompt engineering experiments with specialty-specific few-shot examples to improve per-specialty F1."),
    ("Semantic ICD-10 Matching", "Replace TF-IDF with embedding-based retrieval (SNOMED-to-ICD crosswalk) for clinically-validated code mapping."),
    ("CI/CD Pipeline", "GitHub Actions for automated testing, linting, and deployment on push."),
    ("PostgreSQL Migration", "Move from SQLite to Postgres for production-grade persistence."),
]

for title, detail in planned:
    with st.expander(title):
        st.markdown(detail)

st.divider()

# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------
st.subheader("Links")

link_col1, link_col2, link_col3 = st.columns(3)
link_col1.link_button("GitHub Repository", "https://github.com/ChrisTeceno/clinical-note-intelligence", use_container_width=True)
link_col2.link_button("Portfolio", "https://christeceno.com", use_container_width=True)
link_col3.link_button("LinkedIn", "https://linkedin.com/in/christeceno", use_container_width=True)
