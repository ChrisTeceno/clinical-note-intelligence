# Clinical Note Intelligence Pipeline

An end-to-end clinical data automation system that extracts structured medical information from clinical notes using LLMs, validates ICD-10 codes against CMS standards, and surfaces results through a human-in-the-loop review dashboard.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-clinical.christeceno.com-blue)](https://clinical.christeceno.com)
[![Python](https://img.shields.io/badge/Python-3.11+-informational)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)](https://spark.apache.org)
[![Claude API](https://img.shields.io/badge/Claude-Haiku%204.5-blueviolet)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Live Demo

**[https://clinical.christeceno.com](https://clinical.christeceno.com)**

---

## Architecture

```
Clinical Notes  -->  PySpark Ingestion  -->  Claude API Extraction  -->  ICD-10 Matching  -->  HITL Dashboard
(MTSamples CSV)      (clean + partition)     (tool_use / JSON)           (CMS 2025 codes)       (Streamlit)
                                                                                                      |
                                                                         MIMIC-IV Ground Truth  <-----+
                                                                              Evaluation
```

Data flows from raw clinical text through a PySpark ingestion step into Claude-powered structured extraction. Extracted entities are validated against the full CMS 2025 ICD-10-CM code set (74,260 codes) using fuzzy matching, then surfaced in a Streamlit dashboard for human reviewer action. A separate evaluation module scores extractions against MIMIC-IV demo ground truth, with results tracked across runs for experiment comparison.

---

## Key Features

- **PySpark data pipeline** — local mode with Databricks-compatible design; partitioned Parquet output
- **LLM extraction with structured output** — Claude Haiku 4.5 via `tool_use`; Pydantic-enforced schema with evidence spans, confidence levels, and ICD-10 suggestions
- **ICD-10-CM validation** — fuzzy matching against the full CMS 2025 release (74,260 codes) using `rapidfuzz`; exact, category-level, and name-based match strategies
- **HITL Streamlit dashboard** — seven-page review interface with evidence highlighting, inline editing, and audit logging
- **Evaluation framework** — P/R/F1 scoring against MIMIC-IV demo structured discharge data; Wilson score confidence intervals
- **Experiment tracking** — run history with model, cost, and metric deltas for prompt iteration

---

## Evaluation Results

Scored against MIMIC-IV demo discharge summaries (30 admissions). Synthetic discharge notes were generated from structured MIMIC-IV data (ICD codes, procedures, medications), then run through the extraction pipeline and scored against known ground truth.

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| Diagnoses | 0.870 | 0.766 | 0.815 |
| Procedures | 0.476 | 0.674 | 0.558 |
| Medications | 0.933 | 0.319 | 0.476 |

**Notes on interpretation:**
- Diagnosis extraction is the strongest signal; high precision means low hallucination rate.
- Procedure recall is limited by clinical note verbosity — many procedures are implied, not stated.
- Medication precision is very high (0.93); low recall reflects that discharge summaries reference only a subset of the full medication list present in structured data. Full-text clinical notes are expected to close this gap.
- Evaluation uses synthetic notes generated from structured data. Real MIMIC-IV clinical note evaluation is planned pending PhysioNet credentialing.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Data pipeline | PySpark 3.5 | Local mode; Databricks-ready |
| LLM extraction | Claude Haiku 4.5 (Anthropic API) | `tool_use` for schema-enforced JSON |
| Schema validation | Pydantic v2 | Extraction models with confidence + evidence spans |
| Storage | SQLAlchemy + SQLite / PostgreSQL | SQLite default; Postgres for production |
| Code validation | rapidfuzz | Fuzzy match against CMS ICD-10-CM 2025 |
| Dashboard | Streamlit | Multi-page HITL review interface |
| Visualization | Altair | Analytics and evaluation charts |
| Cloud pipeline | Databricks | Notebook-compatible pipeline design |

---

## Project Structure

```
clinical-note-intelligence/
├── src/clinical_pipeline/
│   ├── config.py              # pydantic-settings env configuration
│   ├── ingestion/             # PySpark CSV -> Parquet pipeline
│   ├── extraction/            # Claude API structured extraction
│   ├── coding/                # ICD-10 fuzzy matching and validation
│   ├── db/                    # SQLAlchemy ORM and persistence
│   ├── evaluation/            # MIMIC-IV ground truth scoring
│   └── audit/                 # Review action logging
├── app/
│   ├── main.py                # Streamlit entry point (7 pages)
│   └── pages/                 # Dashboard, Review Queue, Note Detail,
│                              # Analytics, Failed Extractions, Evaluation, Experiments
├── data/
│   ├── raw/                   # MTSamples CSV (gitignored)
│   ├── processed/             # Partitioned Parquet output
│   ├── reference/             # CMS ICD-10-CM code tables
│   └── evaluation/            # Evaluation results and run history
├── docs/                      # Model cards, QA audit, data governance
├── notebooks/                 # EDA and Databricks-compatible pipeline notebooks
└── tests/                     # pytest suite with fixtures
```

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/christeceno/clinical-note-intelligence.git
cd clinical-note-intelligence
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY

# 2. Install dependencies
make setup
source .venv/bin/activate

# 3. Add data (see Data section below)
# data/raw/mtsamples.csv       — download from Kaggle
# data/reference/              — CMS ICD-10-CM 2025 code tables

# 4. Run the pipeline
make ingest    # PySpark ingestion: CSV -> partitioned Parquet
make extract   # Claude extraction on a 50-note sample

# 5. Launch the dashboard
make run-app   # Opens at http://localhost:8501
```

**Data sources:**
- MTSamples: [Kaggle — Medical Transcriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) (CC0, ~5,000 notes)
- ICD-10-CM: [CMS 2025 Code Tables](https://www.cms.gov/medicare/coding-billing/icd-10-codes) (public domain)

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Dashboard** | KPI overview: notes processed, extraction counts, approval rate, extraction cost. Confidence distribution and specialty breakdown charts. |
| **Review Queue** | Notes pending human review, sorted by extraction confidence. Filter by specialty, status, or confidence tier. |
| **Note Detail** | Full note text alongside extracted diagnoses, procedures, and medications. Evidence spans highlighted inline. Inline edit + accept/reject actions with audit logging. |
| **Analytics** | Aggregate extraction quality metrics, review velocity, approval rate trends, and per-specialty performance breakdown. |
| **Failed Extractions** | Notes where Claude extraction failed or produced schema-invalid output. Shows error type and raw API response for debugging. |
| **Evaluation** | P/R/F1 scores against MIMIC-IV demo ground truth, per entity type. Per-admission breakdown with match method detail (exact code, category, fuzzy name). Wilson score confidence intervals. |
| **Experiments** | Run history table: model, extraction cost, diagnosis/procedure/medication F1, and experiment description. Tracks improvement across prompt iterations. |

---

## Why This Project

Built to demonstrate applied AI engineering at the intersection of clinical data automation and LLM-Ops. The pipeline addresses real-world challenges in clinical note processing: extracting structured entities from unstructured text, validating against authoritative code sets, and building the human-in-the-loop quality assurance that any production clinical AI system requires.

This is not a toy demo. The architecture mirrors production clinical NLP pipelines — PySpark for scale, Pydantic for schema enforcement, fuzzy ICD-10 matching against the real CMS code set, and an evaluation framework tied to credentialed ground-truth data (MIMIC-IV).

The clinical background behind this project is direct: I served as a medic in the USAF and worked as a registered nurse before moving into data science. I have read and written clinical notes. I understand what ICD-10 miscoding costs a health system, and I understand what a reviewer actually needs to trust an LLM extraction. That perspective shaped every design decision here.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Model Card — Claude Extraction](docs/model_cards/claude_extraction.md) | Intended use, limitations, prompt design, performance by specialty |
| [Model Card — ICD-10 Matcher](docs/model_cards/icd10_matcher.md) | Matching algorithm, match-rate metrics, known failure modes |
| [QA Audit Report](docs/qa_audit/qa_report.md) | Extraction quality analysis, error taxonomy, confidence interval methodology |
| [Data Governance](docs/governance/data_lineage.md) | Data lineage, PHI handling policy, MIMIC DUA compliance |

---

## Roadmap

- [ ] Full MIMIC-IV evaluation with real clinical notes (pending PhysioNet credentialing)
- [ ] Medical imaging integration — MIMIC-CXR chest X-ray classification with TorchXRayVision
- [ ] Databricks production deployment with Delta tables
- [ ] Prompt optimization experiments (Sonnet vs. Haiku, few-shot variants)
- [ ] Specialty-specific extraction prompts for high-volume specialties (Orthopedic, Cardiovascular, Gastroenterology)

---

## License

MIT
