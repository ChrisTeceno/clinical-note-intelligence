# Clinical Note Intelligence Pipeline — Implementation Plan

> **Portfolio project for AI Engineering roles**
> Demonstrates: PySpark pipelines, LLM-based extraction (Claude API), CV/multimodal (chest X-rays), HITL validation, and clinical data governance.

---

## Goal

Build an end-to-end clinical data intelligence system that ingests unstructured clinical notes, extracts structured variables (diagnoses, procedures, medications, ICD-10 codes) using Claude, cross-references with medical imaging findings, and surfaces results through a human-in-the-loop validation interface — all documented with model cards and QA audit trails.

---

## Architecture Overview

```
MTSamples (CSV)  ──►  PySpark Ingestion  ──►  Delta Tables (parquet)
                            │
                            ▼
                   Claude API Extraction  ──►  Structured JSON  ──►  Postgres
                            │                                          │
                            ▼                                          ▼
                  ICD-10 Code Mapping  ◄──  CMS Code Tables      Streamlit HITL
                            │                                      Dashboard
                            ▼
              MIMIC-CXR Imaging  ──►  TorchXRayVision  ──►  Multimodal Merge
                            │
                            ▼
                   Model Cards + QA Audit Logs
```

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Data pipeline | PySpark (local mode + Databricks Free Edition) | Demonstrates distributed-ready pipeline design |
| LLM extraction | Claude API (Haiku 4.5 via Batch API) | Structured outputs, cost-effective (~$1/M input tokens, 50% batch discount) |
| Structured output | Pydantic + anthropic SDK | Schema-enforced JSON extraction |
| Storage | PostgreSQL (local) + Delta/Parquet files | Relational for app, columnar for analytics |
| Vision/CV | TorchXRayVision (open-source) | Pre-trained chest X-ray classifiers, no GPU required for inference |
| Frontend | Streamlit | Rapid HITL interface, no frontend framework overhead |
| Documentation | Markdown model cards, JSON audit logs | Industry-standard governance artifacts |

---

## Data Sources

| Source | Size | Access | Credentialing |
|--------|------|--------|---------------|
| [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) | ~5,000 clinical notes | Kaggle CSV download | None (CC0 public domain) |
| [CMS ICD-10 Code Tables](https://www.cms.gov/medicare/coding-billing/icd-10-codes) | ~72,000 codes | Direct download | None |
| [MIMIC-CXR v2.1.0](https://physionet.org/content/mimic-cxr/2.1.0/) | 377,110 chest X-rays + reports | PhysioNet | Required (~1 week approval) |

---

## Phase 1: Data Pipeline + LLM Extraction (MTSamples)

**Objective**: Build the core pipeline — ingest MTSamples, extract structured clinical variables with Claude, map to ICD-10 codes, store in Postgres. This phase is immediately buildable with zero credentialing wait.

**Timeline estimate**: 2 weeks

**Complexity**: Medium — known patterns, some investigation needed for prompt engineering.

### Tasks

#### 1.1 Project scaffolding and environment setup
- **Complexity**: Simple
- Initialize git repo, Python project structure, dependency management

**Files to create:**
```
clinical-note-intelligence/
├── .gitignore
├── .env.example                  # ANTHROPIC_API_KEY, POSTGRES_URL placeholders
├── pyproject.toml                # Project metadata, dependencies (uv/pip)
├── README.md                     # Project overview, setup instructions
├── Makefile                      # Common commands: setup, test, lint, run-pipeline
├── src/
│   └── clinical_pipeline/
│       ├── __init__.py
│       └── config.py             # Settings via pydantic-settings (env vars)
├── tests/
│   └── __init__.py
├── data/
│   └── raw/
│       └── .gitkeep              # MTSamples CSV goes here (gitignored)
├── notebooks/
│   └── 01_eda_mtsamples.ipynb    # Initial exploration notebook
└── docs/
    └── PLAN.md                   # This file (move here after scaffolding)
```

**Dependencies (pyproject.toml):**
```
pyspark >= 3.5
anthropic >= 0.40
pydantic >= 2.0
pydantic-settings >= 2.0
sqlalchemy >= 2.0
psycopg2-binary
streamlit >= 1.35
pandas
pytest
python-dotenv
```

#### 1.2 PySpark ingestion pipeline
- **Complexity**: Medium
- Read MTSamples CSV into PySpark DataFrame
- Clean and normalize: strip HTML, normalize whitespace, handle encoding
- Deduplicate notes (MTSamples has some near-duplicates)
- Schema enforcement with StructType
- Write to partitioned Parquet (partitioned by `medical_specialty`)
- Track data lineage (row counts, null rates, schema hash per run)

**Files to create:**
```
src/clinical_pipeline/
├── ingestion/
│   ├── __init__.py
│   ├── spark_session.py          # SparkSession factory (local vs. Databricks)
│   ├── schema.py                 # StructType definitions for MTSamples
│   ├── cleaners.py               # Text normalization UDFs
│   ├── ingest_mtsamples.py       # Main ingestion pipeline entry point
│   └── data_quality.py           # Row counts, null checks, schema validation
├── data/
│   └── processed/
│       └── .gitkeep              # Parquet output directory
tests/
├── test_cleaners.py              # Unit tests for text normalization
├── test_ingestion.py             # Integration test with small fixture CSV
└── fixtures/
    └── sample_mtsamples.csv      # 20-row test fixture
```

**Databricks notebook (for demonstrating cloud execution):**
```
notebooks/
└── 02_spark_ingestion.py         # Databricks-compatible script (.py notebook format)
```

#### 1.3 Claude-powered structured extraction
- **Complexity**: Medium-Complex
- Define Pydantic models for extraction targets (diagnoses, procedures, medications, ICD-10 suggestions)
- Build prompt templates with few-shot clinical examples
- Use Claude Haiku 4.5 with structured outputs (JSON schema enforcement)
- Implement Batch API pipeline for cost-effective bulk processing
- Add retry logic, rate limiting, error handling
- Store extraction results as JSON alongside Parquet records
- Log every API call: input hash, model, latency, token usage, output hash

**Files to create:**
```
src/clinical_pipeline/
├── extraction/
│   ├── __init__.py
│   ├── models.py                 # Pydantic models: ClinicalExtraction, Diagnosis, Procedure, Medication
│   ├── prompts.py                # Prompt templates + few-shot examples
│   ├── extractor.py              # Main extraction logic (single note + batch)
│   ├── batch_runner.py           # Anthropic Batch API orchestration
│   └── cost_tracker.py           # Token usage + cost logging per run
tests/
├── test_models.py                # Pydantic model validation tests
├── test_extractor.py             # Mock-based extraction tests
└── fixtures/
    └── sample_extractions.json   # Expected output fixtures
```

**Key Pydantic model (sketch):**
```python
class Diagnosis(BaseModel):
    name: str
    icd10_suggestion: str | None
    confidence: Literal["high", "medium", "low"]
    evidence_span: str  # Quoted text from the note

class ClinicalExtraction(BaseModel):
    note_id: str
    diagnoses: list[Diagnosis]
    procedures: list[Procedure]
    medications: list[Medication]
    chief_complaint: str | None
    medical_specialty: str | None
```

#### 1.4 ICD-10 code mapping and validation
- **Complexity**: Simple-Medium
- Download CMS ICD-10-CM code tables (annual release)
- Load into lookup DataFrame / Postgres table
- Fuzzy-match Claude's ICD-10 suggestions against official codes
- Score match quality (exact, partial, no match)
- Flag discrepancies for HITL review

**Files to create:**
```
src/clinical_pipeline/
├── coding/
│   ├── __init__.py
│   ├── icd10_loader.py           # Parse CMS code table files
│   ├── code_matcher.py           # Fuzzy matching logic (Levenshtein + semantic)
│   └── validation.py             # Match quality scoring
data/
└── reference/
    └── .gitkeep                  # ICD-10 code tables go here
tests/
└── test_code_matcher.py
```

#### 1.5 PostgreSQL schema and persistence
- **Complexity**: Simple
- Define SQLAlchemy ORM models mirroring the extraction schema
- Alembic migrations for schema versioning
- Upsert logic for idempotent pipeline reruns

**Files to create:**
```
src/clinical_pipeline/
├── db/
│   ├── __init__.py
│   ├── models.py                 # SQLAlchemy ORM models
│   ├── session.py                # Session factory
│   └── repository.py             # CRUD operations
alembic/
├── alembic.ini
├── env.py
└── versions/
    └── 001_initial_schema.py
```

### Dependencies
- None (this is the foundation phase)

### Risks
| Risk | Mitigation |
|------|------------|
| Claude extraction quality varies by specialty | Build specialty-specific few-shot examples; start with high-volume specialties (Orthopedic, Cardiovascular, Gastroenterology) |
| API costs exceed budget during development | Use Haiku 4.5 + Batch API (effective $0.50/M input tokens); test with 50-note subset before full run |
| MTSamples notes are short/templated vs. real clinical notes | Document this as a known limitation; show pipeline would generalize to longer notes |
| Databricks Free Edition limitations (no GPU, small clusters) | Design pipeline to run in local PySpark mode with a Databricks notebook as a demonstration artifact |

### Verification Criteria
- [ ] `make ingest` processes all ~5,000 MTSamples notes into partitioned Parquet
- [ ] Data quality report shows: row counts, null rates, specialty distribution
- [ ] `make extract --sample 50` runs Claude extraction on 50 notes with structured JSON output
- [ ] All extractions conform to Pydantic schema (zero validation errors)
- [ ] ICD-10 code matching achieves >70% exact or partial match rate on Claude suggestions
- [ ] PostgreSQL contains all extraction results, queryable via SQLAlchemy
- [ ] Total API cost for full 5,000-note run is documented and under $15
- [ ] All tests pass: `make test`

---

## Phase 2: HITL Validation Interface

**Objective**: Build a Streamlit dashboard where a clinical reviewer can validate, correct, and approve LLM-generated extractions. This is the "human-in-the-loop" differentiator that shows production awareness.

**Timeline estimate**: 1.5 weeks

**Complexity**: Medium

### Tasks

#### 2.1 Streamlit app scaffolding
- **Complexity**: Simple
- Multi-page Streamlit app with navigation
- Database connection to Postgres
- Session state management for reviewer workflow

**Files to create:**
```
app/
├── __init__.py
├── main.py                       # Streamlit entry point
├── pages/
│   ├── 01_review_queue.py        # Notes pending review
│   ├── 02_note_detail.py         # Single note review + edit
│   ├── 03_analytics.py           # Aggregate extraction metrics
│   └── 04_audit_log.py           # Full audit trail viewer
├── components/
│   ├── note_viewer.py            # Clinical note display with highlight spans
│   ├── extraction_editor.py      # Editable extraction fields
│   ├── icd10_lookup.py           # Searchable ICD-10 code picker
│   └── confidence_badge.py       # Visual confidence indicators
└── utils/
    ├── db.py                     # DB connection for Streamlit
    └── auth.py                   # Simple password gate (demo-grade)
```

#### 2.2 Review workflow
- **Complexity**: Medium
- Queue view: notes sorted by extraction confidence (lowest first)
- Detail view: side-by-side original note + extracted fields
- Inline editing of diagnoses, procedures, medications, ICD-10 codes
- Accept / Reject / Edit actions per extraction
- Reviewer comments field
- Status tracking: pending → in_review → approved / rejected

**Additional DB tables (Alembic migration):**
```
alembic/versions/
└── 002_review_workflow.py        # review_status, reviewer_id, reviewed_at, reviewer_notes columns
```

#### 2.3 Evidence highlighting
- **Complexity**: Medium
- Highlight `evidence_span` text within the original note
- Visual connection between extracted entity and source text
- Helps reviewer verify extraction accuracy quickly

#### 2.4 Audit trail
- **Complexity**: Simple
- Log every review action: who, what, when, before/after values
- Exportable as CSV/JSON for compliance documentation
- Dashboard page showing review velocity, agreement rates

**Files to create:**
```
src/clinical_pipeline/
├── audit/
│   ├── __init__.py
│   ├── logger.py                 # Audit event logger
│   └── models.py                 # AuditEvent SQLAlchemy model
alembic/versions/
└── 003_audit_trail.py
```

### Dependencies
- Phase 1 complete (extraction results in Postgres)

### Risks
| Risk | Mitigation |
|------|------------|
| Streamlit performance with large datasets | Paginate queries (20 notes per page); use `st.cache_data` aggressively |
| Review workflow state management complexity | Keep it simple: linear status progression, no concurrent editing |

### Verification Criteria
- [ ] Streamlit app launches with `streamlit run app/main.py`
- [ ] Review queue displays notes ordered by confidence score
- [ ] Reviewer can view original note alongside extracted fields
- [ ] Evidence spans are highlighted in the original note text
- [ ] Reviewer can edit any extracted field and save changes
- [ ] Accept/Reject actions update status in database
- [ ] Audit log captures all review actions with timestamps
- [ ] Analytics page shows: review velocity, approval rate, inter-rater agreement (if >1 reviewer)

---

## Phase 3: Medical Imaging (CV/Multimodal)

**Objective**: Integrate chest X-ray analysis using MIMIC-CXR data and TorchXRayVision. Cross-reference imaging findings with text-extracted diagnoses to demonstrate multimodal clinical intelligence.

**Timeline estimate**: 2 weeks

**Complexity**: Complex — requires credentialed data access, image processing pipeline, model integration.

**IMPORTANT**: Start PhysioNet credentialing application during Phase 1 (takes ~1 week for approval). This phase cannot begin until access is granted.

### Tasks

#### 3.1 PhysioNet data access setup
- **Complexity**: Simple (but calendar-time dependent)
- Complete CITI training (if not already done — clinical background may exempt some modules)
- Submit PhysioNet credentialing application
- Sign MIMIC-CXR data use agreement
- Download MIMIC-CXR-JPG (smaller, JPEG version — ~4.7GB vs. 500GB+ for full DICOM)

#### 3.2 Imaging ingestion pipeline
- **Complexity**: Medium
- Download and organize MIMIC-CXR-JPG subset (frontal PA/AP views only)
- Parse associated radiology reports
- Link reports to images via study/subject IDs
- Load metadata into Postgres

**Files to create:**
```
src/clinical_pipeline/
├── imaging/
│   ├── __init__.py
│   ├── mimic_cxr_loader.py       # MIMIC-CXR data loader + metadata parser
│   ├── image_preprocessor.py     # Resize, normalize, tensor conversion
│   ├── report_parser.py          # Radiology report section extractor
│   └── schema.py                 # Imaging data models
data/
└── imaging/
    └── .gitkeep                  # MIMIC-CXR data (gitignored, large)
tests/
├── test_image_preprocessor.py
└── fixtures/
    └── sample_cxr/               # 5 sample images for testing (synthetic or from demo set)
```

#### 3.3 TorchXRayVision classification
- **Complexity**: Medium
- Load pre-trained DenseNet model from TorchXRayVision
- Run inference on MIMIC-CXR images (CPU — no GPU needed for portfolio demo scale)
- Extract pathology predictions: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax
- Store predictions with confidence scores

**Files to create:**
```
src/clinical_pipeline/
├── imaging/
│   ├── classifier.py             # TorchXRayVision inference wrapper
│   ├── predictions.py            # Prediction result models + storage
│   └── batch_classify.py         # Batch inference runner
tests/
└── test_classifier.py            # Test with fixture images
```

#### 3.4 Multimodal cross-referencing
- **Complexity**: Complex
- For MIMIC-CXR cases: extract diagnoses from radiology reports using same Claude pipeline (Phase 1)
- Compare text-extracted diagnoses with vision model predictions
- Compute agreement metrics: concordance rate, discordance analysis
- Flag cases where text and image disagree (high clinical value)

**Files to create:**
```
src/clinical_pipeline/
├── multimodal/
│   ├── __init__.py
│   ├── cross_reference.py        # Text vs. image finding comparison
│   ├── agreement_metrics.py      # Cohen's kappa, concordance calculation
│   └── discordance_report.py     # Generate discordance analysis
notebooks/
└── 03_multimodal_analysis.ipynb  # Analysis notebook with visualizations
```

#### 3.5 Imaging in HITL interface
- **Complexity**: Medium
- Add imaging tab to Streamlit app
- Display chest X-ray alongside radiology report and model predictions
- Reviewer can confirm/override imaging classifications
- Side-by-side text extraction vs. imaging findings view

**Files to create:**
```
app/pages/
├── 05_imaging_review.py          # Image + report review page
app/components/
├── xray_viewer.py                # Image display with overlay annotations
└── findings_comparison.py        # Text vs. image findings side-by-side
```

### Dependencies
- Phase 1 complete (extraction pipeline reused for radiology reports)
- Phase 2 complete (HITL interface extended, not rebuilt)
- PhysioNet credentialing approved

### Risks
| Risk | Mitigation |
|------|------------|
| PhysioNet credentialing delayed | Submit application Day 1; Phase 3 is non-blocking for Phases 1-2 |
| MIMIC-CXR-JPG download is large (4.7GB) | Use a 1,000-image subset for development; document full-scale capability |
| TorchXRayVision model accuracy on MIMIC-CXR | This is a known benchmark — report published metrics; focus on the pipeline, not SOTA accuracy |
| CPU inference too slow for full dataset | Batch processing with progress tracking; 1,000 images is feasible on CPU in <1 hour |
| MIMIC data cannot be shared in public repo | All data paths gitignored; document exact download steps; provide synthetic fixtures for tests |

### Verification Criteria
- [ ] MIMIC-CXR-JPG subset (1,000 images) loaded and metadata in Postgres
- [ ] TorchXRayVision produces pathology predictions for all loaded images
- [ ] Radiology reports processed through Claude extraction pipeline
- [ ] Cross-reference analysis produces concordance/discordance metrics
- [ ] At least 3 example discordance cases documented with clinical interpretation
- [ ] Streamlit imaging review page displays X-ray + findings side-by-side
- [ ] No MIMIC data committed to repository (verified by `git status` + `.gitignore` audit)

---

## Phase 4: Documentation, Governance, and Polish

**Objective**: Create production-grade documentation that demonstrates awareness of clinical AI deployment requirements: model cards, QA audit reports, data governance, and a polished README that tells the portfolio story.

**Timeline estimate**: 1 week

**Complexity**: Medium — writing-intensive, requires synthesizing work from all prior phases.

### Tasks

#### 4.1 Model cards
- **Complexity**: Medium
- One model card per model used (Claude extraction, TorchXRayVision, ICD-10 matcher)
- Follow [Google Model Cards format](https://modelcards.withgoogle.com/about)
- Include: intended use, limitations, ethical considerations, performance metrics, training data provenance

**Files to create:**
```
docs/
├── model_cards/
│   ├── claude_extraction.md      # LLM extraction model card
│   ├── torchxrayvision.md        # Chest X-ray classifier model card
│   └── icd10_matcher.md          # Code matching algorithm card
```

#### 4.2 QA audit report
- **Complexity**: Medium
- Export HITL review data as structured report
- Calculate inter-annotator agreement (if applicable)
- Error taxonomy: what types of extraction errors does Claude make?
- Performance by medical specialty
- Sample size justification and confidence intervals

**Files to create:**
```
docs/
├── qa_audit/
│   ├── qa_report.md              # Main QA audit document
│   ├── error_taxonomy.md         # Categorized extraction errors
│   └── metrics_by_specialty.md   # Per-specialty performance breakdown
notebooks/
└── 04_qa_analysis.ipynb          # Reproducible QA metrics notebook
```

#### 4.3 Data governance documentation
- **Complexity**: Simple
- Data lineage diagram (source → transform → storage → output)
- PHI/PII handling policy (MTSamples is synthetic-ish; MIMIC is de-identified)
- Data retention and access control documentation
- MIMIC-CXR DUA compliance statement

**Files to create:**
```
docs/
├── governance/
│   ├── data_lineage.md           # Full data flow documentation
│   ├── phi_policy.md             # PHI/PII handling (even though data is de-identified)
│   └── mimic_dua_compliance.md   # MIMIC data use agreement compliance
```

#### 4.4 README and portfolio presentation
- **Complexity**: Simple
- Comprehensive README with architecture diagram, setup instructions, screenshots
- "Why this project" section tying clinical background to technical choices
- Demo video or GIF of Streamlit interface
- Clear setup instructions (one-command local dev environment)

**Files to update/create:**
```
README.md                         # Complete rewrite — portfolio-grade
docs/
├── architecture.md               # Detailed architecture documentation
├── setup.md                      # Step-by-step local development setup
└── screenshots/
    └── .gitkeep                  # Streamlit screenshots for README
```

#### 4.5 Databricks notebook polish
- **Complexity**: Simple
- Clean up Databricks-compatible notebooks from Phase 1
- Add markdown explanations, visualizations
- Ensure they run on Databricks Free Edition
- Export as .py and .html for portfolio display

**Files to update:**
```
notebooks/
├── 02_spark_ingestion.py         # Polished Databricks notebook
└── exports/
    └── 02_spark_ingestion.html   # HTML export for viewing without Databricks
```

### Dependencies
- Phases 1, 2, 3 complete (documentation synthesizes all prior work)
- HITL reviews completed (need review data for QA report)

### Risks
| Risk | Mitigation |
|------|------------|
| Documentation becomes stale vs. code | Write docs last; generate metrics programmatically where possible |
| Model card requires metrics not yet computed | Build metrics computation into Phase 1-3 verification; Phase 4 just formats them |

### Verification Criteria
- [ ] All three model cards follow Google Model Cards format with all required sections
- [ ] QA audit report includes quantitative metrics with confidence intervals
- [ ] Error taxonomy documents at least 5 distinct error categories with examples
- [ ] Data governance docs cover lineage, PHI policy, and MIMIC DUA compliance
- [ ] README includes: architecture diagram, setup instructions, screenshots, "why this project" narrative
- [ ] `make setup && make test && make run` works from a clean clone
- [ ] No secrets, credentials, or patient data in the repository
- [ ] Databricks notebook runs on Free Edition without modification

---

## Final Repository Structure

```
clinical-note-intelligence/
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
├── alembic/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
│       ├── 001_initial_schema.py
│       ├── 002_review_workflow.py
│       └── 003_audit_trail.py
├── app/
│   ├── main.py
│   ├── pages/
│   │   ├── 01_review_queue.py
│   │   ├── 02_note_detail.py
│   │   ├── 03_analytics.py
│   │   ├── 04_audit_log.py
│   │   └── 05_imaging_review.py
│   ├── components/
│   │   ├── note_viewer.py
│   │   ├── extraction_editor.py
│   │   ├── icd10_lookup.py
│   │   ├── confidence_badge.py
│   │   ├── xray_viewer.py
│   │   └── findings_comparison.py
│   └── utils/
│       ├── db.py
│       └── auth.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── reference/
│   └── imaging/
├── docs/
│   ├── PLAN.md
│   ├── architecture.md
│   ├── setup.md
│   ├── model_cards/
│   │   ├── claude_extraction.md
│   │   ├── torchxrayvision.md
│   │   └── icd10_matcher.md
│   ├── qa_audit/
│   │   ├── qa_report.md
│   │   ├── error_taxonomy.md
│   │   └── metrics_by_specialty.md
│   ├── governance/
│   │   ├── data_lineage.md
│   │   ├── phi_policy.md
│   │   └── mimic_dua_compliance.md
│   └── screenshots/
├── notebooks/
│   ├── 01_eda_mtsamples.ipynb
│   ├── 02_spark_ingestion.py
│   ├── 03_multimodal_analysis.ipynb
│   ├── 04_qa_analysis.ipynb
│   └── exports/
├── src/
│   └── clinical_pipeline/
│       ├── __init__.py
│       ├── config.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── spark_session.py
│       │   ├── schema.py
│       │   ├── cleaners.py
│       │   ├── ingest_mtsamples.py
│       │   └── data_quality.py
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── prompts.py
│       │   ├── extractor.py
│       │   ├── batch_runner.py
│       │   └── cost_tracker.py
│       ├── coding/
│       │   ├── __init__.py
│       │   ├── icd10_loader.py
│       │   ├── code_matcher.py
│       │   └── validation.py
│       ├── db/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── session.py
│       │   └── repository.py
│       ├── imaging/
│       │   ├── __init__.py
│       │   ├── mimic_cxr_loader.py
│       │   ├── image_preprocessor.py
│       │   ├── report_parser.py
│       │   ├── schema.py
│       │   ├── classifier.py
│       │   ├── predictions.py
│       │   └── batch_classify.py
│       ├── multimodal/
│       │   ├── __init__.py
│       │   ├── cross_reference.py
│       │   ├── agreement_metrics.py
│       │   └── discordance_report.py
│       └── audit/
│           ├── __init__.py
│           ├── logger.py
│           └── models.py
└── tests/
    ├── __init__.py
    ├── test_cleaners.py
    ├── test_ingestion.py
    ├── test_models.py
    ├── test_extractor.py
    ├── test_code_matcher.py
    ├── test_classifier.py
    ├── test_image_preprocessor.py
    └── fixtures/
        ├── sample_mtsamples.csv
        ├── sample_extractions.json
        └── sample_cxr/
```

---

## Dependency Graph

```
Phase 1 ─────────────────────────────► Phase 2 ──────────► Phase 4
  │  (Data pipeline + LLM extraction)    (HITL interface)    (Docs)
  │                                                            ▲
  │                                                            │
  └──► PhysioNet credentialing ──► Phase 3 ────────────────────┘
       (start Day 1, ~1 week)      (Imaging/multimodal)
```

**Parallelism opportunities:**
- Phase 1 tasks 1.1-1.2 are independent of 1.3-1.5 (ingestion vs. extraction can be built in parallel)
- PhysioNet credentialing application should be submitted on Day 1 (calendar-time blocker)
- Phase 2 can begin as soon as Phase 1.5 (Postgres schema) is done, even if extraction tuning continues
- Phase 4 documentation can begin incrementally during Phases 2-3

---

## Scope Assessment

**Overall size**: Large (6-7 weeks of focused work)

**Critical path**: Phase 1 → Phase 2 → Phase 4

Phase 3 (imaging) is on a parallel track gated by PhysioNet credentialing, and while it adds significant portfolio value, the project is demonstrable and impressive after Phases 1 + 2 alone.

---

## Open Questions

1. **Databricks Free Edition vs. local PySpark only?** — Free Edition has [restricted outbound internet access](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations) which may block downloading MTSamples or calling Claude API from within a notebook. May need to run pipeline locally and use Databricks notebooks only for the Spark demonstration portion with pre-uploaded data.

2. **Claude model choice for extraction** — Haiku 4.5 is cheapest ($0.50/M input with batch), but Sonnet 4.5/4.6 may produce meaningfully better clinical extractions. Run a 50-note comparison early in Phase 1 to decide.

3. **Postgres hosting for demo** — Local Postgres works for development. For portfolio demo, consider: (a) SQLite as zero-config alternative, (b) Supabase free tier, or (c) Docker Compose with Postgres.

4. **MIMIC-CXR access timeline** — If credentialing takes longer than expected, Phase 3 can use the [NIH ChestX-ray8 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) as a fallback (no credentialing required, but less prestigious).

5. **Test coverage target** — Recommend 80%+ on extraction and coding modules (core logic), lighter coverage on Streamlit UI code.

---

## Immediate Next Actions (Day 1)

1. Submit PhysioNet credentialing application (unblocks Phase 3)
2. Download MTSamples from Kaggle to `data/raw/`
3. Download CMS ICD-10-CM code tables to `data/reference/`
4. Run Phase 1, Task 1.1 (project scaffolding)
5. Begin EDA notebook on MTSamples (understand data quality, specialty distribution, note lengths)
