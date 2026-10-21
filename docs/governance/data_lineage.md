# Data Lineage

**Version:** 1.1
**Date:** 2026-10-21

This document traces each dataset used in the pipeline from its source through to its role in the system, including license terms and any transformations applied.

---

## Data Sources

### 1. MTSamples (Production Input)

| Attribute | Value |
|-----------|-------|
| Source | MTSamples.com transcription corpus via Kaggle |
| File | `data/raw/mtsamples.csv` |
| License | Creative Commons CC0 1.0 (public domain dedication) |
| Record count | ~4,999 records (raw) |
| Contains PHI | No — transcriptions are de-identified medical transcription samples |
| PHI basis | Synthetic/anonymized transcriptions from a medical transcription training dataset |

**Ingestion flow:**

```
data/raw/mtsamples.csv
    |
    v  PySpark (ingest_mtsamples.py)
    |  - Schema enforcement: description, medical_specialty, sample_name,
    |    transcription, keywords (all string)
    |  - Text cleaning UDFs: whitespace normalization, encoding fixes
    |  - Exact deduplication on transcription column
    |
    v
data/processed/mtsamples.parquet
    - Partitioned by medical_specialty
    - Immutable after write; re-ingestion overwrites
```

**Column provenance:**

| Output column | Source | Transformation |
|---------------|--------|----------------|
| `description` | CSV `description` | clean_text UDF |
| `medical_specialty` | CSV `medical_specialty` | clean_text UDF; used as partition key |
| `sample_name` | CSV `sample_name` | clean_text UDF |
| `transcription` | CSV `transcription` | clean_text UDF; used for deduplication |
| `keywords` | CSV `keywords` | clean_text UDF |

---

### 2. MIMIC-IV Clinical Database Demo (Evaluation Only, default source)

| Attribute | Value |
|-----------|-------|
| Source | PhysioNet — MIMIC-IV Clinical Database Demo v2.2 |
| URL | https://physionet.org/content/mimic-iv-demo/2.2/ |
| License | Open Data Commons Open Database License (ODbL) v1.0 |
| Access | Requires PhysioNet credentialing and acceptance of ODbL terms |
| Contains PHI | No — de-identified per HIPAA Safe Harbor (45 CFR §164.514(b)) |
| PHI basis | Dates shifted; ages capped at 89; patient identifiers replaced with random IDs |

**Usage in this project (evaluation only — not production input):**

```
MIMIC-IV demo hosp/ directory
    - diagnoses_icd.csv.gz    → ground truth diagnoses (ICD-10 only)
    - procedures_icd.csv.gz   → ground truth procedures (ICD-10-PCS)
    - prescriptions.csv.gz    → ground truth medications (prescriptions table)
    - d_icd_diagnoses.csv.gz  → ICD-10 code descriptions
    - d_icd_procedures.csv.gz → ICD-10-PCS code descriptions
    |
    v  ground_truth.py (load_ground_truth / select_admissions)
    |  - Filter to ICD-10 version records only
    |  - Stratified sample: 30 admissions (30% simple, 40% moderate, 30% complex)
    |  - Seed: 42 (reproducible selection)
    |
    v  synthetic_notes.py (generate_batch)
    |  - Claude Haiku 4.5 generates synthetic discharge summaries
    |  - Input: structured diagnosis/procedure/medication lists from MIMIC
    |  - Output: free-text notes (NO ICD codes included)
    |  - Cached to <evaluation_dir>/synthetic_notes/<hadm_id>.txt
    |    (data/evaluation/ for the demo)
    |
    v  run_eval.py
    |  - Extraction run on synthetic notes
    |  - Scored against MIMIC ground truth
    |
    v  data/evaluation/evaluation_results.json
```

The demo is selected with `MIMIC_SOURCE=demo` (the default). Its root is `MIMIC_DEMO_ROOT` (default `~/Downloads/mimic-iv-clinical-database-demo-2.2`).

**MIMIC data is not stored in the repository.** `.gitignore` excludes `*.csv.gz`, `*.csv` under `data/`, `physionet.org/` and `data/full/`. It does not exclude `data/` as a whole: demo-derived output under `data/evaluation/` and `data/optimization/` is tracked, except `synthetic_notes/`, `selected_hadm_ids.json` and `run_history.json` under `data/evaluation/`, which are ignored. Researchers must obtain MIMIC-IV demo access independently from PhysioNet and point `MIMIC_DEMO_ROOT` at the download.

---

### 2b. MIMIC-IV 3.1 Full Dataset (Evaluation Only, credentialed)

| Attribute | Value |
|-----------|-------|
| Source | PhysioNet, MIMIC-IV v3.1 |
| URL | https://physionet.org/content/mimiciv/3.1/ |
| License | PhysioNet Credentialed Health Data License; use is governed by the Data Use Agreement (DUA) |
| Access | Requires PhysioNet credentialing, completed training and a signed DUA |
| Contains PHI | De-identified per HIPAA Safe Harbor, but credentialed: redistribution is restricted by the DUA |
| Selection | `MIMIC_SOURCE=full`; root is `MIMIC_FULL_ROOT` (default `~/physionet.org/files/mimiciv/3.1`) |

**Usage in this project (evaluation only, local machine only):**

```
MIMIC-IV 3.1 hosp/ directory (read-only; icu/ is not read)
    - diagnoses_icd.csv.gz, procedures_icd.csv.gz, prescriptions.csv.gz
    - d_icd_diagnoses.csv.gz, d_icd_procedures.csv.gz
    |
    v  ground_truth.py (load_ground_truth / select_admissions)
    |  - Reads only the columns needed
    |  - Streams prescriptions in chunks, keeping rows for selected admissions
    |  - Same ICD-10 filter and stratified sampling as the demo
    |
    v  synthetic_notes.py (generate_batch), then run_eval.py / run_optimize.py
    |  - Notes are still synthetic; MIMIC-IV-Note is not used
    |
    v  data/full/evaluation/ and data/full/optimization/ (gitignored)
```

Full-data output is written under `data/full/`, never into the tracked demo caches. The Streamlit app shows only aggregate metrics in full mode (see `phi_policy.md`). `--mimic-path` is optional for `run_eval` and `run_optimize`; it defaults to the `hosp/` directory of the active source.

---

### 3. CMS ICD-10-CM 2025 Reference Data

| Attribute | Value |
|-----------|-------|
| Source | Centers for Medicare and Medicaid Services (CMS) |
| URL | https://www.cms.gov/medicare/coding-billing/icd-10-codes |
| Files | `icd10cm_codes_2025.txt`, `icd10cm_codes_addenda_2025.txt`, `icd10cm_order_2025.txt` |
| License | U.S. Government public domain (no copyright restriction) |
| Record count | 74,260 codes loaded into memory |
| Contains PHI | No |

**Ingestion flow:**

```
data/reference/icd10cm_codes_2025.txt
    |
    v  icd10_loader.py (load_from_cms_txt)
    |  - Fixed-width parse: columns 0-6 = code, column 7+ = description
    |  - Normalization: uppercase, dots stripped
    |  - is_header=True for codes with len <= 3 (category-level codes)
    |
    v  ICD10CodeTable (in-memory dict)
    |  - Exact lookup: O(1) by normalized code
    |  - Fuzzy search: over all (code, description) tuples via rapidfuzz
```

The reference files are included in the repository under `data/reference/` and committed to git. These are static public-domain files that change annually with CMS fiscal year updates.

---

### 4. Anthropic API (Claude Haiku 4.5)

| Attribute | Value |
|-----------|-------|
| Role | LLM inference for extraction and synthetic note generation |
| Model | `claude-haiku-4-5-20251001` |
| API key | Loaded from `.env` (`ANTHROPIC_API_KEY`); not committed to git |
| Data transmitted | Clinical note text (transcription string) |
| Data received | Structured JSON via tool_use response |

**Data flow to Anthropic:**

```
Clinical note text
    |
    v  HTTPS POST to api.anthropic.com/v1/messages
    |  - Input tokens: system prompt + user message (note text)
    |  - Output tokens: tool_use JSON block
    |
    v  ClinicalExtraction (returned to caller; not stored at Anthropic)
```

This project was developed and evaluated on synthetic/de-identified data. The API key and any data processing terms are governed by the Anthropic usage agreement applicable to the account. See `docs/governance/phi_policy.md` for PHI handling requirements before processing real patient notes.

---

## Data Flow Diagram

```
[MTSamples CSV]     [MIMIC-IV demo or full 3.1]     [CMS ICD-10-CM]
  (CC0, Kaggle)      (ODbL / PhysioNet DUA)        (public domain)
      |                        |                          |
      v                        v                          v
  PySpark                 ground_truth.py            icd10_loader.py
  Ingestion               + synthetic_notes.py       ICD10CodeTable
      |                        |                          |
      v                        v                          |
  Parquet               Synthetic notes                   |
  (processed/)          (eval cache)                      |
      |                        |                          |
      +------------+-----------+                          |
                   |                                      |
                   v                                      |
          ClinicalExtractor                               |
          (Claude Haiku 4.5)                              |
                   |                                      |
                   v                                      |
          ClinicalExtraction                              |
          (Pydantic model)                                |
                   |                                      |
                   +--------------------------------------+
                   |
                   v
             CodeMatcher
             (exact + fuzzy)
                   |
                   v
           CodeMatch results
           (needs_review flag)
                   |
                   v
          SQLite / PostgreSQL
          (clinical_notes.db)
                   |
                   v
          Streamlit HITL Dashboard
          (human review and approval)
```

---

## Retention and Deletion

| Data | Location | Retention |
|------|----------|-----------|
| MTSamples CSV | `data/raw/` | Gitignored; researcher-managed |
| Processed Parquet | `data/processed/` | Gitignored; re-derivable from CSV |
| MIMIC-IV demo | External (PhysioNet) | Not stored in repo |
| MIMIC-IV 3.1 full | External (PhysioNet, `MIMIC_FULL_ROOT`) | Not stored in repo; `*.csv.gz` and `physionet.org/` gitignored; DUA applies |
| Synthetic eval notes | `data/evaluation/synthetic_notes/` | Gitignored; re-derivable |
| Evaluation results (demo) | `data/evaluation/evaluation_results.json` | Not gitignored; no PHI |
| Full-data output | `data/full/` | Gitignored; local machine only |
| CMS reference files | `data/reference/` | Committed to git; public domain |
| Database | `data/clinical_notes.db` | Gitignored; contains extracted entities |

No real patient data is stored in the repository at any time.
