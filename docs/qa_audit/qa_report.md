# QA Audit Report: Clinical Note Intelligence Pipeline

**Version:** 1.0
**Date:** 2026-03-30
**Evaluated by:** Automated evaluation harness (`src/clinical_pipeline/evaluation/run_eval.py`)
**Evaluation corpus:** MIMIC-IV demo, 30 admissions, synthetic discharge summaries

---

## Executive Summary

The pipeline achieves solid performance on diagnosis extraction (F1 0.815) and demonstrates high precision on medications it does capture (P 0.932). Procedure extraction and medication recall are the primary quality gaps. Neither is primarily a model failure: procedure false positives stem from clinical note narrative context (referenced vs. coded procedures), and medication false negatives reflect a structural mismatch between note coverage and the MAR ground truth. All findings require human-in-the-loop validation before downstream use.

**Zero extraction failures** were recorded across all 30 admissions (no API errors, no Pydantic validation failures).

---

## Pipeline Overview

```
MTSamples CSV
    |
    v
PySpark Ingestion (ingest_mtsamples.py)
    - Schema enforcement (MTSAMPLES_SCHEMA)
    - Text cleaning UDFs (whitespace, encoding)
    - Exact-duplicate removal on transcription
    - Parquet output partitioned by medical_specialty
    |
    v
Claude Extraction (ClinicalExtractor)
    - Model: claude-haiku-4-5-20251001
    - Mechanism: tool_use forced call to extract_clinical_data
    - Output validated by Pydantic ClinicalExtraction
    - Retry: up to 3 attempts on rate-limit / 5xx
    |
    v
ICD-10 Code Matching (CodeMatcher)
    - Step 1: exact lookup against CMS 2025 ICD-10-CM (74,260 codes)
    - Step 2: rapidfuzz WRatio >= 80 on description
    - needs_review=True for partial and none matches
    |
    v
HITL Review (Streamlit dashboard)
    - Coder reviews flagged extractions
    - Accepts, rejects, or corrects each entity
    - Approved codes written to database
```

---

## Quality Metrics

### Overall Extraction Performance

| Entity Type | Precision | Recall | F1 | TP | FP | FN |
|-------------|-----------|--------|----|----|-----|-----|
| Diagnoses | 0.870 | 0.766 | 0.815 | 393 | 59 | 120 |
| Procedures | 0.476 | 0.674 | 0.558 | 60 | 66 | 29 |
| Medications | 0.932 | 0.319 | 0.475 | 373 | 27 | 796 |

### 95% Confidence Intervals (normal approximation)

| Entity Type | Precision CI | Recall CI |
|-------------|-------------|-----------|
| Diagnoses | 0.870 ± 0.031 | 0.766 ± 0.037 |
| Procedures | 0.476 ± 0.087 | 0.674 ± 0.097 |
| Medications | 0.932 ± 0.025 | 0.319 ± 0.027 |

Procedure confidence intervals are wide due to the small sample size (n=126 for precision, n=89 for recall). These estimates should be treated as indicative, not definitive.

### Code Matching Performance (MTSamples run)

| Match Type | Rate | Review Required |
|------------|------|----------------|
| Exact | 71.8% | No |
| Partial (WRatio >= 80) | 27.0% | Yes |
| None | 1.2% | Yes (manual lookup) |

### Extraction Reliability

| Metric | Value |
|--------|-------|
| Admissions processed | 30 / 30 |
| Failed extractions | 0 |
| Pydantic validation errors | 0 |
| API retries triggered | Not instrumented in this run |

---

## Error Taxonomy

See `docs/qa_audit/error_taxonomy.md` for detailed categorization with examples.

### Diagnosis Errors

**FP-D1: Comorbidity over-extraction** (estimated ~40% of diagnosis FPs)
The model extracts comorbidities mentioned in the note that are not included in the final coded diagnosis list. This occurs because clinical notes document the full clinical picture while billing records reflect the principal encounter reason and DRG-relevant comorbidities. Example: a note mentioning `hypothyroidism` as background history produces a false positive if that condition was not coded.

**FP-D2: Lab finding extraction** (estimated ~20% of diagnosis FPs)
Abnormal lab values documented in notes (e.g., `elevated troponin`, `elevated BNP`, `hyperglycemia`) are extracted as diagnoses when they should be signs or findings only, not coded conditions. The system prompt instructs extraction of "explicit" diagnoses but the model treats documented labs as diagnostic entities.

**FP-D3: Differential diagnosis extraction** (estimated ~10% of diagnosis FPs)
Low-confidence extractions from differential diagnosis sections (e.g., `paranoid ideation`, `pericardial effusion` documented as rule-out) are occasionally included. The confidence field helps triage these: `low`-confidence extractions should receive mandatory review.

**FN-D1: Sequencing-dependent diagnoses** (majority of diagnosis FNs)
The 120 false negatives primarily represent secondary and complication codes (e.g., `Y838` — complications of surgical procedures) that are implied by the clinical narrative but not mentioned by name in the note text. The system prompt correctly restricts to explicit mentions, but this produces false negatives against the full ICD-10 coded record.

### Procedure Errors

**FP-P1: Contextually referenced procedures** (majority of procedure FPs)
The dominant error: procedures referenced in the clinical narrative as background or context are extracted even though they are not in the coded procedure record. Examples include `CABG` mentioned in cardiac history, `hemodialysis` referenced as a potential intervention, and vaccine administrations described in the clinical course. The note-to-coded-record gap is wide for procedures.

**FP-P2: Monitoring activities extracted as procedures**
Continuous monitoring activities (`continuous cardiac output monitoring`) are extracted as procedures. These are nursing/monitoring activities that do not typically appear in the ICD-10-PCS coded procedure list.

**FN-P1: Implicitly performed procedures**
Some procedures are implied by the note context but not named. These are correctly not extracted (the system prompt requires explicit mention) but contribute to the 29 false negatives.

### Medication Errors

**FN-Rx1: MAR vs. note coverage mismatch** (primary driver of 796 FNs)
The ground truth is derived from the MIMIC Medication Administration Record, which captures all medications ordered during the hospitalization. Clinical notes enumerate only medications material to the narrative — typically discharge medications, key inpatient medications, and notable adverse events. PRN medications, routine IV fluids, and electrolyte replacements are frequently omitted from note text but present in the MAR. This accounts for the bulk of the false negative count.

**FP-Rx1: Generic formulation variants**
A small number of medication false positives represent route/formulation variants (`normal saline`, `potassium supplementation`) that do not match a specific ground truth drug entry. These are typically over-matched against the broad MAR record.

### Code Matching Errors

**CM-1: Too-broad code accepted as exact match**
When Claude suggests a category-level code (e.g., `E11.9`) and the correct billable code is more specific (e.g., `E11.649`), the exact match succeeds but produces a code with less specificity than the ground truth. These exact matches pass through without a review flag.

**CM-2: WRatio false match on partial lookup**
Description matching can conflate similar-sounding conditions. Examples: `Type 1` vs. `Type 2` diabetes variants; laterality variants (`left` vs. `right`). All partial matches are flagged `needs_review=True` to catch these.

---

## Per-Specialty Performance Variation

Full specialty-level metrics are not available from this evaluation run (the MIMIC demo corpus is not stratified by specialty in the evaluation results). Based on per-admission analysis:

**Higher performance (inferred):** Admissions with cardiovascular, pulmonary, and internal medicine diagnoses show stronger diagnosis F1, likely because these specialties have well-established diagnosis naming conventions that align with ICD-10 descriptions.

**Lower performance (inferred):** Procedurally complex admissions (high procedure count) show worse procedure precision, consistent with the FP-P1 pattern described above. Complex admissions (20+ diagnoses) show more diagnosis false positives as the note accumulates more narrative context.

**Recommendation:** Run a specialty-stratified evaluation pass once a labeled specialty field is available in the evaluation corpus.

---

## Recommendations

| Priority | Recommendation | Rationale |
|----------|---------------|-----------|
| High | Validate on real clinical notes | Synthetic note circularity may inflate metrics |
| High | Confirm BAA/DPA with Anthropic before processing real PHI | Compliance requirement |
| High | Enforce HITL review for all partial code matches and low-confidence extractions | CM-2, FP-D3 error patterns |
| Medium | Add a confidence-based suppression rule for diagnosis FPs | Suppress `low` confidence extractions from automated pass-through |
| Medium | Clarify medication scope with operational team | MAR-vs-note mismatch inflates FN count; define the correct ground truth for the use case |
| Medium | Expand evaluation to 100+ admissions with specialty labels | Procedure CI is too wide at n=30 for production readiness decisions |
| Low | Add ICD-10-CM specificity validation post-match | Catch CM-1 (broad exact matches) |
| Low | Instrument API retry events | Currently not counted; needed for SLA monitoring |

---

## Methodology Notes

**Evaluation harness:** `src/clinical_pipeline/evaluation/run_eval.py` and `scorer.py`

**Match methods used in scoring:**
- `exact_code`: exact ICD-10 code match after normalization
- `category_code`: matching at the 3-character category level (score 80)
- `fuzzy_name`: rapidfuzz WRatio >= 80 on entity name vs. ground truth description
- `drug_name`: case-insensitive drug name match after salt/formulation normalization
- `none`: no match found

A predicted entity is counted as a true positive if it matches any ground truth entity for that admission via any of the above methods. Ground truth entities with no corresponding prediction are false negatives; predicted entities with no matching ground truth are false positives.
