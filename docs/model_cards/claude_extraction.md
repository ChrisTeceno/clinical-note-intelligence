# Model Card: Claude Clinical Extraction Pipeline

**Version:** 1.0
**Date:** 2026-03-30
**Format:** Google Model Cards (Mitchell et al., 2019)

---

## Model Details

### Basic Information

| Field | Value |
|-------|-------|
| Models Evaluated | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), Claude Sonnet 4 (`claude-4-sonnet-20250514`) |
| Provider | Anthropic |
| Output mechanism | `tool_use` with forced tool choice (`extract_clinical_data`) |
| Structured output validation | Pydantic v2 (`ClinicalExtraction`) |
| Max output tokens | 4,096 |
| Retry policy | Up to 3 attempts on rate-limit or 5xx errors |

### What the Model Extracts

Given a free-text clinical note, the pipeline returns a structured `ClinicalExtraction` object containing:

| Field | Type | Optional fields |
|-------|------|-----------------|
| `diagnoses` | List of `Diagnosis` | `icd10_suggestion`, `confidence`, `evidence_span` |
| `procedures` | List of `Procedure` | `cpt_suggestion`, `confidence`, `evidence_span` |
| `medications` | List of `Medication` | `dosage`, `frequency`, `route`, `confidence`, `evidence_span` |
| `chief_complaint` | String or null | — |
| `medical_specialty` | String or null | — |

Each extracted entity carries a confidence level (`high` / `medium` / `low`) and a verbatim `evidence_span` from the source note, enabling downstream human review.

### Confidence Level Definitions

| Level | Criterion |
|-------|-----------|
| `high` | Explicitly stated and unambiguous |
| `medium` | Strongly implied or uses clinical shorthand |
| `low` | Mentioned in differential or uncertain context |

### Prompting Approach

Zero-shot with a system prompt that instructs the model to:
- Extract only information explicitly stated in the note (no inference)
- Provide verbatim evidence spans for every extraction
- Suggest ICD-10-CM codes to the most specific level possible
- Return empty lists (not nulls) for entity types absent from the note

The `tool_use` mechanism with `tool_choice: {"type": "tool", "name": "extract_clinical_data"}` guarantees structured JSON output and eliminates parsing failures from free-text generation.

---

## Intended Use

### Primary Use Case

ICD-10 coding support for clinical documentation workflows. The pipeline surfaces candidate diagnoses, procedures, and medications with code suggestions to reduce manual chart review time for medical coders.

### Intended Users

- Medical coders reviewing clinical notes for billing
- Clinical informatics engineers building coding-assistance tools
- Researchers analyzing population health data from clinical notes

### Out-of-Scope Uses

- **Clinical decision support at point of care.** This pipeline is not validated for real-time treatment recommendations.
- **Autonomous coding without human review.** All outputs require human-in-the-loop (HITL) validation before submission to payers.
- **Legal or regulatory determination.** Extracted codes must be reviewed by a certified coder (CCS/CPC) before use in claims.
- **Notes outside standard discharge summary formats.** Performance on radiology reports, pathology notes, or operative reports has not been evaluated.

---

## Factors

### Medical Specialty

Performance varies by specialty. Notes from procedurally intensive specialties (cardiovascular surgery, orthopedics) tend to have more procedure false positives because the model extracts contextually referenced procedures that do not appear in the final coded record (e.g., procedures considered but not performed). Internal medicine and general medicine notes show the strongest diagnosis extraction F1.

### Note Length and Complexity

The evaluation corpus stratified admissions into three complexity tiers:
- **Simple:** 5–10 diagnoses
- **Moderate:** 11–20 diagnoses
- **Complex:** 20+ diagnoses

Proportional sampling: ~30% simple, ~40% moderate, ~30% complex.

### Note Style

Five note styles were evaluated: standard discharge summary, SOAP-format, H&P-style, brief summary, and detailed narrative. Style affects how medications are enumerated — brief summaries often omit medication reconciliation sections, driving the low medication recall.

### Medication Documentation Conventions

The primary medication recall gap reflects a structural mismatch: the ground truth is derived from the MIMIC-IV Medication Administration Record (MAR), which captures every medication event during hospitalization. Clinical notes typically enumerate only discharge medications or medications material to the admission narrative. This is expected behavior, not a model deficiency, but it inflates the measured false negative count.

---

## Metrics

Evaluated on 30 MIMIC-IV demo admissions (see Evaluation Data section). Confidence intervals use the normal approximation to the binomial proportion.

### Overall Performance — Claude Haiku 4.5

| Entity Type | Precision | Recall | F1 | 95% CI (Precision) | 95% CI (Recall) |
|-------------|-----------|--------|----|--------------------|-----------------|
| Diagnoses | 0.870 | 0.766 | 0.815 | ±0.031 | ±0.037 |
| Procedures | 0.476 | 0.674 | 0.558 | ±0.087 | ±0.097 |
| Medications | 0.932 | 0.319 | 0.475 | ±0.025 | ±0.027 |

### Overall Performance — Claude Sonnet 4

| Entity Type | Precision | Recall | F1 | Delta vs Haiku |
|-------------|-----------|--------|----|----|
| Diagnoses | 0.935 | 0.731 | 0.821 | F1 +0.6% |
| Procedures | 0.472 | 0.663 | 0.551 | F1 -0.7% |
| Medications | 0.952 | 0.308 | 0.465 | F1 -1.0% |

**Key finding:** Sonnet achieves higher precision across all entity types (+6.5% for diagnoses, +2.0% for medications) at the cost of slightly lower recall. This indicates Sonnet is more conservative — it extracts fewer entities but those it extracts are more likely correct. For coding support workflows where false positives create audit risk, Sonnet's precision advantage may justify the higher cost.

### Raw Counts (Haiku)

| Entity Type | TP | FP | FN |
|-------------|----|----|-----|
| Diagnoses | 393 | 59 | 120 |
| Procedures | 60 | 66 | 29 |
| Medications | 373 | 27 | 796 |

### Failure Modes

**Diagnoses — False Positives (59 total):** The model over-extracts comorbidities and incidental findings that appear in the note narrative but are not coded in the ground truth billing record. Common examples: `Hypothyroidism`, `Depression`, `Dysuria`, `Elevated troponin`, `Elevated BNP`. These are clinically valid observations but represent charges not captured in the DRG for that admission.

**Procedures — False Positives (66 total):** The model extracts procedures referenced in the note context (e.g., `Coronary artery bypass grafting (CABG)` mentioned in history, `Hemodialysis`, vaccine administrations) that are not in the coded procedure record. Procedure precision is the weakest metric (0.476).

**Medications — False Negatives (796 total):** The large false negative count reflects the MAR-vs-note mismatch described under Factors. The model's precision on medications it does extract is high (0.932), indicating that what it finds is correct but the note coverage is structurally incomplete relative to the MAR ground truth.

---

## Evaluation Data

| Attribute | Value |
|-----------|-------|
| Source | MIMIC-IV Clinical Database Demo v2.2 |
| License | Open Data Commons Open Database License (ODbL) via PhysioNet |
| Admissions | 30 (ICD-10 coded only) |
| Selection | Stratified random sample by diagnosis count complexity |
| Note type | Synthetic discharge summaries generated from structured MIMIC data |
| Note generation model | Claude Haiku 4.5 (same model as extraction; see Caveats) |
| Ground truth | MIMIC `diagnoses_icd`, `procedures_icd`, `prescriptions` tables |

The evaluation notes are **synthetic**: they were generated by prompting Claude with the structured ground truth (diagnosis descriptions, procedure descriptions, medication names/routes) and asking it to produce a realistic discharge summary. ICD codes were explicitly excluded from the generation prompt. This approach provides a controlled evaluation corpus without requiring access to real patient notes.

---

## Training Data

Not applicable. This pipeline uses zero-shot prompting with no fine-tuning. The model weights are those of Claude Haiku 4.5 as provided by Anthropic. No clinical training data was used.

---

## Ethical Considerations

### Not for Clinical Decision-Making

This system is a coding support tool. It must not be used to guide treatment decisions, triage, or diagnosis at the point of care. Clinicians should not modify treatment plans based on extracted entities without independent clinical judgment.

### Human-in-the-Loop Requirement

All extracted entities and suggested ICD-10 codes require review by a qualified human (medical coder or clinician) before use in claims, registries, or downstream analytics. The pipeline flags all partial-match codes (`needs_review=True`) and confidence < `high` extractions for mandatory review.

### PHI Handling

Clinical note text is sent to the Anthropic API for processing. Before deploying against real patient notes:
- Confirm that a Data Processing Agreement (DPA) or Business Associate Agreement (BAA) is in place with Anthropic.
- Verify that note de-identification meets HIPAA Safe Harbor or Expert Determination standards.
- Review Anthropic's API data retention and usage policies for the applicable contract tier.

This evaluation was conducted entirely on synthetic notes derived from de-identified data (see PHI Policy at `docs/governance/phi_policy.md`).

### Bias and Fairness

Clinical notes often contain demographic information. The model may associate certain conditions disproportionately with demographic groups if those patterns appear in its pre-training corpus. This has not been evaluated for this pipeline. Fairness audits stratified by patient demographics are recommended before production deployment.

---

## Caveats and Recommendations

### Synthetic Evaluation Caveat

The evaluation notes were generated by the same model used for extraction (Claude Haiku 4.5). This creates a potential circularity: the model may extract entities more reliably from notes it generated because those notes mirror its own output patterns. Performance on real clinical notes written by human clinicians may differ — potentially lower for idiomatic phrasing, handwriting transliterations, or uncommon abbreviations.

**Recommendation:** Validate on a held-out set of real, de-identified clinical notes before production deployment.

### Specialty-Dependent Performance

Procedure extraction performs significantly worse than diagnosis extraction (F1 0.558 vs 0.815). Notes from procedurally complex specialties (cardiovascular surgery, interventional radiology) should be treated with higher review scrutiny.

**Recommendation:** Apply specialty-specific confidence thresholds. Consider suppressing low-confidence procedure suggestions in high-volume procedure specialties pending specialty-specific evaluation.

### Medication Recall Gap

The 31.9% medication recall should be interpreted in context of the MAR-vs-note mismatch, not as a standalone quality metric. If the intended use case is medication reconciliation (rather than discharge medication documentation), a dedicated medication extraction pass over nursing notes and MAR data is required.

**Recommendation:** Clarify the operational definition of "medication" in the target workflow before reporting recall to stakeholders.

### ICD-10 Code Suggestions

The model suggests ICD-10 codes but these suggestions feed into a separate fuzzy matching step (`CodeMatcher`) against the CMS 2025 ICD-10-CM table. The suggested code alone should not be used without validation; always use the matched and reviewed output from the full pipeline.
