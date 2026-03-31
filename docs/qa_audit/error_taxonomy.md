# Error Taxonomy: Clinical Extraction Pipeline

**Version:** 1.0
**Date:** 2026-03-30
**Source:** MIMIC-IV demo evaluation, 30 admissions

This document catalogs error patterns observed in the evaluation run, organized by entity type and error class. Each category includes representative examples drawn from the evaluation data.

---

## Diagnosis Errors

### False Positives

#### FP-D1: Comorbidity Over-Extraction

**Description:** The model extracts conditions mentioned in the note narrative as background history or ongoing comorbidities that were not included in the final coded diagnosis list for the admission.

**Root cause:** Clinical notes document the full clinical picture. Billing records reflect only the conditions relevant to the DRG or principal encounter reason. The system prompt instructs extraction of all explicitly mentioned diagnoses, which produces valid extractions that do not match the narrower ground truth.

**Examples from evaluation data:**

| Extracted (FP) | Admission | Context |
|---------------|-----------|---------|
| Hypothyroidism | 20199380 | Background comorbidity in HPI |
| Depression | 20199380 | Psychiatric history section |
| Dysuria | 20199380 | Symptom mentioned but not coded |
| Pericardial effusion | 20297618 | Incidental finding documented |
| Hypo-osmolality | 20297618 | Lab finding documented as diagnosis |
| Paranoid ideation | 20297618 | Psychiatric symptom, not coded |

**Mitigation:** Apply a confidence filter. Require `confidence="high"` for pass-through without review. Flag `medium` and `low` confidence diagnoses for HITL review. Consider a clinical-relevance filter that deprioritizes extractions not matching the principal diagnosis context.

---

#### FP-D2: Lab Finding Extraction

**Description:** Abnormal lab values and physiologic measurements documented in the clinical assessment are extracted as diagnoses. These are signs and findings under ICD-10-CM (R-codes), not billable diagnoses in the context of a hospitalization where an underlying condition is also coded.

**Examples from evaluation data:**

| Extracted (FP) | Correct interpretation |
|---------------|----------------------|
| Elevated troponin | Sign; coded only if underlying ACS is not coded |
| Elevated BNP | Sign of heart failure; not separately coded when HF is present |
| Hyperglycemia | Sign; coded separately only when diabetes is absent |

**Mitigation:** Add a post-processing filter that checks whether an extracted finding (R-code suggestion) co-occurs with a condition that subsumes it. Flag these for coder confirmation rather than auto-accepting.

---

#### FP-D3: Differential Diagnosis Extraction

**Description:** Diagnoses mentioned in a differential diagnosis, rule-out context, or uncertain clinical framing are extracted. The model assigns `confidence="low"` to most of these correctly, but they still appear in the output.

**Examples from evaluation data:**

| Extracted (FP) | Note context |
|---------------|-------------|
| Atherosclerotic heart disease of native coronary arteries | Listed in differential for chest pain workup |

**Mitigation:** Suppress `confidence="low"` extractions from downstream workflows or route them to a separate low-priority review queue.

---

### False Negatives

#### FN-D1: Sequencing and Complication Codes

**Description:** The 120 diagnosis false negatives are predominantly complication codes, external cause codes, and secondary codes that are implied by the clinical narrative but not named explicitly in the note text.

**Examples from evaluation data (ground truth not extracted):**

| Ground truth code | Description | Why missed |
|-------------------|-------------|-----------|
| Y838 | Complication of surgical procedure | Named as "complication" in note without the specific Y-code reference |
| I82422, I82421 variants | Iliofemoral vein conditions | Note describes condition; model suggests slightly different specificity |

**Note:** The system prompt explicitly restricts to information "explicitly stated in the note." Complication and external cause codes are rarely mentioned by name in discharge summaries — they are inferred by coders from the clinical context. This is expected behavior, not a prompt failure.

---

## Procedure Errors

### False Positives

#### FP-P1: Contextually Referenced Procedures

**Description:** Procedures referenced in the note as background, history, or as interventions considered-but-not-performed are extracted as if they were performed during the current admission. This is the dominant procedure error.

**Examples from evaluation data:**

| Extracted (FP) | Likely context |
|---------------|---------------|
| Coronary artery bypass grafting (CABG) with LIMA-to-LAD bypass | Prior surgical history |
| Hemodialysis | Mentioned as potential complication management |
| Pneumococcal 23-valent polysaccharide vaccine administration | Documented in medical history |

**Root cause:** Discharge summaries routinely reference prior procedures, procedures performed at outside facilities, and procedures deferred. The model correctly identifies these as "procedures mentioned in the note" but cannot consistently distinguish documented-as-performed from referenced-historically.

**Mitigation:** Add a temporal/context extraction field to the procedure schema: `performed_this_admission: bool`. Prompt the model to distinguish between procedures performed during the current admission vs. prior history. This would require schema and prompt changes.

---

#### FP-P2: Monitoring Activities Extracted as Procedures

**Description:** Nursing monitoring activities and clinical observations are extracted as coded procedures.

**Examples from evaluation data:**

| Extracted (FP) | Why it is not a coded procedure |
|---------------|-------------------------------|
| Continuous cardiac output monitoring | Nursing activity; not ICD-10-PCS coded |

---

### False Negatives

#### FN-P1: Implicit Procedures

**Description:** Some procedures are so routinely implied by the diagnosis and clinical setting that they may not be mentioned by name in the note. The evaluation records 29 procedure FNs.

**Examples from evaluation data:**

Procedure false negatives include complex multi-step procedures (e.g., vascular interventions with multiple ICD-10-PCS components) where the note describes the overall procedure but the coder assigns several discrete codes. The extraction correctly captures the named procedure but misses the subsidiary component codes.

---

## Medication Errors

### False Negatives (Primary Error Class)

#### FN-Rx1: MAR vs. Note Coverage Mismatch

**Description:** The large medication false negative count (796, driving recall to 0.319) is not a model failure. The MIMIC ground truth is the Medication Administration Record, which captures every medication event during the hospitalization. Clinical notes document only:
- Discharge medications
- Medications material to the admission narrative (e.g., the anticoagulant central to a DVT admission)
- Adverse drug events

PRN medications, maintenance electrolytes, routine IV fluids, and background medications from the patient's home regimen are commonly omitted from the note body.

**Example — Admission 20199380:**
- MAR medications: 31 unique drugs
- Extracted from note: 18 medications
- All 18 extracted were correct matches (precision 1.0)
- 13 MAR medications not mentioned in the note

**Implication:** Measuring medication recall against the MAR is not the appropriate benchmark for note-based extraction. The operationally correct ground truth for this task is "medications documented in the note" — which is not independently available without human annotation.

---

### False Positives

#### FP-Rx1: Generic / Formulation Variants

**Description:** The model extracts broad medication references that do not map to a specific ground truth MAR entry.

**Examples from evaluation data:**

| Extracted (FP) | Issue |
|---------------|-------|
| Normal saline | Generic term; IV fluid not always a discrete MAR entry |
| Potassium supplementation | Route/formulation unspecified; may not match specific potassium chloride entry |
| Pantoprazole | Extracted correctly but not present in this admission's MAR (wrong admission attribution?) |
| Famotidine | Similar issue |

---

## Code Matching Errors

### CM-1: Broad Code Accepted as Exact Match

**Description:** When Claude suggests a 3- or 4-character category code and that code exists in the CMS table, it passes as an exact match with `needs_review=False`. The correct billable code for the patient's condition may be a more specific 5-, 6-, or 7-character subcode.

**Example:**

| Suggested code | Exact match? | Correct specific code | Issue |
|---------------|-------------|-----------------------|-------|
| E11.9 | Yes | E11.649 (long-term insulin use, uncontrolled) | Exact match suppresses review; specificity lost |

**Mitigation:** Flag 3-character (header) codes as requiring specificity review. The `ICD10Code.is_header` field already identifies codes with `len(code) <= 3`. Extend this check to flag any non-terminal code (codes for which more specific subcodes exist).

---

### CM-2: WRatio Confabulation on Partial Match

**Description:** The WRatio scorer can produce high scores for descriptions that sound similar but refer to different clinical entities. This is the primary risk in the 27% partial-match bucket.

**Known risk patterns:**

| Query | Potential spurious match |
|-------|-------------------------|
| "Type 1 diabetes with ketoacidosis" | "Type 2 diabetes with ketoacidosis" |
| "Left femoral fracture" | "Right femoral fracture" |
| "Acute systolic heart failure" | "Acute diastolic heart failure" |

**Mitigation:** All partial matches are flagged `needs_review=True`. Consider adding secondary validation: if the matched description contains a laterality, type qualifier, or number that differs from the query, force-escalate to HITL regardless of WRatio score.

---

## Error Distribution Summary

| Error Code | Entity | Type | Estimated Count | Severity |
|------------|--------|------|-----------------|----------|
| FP-D1 | Diagnoses | FP | ~24 | Medium — valid clinical info, wrong for billing |
| FP-D2 | Diagnoses | FP | ~12 | Low — easily caught by coder |
| FP-D3 | Diagnoses | FP | ~6 | Low — confidence flag helps |
| FN-D1 | Diagnoses | FN | 120 | Medium — complication/sequencing codes need coder |
| FP-P1 | Procedures | FP | ~55 | High — inflates procedure list materially |
| FP-P2 | Procedures | FP | ~11 | Low — obvious to reviewer |
| FN-P1 | Procedures | FN | 29 | Medium — subsidiary PCS codes not captured |
| FN-Rx1 | Medications | FN | ~780 | Low (by design) — MAR mismatch |
| FP-Rx1 | Medications | FP | 27 | Low — generic terms |
| CM-1 | Codes | Match | Unquantified | Medium — specificity risk |
| CM-2 | Codes | Match | ~27% of partials | High — requires mandatory review |
