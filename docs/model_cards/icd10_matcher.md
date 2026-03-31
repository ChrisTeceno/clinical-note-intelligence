# Model Card: ICD-10 Code Matcher

**Version:** 1.0
**Date:** 2026-03-30

---

## Algorithm Details

The `CodeMatcher` is a deterministic rule-based matching algorithm, not a learned model. It takes a Claude-suggested ICD-10 code and/or diagnosis name and returns the best match from the CMS 2025 ICD-10-CM official code table.

### Reference Data

| Attribute | Value |
|-----------|-------|
| Source | CMS ICD-10-CM FY2025 code files |
| File | `icd10cm_codes_2025.txt` (fixed-width format) |
| Total codes loaded | 74,260 |
| License | U.S. Government public domain |
| Addenda file | `icd10cm_codes_addenda_2025.txt` (quarterly updates) |

The code table is loaded at startup into an in-memory dict keyed by normalized code string (uppercase, dots stripped). All lookups are O(1).

### Matching Algorithm

The matcher applies two strategies in sequence, stopping at the first success:

**Step 1 — Exact code lookup**

The suggested code is normalized (stripped, uppercased, dots removed) and looked up directly in the in-memory dict. If found, the match is returned immediately with `match_type="exact"`, `score=100.0`, and `needs_review=False`.

**Step 2 — Fuzzy description matching**

If the exact lookup fails, and a `diagnosis_name` string is available, the algorithm runs `rapidfuzz.process.extractOne` over all 74,260 code descriptions using the `fuzz.WRatio` scorer. The default threshold is 80 (configurable via `partial_threshold`). `WRatio` is a weighted combination of token-sort, token-set, and partial-ratio scorers, which handles word-order variation and substring matches in clinical terminology.

If no match meets the threshold, the result is `match_type="none"` with `needs_review=True`.

### Output Schema

```python
@dataclass(frozen=True)
class CodeMatch:
    suggested_code: str        # Normalized input code from Claude
    suggested_description: str | None   # Diagnosis name passed by caller
    matched_code: str | None   # Official CMS code, or None
    matched_description: str | None     # Official CMS description, or None
    match_type: Literal["exact", "partial", "none"]
    score: float               # 100.0 for exact; WRatio score for partial; 0.0 for none
    needs_review: bool         # True for partial and none; False for exact
```

---

## Intended Use

Map LLM-suggested ICD-10 codes to officially recognized CMS codes as a first-pass validation step in a coding-assistance pipeline. Exact matches can be passed to downstream workflows with low review burden. Partial matches surface the closest official code for human coder review before acceptance.

### Out-of-Scope Uses

- This matcher does not apply ICD-10-CM coding guidelines, principal diagnosis sequencing rules, or excludes1/excludes2 logic.
- It does not validate code applicability to age, sex, or encounter type.
- It does not handle ICD-10-PCS (procedure codes). The extraction pipeline uses ICD-10-CM codes for diagnoses only; procedures use free-text description matching.

---

## Metrics

Match rates from the MTSamples production run (applied to Claude extraction outputs over the MTSamples corpus):

| Match Type | Rate | Notes |
|------------|------|-------|
| Exact | 71.8% | Code passes direct lookup; no review required |
| Partial | 27.0% | WRatio >= 80; flagged for human coder review |
| None | 1.2% | No match found; requires manual code lookup |

These rates are calculated over Claude-suggested codes from the MTSamples extraction run, not the MIMIC evaluation set.

### Threshold Behavior

The default partial match threshold of 80 was chosen empirically to balance coverage against false matches. Lowering the threshold increases partial-match coverage but produces more spurious matches. Raising it reduces review volume but pushes more codes to the `none` category.

| Threshold | Expected behavior |
|-----------|-------------------|
| < 70 | High coverage, high false-match rate; not recommended |
| 80 (default) | Balanced; ~27% of codes require review |
| >= 90 | Low false-match rate; increases `none` rate significantly |

---

## Limitations

### Partial Matches Require Human Review

All partial matches have `needs_review=True`. The WRatio scorer can match superficially similar descriptions that refer to different clinical entities (e.g., `Type 1 diabetes with complications` vs `Type 2 diabetes with complications`). A human coder must confirm the correct code before use.

### Specificity vs. Breadth Tradeoff

When Claude suggests a broad category code (e.g., `E11.9` for unspecified type 2 diabetes) and the correct code for the patient is a more specific subcode (e.g., `E11.649`), the exact match will succeed on the broad code and `needs_review` will be `False`. This means exact matches are not necessarily the most specific or correct code — they are simply valid CMS codes. Coders should still review for specificity.

### No ICD-10-CM Coding Guidelines

The matcher does not know which conditions should be coded as principal vs. secondary diagnoses, nor does it apply the ICD-10-CM Official Guidelines for Coding and Reporting. Sequencing and selection of the principal diagnosis remain human responsibilities.

### No PCS Procedure Codes

The extraction pipeline extracts procedure names as free text; ICD-10-PCS matching is not implemented. Procedure matching in the evaluation uses fuzzy name similarity, not official PCS code lookup.

### Performance on Code Updates

The reference table reflects CMS FY2025 codes. Codes added or revised in subsequent fiscal years will not be found via exact lookup and will fall through to fuzzy matching or `none`. The addenda file (`icd10cm_codes_addenda_2025.txt`) must be applied and the table reloaded to keep pace with annual code updates.

---

## Recommendations

1. **Flag all partial matches for mandatory coder review.** The `needs_review` field is set automatically; downstream systems must not suppress this flag.
2. **Monitor the `none` rate.** A sustained `none` rate above 3–5% may indicate the extraction model is generating non-standard terminology that the matcher cannot resolve. Investigate common `none` patterns and update the system prompt or add a synonym normalization step if needed.
3. **Apply FY2025 addenda before each coding cycle.** Load `icd10cm_codes_addenda_2025.txt` alongside the main code file to catch mid-year revisions.
4. **Do not use this matcher in isolation for claim submission.** Combine with a coding guidelines engine or CCS/CPC coder review workflow.
