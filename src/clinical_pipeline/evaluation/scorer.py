"""Score extraction results against MIMIC ground truth."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from clinical_pipeline.coding.icd10_loader import ICD10CodeTable
from clinical_pipeline.evaluation.ground_truth import GroundTruth
from clinical_pipeline.extraction.models import ClinicalExtraction

logger = logging.getLogger(__name__)

# Suffixes to strip when normalizing drug names
_DRUG_SUFFIXES = re.compile(
    r"\b(hydrochloride|sodium|potassium|sulfate|acetate|tartrate|"
    r"maleate|besylate|mesylate|fumarate|succinate|phosphate|citrate|"
    r"hcl|injection|oral|tablet|capsule|solution|suspension)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MatchDetail:
    """Per-entity match information."""

    ground_truth_value: str
    extracted_value: str | None
    matched: bool
    match_method: str  # "exact_code", "category_code", "fuzzy_name", "drug_name", "none"
    score: float


@dataclass(slots=True)
class EvaluationResult:
    """Precision / recall / F1 for a single entity type on a single admission."""

    entity_type: str  # "diagnoses", "procedures", "medications"
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    details: list[MatchDetail] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _normalize_code(code: str) -> str:
    """Normalize an ICD code: uppercase, strip dots and whitespace."""
    return code.strip().upper().replace(".", "")


def _normalize_drug(name: str) -> str:
    """Normalize a drug name for comparison."""
    name = name.lower().strip()
    name = _DRUG_SUFFIXES.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _evaluate_diagnoses(
    extraction: ClinicalExtraction,
    ground_truth: GroundTruth,
    icd10_table: ICD10CodeTable | None,
) -> EvaluationResult:
    """Evaluate diagnosis extraction against ground truth.

    Matching strategy (in order):
    1. Exact ICD-10 code match (full code)
    2. Category match (first 3 characters of ICD-10 code)
    3. Fuzzy name match (>= 80 similarity on description)
    """
    result = EvaluationResult(entity_type="diagnoses")

    gt_codes = {_normalize_code(d.icd_code) for d in ground_truth.diagnoses}
    gt_categories = {_normalize_code(d.icd_code)[:3] for d in ground_truth.diagnoses}
    gt_descriptions = {d.description.lower(): d.icd_code for d in ground_truth.diagnoses}

    matched_gt_codes: set[str] = set()

    for dx in extraction.diagnoses:
        extracted_code = _normalize_code(dx.icd10_suggestion or "")
        extracted_name = dx.name.strip()
        matched = False

        # Strategy 1: Exact code match
        if extracted_code and extracted_code in gt_codes:
            matched = True
            matched_gt_codes.add(extracted_code)
            result.details.append(
                MatchDetail(
                    ground_truth_value=extracted_code,
                    extracted_value=extracted_code,
                    matched=True,
                    match_method="exact_code",
                    score=100.0,
                )
            )

        # Strategy 2: Category match (first 3 chars)
        if not matched and extracted_code and extracted_code[:3] in gt_categories:
            # Find the specific GT code this category matches
            for gt_code in gt_codes:
                if gt_code[:3] == extracted_code[:3] and gt_code not in matched_gt_codes:
                    matched = True
                    matched_gt_codes.add(gt_code)
                    result.details.append(
                        MatchDetail(
                            ground_truth_value=gt_code,
                            extracted_value=extracted_code,
                            matched=True,
                            match_method="category_code",
                            score=80.0,
                        )
                    )
                    break

        # Strategy 3: Fuzzy name match
        if not matched and extracted_name:
            best_score = 0.0
            best_gt_code = None
            for gt_desc, gt_code in gt_descriptions.items():
                norm_gt = _normalize_code(gt_code)
                if norm_gt in matched_gt_codes:
                    continue
                score = fuzz.WRatio(extracted_name.lower(), gt_desc)
                if score > best_score:
                    best_score = score
                    best_gt_code = norm_gt

            if best_score >= 80.0 and best_gt_code is not None:
                matched = True
                matched_gt_codes.add(best_gt_code)
                result.details.append(
                    MatchDetail(
                        ground_truth_value=best_gt_code,
                        extracted_value=extracted_name,
                        matched=True,
                        match_method="fuzzy_name",
                        score=best_score,
                    )
                )

        if not matched:
            result.details.append(
                MatchDetail(
                    ground_truth_value="",
                    extracted_value=extracted_name or extracted_code,
                    matched=False,
                    match_method="none",
                    score=0.0,
                )
            )

    result.true_positives = len(matched_gt_codes)
    result.false_positives = len(extraction.diagnoses) - result.true_positives
    result.false_negatives = len(gt_codes) - result.true_positives

    return result


def _evaluate_medications(
    extraction: ClinicalExtraction,
    ground_truth: GroundTruth,
) -> EvaluationResult:
    """Evaluate medication extraction against ground truth.

    Matching: normalized drug name containment (either direction).
    """
    result = EvaluationResult(entity_type="medications")

    gt_drugs = [_normalize_drug(m.drug) for m in ground_truth.medications]
    matched_gt_indices: set[int] = set()

    for med in extraction.medications:
        extracted_name = _normalize_drug(med.name)
        matched = False

        for idx, gt_drug in enumerate(gt_drugs):
            if idx in matched_gt_indices:
                continue
            # Check containment in either direction
            if extracted_name in gt_drug or gt_drug in extracted_name:
                matched = True
                matched_gt_indices.add(idx)
                result.details.append(
                    MatchDetail(
                        ground_truth_value=ground_truth.medications[idx].drug,
                        extracted_value=med.name,
                        matched=True,
                        match_method="drug_name",
                        score=100.0,
                    )
                )
                break

        # Fallback: fuzzy match
        if not matched:
            best_score = 0.0
            best_idx = -1
            for idx, gt_drug in enumerate(gt_drugs):
                if idx in matched_gt_indices:
                    continue
                score = fuzz.WRatio(extracted_name, gt_drug)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_score >= 80.0 and best_idx >= 0:
                matched = True
                matched_gt_indices.add(best_idx)
                result.details.append(
                    MatchDetail(
                        ground_truth_value=ground_truth.medications[best_idx].drug,
                        extracted_value=med.name,
                        matched=True,
                        match_method="drug_name",
                        score=best_score,
                    )
                )

        if not matched:
            result.details.append(
                MatchDetail(
                    ground_truth_value="",
                    extracted_value=med.name,
                    matched=False,
                    match_method="none",
                    score=0.0,
                )
            )

    result.true_positives = len(matched_gt_indices)
    result.false_positives = len(extraction.medications) - result.true_positives
    result.false_negatives = len(gt_drugs) - result.true_positives

    return result


def _evaluate_procedures(
    extraction: ClinicalExtraction,
    ground_truth: GroundTruth,
) -> EvaluationResult:
    """Evaluate procedure extraction against ground truth.

    Matching: fuzzy description similarity >= 70.
    """
    result = EvaluationResult(entity_type="procedures")

    gt_descriptions = [p.description.lower() for p in ground_truth.procedures]
    matched_gt_indices: set[int] = set()

    for proc in extraction.procedures:
        extracted_name = proc.name.strip().lower()
        matched = False
        best_score = 0.0
        best_idx = -1

        for idx, gt_desc in enumerate(gt_descriptions):
            if idx in matched_gt_indices:
                continue
            score = fuzz.WRatio(extracted_name, gt_desc)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score >= 70.0 and best_idx >= 0:
            matched = True
            matched_gt_indices.add(best_idx)
            result.details.append(
                MatchDetail(
                    ground_truth_value=ground_truth.procedures[best_idx].description,
                    extracted_value=proc.name,
                    matched=True,
                    match_method="fuzzy_name",
                    score=best_score,
                )
            )

        if not matched:
            result.details.append(
                MatchDetail(
                    ground_truth_value="",
                    extracted_value=proc.name,
                    matched=False,
                    match_method="none",
                    score=0.0,
                )
            )

    result.true_positives = len(matched_gt_indices)
    result.false_positives = len(extraction.procedures) - result.true_positives
    result.false_negatives = len(gt_descriptions) - result.true_positives

    return result


def evaluate_extraction(
    extraction: ClinicalExtraction,
    ground_truth: GroundTruth,
    icd10_table: ICD10CodeTable | None = None,
) -> list[EvaluationResult]:
    """Score a single extraction against its ground truth.

    Returns a list of EvaluationResult, one per entity type
    (diagnoses, procedures, medications).
    """
    return [
        _evaluate_diagnoses(extraction, ground_truth, icd10_table),
        _evaluate_procedures(extraction, ground_truth),
        _evaluate_medications(extraction, ground_truth),
    ]


AggregatedResult = dict[str, dict[str, float]]
"""Mapping of entity_type -> {precision, recall, f1, tp, fp, fn}."""


def aggregate_results(
    all_results: dict[int, list[EvaluationResult]],
) -> AggregatedResult:
    """Aggregate per-admission results into overall metrics by entity type."""
    totals: dict[str, dict[str, int]] = {}

    for results in all_results.values():
        for r in results:
            if r.entity_type not in totals:
                totals[r.entity_type] = {"tp": 0, "fp": 0, "fn": 0}
            totals[r.entity_type]["tp"] += r.true_positives
            totals[r.entity_type]["fp"] += r.false_positives
            totals[r.entity_type]["fn"] += r.false_negatives

    aggregated: AggregatedResult = {}
    for entity_type, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        aggregated[entity_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return aggregated
