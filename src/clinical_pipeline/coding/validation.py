"""Scoring and validation for ICD-10 code matching results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from clinical_pipeline.coding.code_matcher import CodeMatch

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Summary of code matching validation for a set of extractions."""

    total: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    no_matches: int = 0
    flagged_for_review: list[CodeMatch] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.total if self.total > 0 else 0.0

    @property
    def partial_match_rate(self) -> float:
        return self.partial_matches / self.total if self.total > 0 else 0.0

    @property
    def any_match_rate(self) -> float:
        return (self.exact_matches + self.partial_matches) / self.total if self.total > 0 else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "total": self.total,
            "exact_matches": self.exact_matches,
            "partial_matches": self.partial_matches,
            "no_matches": self.no_matches,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "partial_match_rate": round(self.partial_match_rate, 4),
            "any_match_rate": round(self.any_match_rate, 4),
            "flagged_for_review": len(self.flagged_for_review),
        }


def validate_matches(matches: list[CodeMatch]) -> ValidationResult:
    """Score a list of code matches and flag items for human review."""
    result = ValidationResult(total=len(matches))

    for match in matches:
        if match.match_type == "exact":
            result.exact_matches += 1
        elif match.match_type == "partial":
            result.partial_matches += 1
            result.flagged_for_review.append(match)
        else:
            result.no_matches += 1
            result.flagged_for_review.append(match)

    logger.info(
        "Validation: %d total, %d exact (%.1f%%), %d partial (%.1f%%), %d none (%.1f%%)",
        result.total,
        result.exact_matches,
        result.exact_match_rate * 100,
        result.partial_matches,
        result.partial_match_rate * 100,
        result.no_matches,
        (1 - result.any_match_rate) * 100,
    )

    return result
