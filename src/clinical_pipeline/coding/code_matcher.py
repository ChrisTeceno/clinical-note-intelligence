"""Fuzzy-match Claude's ICD-10 suggestions against official codes using rapidfuzz."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process

from clinical_pipeline.coding.icd10_loader import ICD10CodeTable

logger = logging.getLogger(__name__)

EXACT_MATCH_THRESHOLD = 100
PARTIAL_MATCH_THRESHOLD = 80


@dataclass(frozen=True, slots=True)
class CodeMatch:
    suggested_code: str
    suggested_description: str | None
    matched_code: str | None
    matched_description: str | None
    match_type: Literal["exact", "partial", "none"]
    score: float
    needs_review: bool


class CodeMatcher:
    """Matches suggested ICD-10 codes against the official CMS code table."""

    def __init__(
        self,
        code_table: ICD10CodeTable,
        partial_threshold: float = PARTIAL_MATCH_THRESHOLD,
    ) -> None:
        self.code_table = code_table
        self.partial_threshold = partial_threshold
        self._description_choices: dict[str, str] = {
            code: desc for code, desc in code_table.get_all_codes()
        }

    def match_code(
        self, suggested_code: str | None, diagnosis_name: str | None = None
    ) -> CodeMatch:
        """Match a suggested ICD-10 code against official codes.

        Tries exact code match first, then falls back to fuzzy description matching.
        """
        suggested_code = (suggested_code or "").strip().upper().replace(".", "")
        diagnosis_name = (diagnosis_name or "").strip()

        # Try exact code match
        exact = self.code_table.lookup(suggested_code)
        if exact is not None:
            return CodeMatch(
                suggested_code=suggested_code,
                suggested_description=diagnosis_name,
                matched_code=exact.code,
                matched_description=exact.description,
                match_type="exact",
                score=100.0,
                needs_review=False,
            )

        # Try fuzzy matching on description
        if diagnosis_name and self._description_choices:
            result = process.extractOne(
                diagnosis_name,
                self._description_choices,
                scorer=fuzz.WRatio,
                score_cutoff=self.partial_threshold,
            )
            if result is not None:
                matched_desc, score, matched_code = result
                return CodeMatch(
                    suggested_code=suggested_code,
                    suggested_description=diagnosis_name,
                    matched_code=matched_code,
                    matched_description=matched_desc,
                    match_type="partial",
                    score=float(score),
                    needs_review=True,
                )

        return CodeMatch(
            suggested_code=suggested_code,
            suggested_description=diagnosis_name,
            matched_code=None,
            matched_description=None,
            match_type="none",
            score=0.0,
            needs_review=True,
        )
