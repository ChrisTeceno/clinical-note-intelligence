"""Load ICD-10-CM code table from CMS files into a lookup structure."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ICD10Code:
    code: str
    description: str
    is_header: bool = False


class ICD10CodeTable:
    """In-memory lookup table for ICD-10-CM codes."""

    def __init__(self) -> None:
        self._by_code: dict[str, ICD10Code] = {}
        self._descriptions: list[tuple[str, str]] = []  # (code, description) for search

    def add(self, code: ICD10Code) -> None:
        self._by_code[code.code] = code
        self._descriptions.append((code.code, code.description))

    def lookup(self, code: str) -> ICD10Code | None:
        """Exact lookup by code."""
        normalized = code.strip().upper().replace(".", "")
        return self._by_code.get(normalized)

    def get_all_codes(self) -> list[tuple[str, str]]:
        """Return all (code, description) pairs for fuzzy matching."""
        return self._descriptions

    def __len__(self) -> int:
        return len(self._by_code)

    def __contains__(self, code: str) -> bool:
        normalized = code.strip().upper().replace(".", "")
        return normalized in self._by_code


def load_from_cms_txt(path: Path) -> ICD10CodeTable:
    """Load ICD-10-CM codes from the CMS fixed-width text file.

    The CMS ``icd10cm_codes_YYYY.txt`` file uses a fixed format:
    columns 0-6 are the code, and the description starts at column 7 (space-separated).
    """
    table = ICD10CodeTable()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) < 8:
                continue
            code = line[:7].strip()
            description = line[7:].strip()
            if code and description:
                is_header = len(code) <= 3
                table.add(ICD10Code(code=code, description=description, is_header=is_header))

    logger.info("Loaded %d ICD-10-CM codes from %s", len(table), path)
    return table


def load_from_csv(path: Path, code_col: str = "code", desc_col: str = "description") -> ICD10CodeTable:
    """Load ICD-10-CM codes from a CSV file with code and description columns."""
    table = ICD10CodeTable()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row[code_col].strip().upper().replace(".", "")
            description = row[desc_col].strip()
            if code and description:
                is_header = len(code) <= 3
                table.add(ICD10Code(code=code, description=description, is_header=is_header))

    logger.info("Loaded %d ICD-10-CM codes from %s", len(table), path)
    return table
