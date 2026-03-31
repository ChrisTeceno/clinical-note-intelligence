"""Text cleaning functions for clinical notes, usable as PySpark UDFs."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional


def strip_html_tags(text: Optional[str]) -> Optional[str]:
    """Remove HTML tags and decode HTML entities."""
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text


def normalize_whitespace(text: Optional[str]) -> Optional[str]:
    """Collapse runs of whitespace into single spaces and strip edges."""
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_encoding(text: Optional[str]) -> Optional[str]:
    """Normalize Unicode and replace common mojibake patterns."""
    if text is None:
        return None
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_text(text: Optional[str]) -> Optional[str]:
    """Full cleaning pipeline: HTML removal, encoding fix, whitespace normalization."""
    text = strip_html_tags(text)
    text = fix_encoding(text)
    text = normalize_whitespace(text)
    return text


def register_udfs():
    """Register PySpark UDFs. Import PySpark only when needed."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    return {
        "strip_html_tags": F.udf(strip_html_tags, StringType()),
        "normalize_whitespace": F.udf(normalize_whitespace, StringType()),
        "fix_encoding": F.udf(fix_encoding, StringType()),
        "clean_text": F.udf(clean_text, StringType()),
    }
