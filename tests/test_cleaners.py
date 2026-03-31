"""Unit tests for text cleaning functions."""

import pytest

from clinical_pipeline.ingestion.cleaners import (
    clean_text,
    fix_encoding,
    normalize_whitespace,
    strip_html_tags,
)


class TestStripHtmlTags:
    def test_removes_simple_tags(self):
        assert strip_html_tags("<b>bold</b>") == "bold"

    def test_removes_nested_tags(self):
        assert strip_html_tags("<div><p>text</p></div>") == "text"

    def test_decodes_html_entities(self):
        assert strip_html_tags("&amp; &lt; &gt;") == "& < >"

    def test_handles_self_closing_tags(self):
        assert strip_html_tags("line1<br/>line2") == "line1line2"

    def test_returns_none_for_none(self):
        assert strip_html_tags(None) is None

    def test_plain_text_unchanged(self):
        assert strip_html_tags("no html here") == "no html here"


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("too   many   spaces") == "too many spaces"

    def test_collapses_mixed_whitespace(self):
        assert normalize_whitespace("tabs\t\tand\nnewlines") == "tabs and newlines"

    def test_strips_edges(self):
        assert normalize_whitespace("  padded  ") == "padded"

    def test_returns_none_for_none(self):
        assert normalize_whitespace(None) is None


class TestFixEncoding:
    def test_replaces_smart_quotes(self):
        assert fix_encoding("\u201cquoted\u201d") == '"quoted"'

    def test_replaces_smart_apostrophe(self):
        assert fix_encoding("it\u2019s") == "it's"

    def test_replaces_em_dash(self):
        assert fix_encoding("word\u2014word") == "word-word"

    def test_replaces_ellipsis(self):
        assert fix_encoding("wait\u2026") == "wait..."

    def test_replaces_nbsp(self):
        assert fix_encoding("non\u00a0breaking") == "non breaking"

    def test_returns_none_for_none(self):
        assert fix_encoding(None) is None


class TestCleanText:
    def test_full_pipeline(self):
        raw = "  <p>Patient\u2019s temp is &gt; 100\u00b0F.</p>  \n\n  "
        result = clean_text(raw)
        assert "<p>" not in result
        assert "  " not in result
        assert result == "Patient's temp is > 100\u00b0F."

    def test_none_passthrough(self):
        assert clean_text(None) is None

    def test_already_clean_text(self):
        text = "Normal clinical note text."
        assert clean_text(text) == text
