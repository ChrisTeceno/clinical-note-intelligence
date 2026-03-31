"""Retrieve relevant ICD-10 codes to augment the extraction prompt via TF-IDF."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Common English stopwords plus clinical noise words that appear too broadly to be useful
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "not", "no", "nor",
    "this", "that", "these", "those", "it", "its", "as", "if", "when",
    "than", "then", "also", "into", "over", "after", "before", "between",
    "under", "above", "such", "each", "other", "some", "any", "all",
    "both", "more", "most", "only", "very", "same", "about", "up", "out",
    "per", "due", "without", "within", "through", "during",
    "patient", "noted", "history", "performed", "using", "left", "right",
    "status", "type", "unspecified", "other", "encounter",
})

_WORD_RE = re.compile(r"[a-z]{3,}", re.IGNORECASE)


class ICD10Retriever:
    """Retrieve relevant ICD-10 codes to augment the extraction prompt.

    Builds a TF-IDF matrix over all 74K ICD-10-CM code descriptions once,
    then uses cosine similarity to find the most relevant codes for a note.
    """

    def __init__(self, code_table_path: Path) -> None:
        self._codes: list[str] = []
        self._descriptions: list[str] = []
        self._load_codes(code_table_path)

        logger.info("Building TF-IDF matrix over %d ICD-10 descriptions...", len(self._codes))
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[a-zA-Z]{3,}",
            lowercase=True,
            stop_words=list(_STOPWORDS),
            max_features=50_000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._descriptions)
        logger.info("TF-IDF matrix built: %d docs x %d features", *self._tfidf_matrix.shape)

    def _load_codes(self, path: Path) -> None:
        """Parse the CMS fixed-width text file (same format as icd10_loader.load_from_cms_txt)."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if len(line) < 8:
                    continue
                code = line[:7].strip()
                description = line[7:].strip()
                if code and description:
                    self._codes.append(code)
                    self._descriptions.append(description)

    def retrieve(self, note_text: str, top_k: int = 50) -> list[tuple[str, str]]:
        """Find the top_k most relevant ICD-10 codes for this note.

        Returns list of (code, description) tuples, ordered by relevance.

        Strategy:
        1. Extract keywords from the note (words > 2 chars, minus stopwords)
        2. Transform into TF-IDF space using the fitted vectorizer
        3. Compute cosine similarity against all code descriptions
        4. Return top_k matches
        """
        keywords = self._extract_keywords(note_text)
        if not keywords:
            return []

        query = " ".join(keywords)
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                results.append((self._codes[idx], self._descriptions[idx]))

        return results

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from clinical note text."""
        words = _WORD_RE.findall(text.lower())
        return [w for w in words if w not in _STOPWORDS]
