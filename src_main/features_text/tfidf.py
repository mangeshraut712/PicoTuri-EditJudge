"""Lightweight TF-IDF vectorizer helpers for instruction text."""

from __future__ import annotations

from typing import Iterable, Optional

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]

DEFAULT_MAX_FEATURES = 2048


def build_tfidf_vectorizer(max_features: Optional[int] = DEFAULT_MAX_FEATURES) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer tuned for short edit instructions."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        lowercase=True,
        strip_accents="unicode",
    )


def fit_vectorizer(corpus: Iterable[str], max_features: Optional[int] = DEFAULT_MAX_FEATURES) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on the provided corpus."""
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    vectorizer.fit(corpus)
    return vectorizer
