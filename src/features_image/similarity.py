"""
Image similarity helpers.

Legacy support: if Turicreate is installed, we still leverage its image_similarity
toolkit. Otherwise, a histogram-based cosine similarity provides a cross-platform
fallback that does not rely on archived dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image


def _try_turicreate():
    try:
        import turicreate as tc  # type: ignore

        return tc
    except ModuleNotFoundError:
        return None


def _histogram_similarity(original_image: Path | str, edited_image: Path | str, bins: int = 32) -> float:
    """Compute cosine similarity between color histograms."""

    def _load_hist(path: Path | str) -> np.ndarray:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((224, 224))
            # Normalize to [0, 1]
            arr = np.asarray(img, dtype=np.float32) / 255.0
        hist_channels = []
        for channel in range(3):
            channel_values = arr[:, :, channel].ravel()
            hist, _ = np.histogram(channel_values, bins=bins, range=(0.0, 1.0), density=True)
            hist_channels.append(hist)
        hist_vector = np.concatenate(hist_channels)
        norm = np.linalg.norm(hist_vector)
        if norm == 0.0:
            return hist_vector
        return hist_vector / norm

    orig_hist = _load_hist(original_image)
    edit_hist = _load_hist(edited_image)
    similarity = float(np.clip(np.dot(orig_hist, edit_hist), -1.0, 1.0))
    return similarity


def compute_similarity_score(original_image: Path | str, edited_image: Path | str) -> float:
    """
    Compute a similarity score between two images.

    Preference order:
        1. Turicreate image similarity (if installed).
        2. Histogram-based cosine similarity fallback.
    """

    tc = _try_turicreate()
    if tc is not None:
        original = tc.Image(str(original_image))
        edited = tc.Image(str(edited_image))
        data = tc.SFrame({"image": [original, edited], "id": ["original", "edited"]})
        model = tc.image_similarity.create(data, label="id", verbose=False)
        scores = model.similarity(data, k=2)
        target_row = scores[
            (scores["query_label"] == "original") & (scores["reference_label"] == "edited")
        ]
        if len(target_row) == 0:
            return 0.0
        return float(target_row["similarity"][0])

    return _histogram_similarity(original_image, edited_image)


def compute_pair_similarity(pairs: Sequence[tuple[Path | str, Path | str]]) -> List[float]:
    """Batch-compute similarity scores for a sequence of (original, edited) images."""
    return [compute_similarity_score(orig, edit) for orig, edit in pairs]
