"""Join image and text features into a modeling table."""

from __future__ import annotations

import pandas as pd


def build_feature_table(pairs: pd.DataFrame, similarity_column: str = "image_similarity") -> pd.DataFrame:
    """
    Produce the modeling table for the baseline classifier.

    Args:
        pairs: DataFrame containing at least `instruction`, `label`, and the given
            similarity column.
        similarity_column: Column name containing the numeric similarity feature.

    Returns:
        DataFrame subset containing relevant columns and dropping NA rows.
    """

    required_columns = {"instruction", similarity_column, "label"}
    missing = required_columns.difference(pairs.columns)
    if missing:
        raise KeyError(f"Pairs DataFrame missing required columns: {missing}")

    feature_table = pairs[list(required_columns)].dropna().reset_index(drop=True)
    return feature_table
