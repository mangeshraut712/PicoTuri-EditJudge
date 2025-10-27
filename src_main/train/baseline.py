"""Baseline logistic regression training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    import joblib  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    raise ImportError("joblib is required to persist models. Install it with 'pip install joblib'.") from exc

try:
    import pandas as pd  # type: ignore[import]
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    raise ImportError("pandas is required for data processing. Install it with 'pip install pandas'.") from exc

try:
    from sklearn.compose import ColumnTransformer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    raise ImportError(
        "scikit-learn is required for the baseline model. Install it with 'pip install scikit-learn'."
    ) from exc

from src_main.features_image.similarity import compute_pair_similarity
from src_main.features_text.tfidf import build_tfidf_vectorizer
from src_main.fuse.feature_joiner import build_feature_table


@dataclass
class BaselineArtifacts:
    """Container for a trained pipeline and evaluation metrics."""

    pipeline: Pipeline
    metrics: Dict[str, float]
    train_size: int
    test_size: int
    similarity_column: str


def load_pairs(path: Path) -> pd.DataFrame:
    """Load the edit pairs CSV."""
    logging.debug("Loading pairs from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")
    return pd.read_csv(path)


def _resolve_image_path(reference: str, image_root: Optional[Path]) -> Path:
    """Resolve a relative image reference to an absolute path."""
    parsed = urlparse(reference)
    if parsed.scheme in {"http", "https", "s3"}:
        raise ValueError(
            f"Remote image references are not supported for similarity computation: {reference}"
        )

    path = Path(reference)
    if path.is_absolute():
        return path

    if image_root is not None:
        candidate = (image_root / path).resolve()
        if candidate.exists():
            return candidate

    return path.resolve()


def _ensure_similarity_column(
    pairs: pd.DataFrame,
    compute_similarity: bool,
    image_root: Optional[Path],
    similarity_column: str,
) -> pd.DataFrame:
    """Add or recompute similarity scores."""
    if not compute_similarity and similarity_column in pairs.columns:
        return pairs
    logging.info("Computing image similarity scores for %d pairs", len(pairs))
    resolved_pairs = []
    for _, row in pairs.iterrows():
        original_ref = row["original_image"]
        edited_ref = row["edited_image"]
        original_path = _resolve_image_path(str(original_ref), image_root)
        edited_path = _resolve_image_path(str(edited_ref), image_root)
        if not original_path.exists():
            raise FileNotFoundError(f"Original image not found: {original_path}")
        if not edited_path.exists():
            raise FileNotFoundError(f"Edited image not found: {edited_path}")
        resolved_pairs.append((original_path, edited_path))

    similarities = compute_pair_similarity(resolved_pairs)
    updated = pairs.copy()
    updated[similarity_column] = similarities
    return updated


def build_pipeline(seed: int, similarity_column: str = "image_similarity") -> Pipeline:
    """Construct the preprocessing + classifier pipeline."""
    text_vectorizer = build_tfidf_vectorizer()
    preprocessor = ColumnTransformer(
        transformers=[
            ("instruction", text_vectorizer, "instruction"),
            ("image_similarity", "passthrough", [similarity_column]),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        random_state=seed,
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", classifier),
        ]
    )


def train_baseline_model(
    pairs_path: Path,
    test_size: float = 0.3,
    seed: int = 13,
    compute_similarity: bool = False,
    image_root: Optional[Path] = None,
    similarity_column: str = "image_similarity",
) -> Tuple[BaselineArtifacts, Tuple[pd.DataFrame, pd.Series]]:
    """Train the baseline model and return artifacts plus the held-out split."""
    pairs = load_pairs(pairs_path)
    pairs = _ensure_similarity_column(
        pairs,
        compute_similarity=compute_similarity,
        image_root=image_root,
        similarity_column=similarity_column,
    )
    features = build_feature_table(pairs, similarity_column=similarity_column)
    X = features.drop(columns=["label"])
    y = features["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = build_pipeline(seed, similarity_column=similarity_column)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    artifacts = BaselineArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        train_size=len(X_train),
        test_size=len(X_test),
        similarity_column=similarity_column,
    )
    return artifacts, (X_test, y_test)


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> Dict[str, float]:
    """Evaluate the trained model."""
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0.0),
    }

    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    except Exception as exc:  # pragma: no cover - safe fallback
        logging.debug("Unable to compute ROC AUC: %s", exc)
        metrics["roc_auc"] = float("nan")

    return metrics


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Persist the trained pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logging.info("Saved pipeline to %s", path)


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Persist metrics as JSON for CI consumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    logging.info("Saved metrics to %s", path)


def parse_args(raw_args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the PicoTuri baseline model.")
    parser.add_argument("--pairs", type=Path, required=True, help="Path to the edit pairs CSV.")
    parser.add_argument("--model-path", type=Path, required=True, help="Where to save or load the model pipeline.")
    parser.add_argument("--metrics-path", type=Path, help="Optional path for writing evaluation metrics JSON.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction of data to reserve for evaluation.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used across training routines.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and evaluate an existing model.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    parser.add_argument(
        "--compute-similarity",
        action="store_true",
        help="Recompute image similarity scores (uses Turicreate when available, otherwise the built-in histogram fallback).",
    )
    parser.add_argument("--image-root", type=Path, help="Optional root directory containing original and edited images.")
    parser.add_argument(
        "--similarity-column",
        default="image_similarity",
        help="Name of the similarity column to use (default: image_similarity).",
    )
    return parser.parse_args(raw_args)


def main(raw_args: Sequence[str] | None = None) -> None:
    args = parse_args(raw_args)
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.evaluate_only:
        logging.info("Evaluating existing model at %s", args.model_path)
        if not args.model_path.exists():
            raise FileNotFoundError(f"Model not found: {args.model_path}")
        pipeline: Pipeline = joblib.load(args.model_path)
        pairs = load_pairs(args.pairs)
        pairs = _ensure_similarity_column(
            pairs,
            compute_similarity=args.compute_similarity,
            image_root=args.image_root,
            similarity_column=args.similarity_column,
        )
        features = build_feature_table(pairs, similarity_column=args.similarity_column)
        X = features.drop(columns=["label"])
        y = features["label"].astype(int)
        metrics = evaluate_model(pipeline, X, y)
    else:
        artifacts, _ = train_baseline_model(
            args.pairs,
            test_size=args.test_size,
            seed=args.seed,
            compute_similarity=args.compute_similarity,
            image_root=args.image_root,
            similarity_column=args.similarity_column,
        )
        pipeline, metrics = artifacts.pipeline, artifacts.metrics
        save_model(pipeline, args.model_path)

    logging.info("Metrics: %s", metrics)
    if args.metrics_path:
        save_metrics(metrics, args.metrics_path)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
