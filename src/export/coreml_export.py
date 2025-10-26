"""Core ML manifest exporter for PicoTuri-EditJudge."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import used only for type checkers
    from sklearn.pipeline import Pipeline  # type: ignore


def export_coreml_manifest(
    model_path: Path,
    out_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    thresholds_path: Optional[Path] = None,
) -> Path:
    """
    Create a JSON manifest describing how to export the model to Core ML.

    We deliberately avoid converting the scikit-learn pipeline here because the
    TF-IDF + logistic regression combination requires a custom conversion step.
    Instead, this manifest captures all information needed for a manual export
    (e.g., in a Python 3.12 environment with `coremltools` installed).
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    try:
        from joblib import load  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import environment specific
        raise ImportError(
            "joblib is required to load trained pipelines. Install it with 'pip install joblib'."
        ) from exc

    pipeline = load(model_path)
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.resolve()),
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "metadata": metadata or {},
    }

    thresholds = _load_thresholds(thresholds_path)
    if thresholds:
        manifest["thresholds"] = thresholds

    try:
        preprocess = pipeline.named_steps["preprocess"]
        vectorizer = preprocess.named_transformers_["instruction"]
        manifest["tfidf"] = {
            "vocabulary_size": len(getattr(vectorizer, "vocabulary_", {})),
            "max_features": getattr(vectorizer, "max_features", None),
            "ngram_range": getattr(vectorizer, "ngram_range", None),
        }
    except Exception:  # pragma: no cover - pipeline variations
        logging.debug("Unable to extract TF-IDF metadata from pipeline.")

    try:
        classifier = pipeline.named_steps["clf"]
        classes = getattr(classifier, "classes_", None)
        if classes is not None and hasattr(classes, "tolist"):
            classes = classes.tolist()
        manifest["classifier"] = {
            "type": type(classifier).__name__,
            "classes": classes,
        }
    except Exception:  # pragma: no cover - pipeline variations
        logging.debug("Unable to extract classifier metadata from pipeline.")

    out_path = _ensure_json_extension(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logging.info("Wrote Core ML export manifest to %s", out_path)
    _log_coremltools_status()
    return out_path


def _load_thresholds(thresholds_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if thresholds_path is None:
        return None
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")
    with thresholds_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_json_extension(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    logging.warning("Adjusting output path to have a .json extension for the manifest.")
    return path.with_suffix(".json")


def _log_coremltools_status() -> None:
    try:
        import coremltools  # type: ignore

        logging.info("coremltools %s detected – you can perform a true conversion in Python 3.12.", coremltools.__version__)
    except Exception:
        logging.info(
            "coremltools not installed in this environment. "
            "Install it in a Python 3.12 environment to generate a .mlmodel bundle."
        )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Core ML export manifest.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the trained sklearn pipeline (.joblib).")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSON path for the manifest.")
    parser.add_argument("--thresholds", type=Path, help="Optional thresholds JSON to embed in the manifest.")
    parser.add_argument("--metadata", type=str, nargs="*", help="Optional key=value pairs to embed in metadata.")
    return parser.parse_args(list(args) if args is not None else None)


def _parse_metadata(kv_pairs: Optional[Iterable[str]]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if not kv_pairs:
        return metadata
    for pair in kv_pairs:
        if "=" not in pair:
            logging.warning("Ignoring malformed metadata item '%s'; expected key=value.", pair)
            continue
        key, value = pair.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = _parse_metadata(args.metadata)
    export_coreml_manifest(args.model_path, args.out, metadata=metadata, thresholds_path=args.thresholds)


if __name__ == "__main__":
    main()
