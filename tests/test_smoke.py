from pathlib import Path

import pandas as pd
from PIL import Image

from src.features_image import compute_similarity_score
from src.train.baseline import train_baseline_model


DATA_PATH = Path("data/manifests/sample_pairs.csv")


def test_sample_manifest_present():
    assert DATA_PATH.exists(), "Sample manifest is missing."
    df = pd.read_csv(DATA_PATH)
    assert not df.empty
    assert {"instruction", "image_similarity", "label"}.issubset(df.columns)


def test_baseline_training_runs():
    artifacts, _ = train_baseline_model(DATA_PATH, test_size=0.3, seed=13)
    predictions = artifacts.pipeline.predict(
        pd.DataFrame([{"instruction": "Test instruction", "image_similarity": 0.75}])
    )
    assert predictions.shape == (1,)
    metrics = artifacts.metrics
    assert {"accuracy", "f1", "roc_auc"}.issubset(metrics.keys())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0 or pd.isna(metrics["f1"])


def test_histogram_similarity_fallback(tmp_path):
    original_path = tmp_path / "orig.png"
    edited_same_path = tmp_path / "edit_same.png"
    edited_diff_path = tmp_path / "edit_diff.png"

    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(original_path)
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(edited_same_path)
    Image.new("RGB", (64, 64), color=(20, 200, 200)).save(edited_diff_path)

    same_score = compute_similarity_score(original_path, edited_same_path)
    diff_score = compute_similarity_score(original_path, edited_diff_path)

    assert same_score > diff_score
    assert 0.0 <= same_score <= 1.0
    assert -1.0 <= diff_score <= 1.0
