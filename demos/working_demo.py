#!/usr/bin/env python3
"""
PicoTuri-EditJudge Demo

This script demonstrates the complete pipeline:
1. Train a model on sample data
2. Make predictions
3. Check Core ML export readiness

Usage: python working_demo.py
"""

import sys
from pathlib import Path

# Add project root to Python path to enable src module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("🎯 PicoTuri-EditJudge Demo")
    print("=" * 32)

    # Check dependencies
    try:
        import pandas as pd  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        import joblib  # type: ignore
        print("✅ Dependencies loaded")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install -r requirements-dev.txt")
        return False

    # Check sample data
    sample_file = "data/manifests/sample_pairs.csv"
    if not Path(sample_file).exists():
        print("❌ Sample data missing")
        return False

    # Load and show data
    df = pd.read_csv(sample_file)
    print(f"📊 Sample data: {len(df)} edit pairs")
    print(f"📝 Example: {df.iloc[0]['instruction'][:50]}...")

    # Train demo model
    try:
        from src.train.baseline import train_baseline_model

        print("🏃 Training model...")
        artifacts, (X_test, y_test) = train_baseline_model(
            Path(sample_file), test_size=0.5, seed=42, compute_similarity=False
        )

        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

    # Test predictions
    try:
        test_predictions = [
            {"instruction": "brighten this photo", "image_similarity": 0.7},
            {"instruction": "add blue sky", "image_similarity": 0.8}
        ]

        test_df = pd.DataFrame(test_predictions)
        predictions = artifacts.pipeline.predict_proba(test_df)[:, 1]

        print("🔮 Predictions on sample inputs:")
        for i, pred in enumerate(predictions):
            print(".3f")
    except Exception as e:
        print(f"⚠️ Prediction test failed: {e}")

    # Check Core ML readiness
    ios_model = Path("examples/ios/EditJudgeDemo/PicoTuriEditJudge.mlmodel")
    manifest = Path("artifacts/editjudge_coreml_manifest.json")

    print("📱 Core ML status:")
    if ios_model.exists():
        size_kb = ios_model.stat().st_size / 1024
        print(".0f")
        print("   Ready for iOS demo!")
    elif manifest.exists():
        print("   Manifest ready - use Python 3.12 with coremltools to build .mlmodel")
    else:
        print("   No Core ML artifacts - training required")

    print("\n🎉 Demo complete! Project is working.")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
