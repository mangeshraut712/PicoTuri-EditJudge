"""Training entry points."""

from .baseline import BaselineArtifacts, train_baseline_model, evaluate_model

__all__ = ["BaselineArtifacts", "train_baseline_model", "evaluate_model"]
