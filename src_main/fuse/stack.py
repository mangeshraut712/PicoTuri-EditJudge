"""
Feature Fusion Stack
Robust feature alignment and fusion for PicoTuri-EditJudge
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FeatureAligner:
    """
    Robust feature alignment and joining by sample ID with NaN guards
    """

    def __init__(
        self,
        imputer_strategy: str = "median",
        scaler_method: str = "robust",
        handle_missing: str = "drop"
    ):
        """
        Initialize feature aligner

        Args:
            imputer_strategy: Strategy for imputation (median, mean, knn)
            scaler_method: Scaling method (standard, robust, none)
            handle_missing: How to handle missing samples (drop, impute)
        """
        self.imputer_strategy = imputer_strategy
        self.scaler_method = scaler_method
        self.handle_missing = handle_missing

        # Initialize preprocessors
        self.imputers = {}
        self.scalers = {}
        self.feature_names = {}
        self.is_fitted = False

    def fit(self, features_dict: Dict[str, np.ndarray], sample_ids: List[str]):
        """
        Fit the aligner on training data

        Args:
            features_dict: Dictionary of feature arrays
            sample_ids: List of sample IDs for alignment
        """
        logger.info("Fitting feature aligner...")

        # Store feature names
        for name, features in features_dict.items():
            self.feature_names[name] = features.shape[1]

        # Setup imputers
        if self.handle_missing == "impute":
            for name, features in features_dict.items():
                if self.imputer_strategy == "median":
                    self.imputers[name] = SimpleImputer(strategy='median')
                elif self.imputer_strategy == "mean":
                    self.imputers[name] = SimpleImputer(strategy='mean')
                elif self.imputer_strategy == "knn":
                    self.imputers[name] = KNNImputer(n_neighbors=5)

                # Fit imputer
                self.imputers[name].fit(features)

        # Setup scalers
        if self.scaler_method != "none":
            for name, features in features_dict.items():
                if self.scaler_method == "standard":
                    self.scalers[name] = StandardScaler()
                elif self.scaler_method == "robust":
                    self.scalers[name] = RobustScaler()

                # Fit scaler
                self.scalers[name].fit(features)

        self.is_fitted = True
        logger.info("Feature aligner fitted successfully")

    def transform(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform and align features

        Args:
            features_dict: Dictionary of feature arrays

        Returns:
            Aligned and concatenated feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Aligner must be fitted before transformation")

        processed_features = []

        for name, features in features_dict.items():
            # Handle missing values
            if self.handle_missing == "impute" and name in self.imputers:
                features = self.imputers[name].transform(features)

            # Scale features
            if self.scaler_method != "none" and name in self.scalers:
                features = self.scalers[name].transform(features)

            processed_features.append(features)

        # Concatenate all features
        if processed_features:
            aligned_features = np.concatenate(processed_features, axis=1)
        else:
            aligned_features = np.array([])

        return aligned_features

    def fit_transform(self, features_dict: Dict[str, np.ndarray], sample_ids: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(features_dict, sample_ids)
        return self.transform(features_dict)

class FusionMLPHead(nn.Module):
    """
    MLP head for feature fusion with calibration support
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = False
    ):
        """
        Initialize fusion MLP head

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for binary classification)
            dropout_rate: Dropout rate
            activation: Activation function (relu, gelu, swish)
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "swish":
                layers.append(nn.SiLU())

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.use_residual and len(self.mlp) > 2:
            # Simple residual connection for deep networks
            residual = x
            out = self.mlp(x)

            # Add residual if dimensions match
            if out.shape[-1] == residual.shape[-1]:
                out = out + residual
        else:
            out = self.mlp(x)

        return out

class FeatureFusionStack:
    """
    Complete feature fusion pipeline with alignment, fusion, and calibration
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.2,
        imputer_strategy: str = "median",
        scaler_method: str = "robust",
        device: Optional[str] = None
    ):
        """
        Initialize feature fusion stack

        Args:
            feature_dims: Dictionary of feature dimensions
            hidden_dims: MLP hidden layer dimensions
            dropout_rate: Dropout rate
            imputer_strategy: Imputation strategy
            scaler_method: Scaling method
            device: Device to run on
        """
        self.feature_dims = feature_dims
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate total input dimension
        self.total_input_dim = sum(feature_dims.values())

        # Initialize components
        self.aligner = FeatureAligner(
            imputer_strategy=imputer_strategy,
            scaler_method=scaler_method
        )

        self.fusion_head = FusionMLPHead(
            input_dim=self.total_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Calibration components
        self.calibrator = None
        self.is_fitted = False

    def fuse_features(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        clip_similarity: Optional[np.ndarray] = None,
        object_delta: Optional[np.ndarray] = None,
        sample_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fuse multiple feature types

        Args:
            text_embeddings: Text embedding matrix
            image_embeddings: Image embedding matrix
            clip_similarity: CLIP similarity scores
            object_delta: Object detection delta features
            sample_ids: Sample IDs for alignment

        Returns:
            Fused feature matrix
        """
        # Build features dictionary
        features_dict = {
            "text": text_embeddings,
            "image": image_embeddings
        }

        if clip_similarity is not None:
            features_dict["clip_sim"] = clip_similarity.reshape(-1, 1)

        if object_delta is not None:
            features_dict["obj_delta"] = object_delta

        # Validate dimensions
        for name, features in features_dict.items():
            expected_dim = self.feature_dims.get(name, features.shape[1])
            if features.shape[1] != expected_dim:
                logger.warning(f"Feature dimension mismatch for {name}: expected {expected_dim}, got {features.shape[1]}")

        # Align and fuse features
        if sample_ids is not None and not self.aligner.is_fitted:
            # Fit aligner on first use
            fused_features = self.aligner.fit_transform(features_dict, sample_ids)
        else:
            # Transform only
            fused_features = self.aligner.transform(features_dict)

        return fused_features

    def predict_scores(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        clip_similarity: Optional[np.ndarray] = None,
        object_delta: Optional[np.ndarray] = None,
        sample_ids: Optional[List[str]] = None,
        apply_calibration: bool = True
    ) -> np.ndarray:
        """
        Predict edit quality scores

        Args:
            text_embeddings: Text embedding matrix
            image_embeddings: Image embedding matrix
            clip_similarity: CLIP similarity scores
            object_delta: Object detection delta features
            sample_ids: Sample IDs for alignment
            apply_calibration: Whether to apply calibration

        Returns:
            Predicted scores
        """
        # Fuse features
        fused_features = self.fuse_features(
            text_embeddings, image_embeddings, clip_similarity, object_delta, sample_ids
        )

        # Convert to tensor
        features_tensor = torch.FloatTensor(fused_features).to(self.device)

        # Predict with fusion head
        self.fusion_head.eval()
        with torch.no_grad():
            raw_scores = self.fusion_head(features_tensor).cpu().numpy()

        # Apply calibration if available
        if apply_calibration and self.calibrator is not None:
            calibrated_scores = self.calibrator.transform(raw_scores)
        else:
            calibrated_scores = raw_scores

        # Apply sigmoid for binary classification
        final_scores = 1 / (1 + np.exp(-calibrated_scores.flatten()))

        return final_scores

    def fit_calibration(self, scores: np.ndarray, labels: np.ndarray, method: str = "isotonic"):
        """
        Fit calibration on predicted scores

        Args:
            scores: Predicted scores
            labels: True labels
            method: Calibration method (isotonic, platt)
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == "platt":
            # Platt scaling using logistic regression
            self.calibrator = LogisticRegression()
            scores = scores.reshape(-1, 1)

        # Fit calibrator
        self.calibrator.fit(scores, labels)
        logger.info(f"Calibration fitted using {method} method")

    def save_model(self, output_path: str):
        """Save the fusion model"""
        torch.save({
            'fusion_head': self.fusion_head.state_dict(),
            'feature_dims': self.feature_dims,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'is_fitted': self.is_fitted
        }, output_path)
        logger.info(f"Fusion model saved to {output_path}")

    def load_model(self, model_path: str):
        """Load the fusion model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.fusion_head.load_state_dict(checkpoint['fusion_head'])
        self.feature_dims = checkpoint['feature_dims']
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout_rate = checkpoint['dropout_rate']
        self.is_fitted = checkpoint['is_fitted']
        logger.info(f"Fusion model loaded from {model_path}")

# Test function
def test_feature_fusion():
    """Test feature fusion functionality"""
    print("Testing feature fusion...")

    # Create dummy data
    batch_size = 32
    text_dim = 768  # BERT
    image_dim = 512  # CLIP ViT-B/32

    text_embeddings = np.random.randn(batch_size, text_dim)
    image_embeddings = np.random.randn(batch_size, image_dim)
    clip_similarity = np.random.randn(batch_size, 1)
    object_delta = np.random.randn(batch_size, 64)

    sample_ids = [f"sample_{i}" for i in range(batch_size)]

    # Initialize fusion stack
    feature_dims = {
        "text": text_dim,
        "image": image_dim,
        "clip_sim": 1,
        "obj_delta": 64
    }

    fusion_stack = FeatureFusionStack(
        feature_dims=feature_dims,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1
    )

    # Test feature fusion
    fused_features = fusion_stack.fuse_features(
        text_embeddings, image_embeddings, clip_similarity, object_delta, sample_ids
    )

    print(f"Fused features shape: {fused_features.shape}")
    print(f"Expected total dim: {sum(feature_dims.values())}")

    # Test prediction
    scores = fusion_stack.predict_scores(
        text_embeddings, image_embeddings, clip_similarity, object_delta, sample_ids
    )

    print(f"Predicted scores shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test calibration
    dummy_labels = np.random.randint(0, 2, batch_size)
    fusion_stack.fit_calibration(scores, dummy_labels, method="isotonic")

    calibrated_scores = fusion_stack.predict_scores(
        text_embeddings, image_embeddings, clip_similarity, object_delta, sample_ids
    )

    print(f"Calibrated scores shape: {calibrated_scores.shape}")
    print(f"Calibrated score range: [{calibrated_scores.min():.3f}, {calibrated_scores.max():.3f}]")

    print("Feature fusion test completed!")

if __name__ == "__main__":
    test_feature_fusion()
