"""
Domain Calibration and Drift Detection
Adaptive calibration and domain shift detection for PicoTuri-EditJudge
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, expected_calibration_error
import logging
from pathlib import Path
import pickle
import json
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CalibrationMethod(Enum):
    """Available calibration methods"""
    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    BINARY_CALIBRATION = "binary_calibration"
    NONE = "none"

class DriftDetectionMethod(Enum):
    """Available drift detection methods"""
    KS_TEST = "ks_test"  # Kolmogorov-Smirnov test
    PSI = "psi"         # Population Stability Index
    KL_DIVERGENCE = "kl_divergence"
    WASSERSTEIN = "wasserstein"
    CHISQUARE = "chisquare"

@dataclass
class CalibrationConfig:
    """Configuration for calibration"""
    method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING
    temperature_init: float = 1.0
    regularization_strength: float = 1e-4
    max_iter: int = 1000
    lr: float = 0.01
    cv_folds: int = 5
    
@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection"""
    method: DriftDetectionMethod = DriftDetectionMethod.KS_TEST
    significance_level: float = 0.05
    window_size: int = 1000
    update_frequency: int = 100
    min_samples: int = 50
    
@dataclass
class CalibrationMetrics:
    """Calibration performance metrics"""
    method: str
    temperature: Optional[float] = None
    brier_score: float = 0.0
    ece: float = 0.0
    mce: float = 0.0
    nll: float = 0.0
    accuracy: float = 0.0
    
@dataclass
class DriftMetrics:
    """Drift detection metrics"""
    method: str
    statistic: float = 0.0
    p_value: float = 1.0
    drift_detected: bool = False
    confidence: float = 0.0
    sample_size: int = 0

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaling
        
        Args:
            temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Input logits
            
        Returns:
            Calibrated probabilities
        """
        return F.softmax(logits / self.temperature, dim=-1)

class PlattScaling:
    """
    Platt scaling (logistic regression) for calibration
    """
    
    def __init__(self, regularization_strength: float = 1e-4):
        """
        Initialize Platt scaling
        
        Args:
            regularization_strength: Regularization strength
        """
        self.regularization_strength = regularization_strength
        self.model = LogisticRegression(
            C=1.0 / regularization_strength,
            max_iter=1000
        )
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, targets: np.ndarray):
        """
        Fit Platt scaling model
        
        Args:
            logits: Model logits
            targets: True targets
        """
        # Convert logits to probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))
        
        # Fit logistic regression
        self.model.fit(probs.reshape(-1, 1), targets)
        self.is_fitted = True
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities
        
        Args:
            logits: Model logits
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Platt scaling model not fitted")
        
        probs = 1.0 / (1.0 + np.exp(-logits))
        return self.model.predict_proba(probs.reshape(-1, 1))[:, 1]

class BinaryCalibrator:
    """
    Binary calibration for quality assessment scores
    """
    
    def __init__(self, method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING):
        """
        Initialize binary calibrator
        
        Args:
            method: Calibration method to use
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
        # Initialize calibrator based on method
        if method == CalibrationMethod.TEMPERATURE_SCALING:
            self.calibrator = TemperatureScaling()
        elif method == CalibrationMethod.PLATT_SCALING:
            self.calibrator = PlattScaling()
        elif method == CalibrationMethod.ISOTONIC_REGRESSION:
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
    
    def fit(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2
    ) -> CalibrationMetrics:
        """
        Fit calibration model
        
        Args:
            scores: Model scores (logits or probabilities)
            targets: True targets (0 or 1)
            validation_split: Fraction of data for validation
            
        Returns:
            Calibration metrics
        """
        # Split data
        n_samples = len(scores)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]
        train_scores, train_targets = scores[train_idx], targets[train_idx]
        val_scores, val_targets = scores[val_idx], targets[val_idx]
        
        # Fit calibrator
        if self.method == CalibrationMethod.TEMPERATURE_SCALING:
            self._fit_temperature_scaling(train_scores, train_targets)
        elif self.method == CalibrationMethod.PLATT_SCALING:
            self.calibrator.fit(train_scores, train_targets)
        elif self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            self.calibrator.fit(train_scores, train_targets)
        
        self.is_fitted = True
        
        # Evaluate on validation set
        calibrated_probs = self.predict_proba(val_scores)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            val_scores, val_targets, calibrated_probs
        )
        
        return metrics
    
    def _fit_temperature_scaling(
        self,
        scores: np.ndarray,
        targets: np.ndarray
    ):
        """Fit temperature scaling using NLL optimization"""
        # Convert to tensors
        logits = torch.tensor(scores, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS(
            [self.calibrator.temperature],
            lr=0.01,
            max_iter=100
        )
        
        def nll_loss(temp):
            """Negative log likelihood loss"""
            probs = F.softmax(logits / temp, dim=-1)
            # For binary case, take probability of positive class
            pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            return -torch.mean(
                targets * torch.log(pos_probs + 1e-10) +
                (1 - targets) * torch.log(1 - pos_probs + 1e-10)
            )
        
        def closure():
            optimizer.zero_grad()
            loss = nll_loss(self.calibrator.temperature)
            loss.backward()
            return loss
        
        optimizer.step(closure)
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities
        
        Args:
            scores: Model scores
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted")
        
        if self.method == CalibrationMethod.TEMPERATURE_SCALING:
            with torch.no_grad():
                logits = torch.tensor(scores, dtype=torch.float32)
                if len(logits.shape) == 1:
                    logits = torch.stack([1 - logits, logits], dim=1)
                probs = self.calibrator(logits)
                return probs[:, 1].numpy()
        elif self.method == CalibrationMethod.PLATT_SCALING:
            return self.calibrator.predict_proba(scores)
        elif self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            return self.calibrator.predict(scores)
        else:
            return scores
    
    def _calculate_metrics(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        calibrated_probs: np.ndarray
    ) -> CalibrationMetrics:
        """Calculate calibration metrics"""
        # Brier score
        brier_score = brier_score_loss(targets, calibrated_probs)
        
        # Expected Calibration Error (ECE)
        ece = expected_calibration_error(targets, calibrated_probs)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(targets, calibrated_probs)
        
        # Negative Log Likelihood
        nll = -np.mean(
            targets * np.log(calibrated_probs + 1e-10) +
            (1 - targets) * np.log(1 - calibrated_probs + 1e-10)
        )
        
        # Accuracy (at 0.5 threshold)
        predictions = (calibrated_probs >= 0.5).astype(int)
        accuracy = np.mean(predictions == targets)
        
        # Temperature for temperature scaling
        temperature = None
        if self.method == CalibrationMethod.TEMPERATURE_SCALING:
            temperature = self.calibrator.temperature.item()
        
        return CalibrationMetrics(
            method=self.method.value,
            temperature=temperature,
            brier_score=brier_score,
            ece=ece,
            mce=mce,
            nll=nll,
            accuracy=accuracy
        )
    
    def _calculate_mce(
        self,
        targets: np.ndarray,
        probs: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_edges[:-1]
        bin_uppers = bin_edges[1:]
        
        mce = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                
                # Update MCE
                mce = max(mce, np.abs(accuracy_in_bin - avg_confidence_in_bin))
        
        return mce
    
    def save(self, save_path: str):
        """Save calibrator"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save calibrator state
        if self.method == CalibrationMethod.TEMPERATURE_SCALING:
            torch.save({
                'temperature': self.calibrator.temperature,
                'method': self.method.value
            }, save_path / "temperature_scaling.pt")
        else:
            with open(save_path / "calibrator.pkl", "wb") as f:
                pickle.dump(self.calibrator, f)
        
        # Save metadata
        metadata = {
            "method": self.method.value,
            "is_fitted": self.is_fitted
        }
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, load_path: str) -> "BinaryCalibrator":
        """Load calibrator"""
        load_path = Path(load_path)
        
        # Load metadata
        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create calibrator
        calibrator = cls(CalibrationMethod(metadata["method"]))
        calibrator.is_fitted = metadata["is_fitted"]
        
        # Load calibrator state
        if calibrator.method == CalibrationMethod.TEMPERATURE_SCALING:
            checkpoint = torch.load(load_path / "temperature_scaling.pt")
            calibrator.calibrator.temperature = checkpoint["temperature"]
        else:
            with open(load_path / "calibrator.pkl", "rb") as f:
                calibrator.calibrator = pickle.load(f)
        
        return calibrator

class DriftDetector:
    """
    Drift detection for domain shift monitoring
    """
    
    def __init__(self, config: DriftDetectionConfig):
        """
        Initialize drift detector
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.reference_data = None
        self.current_window = []
        self.drift_history = []
        
    def fit_reference(self, data: np.ndarray):
        """
        Fit reference distribution
        
        Args:
            data: Reference data
        """
        self.reference_data = data.copy()
        logger.info(f"Reference distribution fitted with {len(data)} samples")
    
    def detect_drift(self, data: np.ndarray) -> DriftMetrics:
        """
        Detect drift in new data
        
        Args:
            data: New data to test
            
        Returns:
            Drift metrics
        """
        if self.reference_data is None:
            raise RuntimeError("Reference distribution not fitted")
        
        if len(data) < self.config.min_samples:
            logger.warning(f"Insufficient samples for drift detection: {len(data)}")
            return DriftMetrics(
                method=self.config.method.value,
                sample_size=len(data)
            )
        
        # Perform drift test based on method
        if self.config.method == DriftDetectionMethod.KS_TEST:
            return self._ks_test(data)
        elif self.config.method == DriftDetectionMethod.PSI:
            return self._psi_test(data)
        elif self.config.method == DriftDetectionMethod.KL_DIVERGENCE:
            return self._kl_divergence_test(data)
        elif self.config.method == DriftDetectionMethod.WASSERSTEIN:
            return self._wasserstein_test(data)
        elif self.config.method == DriftDetectionMethod.CHISQUARE:
            return self._chisquare_test(data)
        else:
            raise ValueError(f"Unknown drift detection method: {self.config.method}")
    
    def _ks_test(self, data: np.ndarray) -> DriftMetrics:
        """Kolmogorov-Smirnov test for drift detection"""
        from scipy import stats
        
        statistic, p_value = stats.ks_2samp(self.reference_data, data)
        
        drift_detected = p_value < self.config.significance_level
        confidence = 1 - p_value
        
        return DriftMetrics(
            method=self.config.method.value,
            statistic=statistic,
            p_value=p_value,
            drift_detected=drift_detected,
            confidence=confidence,
            sample_size=len(data)
        )
    
    def _psi_test(self, data: np.ndarray, n_bins: int = 10) -> DriftMetrics:
        """Population Stability Index test"""
        # Create bins
        min_val = min(np.min(self.reference_data), np.min(data))
        max_val = max(np.max(self.reference_data), np.max(data))
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate frequencies
        ref_hist, _ = np.histogram(self.reference_data, bins=bins)
        cur_hist, _ = np.histogram(data, bins=bins)
        
        # Convert to proportions
        ref_prop = ref_hist / len(self.reference_data)
        cur_prop = cur_hist / len(data)
        
        # Calculate PSI
        psi = np.sum((cur_prop - ref_prop) * np.log((cur_prop + 1e-10) / (ref_prop + 1e-10)))
        
        # Drift detected if PSI > threshold (commonly 0.25)
        drift_detected = psi > 0.25
        confidence = min(psi / 0.25, 1.0)
        
        return DriftMetrics(
            method=self.config.method.value,
            statistic=psi,
            p_value=1.0 - confidence,  # Convert to p-value-like metric
            drift_detected=drift_detected,
            confidence=confidence,
            sample_size=len(data)
        )
    
    def _kl_divergence_test(self, data: np.ndarray, n_bins: int = 10) -> DriftMetrics:
        """KL divergence test"""
        # Create bins
        min_val = min(np.min(self.reference_data), np.min(data))
        max_val = max(np.max(self.reference_data), np.max(data))
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate densities
        ref_hist, _ = np.histogram(self.reference_data, bins=bins, density=True)
        cur_hist, _ = np.histogram(data, bins=bins, density=True)
        
        # Calculate KL divergence
        kl_div = np.sum(ref_hist * np.log((ref_hist + 1e-10) / (cur_hist + 1e-10)))
        
        # Drift detected if KL divergence is high
        drift_detected = kl_div > 0.5  # Threshold can be tuned
        confidence = min(kl_div / 0.5, 1.0)
        
        return DriftMetrics(
            method=self.config.method.value,
            statistic=kl_div,
            p_value=1.0 - confidence,
            drift_detected=drift_detected,
            confidence=confidence,
            sample_size=len(data)
        )
    
    def _wasserstein_test(self, data: np.ndarray) -> DriftMetrics:
        """Wasserstein distance test"""
        from scipy import stats
        
        statistic = stats.wasserstein_distance(self.reference_data, data)
        
        # Drift detected if distance is high
        drift_detected = statistic > 0.1  # Threshold can be tuned
        confidence = min(statistic / 0.1, 1.0)
        
        return DriftMetrics(
            method=self.config.method.value,
            statistic=statistic,
            p_value=1.0 - confidence,
            drift_detected=drift_detected,
            confidence=confidence,
            sample_size=len(data)
        )
    
    def _chisquare_test(self, data: np.ndarray, n_bins: int = 10) -> DriftMetrics:
        """Chi-square test for drift detection"""
        from scipy import stats
        
        # Create bins
        min_val = min(np.min(self.reference_data), np.min(data))
        max_val = max(np.max(self.reference_data), np.max(data))
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate frequencies
        ref_hist, _ = np.histogram(self.reference_data, bins=bins)
        cur_hist, _ = np.histogram(data, bins=bins)
        
        # Perform chi-square test
        statistic, p_value = stats.chisquare(cur_hist, ref_hist)
        
        drift_detected = p_value < self.config.significance_level
        confidence = 1 - p_value
        
        return DriftMetrics(
            method=self.config.method.value,
            statistic=statistic,
            p_value=p_value,
            drift_detected=drift_detected,
            confidence=confidence,
            sample_size=len(data)
        )
    
    def update_reference(self, data: np.ndarray):
        """Update reference distribution with new data"""
        if self.reference_data is None:
            self.reference_data = data.copy()
        else:
            # Combine with existing reference data
            self.reference_data = np.concatenate([self.reference_data, data])
        
        logger.info(f"Reference distribution updated with {len(data)} samples")

class DomainAdapter:
    """
    Domain adaptation system with calibration and drift detection
    """
    
    def __init__(
        self,
        calibration_config: CalibrationConfig,
        drift_config: DriftDetectionConfig
    ):
        """
        Initialize domain adapter
        
        Args:
            calibration_config: Calibration configuration
            drift_config: Drift detection configuration
        """
        self.calibration_config = calibration_config
        self.drift_config = drift_config
        
        # Initialize components
        self.calibrator = BinaryCalibrator(calibration_config.method)
        self.drift_detector = DriftDetector(drift_config)
        
        # State
        self.is_calibrated = False
        self.calibration_metrics = None
        self.drift_history = []
        
        logger.info("Domain adapter initialized")
    
    def calibrate(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2
    ) -> CalibrationMetrics:
        """
        Calibrate the model
        
        Args:
            scores: Model scores
            targets: True targets
            validation_split: Validation split fraction
            
        Returns:
            Calibration metrics
        """
        logger.info(f"Calibrating with {len(scores)} samples")
        
        # Fit calibrator
        self.calibration_metrics = self.calibrator.fit(
            scores, targets, validation_split
        )
        
        self.is_calibrated = True
        
        logger.info(f"Calibration completed. ECE: {self.calibration_metrics.ece:.4f}")
        
        return self.calibration_metrics
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities
        
        Args:
            scores: Model scores
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_calibrated:
            logger.warning("Model not calibrated, returning raw scores")
            return scores
        
        return self.calibrator.predict_proba(scores)
    
    def detect_drift(self, scores: np.ndarray) -> DriftMetrics:
        """
        Detect drift in scores
        
        Args:
            scores: New scores to test
            
        Returns:
            Drift metrics
        """
        drift_metrics = self.drift_detector.detect_drift(scores)
        self.drift_history.append(drift_metrics)
        
        if drift_metrics.drift_detected:
            logger.warning(
                f"Drift detected! Method: {drift_metrics.method}, "
                f"Statistic: {drift_metrics.statistic:.4f}, "
                f"Confidence: {drift_metrics.confidence:.4f}"
            )
        
        return drift_metrics
    
    def update_reference_distribution(self, scores: np.ndarray):
        """Update reference distribution for drift detection"""
        self.drift_detector.update_reference(scores)
        logger.info(f"Reference distribution updated with {len(scores)} samples")
    
    def save(self, save_path: str):
        """Save domain adapter"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save calibrator
        self.calibrator.save(save_path / "calibrator")
        
        # Save drift detector
        drift_data = {
            "reference_data": self.drift_detector.reference_data.tolist() if self.drift_detector.reference_data is not None else None,
            "drift_history": [
                {
                    "method": m.method,
                    "statistic": m.statistic,
                    "p_value": m.p_value,
                    "drift_detected": m.drift_detected,
                    "confidence": m.confidence,
                    "sample_size": m.sample_size
                } for m in self.drift_history
            ]
        }
        
        with open(save_path / "drift_detector.json", "w") as f:
            json.dump(drift_data, f, indent=2)
        
        # Save metadata
        metadata = {
            "is_calibrated": self.is_calibrated,
            "calibration_config": {
                "method": self.calibration_config.method.value,
                "temperature_init": self.calibration_config.temperature_init,
                "regularization_strength": self.calibration_config.regularization_strength,
                "max_iter": self.calibration_config.max_iter,
                "lr": self.calibration_config.lr,
                "cv_folds": self.calibration_config.cv_folds
            },
            "drift_config": {
                "method": self.drift_config.method.value,
                "significance_level": self.drift_config.significance_level,
                "window_size": self.drift_config.window_size,
                "update_frequency": self.drift_config.update_frequency,
                "min_samples": self.drift_config.min_samples
            }
        }
        
        if self.calibration_metrics:
            metadata["calibration_metrics"] = {
                "method": self.calibration_metrics.method,
                "temperature": self.calibration_metrics.temperature,
                "brier_score": self.calibration_metrics.brier_score,
                "ece": self.calibration_metrics.ece,
                "mce": self.calibration_metrics.mce,
                "nll": self.calibration_metrics.nll,
                "accuracy": self.calibration_metrics.accuracy
            }
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Domain adapter saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> "DomainAdapter":
        """Load domain adapter"""
        load_path = Path(load_path)
        
        # Load metadata
        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create configurations
        calib_config = CalibrationConfig(
            method=CalibrationMethod(metadata["calibration_config"]["method"]),
            temperature_init=metadata["calibration_config"]["temperature_init"],
            regularization_strength=metadata["calibration_config"]["regularization_strength"],
            max_iter=metadata["calibration_config"]["max_iter"],
            lr=metadata["calibration_config"]["lr"],
            cv_folds=metadata["calibration_config"]["cv_folds"]
        )
        
        drift_config = DriftDetectionConfig(
            method=DriftDetectionMethod(metadata["drift_config"]["method"]),
            significance_level=metadata["drift_config"]["significance_level"],
            window_size=metadata["drift_config"]["window_size"],
            update_frequency=metadata["drift_config"]["update_frequency"],
            min_samples=metadata["drift_config"]["min_samples"]
        )
        
        # Create domain adapter
        adapter = cls(calib_config, drift_config)
        
        # Load calibrator
        adapter.calibrator = BinaryCalibrator.load(load_path / "calibrator")
        adapter.is_calibrated = metadata["is_calibrated"]
        
        # Load drift detector
        with open(load_path / "drift_detector.json", "r") as f:
            drift_data = json.load(f)
        
        if drift_data["reference_data"] is not None:
            adapter.drift_detector.reference_data = np.array(drift_data["reference_data"])
        
        # Load drift history
        adapter.drift_history = [
            DriftMetrics(
                method=m["method"],
                statistic=m["statistic"],
                p_value=m["p_value"],
                drift_detected=m["drift_detected"],
                confidence=m["confidence"],
                sample_size=m["sample_size"]
            ) for m in drift_data["drift_history"]
        ]
        
        # Load calibration metrics
        if "calibration_metrics" in metadata:
            adapter.calibration_metrics = CalibrationMetrics(
                method=metadata["calibration_metrics"]["method"],
                temperature=metadata["calibration_metrics"]["temperature"],
                brier_score=metadata["calibration_metrics"]["brier_score"],
                ece=metadata["calibration_metrics"]["ece"],
                mce=metadata["calibration_metrics"]["mce"],
                nll=metadata["calibration_metrics"]["nll"],
                accuracy=metadata["calibration_metrics"]["accuracy"]
            )
        
        logger.info(f"Domain adapter loaded from {load_path}")
        return adapter

# Test function
def test_domain_adaptation():
    """Test domain adaptation functionality"""
    print("Testing Domain Adaptation...")
    
    # Create configurations
    calib_config = CalibrationConfig(
        method=CalibrationMethod.TEMPERATURE_SCALING,
        temperature_init=1.0
    )
    
    drift_config = DriftDetectionConfig(
        method=DriftDetectionMethod.KS_TEST,
        significance_level=0.05
    )
    
    # Create domain adapter
    adapter = DomainAdapter(calib_config, drift_config)
    
    # Generate mock data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate scores (logits) and targets
    scores = np.random.normal(0, 1, n_samples)
    targets = (scores > 0).astype(int)
    
    # Calibrate
    metrics = adapter.calibrate(scores, targets)
    print(f"Calibration metrics: ECE={metrics.ece:.4f}, Brier={metrics.brier_score:.4f}")
    
    # Test prediction
    test_scores = np.random.normal(0.2, 1, 100)  # Slight shift
    calibrated_probs = adapter.predict_proba(test_scores)
    print(f"Calibrated probabilities shape: {calibrated_probs.shape}")
    
    # Test drift detection
    adapter.drift_detector.fit_reference(scores)
    drift_metrics = adapter.detect_drift(test_scores)
    print(f"Drift metrics: detected={drift_metrics.drift_detected}, p_value={drift_metrics.p_value:.4f}")
    
    # Test save/load
    adapter.save("/tmp/test_domain_adapter")
    loaded_adapter = DomainAdapter.load("/tmp/test_domain_adapter")
    
    print("Domain adaptation test completed!")

if __name__ == "__main__":
    test_domain_adaptation()
