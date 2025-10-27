"""
Training Head with Calibration
MLP head training with Platt / Isotonic calibration for PicoTuri-EditJudge
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EditJudgeDataset(Dataset):
    """Dataset for edit quality judgment training"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_ids: Optional[List[str]] = None
    ):
        """
        Initialize dataset

        Args:
            features: Feature matrix
            labels: Binary labels (0 / 1)
            sample_ids: Optional sample IDs
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sample_ids = sample_ids or [f"sample_{i}" for i in range(len(features))]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'sample_id': self.sample_ids[idx]
        }

class CalibrationMethod:
    """Calibration methods for probability calibration"""

    @staticmethod
    def platt_scaling(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        """
        Platt scaling using logistic regression

        Args:
            scores: Predicted scores
            labels: True labels

        Returns:
            Fitted logistic regression model
        """
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(scores.reshape(-1, 1), labels)
        return lr

    @staticmethod
    def isotonic_regression(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
        """
        Isotonic regression calibration

        Args:
            scores: Predicted scores
            labels: True labels

        Returns:
            Fitted isotonic regression model
        """
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(scores, labels)
        return iso

    @staticmethod
    def temperature_scaling(
        scores: np.ndarray,
        labels: np.ndarray,
        model: nn.Module,
        loader: DataLoader,
        device: str = "cpu",
    ) -> float:
        """
        Temperature scaling calibration

        Args:
            scores: Predicted scores (not used directly)
            labels: True labels (not used directly)
            model: Neural network model
            loader: Data loader
            device: Device to run on

        Returns:
            Optimal temperature
        """
        model.eval()
        nll_criterion = nn.BCELoss()

        # Collect logits and labels
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in loader:
                features = batch['features'].to(device)
                batch_labels = batch['labels'].to(device)

                logits = model(features).squeeze()
                logits_list.append(logits.cpu())
                labels_list.append(batch_labels.cpu())

        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list)

        # Optimize temperature
        log_temperature = torch.tensor(0.0, requires_grad=True, device=device)
        optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=50)

        def eval_loss():
            temperature = torch.exp(log_temperature)
            scaled_logits = all_logits / temperature
            loss = nll_criterion(torch.sigmoid(scaled_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        optimal_temperature = torch.exp(log_temperature).item()
        logger.info(f"Optimal temperature: {optimal_temperature:.4f}")

        return optimal_temperature

class TrainingHead:
    """
    Training head for MLP with calibration support
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        save_dir: str = "models"
    ):
        """
        Initialize training head

        Args:
            model: Neural network model to train
            device: Device to run on
            save_dir: Directory to save models
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_acc': [],
            'val_acc': []
        }

        # Calibration
        self.calibrator = None
        self.calibration_method = None
        self.temperature = None

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the model

        Args:
            train_features: Training features
            train_labels: Training labels
            val_features: Validation features
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            validation_split: Validation split if no val data provided

        Returns:
            Training history
        """
        # Create validation split if not provided
        if val_features is None:
            train_features, val_features, train_labels, val_labels = train_test_split(
                train_features, train_labels, test_size=validation_split,
                random_state=42, stratify=train_labels
            )

        # Create datasets and loaders
        train_dataset = EditJudgeDataset(train_features, train_labels)
        val_dataset = EditJudgeDataset(val_features, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup training
        self.model.to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(features).squeeze()
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)

            train_auc = roc_auc_score(train_targets, train_predictions)
            val_auc = roc_auc_score(val_targets, val_predictions)

            train_acc = accuracy_score(train_targets, np.array(train_predictions) > 0.5)
            val_acc = accuracy_score(val_targets, np.array(val_predictions) > 0.5)

            # Update history
            self.training_history['train_loss'].append(train_loss_avg)
            self.training_history['val_loss'].append(val_loss_avg)
            self.training_history['train_auc'].append(train_auc)
            self.training_history['val_auc'].append(val_auc)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss_avg)

            # Early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                self.save_model("best_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss_avg:.4f}, "
                    f"Val Loss: {val_loss_avg:.4f}, Train AUC: {train_auc:.4f}, "
                    f"Val AUC: {val_auc:.4f}"
                )

        # Load best model
        self.load_model("best_model.pth")

        logger.info("Training completed!")
        return self.training_history

    def calibrate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = "isotonic",
        batch_size: int = 32
    ):
        """
        Calibrate the model

        Args:
            features: Features for calibration
            labels: True labels
            method: Calibration method (isotonic, platt, temperature)
            batch_size: Batch size for inference
        """
        logger.info(f"Calibrating model using {method} method...")

        # Get predictions
        dataset = EditJudgeDataset(features, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:
                batch_features = batch['features'].to(self.device)
                outputs = self.model(batch_features).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(probs)

        predictions = np.array(predictions)

        # Apply calibration
        if method == "isotonic":
            self.calibrator = CalibrationMethod.isotonic_regression(predictions, labels)
        elif method == "platt":
            self.calibrator = CalibrationMethod.platt_scaling(predictions, labels)
        elif method == "temperature":
            self.temperature = CalibrationMethod.temperature_scaling(
                predictions, labels, self.model, loader, self.device
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.calibration_method = method
        logger.info("Calibration completed!")

    def predict(
        self,
        features: np.ndarray,
        batch_size: int = 32,
        apply_calibration: bool = True
    ) -> np.ndarray:
        """
        Make predictions

        Args:
            features: Input features
            batch_size: Batch size
            apply_calibration: Whether to apply calibration

        Returns:
            Predicted probabilities
        """
        dataset = EditJudgeDataset(features, np.zeros(len(features)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:
                batch_features = batch['features'].to(self.device)
                outputs = self.model(batch_features).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(probs)

        predictions = np.array(predictions)

        # Apply calibration
        if apply_calibration and self.calibrator is not None:
            if self.calibration_method == "platt":
                predictions = self.calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
            else:
                predictions = self.calibrator.transform(predictions)
        elif apply_calibration and self.temperature is not None:
            # Apply temperature scaling
            dataset = EditJudgeDataset(features, np.zeros(len(features)))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            temp_predictions = []
            with torch.no_grad():
                for batch in loader:
                    batch_features = batch['features'].to(self.device)
                    outputs = self.model(batch_features).squeeze()
                    temp_probs = torch.sigmoid(outputs / self.temperature).cpu().numpy()
                    temp_predictions.extend(temp_probs)

            predictions = np.array(temp_predictions)

        return predictions

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        apply_calibration: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            features: Test features
            labels: True labels
            batch_size: Batch size
            apply_calibration: Whether to apply calibration

        Returns:
            Evaluation metrics
        """
        predictions = self.predict(features, batch_size, apply_calibration)

        # Calculate metrics
        auc = roc_auc_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions > 0.5)
        f1 = f1_score(labels, predictions > 0.5)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(labels, predictions)
        pr_auc = np.trapz(precision, recall)

        metrics = {
            'auc': auc,
            'accuracy': accuracy,
            'f1': f1,
            'pr_auc': pr_auc,
            'calibration_method': self.calibration_method if apply_calibration else 'none'
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_model(self, filename: str):
        """Save model and training state"""
        save_path = self.save_dir / filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'calibration_method': self.calibration_method,
            'temperature': self.temperature
        }, save_path)

        # Save calibrator separately if it exists
        if self.calibrator is not None:
            import pickle
            with open(self.save_dir / f"{filename}.calibrator.pkl", 'wb') as f:
                pickle.dump(self.calibrator, f)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, filename: str):
        """Load model and training state"""
        load_path = self.save_dir / filename

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        self.calibration_method = checkpoint['calibration_method']
        self.temperature = checkpoint['temperature']

        # Load calibrator if it exists
        calibrator_path = self.save_dir / f"{filename}.calibrator.pkl"
        if calibrator_path.exists():
            import pickle
            with open(calibrator_path, 'rb') as f:
                self.calibrator = pickle.load(f)

        logger.info(f"Model loaded from {load_path}")

    def save_training_history(self, filename: str):
        """Save training history as JSON"""
        history_path = self.save_dir / filename

        # Convert numpy types to Python types for JSON serialization
        json_history = {}
        for key, values in self.training_history.items():
            json_history[key] = [float(v) for v in values]

        with open(history_path, 'w') as f:
            json.dump(json_history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")

# Test function
def test_training_head():
    """Test training head functionality"""
    print("Testing training head...")

    from src_main.fuse.stack import FeatureFusionStack, FusionMLPHead

    # Create dummy data
    n_samples = 1000
    input_dim = 1343  # BERT(768) + CLIP(512) + CLIP_sim(1) + obj_delta(64)

    features = np.random.randn(n_samples, input_dim)
    labels = np.random.randint(0, 2, n_samples)

    # Split data
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Create model
    model = FusionMLPHead(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=1,
        dropout_rate=0.2
    )

    # Initialize training head
    trainer = TrainingHead(model, save_dir="test_models")

    # Train model
    history = trainer.train(
        train_features, train_labels,
        epochs=20,  # Small number for testing
        batch_size=32,
        learning_rate=1e-3
    )

    print(f"Training completed. Final val AUC: {history['val_auc'][-1]:.4f}")

    # Calibrate model
    val_features, calib_features, val_labels, calib_labels = train_test_split(
        train_features, train_labels, test_size=0.3, random_state=42
    )

    trainer.calibrate(calib_features, calib_labels, method="isotonic")

    # Evaluate model
    metrics = trainer.evaluate(test_features, test_labels)
    print(f"Test metrics: {metrics}")

    # Test prediction
    sample_predictions = trainer.predict(test_features[:10])
    print(f"Sample predictions: {sample_predictions}")

    print("Training head test completed!")

if __name__ == "__main__":
    test_training_head()
