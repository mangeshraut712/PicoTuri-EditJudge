"""
R2: Fusion Architecture Ablations
Experiment runner for fusion architecture comparison
"""

import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.features_text.bert import BERTTextEmbedder
from src.features_image.clip import CLIPImageEmbedder
from src.fuse.fusion import FusionHead
from experiments.utils import (
    get_device, benchmark_model, 
    format_time, MemoryMonitor
)

logger = logging.getLogger(__name__)

def load_models(config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Load models based on configuration."""
    device = get_device()
    
    # Load text encoder
    text_config = config['models']['text_encoder']
    text_encoder = BERTTextEmbedder(
        model_name=text_config['name'],
        max_length=text_config['max_length'],
        device=device
    )
    
    # Load image encoder
    image_config = config['models']['image_encoder']
    image_encoder = CLIPImageEmbedder(
        model_name=image_config['name'],
        image_size=image_config['image_size'],
        device=device
    )
    
    # Load fusion head
    fusion_config = config['models']['fusion_head']
    if fusion_config['type'] == 'logistic_regression':
        fusion_head = None  # Will be created during training
    else:
        fusion_head = FusionHead(
            input_dim=fusion_config['input_dim'],
            hidden_dims=fusion_config['hidden_dims'],
            output_dim=fusion_config['output_dim'],
            dropout=fusion_config['dropout']
        )
    
    return text_encoder, image_encoder, fusion_head

def extract_features_with_similarity(text_encoder, image_encoder, texts: list, image_paths: list, 
                                   config: Dict[str, Any]) -> np.ndarray:
    """Extract features with similarity and delta features."""
    logger.info(f"Extracting features for {len(texts)} samples")
    
    # Extract basic features
    text_embeddings = []
    for i, text in enumerate(texts):
        if i % 100 == 0:
            logger.info(f"Processing text {i}/{len(texts)}")
        embedding = text_encoder.encode(text)
        text_embeddings.append(embedding)
    
    text_features = np.array(text_embeddings)
    
    # Synthetic image features (in real scenario, would load and process actual images)
    image_features = np.random.randn(len(image_paths), image_encoder.hidden_size)
    
    # Combine basic features
    combined_features = np.concatenate([text_features, image_features], axis=1)
    
    # Add similarity features if enabled
    if config['features']['cross_modal_similarity']:
        # Synthetic cross-modal similarity features
        similarity_features = np.random.randn(len(texts), 256)
        combined_features = np.concatenate([combined_features, similarity_features], axis=1)
    
    # Add delta features if enabled
    if config['features']['delta_features']:
        # Synthetic delta features
        delta_features = np.random.randn(len(texts), 256)
        combined_features = np.concatenate([combined_features, delta_features], axis=1)
    
    logger.info(f"Feature extraction completed. Final shape: {combined_features.shape}")
    
    return combined_features

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray,
                            config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Train logistic regression fusion head."""
    logger.info("Training logistic regression fusion head")
    
    # Create logistic regression model
    fusion_config = config['models']['fusion_head']
    clf = LogisticRegression(
        penalty=fusion_config['regularization'],
        C=fusion_config['C'],
        max_iter=1000,
        random_state=42
    )
    
    # Train model
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_proba)
    }
    
    logger.info(f"Logistic regression training completed. Val AUC: {metrics['auc']:.4f}")
    
    return clf, metrics

def train_mlp_fusion(fusion_head, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_val: torch.Tensor, y_val: torch.Tensor,
                   config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Train MLP fusion head."""
    logger.info("Training MLP fusion head")
    
    device = get_device()
    fusion_head = fusion_head.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        fusion_head.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        # Training
        fusion_head.train()
        epoch_train_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = fusion_head(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(fusion_head.parameters(), config['training']['gradient_clip_norm'])
            
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / (len(X_train) // batch_size)
        train_losses.append(avg_train_loss)
        
        # Validation
        fusion_head.eval()
        with torch.no_grad():
            val_outputs = fusion_head(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)
        
        if epoch % 2 == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Final evaluation
    fusion_head.eval()
    with torch.no_grad():
        test_outputs = fusion_head(X_val).squeeze()
        test_probs = torch.sigmoid(test_outputs)
        
        # Calculate metrics
        test_preds = (test_probs > 0.5).float()
        accuracy = (test_preds == y_val).float().mean().item()
        
        # Convert to numpy for sklearn metrics
        y_val_np = y_val.cpu().numpy()
        test_probs_np = test_probs.cpu().numpy()
        test_preds_np = test_preds.cpu().numpy()
        
        test_auc = roc_auc_score(y_val_np, test_probs_np)
        test_f1 = f1_score(y_val_np, test_preds_np)
        test_precision = precision_score(y_val_np, test_preds_np)
        test_recall = recall_score(y_val_np, test_preds_np)
    
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'test_accuracy': accuracy,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall
    }
    
    logger.info(f"MLP training completed. Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")
    
    return fusion_head, training_results

def apply_calibration(model, X_val: np.ndarray, y_val: np.ndarray, 
                     config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Apply calibration to model predictions."""
    logger.info(f"Applying {config['evaluation']['calibration']['method']} calibration")
    
    # Get validation predictions
    if hasattr(model, 'predict_proba'):
        # sklearn model
        val_proba = model.predict_proba(X_val)[:, 1]
    else:
        # PyTorch model
        device = get_device()
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        with torch.no_grad():
            val_logits = model(X_val_tensor).squeeze()
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
    
    calibration_method = config['evaluation']['calibration']['method']
    
    if calibration_method == "platt_scaling":
        # Platt scaling (logistic regression calibration)
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        calibrated_model.fit(X_val, y_val)
        
        # Get calibrated predictions
        calibrated_proba = calibrated_model.predict_proba(X_val)[:, 1]
        
    elif calibration_method == "isotonic_regression":
        # Isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        calibrated_proba = iso_reg.fit_transform(val_proba, y_val)
        
        # Create a wrapper for the calibrated model
        class CalibratedModel:
            def __init__(self, base_model, iso_reg):
                self.base_model = base_model
                self.iso_reg = iso_reg
            
            def predict_proba(self, X):
                if hasattr(self.base_model, 'predict_proba'):
                    proba = self.base_model.predict_proba(X)[:, 1]
                else:
                    device = get_device()
                    X_tensor = torch.FloatTensor(X).to(device)
                    with torch.no_grad():
                        logits = self.base_model(X_tensor).squeeze()
                        proba = torch.sigmoid(logits).cpu().numpy()
                calibrated = self.iso_reg.transform(proba)
                return np.column_stack([1 - calibrated, calibrated])
        
        calibrated_model = CalibratedModel(model, iso_reg)
    
    else:
        logger.warning(f"Unknown calibration method: {calibration_method}")
        return model, {}
    
    # Calculate calibration metrics
    ece = calculate_ece(y_val, calibrated_proba)
    brier = brier_score_loss(y_val, calibrated_proba)
    
    calibration_metrics = {
        'ece': ece,
        'brier_score': brier,
        'method': calibration_method
    }
    
    logger.info(f"Calibration completed. ECE: {ece:.4f}, Brier: {brier:.4f}")
    
    return calibrated_model, calibration_metrics

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Update ECE
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece

def run_r2_experiment(config: Dict[str, Any], dirs: Dict[str, Path], seed: int) -> Dict[str, Any]:
    """Run R2 fusion architecture experiment."""
    logger.info(f"Running R2 experiment with seed {seed}")
    
    # Setup memory monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    start_time = time.time()
    
    try:
        # Load models
        logger.info("Loading models...")
        text_encoder, image_encoder, fusion_head = load_models(config)
        
        # Create synthetic dataset
        logger.info("Creating synthetic dataset...")
        instructions = [
            "Make the image brighter",
            "Add more contrast", 
            "Adjust the saturation",
            "Crop the image",
            "Apply a vintage filter"
        ]
        
        num_samples = 1000
        texts = np.random.choice(instructions, num_samples)
        scores = np.random.beta(2, 2, num_samples)
        image_paths = [f"synthetic_image_{i}.jpg" for i in range(num_samples)]
        
        # Extract features
        features = extract_features_with_similarity(text_encoder, image_encoder, list(texts), list(image_paths), config)
        
        # Split data
        train_size = int(len(features) * config['training']['data']['train_split'])
        val_size = int(len(features) * config['training']['data']['val_split'])
        
        indices = np.random.permutation(len(features))
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        X_train, y_train = features[train_idx], scores[train_idx]
        X_val, y_val = features[val_idx], scores[val_idx]
        X_test, y_test = features[test_idx], scores[test_idx]
        
        # Train fusion head
        fusion_config = config['models']['fusion_head']
        if fusion_config['type'] == 'logistic_regression':
            trained_model, training_results = train_logistic_regression(X_train, y_train, X_val, y_val, config)
        else:
            # Convert to tensors for MLP
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            trained_model, training_results = train_mlp_fusion(
                fusion_head, X_train_tensor, y_train_tensor, 
                X_val_tensor, y_val_tensor, config
            )
        
        # Apply calibration
        calibrated_model, calibration_results = apply_calibration(trained_model, X_val, y_val, config)
        
        # Final evaluation on test set
        if hasattr(calibrated_model, 'predict_proba'):
            test_proba = calibrated_model.predict_proba(X_test)[:, 1]
        else:
            device = get_device()
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            with torch.no_grad():
                test_logits = calibrated_model(X_test_tensor).squeeze()
                test_proba = torch.sigmoid(test_logits).cpu().numpy()
        
        test_preds = (test_proba > 0.5).astype(int)
        
        # Calculate final metrics
        final_metrics = {
            'test_accuracy': accuracy_score(y_test, test_preds),
            'test_f1': f1_score(y_test, test_preds),
            'test_precision': precision_score(y_test, test_preds),
            'test_recall': recall_score(y_test, test_preds),
            'test_auc': roc_auc_score(y_test, test_proba),
            'test_ece': calculate_ece(y_test, test_proba),
            'test_brier': brier_score_loss(y_test, test_proba)
        }
        
        # Benchmark model
        if fusion_config['type'] != 'logistic_regression':
            benchmark_results = benchmark_models(trained_model, config)
        else:
            benchmark_results = {'sklearn_model': 'no_benchmark'}
        
        # Compile results
        total_time = time.time() - start_time
        memory_monitor.update_peak()
        
        results = {
            'experiment_type': 'r2_fusion',
            'seed': seed,
            'config': config,
            'training': training_results,
            'calibration': calibration_results,
            'final_metrics': final_metrics,
            'benchmark': benchmark_results,
            'timing': {
                'total_time_seconds': total_time,
                'formatted_time': format_time(total_time)
            },
            'memory': {
                'peak_memory_mb': memory_monitor.get_peak_memory() / (1024 * 1024)
            },
            'status': 'completed'
        }
        
        # Extract key metrics for easy access
        results['metrics'] = {
            'auc': final_metrics['test_auc'],
            'f1': final_metrics['test_f1'],
            'accuracy': final_metrics['test_accuracy'],
            'ece': final_metrics['test_ece'],
            'brier_score': final_metrics['test_brier']
        }
        
        # Save trained model
        if config['output']['save_model']:
            if fusion_config['type'] != 'logistic_regression':
                model_path = dirs['models'] / f"fusion_head_seed_{seed}.pt"
                torch.save(trained_model.state_dict(), model_path)
            else:
                import joblib
                model_path = dirs['models'] / f"lr_model_seed_{seed}.joblib"
                joblib.dump(trained_model, model_path)
            logger.info(f"Model saved to {model_path}")
        
        logger.info(f"R2 experiment completed successfully in {format_time(total_time)}")
        
        return results
        
    except Exception as e:
        logger.error(f"R2 experiment failed: {e}")
        return {
            'experiment_type': 'r2_fusion',
            'seed': seed,
            'error': str(e),
            'status': 'failed'
        }

def benchmark_models(model, config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark trained model performance."""
    if hasattr(model, 'predict_proba'):
        # sklearn model - skip detailed benchmarking
        return {'model_type': 'sklearn', 'benchmark_skipped': True}
    
    # PyTorch model benchmarking
    device = get_device()
    model.eval()
    
    sample_input = torch.randn(1, config['models']['fusion_head']['input_dim']).to(device)
    
    stats = benchmark_model(
        model,
        sample_input,
        num_runs=100,
        warmup_runs=20
    )
    
    return {'model_type': 'pytorch', 'latency_stats': stats}
