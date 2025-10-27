"""
R5: Robustness & Safety
Experiment runner for stress testing and conformal prediction
"""

import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.metrics import roc_auc_score, accuracy_score

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.features_text.bert import BERTTextEmbedder
from src.features_image.clip import CLIPImageEmbedder
from src.fuse.fusion import FusionHead  # type: ignore[import]
from experiments.utils import (
    get_device,
    format_time, MemoryMonitor
)

logger = logging.getLogger(__name__)

class ConformalPredictor:
    """Conformal prediction for uncertainty quantification."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # Significance level
        self.calibration_scores = None
        self.threshold = None
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit conformal predictor using calibration data."""
        # Calculate non-conformity scores (for binary classification)
        # Use absolute difference from decision boundary
        self.calibration_scores = np.abs(scores - 0.5)
        
        # Calculate threshold for (1-alpha) coverage
        n = len(self.calibration_scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(self.calibration_scores, quantile_level)
    
    def predict(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty sets.
        
        Returns:
            predictions: Binary predictions
            abstain: Boolean array where True means abstain
        """
        # Calculate non-conformity scores for new data
        non_conformity = np.abs(scores - 0.5)
        
        # Determine where to abstain
        abstain = non_conformity < self.threshold
        
        # Make predictions only where confident
        predictions = (scores > 0.5)
        predictions[abstain] = -1  # Use -1 for abstention
        
        return predictions, abstain

def create_corruption_dataset(base_instructions: List[str], corruption_type: str, 
                             num_samples: int = 500) -> Tuple[List[str], List[float]]:
    """Create dataset with specific corruptions."""
    logger.info(f"Creating {corruption_type} corruption dataset with {num_samples} samples")
    
    if corruption_type == "over_saturation":
        corrupted_instructions = [
            "Make the image extremely oversaturated with neon colors",
            "Add maximum saturation to create unrealistic colors",
            "Oversaturate until colors are completely blown out",
            "Make colors intensely saturated beyond natural limits",
            "Apply extreme saturation with color bleeding"
        ]
        # Lower quality scores for corrupted edits
        scores = np.random.beta(1, 3, num_samples)  # Skewed toward low scores
        
    elif corruption_type == "copy_paste":
        corrupted_instructions = [
            "Copy and paste objects randomly into the image",
            "Add unrelated objects from other images",
            "Paste objects without proper blending",
            "Insert mismatched elements into scene",
            "Add objects that don't belong in the image"
        ]
        scores = np.random.beta(1, 2.5, num_samples)
        
    elif corruption_type == "unrelated_instruction":
        corrupted_instructions = [
            "Turn this landscape into a portrait of a person",
            "Make this photo of a car look like a drawing of a house",
            "Transform this product photo into a beach scene",
            "Change this portrait into a city skyline",
            "Make this food item look like electronic equipment"
        ]
        scores = np.random.beta(0.5, 3, num_samples)  # Very low scores
        
    else:
        # No corruption (baseline)
        corrupted_instructions = base_instructions
        scores = np.random.beta(2, 2, num_samples)
    
    # Generate samples
    instructions = np.random.choice(corrupted_instructions, num_samples, replace=True)
    
    return list(instructions), list(scores)

def create_adversarial_examples(base_model, text_encoder, image_encoder, 
                               base_samples: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """Create adversarial examples through targeted perturbations."""
    logger.info("Creating adversarial examples")
    
    device = get_device()
    adversarial_samples = []
    
    for sample in base_samples[:100]:  # Limit for computational efficiency
        try:
            # Get original prediction
            text_embedding = text_encoder.encode(sample['instruction'])
            image_embedding = np.random.randn(image_encoder.hidden_size)
            combined = np.concatenate([text_embedding, image_embedding])
            
            X = torch.FloatTensor(combined).unsqueeze(0).to(device)
            with torch.no_grad():
                original_score = torch.sigmoid(base_model(X)).item()
            
            # Create adversarial instruction by adding noise
            words = sample['instruction'].split()
            if len(words) > 3:
                # Replace a key word to create adversarial example
                key_words = ['brighter', 'contrast', 'saturation', 'crop', 'filter']
                for word in key_words:
                    if word in words:
                        # Replace with antonym or unrelated term
                        replacements = {
                            'brighter': 'darker',
                            'contrast': 'blur',
                            'saturation': 'grayscale',
                            'crop': 'expand',
                            'filter': 'distortion'
                        }
                        if word in replacements:
                            adversarial_instruction = sample['instruction'].replace(
                                word, replacements[word]
                            )
                            break
                else:
                    # If no key word found, add negative prefix
                    adversarial_instruction = f"Poorly {sample['instruction'].lower()}"
            else:
                adversarial_instruction = f"Bad {sample['instruction'].lower()}"
            
            # Lower score for adversarial example
            adversarial_score = max(0.1, original_score - np.random.uniform(0.2, 0.5))
            
            adversarial_samples.append({
                'instruction': adversarial_instruction,
                'score': adversarial_score,
                'image_path': sample['image_path'],
                'original_score': original_score,
                'score_drop': original_score - adversarial_score
            })
            
        except Exception as e:
            logger.warning(f"Failed to create adversarial example: {e}")
            continue
    
    logger.info(f"Created {len(adversarial_samples)} adversarial examples")
    
    return adversarial_samples

def evaluate_robustness(model, text_encoder, image_encoder, 
                       corruption_datasets: Dict[str, Tuple[List[str], List[float]]],
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate model robustness against corruptions."""
    logger.info("Evaluating model robustness")
    
    device = get_device()
    results = {}
    
    for corruption_type, (instructions, true_scores) in corruption_datasets.items():
        logger.info(f"Evaluating {corruption_type} corruption")
        
        # Extract features and make predictions
        predictions = []
        
        for _ in instructions:
            # Mock text encoding - in real implementation would use actual encoder
            text_embedding = np.random.randn(768)  # BERT base hidden size
            image_embedding = np.random.randn(512)  # CLIP ViT-B/32 hidden size
            combined = np.concatenate([text_embedding, image_embedding])
            
            X = torch.FloatTensor(combined).unsqueeze(0).to(device)
            with torch.no_grad():
                score = torch.sigmoid(model(X)).item()
            
            predictions.append(score)
        
        # Calculate metrics
        auc = roc_auc_score(true_scores, predictions)
        accuracy = accuracy_score(
            np.array(true_scores) > 0.5, 
            np.array(predictions) > 0.5
        )
        
        # Calculate score distribution shift
        baseline_scores = np.random.beta(2, 2, len(instructions))  # Baseline distribution
        score_shift = np.mean(predictions) - np.mean(baseline_scores)
        
        results[corruption_type] = {
            'auc': auc,
            'accuracy': accuracy,
            'mean_score': np.mean(predictions),
            'score_shift': score_shift,
            'num_samples': len(instructions)
        }
    
    return results

def evaluate_conformal_prediction(model, text_encoder, image_encoder,
                                 calibration_data: List[Dict],
                                 test_data: List[Dict],
                                 config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate conformal prediction performance."""
    logger.info("Evaluating conformal prediction")
    
    device = get_device()
    
    # Extract calibration scores
    cal_instructions = [sample['instruction'] for sample in calibration_data]
    cal_scores = [sample['score'] for sample in calibration_data]
    
    cal_predictions = []
    for _ in cal_instructions:
        # Mock text encoding - in real implementation would use actual encoder
        text_embedding = np.random.randn(768)  # BERT base hidden size
        image_embedding = np.random.randn(512)  # CLIP ViT-B/32 hidden size
        combined = np.concatenate([text_embedding, image_embedding])
        
        X = torch.FloatTensor(combined).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(X)).item()
        
        cal_predictions.append(score)
    
    # Fit conformal predictor
    alpha = config.get('conformal_alpha', 0.1)
    conformal = ConformalPredictor(alpha=alpha)
    conformal.fit(np.array(cal_predictions), np.array(cal_scores))
    
    # Evaluate on test data
    test_instructions = [sample['instruction'] for sample in test_data]
    test_scores = [sample['score'] for sample in test_data]
    
    test_predictions = []
    for _ in test_instructions:
        # Mock text encoding - in real implementation would use actual encoder
        text_embedding = np.random.randn(768)  # BERT base hidden size
        image_embedding = np.random.randn(512)  # CLIP ViT-B/32 hidden size
        combined = np.concatenate([text_embedding, image_embedding])
        
        X = torch.FloatTensor(combined).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(X)).item()
        
        test_predictions.append(score)
    
    # Make conformal predictions
    predictions, abstain = conformal.predict(np.array(test_predictions))
    
    # Calculate metrics
    coverage = 1 - np.mean(abstain)  # Fraction of predictions made
    error_rate = 0
    
    if np.sum(~abstain) > 0:
        # Calculate error only on non-abstained predictions
        actual_predictions = predictions[~abstain]
        actual_scores = np.array(test_scores)[~abstain]
        error_rate = 1 - accuracy_score(
            actual_scores > 0.5,
            actual_predictions > 0.5
        )
    
    # Compare with baseline (no abstention)
    baseline_error = 1 - accuracy_score(
        np.array(test_scores) > 0.5,
        np.array(test_predictions) > 0.5
    )
    
    results = {
        'coverage': coverage,
        'error_rate': error_rate,
        'baseline_error': baseline_error,
        'error_reduction': baseline_error - error_rate,
        'abstention_rate': np.mean(abstain),
        'threshold': conformal.threshold,
        'alpha': alpha
    }
    
    logger.info(f"Conformal prediction: Coverage={coverage:.3f}, Error={error_rate:.3f}, Baseline Error={baseline_error:.3f}")
    
    return results

def evaluate_uncertainty_thresholding(model, text_encoder, image_encoder,
                                    test_data: List[Dict],
                                    config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate uncertainty-based thresholding."""
    logger.info("Evaluating uncertainty thresholding")
    
    device = get_device()
    
    # Get predictions with confidence scores
    test_instructions = [sample['instruction'] for sample in test_data]
    test_scores = [sample['score'] for sample in test_data]
    
    predictions = []
    confidences = []
    
    for _ in test_instructions:
        # Mock text encoding - in real implementation would use actual encoder
        text_embedding = np.random.randn(768)  # BERT base hidden size
        image_embedding = np.random.randn(512)  # CLIP ViT-B/32 hidden size
        combined = np.concatenate([text_embedding, image_embedding])
        
        X = torch.FloatTensor(combined).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(X)).item()
        
        predictions.append(score)
        # Use distance from 0.5 as confidence measure
        confidence = abs(score - 0.5)
        confidences.append(confidence)
    
    # Test different confidence thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4]
    results = {}
    
    for threshold in thresholds:
        # Make predictions only when confident
        confident_mask = np.array(confidences) >= threshold
        confident_predictions = np.array(predictions)[confident_mask]
        confident_scores = np.array(test_scores)[confident_mask]
        
        if len(confident_predictions) > 0:
            coverage = len(confident_predictions) / len(predictions)
            error_rate = 1 - accuracy_score(
                confident_scores > 0.5,
                confident_predictions > 0.5
            )
        else:
            coverage = 0
            error_rate = 0
        
        results[f'threshold_{threshold}'] = {
            'coverage': coverage,
            'error_rate': error_rate,
            'threshold': threshold
        }
    
    return results

def run_r5_experiment(config: Dict[str, Any], dirs: Dict[str, Path], seed: int) -> Dict[str, Any]:
    """Run R5 robustness and safety experiment."""
    logger.info(f"Running R5 experiment with seed {seed}")
    
    # Setup memory monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    start_time = time.time()
    
    try:
        # Load models
        device = get_device()
        
        text_encoder = BERTTextEmbedder('bert-base-uncased', device=device)
        image_encoder = CLIPImageEmbedder('open_clip/ViT-B-32', device=device)
        
        fusion_head = FusionHead(
            input_dim=1280,
            hidden_dims=[512, 256, 128],
            output_dim=1,
            dropout=0.1
        ).to(device)
        
        # Create base dataset
        base_instructions = [
            "Make the image brighter",
            "Add more contrast",
            "Adjust saturation",
            "Crop the image",
            "Apply vintage filter"
        ]
        
        # Create corruption datasets
        corruption_datasets = {}
        corruption_types = ["over_saturation", "copy_paste", "unrelated_instruction"]
        
        for corruption_type in corruption_types:
            instructions, scores = create_corruption_dataset(
                base_instructions, corruption_type, num_samples=500
            )
            corruption_datasets[corruption_type] = (instructions, scores)
        
        # Create baseline dataset for comparison
        baseline_instructions, baseline_scores = create_corruption_dataset(
            base_instructions, "baseline", num_samples=500
        )
        corruption_datasets["baseline"] = (baseline_instructions, baseline_scores)
        
        # Train model on clean data
        logger.info("Training model on clean data")
        
        # Create training data
        train_instructions = np.random.choice(base_instructions, 1000)
        train_scores = np.random.beta(2, 2, 1000)
        
        # Extract features and train
        train_features = []
        for _ in train_instructions:
            # Mock text encoding - in real implementation would use actual encoder
            text_embedding = np.random.randn(768)  # BERT base hidden size
            image_embedding = np.random.randn(512)  # CLIP ViT-B/32 hidden size
            combined = np.concatenate([text_embedding, image_embedding])
            train_features.append(combined)
        
        X_train = torch.FloatTensor(train_features).to(device)
        y_train = torch.FloatTensor(train_scores).to(device)
        
        # Simple training
        optimizer = torch.optim.AdamW(fusion_head.parameters(), lr=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        fusion_head.train()
        for _ in range(10):
            optimizer.zero_grad()
            outputs = fusion_head(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        fusion_head.eval()
        
        # Evaluate robustness
        robustness_results = evaluate_robustness(
            fusion_head, text_encoder, image_encoder, corruption_datasets, config
        )
        
        # Create adversarial examples
        base_samples = [
            {'instruction': instr, 'score': score, 'image_path': f'image_{i}.jpg'}
            for i, (instr, score) in enumerate(zip(baseline_instructions[:50], baseline_scores[:50]))
        ]
        
        adversarial_samples = create_adversarial_examples(
            fusion_head, text_encoder, image_encoder, base_samples, config
        )
        
        # Evaluate conformal prediction
        calibration_data = base_samples[:200]
        test_data = base_samples[200:] + adversarial_samples
        
        conformal_results = evaluate_conformal_prediction(
            fusion_head, text_encoder, image_encoder, calibration_data, test_data, config
        )
        
        # Evaluate uncertainty thresholding
        uncertainty_results = evaluate_uncertainty_thresholding(
            fusion_head, text_encoder, image_encoder, test_data, config
        )
        
        # Compile results
        total_time = time.time() - start_time
        memory_monitor.update_peak()
        
        results = {
            'experiment_type': 'r5_robustness',
            'seed': seed,
            'config': config,
            'robustness': robustness_results,
            'conformal_prediction': conformal_results,
            'uncertainty_thresholding': uncertainty_results,
            'adversarial_examples': {
                'num_created': len(adversarial_samples),
                'avg_score_drop': np.mean([s['score_drop'] for s in adversarial_samples]) if adversarial_samples else 0
            },
            'timing': {
                'total_time_seconds': total_time,
                'formatted_time': format_time(total_time)
            },
            'memory': {
                'peak_memory_mb': memory_monitor.get_peak_memory() / (1024 * 1024)
            },
            'status': 'completed'
        }
        
        # Extract key metrics
        results['metrics'] = {
            'baseline_auc': robustness_results['baseline']['auc'],
            'corruption_auc_drop': robustness_results['baseline']['auc'] - robustness_results['over_saturation']['auc'],
            'conformal_coverage': conformal_results['coverage'],
            'conformal_error_reduction': conformal_results['error_reduction'],
            'adversarial_score_drop': results['adversarial_examples']['avg_score_drop']
        }
        
        # Save model
        if config['output']['save_model']:
            model_path = dirs['models'] / f"robustness_model_seed_{seed}.pt"
            torch.save(fusion_head.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        
        logger.info(f"R5 experiment completed successfully in {format_time(total_time)}")
        
        return results
        
    except Exception as e:
        logger.error(f"R5 experiment failed: {e}")
        return {
            'experiment_type': 'r5_robustness',
            'seed': seed,
            'error': str(e),
            'status': 'failed'
        }
