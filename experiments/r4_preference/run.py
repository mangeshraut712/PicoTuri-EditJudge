"""
R4: Preference Learning for Ranking
Experiment runner for pairwise and listwise learning
"""

import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau

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

class PreferenceLoss(torch.nn.Module):
    """Pairwise preference learning loss."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores_good: torch.Tensor, scores_bad: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise preference loss.
        
        Args:
            scores_good: Scores for preferred edits
            scores_bad: Scores for non-preferred edits
            
        Returns:
            Pairwise hinge loss
        """
        # Preference: good should have higher score than bad
        loss = torch.relu(self.margin - (scores_good - scores_bad))
        return loss.mean()

class ListwiseLoss(torch.nn.Module):
    """Listwise learning to rank loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        """
        Compute listwise loss using ListNet approach.
        
        Args:
            scores: Predicted scores
            relevance: True relevance scores
            
        Returns:
            Listwise cross-entropy loss
        """
        # Compute probability distributions
        scores_probs = torch.softmax(scores, dim=0)
        relevance_probs = torch.softmax(relevance, dim=0)
        
        # Cross-entropy loss
        loss = -torch.sum(relevance_probs * torch.log(scores_probs + 1e-8))
        return loss

def create_preference_dataset(num_samples: int = 1000) -> Tuple[List[Dict], List[Dict]]:
    """Create pairwise preference dataset."""
    logger.info(f"Creating preference dataset with {num_samples} pairs")
    
    instructions = [
        "Make the image brighter",
        "Add more contrast",
        "Adjust saturation",
        "Crop the image",
        "Apply vintage filter",
        "Enhance colors",
        "Add blur effect",
        "Sharpen details"
    ]
    
    preference_pairs = []
    individual_samples = []
    
    for i in range(num_samples):
        # Create a pair of edits for the same image
        base_instruction = np.random.choice(instructions)
        
        # Generate "good" edit (higher quality)
        good_instruction = f"Carefully {base_instruction.lower()} with professional quality"
        good_score = np.random.beta(3, 1)  # Skewed toward higher scores
        
        # Generate "bad" edit (lower quality)
        bad_instruction = f"Poorly {base_instruction.lower()} with artifacts"
        bad_score = np.random.beta(1, 3)  # Skewed toward lower scores
        
        # Ensure preference ordering
        if good_score <= bad_score:
            good_score, bad_score = bad_score, good_score
            good_instruction, bad_instruction = bad_instruction, good_instruction
        
        pair = {
            'image_id': f"image_{i}",
            'good_edit': {
                'instruction': good_instruction,
                'score': good_score,
                'image_path': f"image_{i}_good.jpg"
            },
            'bad_edit': {
                'instruction': bad_instruction,
                'score': bad_score,
                'image_path': f"image_{i}_bad.jpg"
            }
        }
        
        preference_pairs.append(pair)
        
        # Also store individual samples for listwise learning
        individual_samples.extend([
            {
                'instruction': good_instruction,
                'score': good_score,
                'image_path': f"image_{i}_good.jpg",
                'image_id': f"image_{i}"
            },
            {
                'instruction': bad_instruction,
                'score': bad_score,
                'image_path': f"image_{i}_bad.jpg",
                'image_id': f"image_{i}"
            }
        ])
    
    return preference_pairs, individual_samples

def extract_features_for_samples(text_encoder, image_encoder, samples: List[Dict]) -> np.ndarray:
    """Extract features for a list of samples."""
    logger.info(f"Extracting features for {len(samples)} samples")
    
    features = []
    
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            logger.info(f"Processing sample {i}/{len(samples)}")
        
        # Text embedding
        text_embedding = text_encoder.encode(sample['instruction'])
        
        # Synthetic image embedding
        image_embedding = np.random.randn(image_encoder.hidden_size)
        
        # Combine features
        combined = np.concatenate([text_embedding, image_embedding])
        features.append(combined)
    
    return np.array(features)

def train_pointwise_model(fusion_head, features: np.ndarray, scores: np.ndarray, 
                         config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Train pointwise model (standard regression)."""
    logger.info("Training pointwise model")
    
    device = get_device()
    fusion_head = fusion_head.to(device)
    
    X = torch.FloatTensor(features).to(device)
    y = torch.FloatTensor(scores).to(device)
    
    # Split data
    train_size = int(len(X) * 0.8)
    indices = torch.randperm(len(X))
    X_train, y_train = X[indices[:train_size]], y[indices[:train_size]]
    X_val, y_val = X[indices[train_size:]], y[indices[train_size:]]
    
    # Training setup
    optimizer = torch.optim.AdamW(
        fusion_head.parameters(),
        lr=config['training']['learning_rate']
    )
    
    criterion = torch.nn.MSELoss()
    
    # Training loop
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    
    train_losses = []
    
    for epoch in range(epochs):
        fusion_head.train()
        epoch_loss = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = fusion_head(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(X_train) // batch_size)
        train_losses.append(avg_loss)
        
        if epoch % 2 == 0:
            logger.info(f"Pointwise epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # Validation
    fusion_head.eval()
    with torch.no_grad():
        val_outputs = fusion_head(X_val).squeeze()
        val_loss = criterion(val_outputs, y_val).item()
    
    training_results = {
        'train_losses': train_losses,
        'final_train_loss': train_losses[-1],
        'val_loss': val_loss
    }
    
    logger.info(f"Pointwise training completed. Val loss: {val_loss:.4f}")
    
    return fusion_head, training_results

def train_pairwise_model(fusion_head, preference_pairs: List[Dict], config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Train pairwise preference model."""
    logger.info("Training pairwise preference model")
    
    device = get_device()
    fusion_head = fusion_head.to(device)
    
    # Extract features for preference pairs
    good_samples = [pair['good_edit'] for pair in preference_pairs]
    bad_samples = [pair['bad_edit'] for pair in preference_pairs]
    
    # Load encoders
    text_encoder = BERTTextEmbedder('bert-base-uncased', device=device)
    image_encoder = CLIPImageEmbedder('open_clip/ViT-B-32', device=device)
    
    good_features = extract_features_for_samples(text_encoder, image_encoder, good_samples)
    bad_features = extract_features_for_samples(text_encoder, image_encoder, bad_samples)
    
    X_good = torch.FloatTensor(good_features).to(device)
    X_bad = torch.FloatTensor(bad_features).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        fusion_head.parameters(),
        lr=config['training']['learning_rate']
    )
    
    criterion = PreferenceLoss(margin=1.0)
    
    # Training loop
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    
    train_losses = []
    
    for epoch in range(epochs):
        fusion_head.train()
        epoch_loss = 0
        
        for i in range(0, len(X_good), batch_size):
            batch_good = X_good[i:i+batch_size]
            batch_bad = X_bad[i:i+batch_size]
            
            optimizer.zero_grad()
            
            scores_good = fusion_head(batch_good).squeeze()
            scores_bad = fusion_head(batch_bad).squeeze()
            
            loss = criterion(scores_good, scores_bad)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(X_good) // batch_size)
        train_losses.append(avg_loss)
        
        if epoch % 2 == 0:
            logger.info(f"Pairwise epoch {epoch}: Loss = {avg_loss:.4f}")
    
    training_results = {
        'train_losses': train_losses,
        'final_train_loss': train_losses[-1]
    }
    
    logger.info(f"Pairwise training completed. Final loss: {train_losses[-1]:.4f}")
    
    return fusion_head, training_results

def train_listwise_model(fusion_head, individual_samples: List[Dict], config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Train listwise learning to rank model."""
    logger.info("Training listwise model")
    
    device = get_device()
    fusion_head = fusion_head.to(device)
    
    # Group samples by image for listwise training
    image_groups = {}
    for sample in individual_samples:
        image_id = sample['image_id']
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(sample)
    
    # Create training data (lists of edits for each image)
    text_encoder = BERTTextEmbedder('bert-base-uncased', device=device)
    image_encoder = CLIPImageEmbedder('open_clip/ViT-B-32', device=device)
    
    list_data = []
    for image_id, samples in image_groups.items():
        if len(samples) >= 2:  # Need at least 2 samples for ranking
            features = extract_features_for_samples(text_encoder, image_encoder, samples)
            scores = [sample['score'] for sample in samples]
            list_data.append((features, scores))
    
    # Training setup
    optimizer = torch.optim.AdamW(
        fusion_head.parameters(),
        lr=config['training']['learning_rate']
    )
    
    criterion = ListwiseLoss()
    
    # Training loop
    epochs = config['training']['epochs']
    train_losses = []
    
    for epoch in range(epochs):
        fusion_head.train()
        epoch_loss = 0
        
        for features, scores in list_data:
            X = torch.FloatTensor(features).to(device)
            y = torch.FloatTensor(scores).to(device)
            
            optimizer.zero_grad()
            
            predicted_scores = fusion_head(X).squeeze()
            loss = criterion(predicted_scores, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(list_data)
        train_losses.append(avg_loss)
        
        if epoch % 2 == 0:
            logger.info(f"Listwise epoch {epoch}: Loss = {avg_loss:.4f}")
    
    training_results = {
        'train_losses': train_losses,
        'final_train_loss': train_losses[-1]
    }
    
    logger.info(f"Listwise training completed. Final loss: {train_losses[-1]:.4f}")
    
    return fusion_head, training_results

def evaluate_ranking_performance(models: Dict[str, Any], test_samples: List[Dict], 
                                config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate ranking performance using NDCG and Kendall's tau."""
    logger.info("Evaluating ranking performance")
    
    device = get_device()
    
    # Group test samples by image
    image_groups = {}
    for sample in test_samples:
        image_id = sample['image_id']
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(sample)
    
    # Extract features for evaluation
    text_encoder = BERTTextEmbedder('bert-base-uncased', device=device)
    image_encoder = CLIPImageEmbedder('open_clip/ViT-B-32', device=device)
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        
        all_ndcg_scores = []
        all_kendall_scores = []
        
        for image_id, samples in image_groups.items():
            if len(samples) >= 2:
                # Extract features
                features = extract_features_for_samples(text_encoder, image_encoder, samples)
                true_scores = [sample['score'] for sample in samples]
                
                # Get predictions
                X = torch.FloatTensor(features).to(device)
                with torch.no_grad():
                    predicted_scores = model(X).squeeze().cpu().numpy()
                
                # Calculate NDCG@k
                for k in [5, 10]:
                    if len(samples) >= k:
                        # Reshape for sklearn's ndcg_score
                        true_scores_2d = np.array([true_scores])
                        pred_scores_2d = np.array([predicted_scores])
                        
                        ndcg = ndcg_score(true_scores_2d, pred_scores_2d, k=k)
                        all_ndcg_scores.append(ndcg)
                
                # Calculate Kendall's tau
                kendall_val, _ = kendalltau(true_scores, predicted_scores)
                all_kendall_scores.append(kendall_val)
        
        # Aggregate results
        results[model_name] = {
            'ndcg_at_5': np.mean([s for s in all_ndcg_scores if not np.isnan(s)]) if all_ndcg_scores else 0,
            'ndcg_at_10': np.mean([s for s in all_ndcg_scores if not np.isnan(s)]) if all_ndcg_scores else 0,
            'kendall_tau': np.mean([s for s in all_kendall_scores if not np.isnan(s)]) if all_kendall_scores else 0,
            'num_queries': len(image_groups)
        }
    
    logger.info("Ranking evaluation completed")
    
    return results

def run_r4_experiment(config: Dict[str, Any], dirs: Dict[str, Path], seed: int) -> Dict[str, Any]:
    """Run R4 preference learning experiment."""
    logger.info(f"Running R4 experiment with seed {seed}")
    
    # Setup memory monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    start_time = time.time()
    
    try:
        # Create datasets
        preference_pairs, individual_samples = create_preference_dataset(num_samples=1000)
        
        # Split data
        train_size = int(len(preference_pairs) * 0.8)
        train_pairs = preference_pairs[:train_size]
        # test_pairs = preference_pairs[train_size:]  # Not used in this experiment
        
        train_samples = individual_samples[:train_size * 2]
        test_samples = individual_samples[train_size * 2:]
        
        # Create models
        device = get_device()
        
        # Pointwise model
        pointwise_head = FusionHead(
            input_dim=1280,
            hidden_dims=[512, 256, 128],
            output_dim=1,
            dropout=0.1
        )
        
        # Pairwise model
        pairwise_head = FusionHead(
            input_dim=1280,
            hidden_dims=[512, 256, 128],
            output_dim=1,
            dropout=0.1
        )
        
        # Listwise model
        listwise_head = FusionHead(
            input_dim=1280,
            hidden_dims=[512, 256, 128],
            output_dim=1,
            dropout=0.1
        )
        
        # Train models
        logger.info("Training preference learning models")
        
        # Pointwise training
        pointwise_features = extract_features_for_samples(
            BERTTextEmbedder('bert-base-uncased', device=device),
            CLIPImageEmbedder('open_clip/ViT-B-32', device=device),
            train_samples
        )
        pointwise_scores = [sample['score'] for sample in train_samples]
        
        pointwise_model, pointwise_results = train_pointwise_model(
            pointwise_head, pointwise_features, np.array(pointwise_scores), config
        )
        
        # Pairwise training
        pairwise_model, pairwise_results = train_pairwise_model(
            pairwise_head, train_pairs, config
        )
        
        # Listwise training
        listwise_model, listwise_results = train_listwise_model(
            listwise_head, train_samples, config
        )
        
        # Evaluate ranking performance
        models = {
            'pointwise': pointwise_model,
            'pairwise': pairwise_model,
            'listwise': listwise_model
        }
        
        ranking_results = evaluate_ranking_performance(models, test_samples, config)
        
        # Compile results
        total_time = time.time() - start_time
        memory_monitor.update_peak()
        
        results = {
            'experiment_type': 'r4_preference',
            'seed': seed,
            'config': config,
            'training': {
                'pointwise': pointwise_results,
                'pairwise': pairwise_results,
                'listwise': listwise_results
            },
            'evaluation': ranking_results,
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
            'pointwise_ndcg5': ranking_results['pointwise']['ndcg_at_5'],
            'pairwise_ndcg5': ranking_results['pairwise']['ndcg_at_5'],
            'listwise_ndcg5': ranking_results['listwise']['ndcg_at_5'],
            'pointwise_kendall': ranking_results['pointwise']['kendall_tau'],
            'pairwise_kendall': ranking_results['pairwise']['kendall_tau'],
            'listwise_kendall': ranking_results['listwise']['kendall_tau']
        }
        
        # Save models
        if config['output']['save_model']:
            for model_name, model in models.items():
                model_path = dirs['models'] / f"{model_name}_model_seed_{seed}.pt"
                torch.save(model.state_dict(), model_path)
                logger.info(f"{model_name} model saved to {model_path}")
        
        logger.info(f"R4 experiment completed successfully in {format_time(total_time)}")
        
        return results
        
    except Exception as e:
        logger.error(f"R4 experiment failed: {e}")
        return {
            'experiment_type': 'r4_preference',
            'seed': seed,
            'error': str(e),
            'status': 'failed'
        }
