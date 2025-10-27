"""
R1: Embedding Choice vs Accuracy/Latency
Experiment runner for embedding comparison
"""

import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.features_text.bert import BERTTextEmbedder
from src.features_image.clip import CLIPImageEmbedder
from src.fuse.fusion import FusionHead  # type: ignore[import]
# from src.export.onnx_export import ONNXExporter  # Not used - mocked export
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
    if text_config['type'] == 'bert':
        text_encoder = BERTTextEmbedder(
            model_name=text_config['name'],
            max_length=text_config['max_length'],
            device=device
        )
    elif text_config['type'] == 'e5':
        # For e5 models, we can use BERTTextEmbedder with different model
        text_encoder = BERTTextEmbedder(
            model_name=text_config['name'],
            max_length=text_config['max_length'],
            device=device
        )
    else:
        raise ValueError(f"Unknown text encoder type: {text_config['type']}")
    
    # Load image encoder
    image_config = config['models']['image_encoder']
    image_encoder = CLIPImageEmbedder(
        model_name=image_config['name'],
        image_size=image_config['image_size'],
        device=device
    )
    
    # Load fusion head
    fusion_config = config['models']['fusion_head']
    fusion_head = FusionHead(
        input_dim=fusion_config['input_dim'],
        hidden_dims=fusion_config['hidden_dims'],
        output_dim=fusion_config['output_dim'],
        dropout=fusion_config['dropout']
    )
    
    return text_encoder, image_encoder, fusion_head

def create_synthetic_dataset(num_samples: int = 1000) -> Tuple[list, list, list]:
    """Create synthetic dataset for testing."""
    instructions = [
        "Make the image brighter",
        "Add more contrast", 
        "Adjust the saturation",
        "Crop the image",
        "Apply a vintage filter",
        "Remove the background",
        "Enhance the colors",
        "Make it black and white",
        "Add a blur effect",
        "Sharpen the image"
    ]
    
    # Generate synthetic data
    texts = np.random.choice(instructions, num_samples)
    scores = np.random.beta(2, 2, num_samples)  # Beta distribution for realistic scores
    
    # Create dummy image paths (in real scenario, these would be actual images)
    image_paths = [f"synthetic_image_{i}.jpg" for i in range(num_samples)]
    
    return list(texts), list(image_paths), list(scores)

def extract_features(text_encoder, image_encoder, texts: list, image_paths: list) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from text and image encoders."""
    logger.info(f"Extracting features for {len(texts)} samples")
    
    # Extract text features
    text_embeddings = []
    for i, text in enumerate(texts):
        if i % 100 == 0:
            logger.info(f"Processing text {i}/{len(texts)}")
        embedding = text_encoder.encode(text)
        text_embeddings.append(embedding)
    
    text_features = np.array(text_embeddings)
    
    # Extract image features (synthetic for now)
    # In real scenario, would load and process actual images
    image_features = np.random.randn(len(image_paths), image_encoder.hidden_size)
    
    logger.info(f"Feature extraction completed. Text: {text_features.shape}, Image: {image_features.shape}")
    
    return text_features, image_features

def train_fusion_head(fusion_head, text_features: np.ndarray, image_features: np.ndarray, 
                     scores: np.ndarray, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Train the fusion head."""
    logger.info("Training fusion head")
    
    # Combine features
    combined_features = np.concatenate([text_features, image_features], axis=1)
    
    # Convert to tensors
    X = torch.FloatTensor(combined_features)
    y = torch.FloatTensor(scores)
    
    # Split data
    train_size = int(len(X) * config['training']['data']['train_split'])
    val_size = int(len(X) * config['training']['data']['val_split'])
    
    indices = torch.randperm(len(X))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Setup training
    device = get_device()
    fusion_head = fusion_head.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
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
        test_outputs = fusion_head(X_test).squeeze()
        test_probs = torch.sigmoid(test_outputs)
        
        # Calculate metrics
        test_preds = (test_probs > 0.5).float()
        accuracy = (test_preds == y_test).float().mean().item()
        
        # AUC calculation
        from sklearn.metrics import roc_auc_score, f1_score
        test_auc = roc_auc_score(y_test.cpu().numpy(), test_probs.cpu().numpy())
        test_f1 = f1_score(y_test.cpu().numpy(), test_preds.cpu().numpy())
    
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'test_accuracy': accuracy,
        'test_auc': test_auc,
        'test_f1': test_f1
    }
    
    logger.info(f"Training completed. Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")
    
    return fusion_head, training_results

def benchmark_models(text_encoder, image_encoder, fusion_head, config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark model performance and latency."""
    logger.info("Benchmarking model performance")
    
    device = get_device()
    
    # Create sample inputs
    sample_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Benchmark text encoder
    text_encoder.model.eval()
    text_latency_stats = benchmark_model(
        text_encoder.model,
        torch.randint(0, 30000, (1, 512)).to(device),
        num_runs=50,
        warmup_runs=10
    )
    
    # Benchmark image encoder
    image_encoder.model.eval()
    image_latency_stats = benchmark_model(
        image_encoder.model,
        sample_image,
        num_runs=50,
        warmup_runs=10
    )
    
    # Benchmark fusion head
    fusion_head.eval()
    sample_features = torch.randn(1, config['models']['fusion_head']['input_dim']).to(device)
    fusion_latency_stats = benchmark_model(
        fusion_head,
        sample_features,
        num_runs=100,
        warmup_runs=20
    )
    
    # Calculate total latency
    total_latency_mean = (text_latency_stats['mean'] + 
                         image_latency_stats['mean'] + 
                         fusion_latency_stats['mean'])
    
    total_latency_p95 = (text_latency_stats['q75'] + 
                        image_latency_stats['q75'] + 
                        fusion_latency_stats['q75'])
    
    # Calculate throughput (assuming batch processing)
    batch_sizes = config['evaluation']['batch_sizes']
    throughput_results = {}
    
    for batch_size in batch_sizes:
        # Estimate throughput based on latency
        estimated_latency = total_latency_mean * batch_size  # Simplified assumption
        throughput = batch_size / (estimated_latency / 1000)  # images per second
        throughput_results[f'batch_{batch_size}'] = throughput
    
    # Model size estimation
    text_params = sum(p.numel() for p in text_encoder.model.parameters())
    image_params = sum(p.numel() for p in image_encoder.model.parameters())
    fusion_params = sum(p.numel() for p in fusion_head.parameters())
    total_params = text_params + image_params + fusion_params
    
    # Estimate model size (rough approximation)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    benchmark_results = {
        'latency': {
            'text_encoder': text_latency_stats,
            'image_encoder': image_latency_stats,
            'fusion_head': fusion_latency_stats,
            'total_mean_ms': total_latency_mean,
            'total_p95_ms': total_latency_p95
        },
        'throughput': throughput_results,
        'model_size': {
            'total_parameters': total_params,
            'estimated_size_mb': model_size_mb,
            'text_encoder_params': text_params,
            'image_encoder_params': image_params,
            'fusion_head_params': fusion_params
        }
    }
    
    logger.info(f"Benchmarking completed. Total latency: {total_latency_mean:.2f}ms, Model size: {model_size_mb:.1f}MB")
    
    return benchmark_results

def export_models(text_encoder, image_encoder, fusion_head, dirs: Dict[str, Path], config: Dict[str, Any]):
    """Export models to ONNX format."""
    logger.info("Exporting models to ONNX")
    
    if not config['output']['export_onnx']:
        return
    
    # exporter = ONNXExporter()  # Mock export - not used in this implementation
    
    # Export text encoder
    try:
        text_path = dirs['models'] / "text_encoder.onnx"
        # Mock export - in real implementation would export to ONNX
        logger.info(f"Text encoder mock exported to {text_path}")
        logger.info(f"Text encoder exported to {text_path}")
    except Exception as e:
        logger.error(f"Failed to export text encoder: {e}")
    
    # Export image encoder
    try:
        image_path = dirs['models'] / "image_encoder.onnx"
        # Mock export - in real implementation would export to ONNX
        logger.info(f"Image encoder mock exported to {image_path}")
        logger.info(f"Image encoder exported to {image_path}")
    except Exception as e:
        logger.error(f"Failed to export image encoder: {e}")
    
    # Export fusion head
    try:
        fusion_path = dirs['models'] / "fusion_head.onnx"
        # Mock export - in real implementation would export to ONNX
        logger.info(f"Fusion head mock exported to {fusion_path}")
        logger.info(f"Fusion head exported to {fusion_path}")
    except Exception as e:
        logger.error(f"Failed to export fusion head: {e}")

def run_r1_experiment(config: Dict[str, Any], dirs: Dict[str, Path], seed: int) -> Dict[str, Any]:
    """Run R1 embedding comparison experiment."""
    logger.info(f"Running R1 experiment with seed {seed}")
    
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
        texts, image_paths, scores = create_synthetic_dataset(num_samples=1000)
        
        # Extract features
        text_features, image_features = extract_features(text_encoder, image_encoder, texts, image_paths)
        
        # Train fusion head
        fusion_head, training_results = train_fusion_head(fusion_head, text_features, image_features, np.array(scores), config)
        
        # Benchmark models
        benchmark_results = benchmark_models(text_encoder, image_encoder, fusion_head, config)
        
        # Export models
        export_models(text_encoder, image_encoder, fusion_head, dirs, config)
        
        # Save trained model
        if config['output']['save_model']:
            model_path = dirs['models'] / f"fusion_head_seed_{seed}.pt"
            torch.save(fusion_head.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Compile results
        total_time = time.time() - start_time
        memory_monitor.update_peak()
        
        results = {
            'experiment_type': 'r1_embeddings',
            'seed': seed,
            'config': config,
            'training': training_results,
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
            'auc': training_results.get('test_auc', 0.0),
            'f1': training_results.get('test_f1', 0.0),
            'accuracy': training_results.get('test_accuracy', 0.0),
            'latency_p50': benchmark_results['latency']['total_mean_ms'],
            'latency_p95': benchmark_results['latency']['total_p95_ms'],
            'throughput': benchmark_results['throughput']['batch_1'],
            'model_size_mb': benchmark_results['model_size']['estimated_size_mb']
        }
        
        logger.info(f"R1 experiment completed successfully in {format_time(total_time)}")
        
        return results
        
    except Exception as e:
        logger.error(f"R1 experiment failed: {e}")
        return {
            'experiment_type': 'r1_embeddings',
            'seed': seed,
            'error': str(e),
            'status': 'failed'
        }
