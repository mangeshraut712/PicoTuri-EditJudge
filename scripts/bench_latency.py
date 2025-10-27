#!/usr/bin/env python3
"""
Latency Benchmarking Script
Standardized latency protocol across platforms and models
"""

import argparse
import json
import logging
import time
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.utils import setup_logging, get_device, benchmark_model, compute_statistics, format_time

logger = logging.getLogger(__name__)

class LatencyBenchmark:
    """Comprehensive latency benchmarking suite."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.results = {}
        
    def benchmark_text_encoders(self) -> Dict[str, Any]:
        """Benchmark text encoder models."""
        logger.info("Benchmarking text encoders...")
        
        text_models = {
            'bert-base': 'bert-base-uncased',
            'e5-small': 'intfloat/e5-small-v2',
            'minilm': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        
        results = {}
        
        for model_name, model_path in text_models.items():
            logger.info(f"Benchmarking {model_name}")
            
            try:
                from src.features_text.bert import BERTTextEmbedder
                
                encoder = BERTTextEmbedder(model_path, device=self.device)
                
                # Create sample input
                sample_input = torch.randint(0, 30000, (1, 512)).to(self.device)
                
                # Benchmark
                stats = benchmark_model(
                    encoder.model,
                    sample_input,
                    num_runs=100,
                    warmup_runs=20
                )
                
                results[model_name] = stats
                logger.info(f"{model_name}: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def benchmark_image_encoders(self) -> Dict[str, Any]:
        """Benchmark image encoder models."""
        logger.info("Benchmarking image encoders...")
        
        image_models = {
            'clip-vit-b32': 'open_clip/ViT-B-32',
            'clip-vit-l14': 'open_clip/ViT-L-14',
            'efficientnet-b0': 'efficientnet_b0'
        }
        
        results = {}
        
        for model_name, model_path in image_models.items():
            logger.info(f"Benchmarking {model_name}")
            
            try:
                from src.features_image.clip import CLIPImageEmbedder
                
                if model_name.startswith('clip'):
                    encoder = CLIPImageEmbedder(model_path, device=self.device)
                else:
                    # For EfficientNet, we'd need a different encoder
                    logger.warning(f"Skipping {model_name} - not implemented")
                    continue
                
                # Create sample input
                sample_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Benchmark
                stats = benchmark_model(
                    encoder.model,
                    sample_input,
                    num_runs=50,
                    warmup_runs=10
                )
                
                results[model_name] = stats
                logger.info(f"{model_name}: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def benchmark_fusion_heads(self) -> Dict[str, Any]:
        """Benchmark fusion head models."""
        logger.info("Benchmarking fusion heads...")
        
        fusion_configs = {
            'small': {'input_dim': 768, 'hidden_dims': [256, 128]},
            'medium': {'input_dim': 1280, 'hidden_dims': [512, 256, 128]},
            'large': {'input_dim': 1536, 'hidden_dims': [1024, 512, 256]}
        }
        
        results = {}
        
        for config_name, config in fusion_configs.items():
            logger.info(f"Benchmarking fusion head {config_name}")
            
            try:
                from src.fuse.fusion import FusionHead
                
                fusion_head = FusionHead(
                    input_dim=config['input_dim'],
                    hidden_dims=config['hidden_dims'],
                    output_dim=1,
                    dropout=0.1
                ).to(self.device)
                
                # Create sample input
                sample_input = torch.randn(1, config['input_dim']).to(self.device)
                
                # Benchmark
                stats = benchmark_model(
                    fusion_head,
                    sample_input,
                    num_runs=200,
                    warmup_runs=50
                )
                
                results[config_name] = stats
                logger.info(f"fusion-{config_name}: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
                
            except Exception as e:
                logger.error(f"Failed to benchmark fusion head {config_name}: {e}")
                results[config_name] = {'error': str(e)}
        
        return results
    
    def benchmark_end_to_end(self) -> Dict[str, Any]:
        """Benchmark end-to-end pipeline."""
        logger.info("Benchmarking end-to-end pipeline...")
        
        # Define pipeline configurations
        pipelines = {
            'baseline': {
                'text_encoder': 'bert-base',
                'image_encoder': 'clip-vit-b32',
                'fusion_head': 'medium'
            },
            'high_performance': {
                'text_encoder': 'e5-small',
                'image_encoder': 'clip-vit-l14',
                'fusion_head': 'large'
            },
            'lightweight': {
                'text_encoder': 'minilm',
                'image_encoder': 'clip-vit-b32',
                'fusion_head': 'small'
            }
        }
        
        results = {}
        
        for pipeline_name, pipeline_config in pipelines.items():
            logger.info(f"Benchmarking pipeline {pipeline_name}")
            
            try:
                # Load models
                from src.features_text.bert import BERTTextEmbedder
                from src.features_image.clip import CLIPImageEmbedder
                from src.fuse.fusion import FusionHead
                
                # Text encoder
                text_model_path = {
                    'bert-base': 'bert-base-uncased',
                    'e5-small': 'intfloat/e5-small-v2',
                    'minilm': 'sentence-transformers/all-MiniLM-L6-v2'
                }[pipeline_config['text_encoder']]
                
                text_encoder = BERTTextEmbedder(text_model_path, device=self.device)
                
                # Image encoder
                image_model_path = {
                    'clip-vit-b32': 'open_clip/ViT-B-32',
                    'clip-vit-l14': 'open_clip/ViT-L-14'
                }[pipeline_config['image_encoder']]
                
                image_encoder = CLIPImageEmbedder(image_model_path, device=self.device)
                
                # Fusion head
                fusion_config = {
                    'small': {'input_dim': 768, 'hidden_dims': [256, 128]},
                    'medium': {'input_dim': 1280, 'hidden_dims': [512, 256, 128]},
                    'large': {'input_dim': 1536, 'hidden_dims': [1024, 512, 256]}
                }[pipeline_config['fusion_head']]
                
                fusion_head = FusionHead(
                    input_dim=fusion_config['input_dim'],
                    hidden_dims=fusion_config['hidden_dims'],
                    output_dim=1,
                    dropout=0.1
                ).to(self.device)
                
                # Benchmark end-to-end latency
                def end_to_end_inference():
                    # Text encoding
                    text_input = torch.randint(0, 30000, (1, 512)).to(self.device)
                    with torch.no_grad():
                        text_embedding = text_encoder.model(text_input)
                    
                    # Image encoding
                    image_input = torch.randn(1, 3, 224, 224).to(self.device)
                    with torch.no_grad():
                        image_embedding = image_encoder.model(image_input)
                    
                    # Fusion
                    combined = torch.cat([text_embedding, image_embedding], dim=1)
                    with torch.no_grad():
                        score = fusion_head(combined)
                    
                    return score
                
                # Benchmark
                stats = benchmark_model(
                    None,  # We'll use the custom function
                    None,
                    num_runs=30,
                    warmup_runs=10,
                    custom_function=end_to_end_inference
                )
                
                results[pipeline_name] = stats
                logger.info(f"Pipeline {pipeline_name}: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
                
            except Exception as e:
                logger.error(f"Failed to benchmark pipeline {pipeline_name}: {e}")
                results[pipeline_name] = {'error': str(e)}
        
        return results
    
    def benchmark_batch_sizes(self) -> Dict[str, Any]:
        """Benchmark different batch sizes."""
        logger.info("Benchmarking batch sizes...")
        
        # Use baseline pipeline for batch testing
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        results = {}
        
        try:
            from src.features_text.bert import BERTTextEmbedder
            from src.features_image.clip import CLIPImageEmbedder
            from src.fuse.fusion import FusionHead
            
            # Load models
            text_encoder = BERTTextEmbedder('bert-base-uncased', device=self.device)
            image_encoder = CLIPImageEmbedder('open_clip/ViT-B-32', device=self.device)
            fusion_head = FusionHead(input_dim=1280, hidden_dims=[512, 256, 128], output_dim=1).to(self.device)
            
            for batch_size in batch_sizes:
                logger.info(f"Benchmarking batch size {batch_size}")
                
                def batch_inference():
                    # Text encoding
                    text_input = torch.randint(0, 30000, (batch_size, 512)).to(self.device)
                    with torch.no_grad():
                        text_embedding = text_encoder.model(text_input)
                    
                    # Image encoding
                    image_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                    with torch.no_grad():
                        image_embedding = image_encoder.model(image_input)
                    
                    # Fusion
                    combined = torch.cat([text_embedding, image_embedding], dim=1)
                    with torch.no_grad():
                        scores = fusion_head(combined)
                    
                    return scores
                
                # Benchmark
                stats = benchmark_model(
                    None,
                    None,
                    num_runs=20,
                    warmup_runs=5,
                    custom_function=batch_inference
                )
                
                # Calculate throughput
                throughput = batch_size / (stats['mean'] / 1000)  # images per second
                
                results[f'batch_{batch_size}'] = {
                    **stats,
                    'throughput_img_per_sec': throughput
                }
                
                logger.info(f"Batch {batch_size}: {stats['mean']:.2f}ms, {throughput:.1f} img/s")
        
        except Exception as e:
            logger.error(f"Failed to benchmark batch sizes: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting full latency benchmark suite")
        
        start_time = time.time()
        
        # Run all benchmarks
        self.results = {
            'text_encoders': self.benchmark_text_encoders(),
            'image_encoders': self.benchmark_image_encoders(),
            'fusion_heads': self.benchmark_fusion_heads(),
            'end_to_end': self.benchmark_end_to_end(),
            'batch_sizes': self.benchmark_batch_sizes(),
            'system_info': self.get_system_info(),
            'timestamp': time.time()
        }
        
        total_time = time.time() - start_time
        self.results['benchmark_time'] = {
            'total_seconds': total_time,
            'formatted': format_time(total_time)
        }
        
        logger.info(f"Full benchmark completed in {format_time(total_time)}")
        
        return self.results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        import psutil
        import platform
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'device': self.device,
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name()
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

def generate_report(results: Dict[str, Any], output_path: str):
    """Generate a human-readable benchmark report."""
    report_lines = [
        "# Latency Benchmark Report",
        "",
        f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}",
        f"**Benchmark Time**: {results['benchmark_time']['formatted']}",
        "",
        "## System Information",
        "",
        f"- **Platform**: {results['system_info']['platform']}",
        f"- **Device**: {results['system_info']['device']}",
        f"- **CPU Cores**: {results['system_info']['cpu_count']}",
        f"- **Memory**: {results['system_info']['memory_gb']:.1f} GB",
        f"- **PyTorch**: {results['system_info']['torch_version']}",
    ]
    
    if 'gpu_name' in results['system_info']:
        report_lines.extend([
            f"- **GPU**: {results['system_info']['gpu_name']}",
            f"- **GPU Memory**: {results['system_info']['gpu_memory_gb']:.1f} GB",
        ])
    
    report_lines.extend([
        "",
        "## Text Encoder Latency (ms)",
        "",
        "| Model | Mean | Std | Min | Max | P50 | P95 |",
        "|-------|------|-----|-----|-----|-----|-----|",
    ])
    
    for model_name, stats in results['text_encoders'].items():
        if 'error' not in stats:
            report_lines.append(
                f"| {model_name} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['median']:.2f} | {stats['q75']:.2f} |"
            )
        else:
            report_lines.append(f"| {model_name} | ERROR | - | - | - | - | - |")
    
    report_lines.extend([
        "",
        "## Image Encoder Latency (ms)",
        "",
        "| Model | Mean | Std | Min | Max | P50 | P95 |",
        "|-------|------|-----|-----|-----|-----|-----|",
    ])
    
    for model_name, stats in results['image_encoders'].items():
        if 'error' not in stats:
            report_lines.append(
                f"| {model_name} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['median']:.2f} | {stats['q75']:.2f} |"
            )
        else:
            report_lines.append(f"| {model_name} | ERROR | - | - | - | - | - |")
    
    report_lines.extend([
        "",
        "## End-to-End Pipeline Latency (ms)",
        "",
        "| Pipeline | Mean | Std | Min | Max | P50 | P95 |",
        "|----------|------|-----|-----|-----|-----|-----|",
    ])
    
    for pipeline_name, stats in results['end_to_end'].items():
        if 'error' not in stats:
            report_lines.append(
                f"| {pipeline_name} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['median']:.2f} | {stats['q75']:.2f} |"
            )
        else:
            report_lines.append(f"| {pipeline_name} | ERROR | - | - | - | - | - |")
    
    report_lines.extend([
        "",
        "## Batch Size Performance",
        "",
        "| Batch Size | Latency (ms) | Throughput (img/s) |",
        "|------------|--------------|-------------------|",
    ])
    
    for batch_key, stats in results['batch_sizes'].items():
        if 'error' not in stats:
            batch_size = batch_key.split('_')[1]
            report_lines.append(
                f"| {batch_size} | {stats['mean']:.2f} | {stats['throughput_img_per_sec']:.1f} |"
            )
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run latency benchmarks")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--report", type=str, default="benchmark_report.md", help="Output file for report")
    parser.add_argument("--models", type=str, default="all", help="Models to benchmark (all, text, image, fusion, e2e)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Create benchmark configuration
    config = {
        'device': get_device(),
        'num_runs': 100,
        'warmup_runs': 20
    }
    
    # Run benchmark
    benchmark = LatencyBenchmark(config)
    
    if args.models == "all":
        results = benchmark.run_full_benchmark()
    else:
        # Run specific benchmarks
        results = {'system_info': benchmark.get_system_info()}
        
        if "text" in args.models:
            results['text_encoders'] = benchmark.benchmark_text_encoders()
        if "image" in args.models:
            results['image_encoders'] = benchmark.benchmark_image_encoders()
        if "fusion" in args.models:
            results['fusion_heads'] = benchmark.benchmark_fusion_heads()
        if "e2e" in args.models:
            results['end_to_end'] = benchmark.benchmark_end_to_end()
    
    # Save results
    benchmark.save_results(args.output)
    generate_report(results, args.report)
    
    logger.info(f"Benchmark completed. Results saved to {args.output}, report saved to {args.report}")

if __name__ == "__main__":
    main()
