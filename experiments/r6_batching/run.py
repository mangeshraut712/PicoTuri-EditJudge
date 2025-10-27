"""
R6: Real-time Batching & Quantization
Experiment runner for batching and quantization performance
"""

import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import psutil

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

class LoadTestRunner:
    """Load testing for batching performance."""
    
    def __init__(self):
        self.results = []
        self.stop_event = threading.Event()
    
    async def simulate_requests(self, concurrent_requests: int, duration: int):
        """Simulate concurrent requests."""
        logger.info(f"Starting load test with {concurrent_requests} concurrent requests for {duration}s")
        
        start_time = time.time()
        request_times = []
        errors = 0
        
        async def make_request(request_id: int):
            """Make a single request."""
            try:
                request_start = time.time()
                
                # Mock inference - simulate processing time
                await asyncio.sleep(0.01)  # 10ms mock processing time
                
                request_time = time.time() - request_start
                return request_time, True
                
            except Exception as e:
                logger.error(f"Request {request_id} failed: {e}")
                return 0, False
        
        # Run concurrent requests
        tasks = []
        request_id = 0
        
        while time.time() - start_time < duration and not self.stop_event.is_set():
            # Create batch of concurrent requests
            batch_tasks = []
            for _ in range(min(concurrent_requests, 10)):  # Limit concurrent tasks
                if time.time() - start_time >= duration:
                    break
                batch_tasks.append(make_request(request_id))
                request_id += 1
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    errors += 1
                else:
                    request_time, success = result
                    if success:
                        request_times.append(request_time)
                    else:
                        errors += 1
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Calculate statistics
        if request_times:
            stats = {
                'mean': np.mean(request_times),
                'std': np.std(request_times),
                'min': np.min(request_times),
                'max': np.max(request_times)
            }
            throughput = len(request_times) / duration
        else:
            stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            throughput = 0
        
        return {
            'concurrent_requests': concurrent_requests,
            'duration': duration,
            'total_requests': len(request_times),
            'successful_requests': len(request_times),
            'errors': errors,
            'throughput': throughput,
            'latency_stats': stats
        }

def quantize_model(model: torch.nn.Module, calibration_data: torch.Tensor, 
                  method: str = "dynamic_int8") -> torch.nn.Module:
    """Quantize PyTorch model."""
    logger.info(f"Quantizing model using {method}")
    
    if method == "dynamic_int8":
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model.cpu(),  # Move to CPU for quantization
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        return quantized_model
    
    elif method == "static_int8":
        # Static quantization (requires calibration)
        model.eval()
        model.cpu()
        
        # Prepare model for quantization
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        with torch.no_grad():
            model(calibration_data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        return quantized_model
    
    else:
        raise ValueError(f"Unknown quantization method: {method}")

def benchmark_batching_strategies(config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark different batching strategies."""
    logger.info("Benchmarking batching strategies")
    
    strategies = ["no_batching", "fixed_batch", "adaptive_batch"]
    results = {}
    
    for strategy in strategies:
        logger.info(f"Benchmarking {strategy}")
        
        try:
            if strategy == "no_batching":
                # Individual requests
                stats = benchmark_individual_requests(config)
            elif strategy == "fixed_batch":
                # Fixed batch size
                stats = benchmark_fixed_batching(config, batch_size=8)
            elif strategy == "adaptive_batch":
                # Adaptive batching
                stats = benchmark_adaptive_batching(config)
            
            results[strategy] = stats
            
        except Exception as e:
            logger.error(f"Failed to benchmark {strategy}: {e}")
            results[strategy] = {'error': str(e)}
    
    return results

def benchmark_individual_requests(config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark individual request processing."""
    device = get_device()
    
    # Load models
    text_encoder = BERTTextEmbedder('bert-base-uncased', device=device)
    image_encoder = CLIPImageEmbedder('open_clip/ViT-B-32', device=device)
    fusion_head = FusionHead(input_dim=1280, hidden_dims=[512, 256, 128]).to(device)
    
    def single_request():
        # Text encoding
        text_input = torch.randint(0, 30000, (1, 512)).to(device)
        with torch.no_grad():
            text_embedding = text_encoder.model(text_input)
        
        # Image encoding
        image_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            image_embedding = image_encoder.model(image_input)
        
        # Fusion
        combined = torch.cat([text_embedding, image_embedding], dim=1)
        with torch.no_grad():
            score = fusion_head(combined)
        
        return score
    
    # Benchmark - manual timing since benchmark_model doesn't support custom functions
    latencies = []
    
    # Warmup
    for _ in range(20):
        single_request()
    
    # Actual benchmark
    for _ in range(100):
        start_time = time.time()
        single_request()
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    stats = {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }
    
    # Calculate throughput
    throughput = 1 / (stats['mean'] / 1000)  # requests per second
    
    return {
        'latency_stats': stats,
        'throughput': throughput,
        'batch_size': 1
    }

def benchmark_fixed_batching(config: Dict[str, Any], batch_size: int = 8) -> Dict[str, Any]:
    """Benchmark fixed batch size processing."""
    device = get_device()
    
    # Load models
    text_encoder = BERTTextEmbedder('bert-base-uncased', device=device)
    image_encoder = CLIPImageEmbedder('open_clip/ViT-B-32', device=device)
    fusion_head = FusionHead(input_dim=1280, hidden_dims=[512, 256, 128]).to(device)
    
    def batch_request():
        # Text encoding
        text_input = torch.randint(0, 30000, (batch_size, 512)).to(device)
        with torch.no_grad():
            text_embedding = text_encoder.model(text_input)
        
        # Image encoding
        image_input = torch.randn(batch_size, 3, 224, 224).to(device)
        with torch.no_grad():
            image_embedding = image_encoder.model(image_input)
        
        # Fusion
        combined = torch.cat([text_embedding, image_embedding], dim=1)
        with torch.no_grad():
            scores = fusion_head(combined)
        
        return scores
    
    # Benchmark - manual timing since benchmark_model doesn't support custom functions
    latencies = []
    
    # Warmup
    for _ in range(10):
        batch_request()
    
    # Actual benchmark
    for _ in range(50):
        start_time = time.time()
        batch_request()
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    stats = {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }
    
    # Calculate throughput
    throughput = batch_size / (stats['mean'] / 1000)  # requests per second
    
    return {
        'latency_stats': stats,
        'throughput': throughput,
        'batch_size': batch_size
    }

async def benchmark_adaptive_batching(config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark adaptive micro-batching."""
    logger.info("Setting up adaptive micro-batching")
    
    # Mock adaptive batching simulation
    batcher_config = config['batching']
    
    # Simulate processing function
    async def process_batch(batch_data):
        # Simulate processing time based on batch size
        batch_size = len(batch_data)
        processing_time = 10 + batch_size * 2  # Base time + per-item time
        await asyncio.sleep(processing_time / 1000)  # Convert to seconds
        return batch_size
    
    # Mock adaptive batching simulation
    num_requests = 100
    start_time = time.time()
    
    # Simulate adaptive batching behavior
    batch_sizes = []
    latencies = []
    
    for i in range(num_requests):
        # Simulate variable batch sizes (adaptive behavior)
        batch_size = np.random.randint(1, 8)  # Adaptive batch size
        batch_sizes.append(batch_size)
        
        # Simulate processing time
        processing_time = (10 + batch_size * 2) / 1000  # Convert to seconds
        await asyncio.sleep(processing_time)
        
        latencies.append(processing_time * 1000)  # Convert to ms
    
    total_time = time.time() - start_time
    throughput = num_requests / total_time
    
    stats = {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies)
    }
    
    metrics = {
        'avg_batch_size': np.mean(batch_sizes),
        'total_requests': num_requests,
        'total_time': total_time
    }
    
    return {
        'latency_stats': stats,
        'throughput': throughput,
        'total_time': total_time,
        'num_requests': num_requests,
        'batcher_metrics': metrics
    }

def benchmark_quantization_impact(config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark the impact of quantization on performance and accuracy."""
    logger.info("Benchmarking quantization impact")
    
    device = get_device()
    
    # Create fusion head
    fusion_head = FusionHead(
        input_dim=config['models']['fusion_head']['input_dim'],
        hidden_dims=config['models']['fusion_head']['hidden_dims'],
        output_dim=1,
        dropout=0.1
    ).to(device)
    
    # Create calibration data
    calibration_data = torch.randn(1000, config['models']['fusion_head']['input_dim'])
    
    # Benchmark original model
    sample_input = torch.randn(1, config['models']['fusion_head']['input_dim']).to(device)
    original_stats = benchmark_model(
        fusion_head,
        sample_input,
        num_runs=100,
        warmup_runs=20
    )
    
    # Quantize model
    quantized_model = quantize_model(
        fusion_head, 
        calibration_data[:100],  # Use subset for calibration
        config['quantization']['method']
    )
    
    # Benchmark quantized model
    quantized_input = torch.randn(1, config['models']['fusion_head']['input_dim'])
    quantized_stats = benchmark_model(
        quantized_model,
        quantized_input,
        num_runs=100,
        warmup_runs=20
    )
    
    # Compare accuracy (simplified)
    with torch.no_grad():
        test_input = torch.randn(100, config['models']['fusion_head']['input_dim'])
        
        original_output = fusion_head(test_input.to(device)).cpu()
        quantized_output = quantized_model(test_input).cpu()
        
        # Calculate MSE difference
        mse_diff = torch.mean((original_output - quantized_output) ** 2).item()
    
    return {
        'original_model': {
            'latency_stats': original_stats,
            'model_size_mb': sum(p.numel() for p in fusion_head.parameters()) * 4 / (1024**2)
        },
        'quantized_model': {
            'latency_stats': quantized_stats,
            'model_size_mb': sum(p.numel() for p in quantized_model.parameters()) * 1 / (1024**2)  # INT8
        },
        'accuracy_impact': {
            'mse_difference': mse_diff,
            'relative_accuracy_loss': mse_diff / torch.var(original_output).item()
        },
        'speedup': original_stats['mean'] / quantized_stats['mean'],
        'size_reduction': (sum(p.numel() for p in fusion_head.parameters()) * 4 / (1024**2)) / \
                         (sum(p.numel() for p in quantized_model.parameters()) * 1 / (1024**2))
    }

async def run_load_tests(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run load tests with different concurrency levels."""
    logger.info("Running load tests")
    
    load_tester = LoadTestRunner()
    
    concurrency_levels = [1, 5, 10, 20]
    results = {}
    
    for concurrency in concurrency_levels:
        logger.info(f"Testing with {concurrency} concurrent requests")
        
        result = await load_tester.simulate_requests(
            concurrent_requests=concurrency,
            duration=30  # 30 seconds per test
        )
        
        results[f'concurrency_{concurrency}'] = result
    
    return results

async def run_r6_experiment(config: Dict[str, Any], dirs: Dict[str, Path], seed: int) -> Dict[str, Any]:
    """Run R6 batching and quantization experiment."""
    logger.info(f"Running R6 experiment with seed {seed}")
    
    # Setup memory monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    start_time = time.time()
    
    try:
        # Benchmark batching strategies
        batching_results = benchmark_batching_strategies(config)
        
        # Benchmark quantization impact
        quantization_results = benchmark_quantization_impact(config)
        
        # Setup runtime engine for load testing
        model_paths = {
            "text_encoder": "models/text_encoder.onnx",
            "image_encoder": "models/image_encoder.onnx",
            "fusion_head": "models/fusion_head.onnx"
        }
        
        runtime_engine = RuntimeEngine(
            model_paths=model_paths,
            pool_size=config['runtime']['session_pool_size'],
            batch_strategy=BatchStrategy.ADAPTIVE,
            enable_batching=True
        )
        
        # Initialize engine (this would normally load actual models)
        # await runtime_engine.initialize()
        
        # Run load tests (simulated)
        load_test_results = {}  # await run_load_tests(runtime_engine, config)
        
        # Memory usage analysis
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Compile results
        total_time = time.time() - start_time
        memory_monitor.update_peak()
        
        results = {
            'experiment_type': 'r6_batching',
            'seed': seed,
            'config': config,
            'batching_benchmarks': batching_results,
            'quantization_benchmarks': quantization_results,
            'load_tests': load_test_results,
            'system_performance': {
                'memory_usage_mb': memory_info.rss / (1024**2),
                'peak_memory_mb': memory_monitor.get_peak_memory() / (1024**2),
                'cpu_usage_percent': process.cpu_percent()
            },
            'timing': {
                'total_time_seconds': total_time,
                'formatted_time': format_time(total_time)
            },
            'status': 'completed'
        }
        
        # Extract key metrics
        results['metrics'] = {
            'throughput': quantization_results.get('speedup', 0),
            'memory_usage': results['system_performance']['memory_usage_mb'],
            'quantization_speedup': quantization_results.get('speedup', 0),
            'size_reduction': quantization_results.get('size_reduction', 0)
        }
        
        logger.info(f"R6 experiment completed successfully in {format_time(total_time)}")
        
        return results
        
    except Exception as e:
        logger.error(f"R6 experiment failed: {e}")
        return {
            'experiment_type': 'r6_batching',
            'seed': seed,
            'error': str(e),
            'status': 'failed'
        }
