"""
R7: Cross-platform Inference Parity
Experiment runner for cross-platform model comparison
"""

import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import subprocess
import tempfile
import os

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.features_text.bert import BERTTextEmbedder
from src.features_image.clip import CLIPImageEmbedder
from src.fuse.fusion import FusionHead
from src.export.onnx_export import ONNXExporter
from experiments.utils import (
    setup_logging, get_device, benchmark_model, 
    compute_statistics, format_time, MemoryMonitor
)

logger = logging.getLogger(__name__)

class CrossPlatformRunner:
    """Runner for cross-platform inference testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.results = {}
    
    def export_models_to_onnx(self, text_encoder, image_encoder, fusion_head, 
                             export_dir: Path) -> Dict[str, str]:
        """Export models to ONNX format."""
        logger.info("Exporting models to ONNX")
        
        exporter = ONNXExporter()
        
        # Export text encoder
        text_path = export_dir / "text_encoder.onnx"
        try:
            exporter.export_text_encoder(
                model=text_encoder.model,
                output_path=str(text_path),
                input_shape=(1, 512)
            )
            logger.info(f"Text encoder exported to {text_path}")
        except Exception as e:
            logger.error(f"Failed to export text encoder: {e}")
            text_path = None
        
        # Export image encoder
        image_path = export_dir / "image_encoder.onnx"
        try:
            exporter.export_image_encoder(
                model=image_encoder.model,
                output_path=str(image_path),
                input_shape=(1, 3, 224, 224)
            )
            logger.info(f"Image encoder exported to {image_path}")
        except Exception as e:
            logger.error(f"Failed to export image encoder: {e}")
            image_path = None
        
        # Export fusion head
        fusion_path = export_dir / "fusion_head.onnx"
        try:
            exporter.export_fusion_head(
                model=fusion_head,
                output_path=str(fusion_path),
                input_shape=(1, 1280)
            )
            logger.info(f"Fusion head exported to {fusion_path}")
        except Exception as e:
            logger.error(f"Failed to export fusion head: {e}")
            fusion_path = None
        
        return {
            'text_encoder': str(text_path) if text_path else None,
            'image_encoder': str(image_path) if image_path else None,
            'fusion_head': str(fusion_path) if fusion_path else None
        }
    
    def benchmark_pytorch_model(self, model, model_name: str, input_data: torch.Tensor, 
                               num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark PyTorch model."""
        logger.info(f"Benchmarking PyTorch {model_name}")
        
        model.eval()
        
        stats = benchmark_model(
            model,
            input_data,
            num_runs=num_runs,
            warmup_runs=20
        )
        
        return {
            'backend': 'pytorch',
            'device': str(self.device),
            'latency_stats': stats,
            'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
        }
    
    def benchmark_onnx_cpu(self, model_path: str, model_name: str, input_data: np.ndarray,
                          num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark ONNX model on CPU."""
        logger.info(f"Benchmarking ONNX CPU {model_name}")
        
        try:
            import onnxruntime as ort
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: input_data.astype(np.float32)}
            
            # Benchmark
            latencies = []
            
            # Warmup
            for _ in range(20):
                session.run(None, ort_inputs)
            
            # Actual benchmark
            for _ in range(num_runs):
                start_time = time.time()
                outputs = session.run(None, ort_inputs)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            stats = compute_statistics(latencies)
            
            # Get model size
            model_size_mb = os.path.getsize(model_path) / (1024**2)
            
            return {
                'backend': 'onnx_cpu',
                'device': 'cpu',
                'latency_stats': stats,
                'model_size_mb': model_size_mb,
                'providers': session.get_providers()
            }
            
        except Exception as e:
            logger.error(f"Failed to benchmark ONNX CPU {model_name}: {e}")
            return {'error': str(e)}
    
    def benchmark_onnx_gpu(self, model_path: str, model_name: str, input_data: np.ndarray,
                          num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark ONNX model on GPU."""
        logger.info(f"Benchmarking ONNX GPU {model_name}")
        
        try:
            import onnxruntime as ort
            
            # Check GPU availability
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)
            
            if 'CUDAExecutionProvider' not in session.get_providers():
                logger.warning("CUDA not available for ONNX, falling back to CPU")
                return self.benchmark_onnx_cpu(model_path, model_name, input_data, num_runs)
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: input_data.astype(np.float32)}
            
            # Benchmark
            latencies = []
            
            # Warmup
            for _ in range(20):
                session.run(None, ort_inputs)
            
            # Actual benchmark
            for _ in range(num_runs):
                start_time = time.time()
                outputs = session.run(None, ort_inputs)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)
            
            stats = compute_statistics(latencies)
            
            # Get model size
            model_size_mb = os.path.getsize(model_path) / (1024**2)
            
            return {
                'backend': 'onnx_gpu',
                'device': 'cuda',
                'latency_stats': stats,
                'model_size_mb': model_size_mb,
                'providers': session.get_providers()
            }
            
        except Exception as e:
            logger.error(f"Failed to benchmark ONNX GPU {model_name}: {e}")
            return {'error': str(e)}
    
    def benchmark_coreml(self, model_path: str, model_name: str, input_data: np.ndarray,
                        num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark Core ML model (macOS only)."""
        logger.info(f"Benchmarking Core ML {model_name}")
        
        try:
            import coremltools as ct
            
            # Load Core ML model
            model = ct.models.MLModel(model_path)
            
            # Prepare input (Core ML uses dictionaries)
            input_dict = {'input': input_data.astype(np.float32)}
            
            # Benchmark
            latencies = []
            
            # Warmup
            for _ in range(20):
                model.predict(input_dict)
            
            # Actual benchmark
            for _ in range(num_runs):
                start_time = time.time()
                outputs = model.predict(input_dict)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)
            
            stats = compute_statistics(latencies)
            
            # Get model size
            model_size_mb = os.path.getsize(model_path) / (1024**2)
            
            return {
                'backend': 'coreml',
                'device': 'cpu',  # Core ML runs on CPU or Neural Engine
                'latency_stats': stats,
                'model_size_mb': model_size_mb
            }
            
        except Exception as e:
            logger.error(f"Failed to benchmark Core ML {model_name}: {e}")
            return {'error': str(e)}
    
    def export_to_coreml(self, pytorch_model, model_name: str, input_shape: Tuple[int, ...],
                        export_dir: Path) -> str:
        """Export PyTorch model to Core ML format."""
        logger.info(f"Exporting {model_name} to Core ML")
        
        try:
            import coremltools as ct
            
            # Create sample input
            sample_input = torch.randn(input_shape)
            
            # Trace the model
            traced_model = torch.jit.trace(pytorch_model, sample_input)
            
            # Convert to Core ML
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape, dtype=np.float32)]
            )
            
            # Save model
            coreml_path = export_dir / f"{model_name}.mlmodel"
            mlmodel.save(str(coreml_path))
            
            logger.info(f"Core ML model saved to {coreml_path}")
            
            return str(coreml_path)
            
        except Exception as e:
            logger.error(f"Failed to export {model_name} to Core ML: {e}")
            return None
    
    def compare_predictions(self, pytorch_model, onnx_path: str, coreml_path: str,
                           input_data: torch.Tensor, model_name: str) -> Dict[str, Any]:
        """Compare predictions across different backends."""
        logger.info(f"Comparing predictions for {model_name}")
        
        try:
            # PyTorch prediction
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(input_data).cpu().numpy()
            
            # ONNX prediction
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: input_data.cpu().numpy().astype(np.float32)}
            onnx_outputs = session.run(None, ort_inputs)[0]
            
            # Core ML prediction (if available)
            coreml_output = None
            if coreml_path and os.path.exists(coreml_path):
                try:
                    import coremltools as ct
                    model = ct.models.MLModel(coreml_path)
                    input_dict = {'input': input_data.cpu().numpy().astype(np.float32)}
                    coreml_output = model.predict(input_dict)['output']
                except Exception as e:
                    logger.warning(f"Core ML prediction failed: {e}")
            
            # Calculate differences
            onnx_diff = np.abs(pytorch_output - onnx_outputs).mean()
            
            comparison = {
                'pytorch_shape': pytorch_output.shape,
                'onnx_shape': onnx_outputs.shape,
                'onnx_mean_diff': float(onnx_diff),
                'onnx_max_diff': float(np.abs(pytorch_output - onnx_outputs).max())
            }
            
            if coreml_output is not None:
                coreml_diff = np.abs(pytorch_output - coreml_output).mean()
                comparison.update({
                    'coreml_shape': coreml_output.shape,
                    'coreml_mean_diff': float(coreml_diff),
                    'coreml_max_diff': float(np.abs(pytorch_output - coreml_output).max())
                })
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare predictions for {model_name}: {e}")
            return {'error': str(e)}
    
    def run_cross_platform_benchmark(self, text_encoder, image_encoder, fusion_head,
                                   export_dir: Path) -> Dict[str, Any]:
        """Run comprehensive cross-platform benchmark."""
        logger.info("Running cross-platform benchmark")
        
        results = {}
        
        # Export models to ONNX
        onnx_paths = self.export_models_to_onnx(text_encoder, image_encoder, fusion_head, export_dir)
        
        # Export fusion head to Core ML (macOS only)
        coreml_path = None
        if sys.platform == "darwin":
            coreml_path = self.export_to_coreml(
                fusion_head, "fusion_head", (1, 1280), export_dir
            )
        
        # Benchmark each model across backends
        models_to_test = [
            ('text_encoder', text_encoder, (1, 512)),
            ('image_encoder', image_encoder, (1, 3, 224, 224)),
            ('fusion_head', fusion_head, (1, 1280))
        ]
        
        for model_name, model, input_shape in models_to_test:
            logger.info(f"Benchmarking {model_name}")
            
            model_results = {}
            
            # PyTorch benchmark
            pytorch_input = torch.randn(input_shape).to(self.device)
            model_results['pytorch'] = self.benchmark_pytorch_model(
                model, model_name, pytorch_input
            )
            
            # ONNX benchmarks
            if onnx_paths[model_name]:
                onnx_input = np.random.randn(*input_shape).astype(np.float32)
                
                model_results['onnx_cpu'] = self.benchmark_onnx_cpu(
                    onnx_paths[model_name], model_name, onnx_input
                )
                
                if torch.cuda.is_available():
                    model_results['onnx_gpu'] = self.benchmark_onnx_gpu(
                        onnx_paths[model_name], model_name, onnx_input
                    )
            
            # Core ML benchmark (fusion head only, macOS only)
            if model_name == 'fusion_head' and coreml_path:
                coreml_input = np.random.randn(*input_shape).astype(np.float32)
                model_results['coreml'] = self.benchmark_coreml(
                    coreml_path, model_name, coreml_input
                )
            
            # Compare predictions
            if onnx_paths[model_name]:
                pytorch_input = torch.randn(input_shape).to(self.device)
                comparison = self.compare_predictions(
                    model, onnx_paths[model_name], coreml_path, pytorch_input, model_name
                )
                model_results['prediction_comparison'] = comparison
            
            results[model_name] = model_results
        
        # Summary statistics
        summary = self.generate_summary(results)
        results['summary'] = summary
        
        return results
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all models and backends."""
        logger.info("Generating cross-platform summary")
        
        summary = {
            'backend_comparison': {},
            'latency_analysis': {},
            'accuracy_analysis': {}
        }
        
        # Collect latency data
        backend_latencies = {}
        
        for model_name, model_results in results.items():
            if model_name == 'summary':
                continue
                
            for backend, backend_results in model_results.items():
                if backend in ['pytorch', 'onnx_cpu', 'onnx_gpu', 'coreml'] and 'latency_stats' in backend_results:
                    if backend not in backend_latencies:
                        backend_latencies[backend] = []
                    backend_latencies[backend].append(backend_results['latency_stats']['mean'])
        
        # Calculate average latencies per backend
        for backend, latencies in backend_latencies.items():
            if latencies:
                summary['latency_analysis'][backend] = {
                    'avg_latency_ms': np.mean(latencies),
                    'std_latency_ms': np.std(latencies),
                    'num_models': len(latencies)
                }
        
        # Calculate prediction accuracy differences
        prediction_diffs = {}
        for model_name, model_results in results.items():
            if model_name == 'summary':
                continue
                
            if 'prediction_comparison' in model_results:
                comparison = model_results['prediction_comparison']
                if 'onnx_mean_diff' in comparison:
                    if 'onnx' not in prediction_diffs:
                        prediction_diffs['onnx'] = []
                    prediction_diffs['onnx'].append(comparison['onnx_mean_diff'])
                
                if 'coreml_mean_diff' in comparison:
                    if 'coreml' not in prediction_diffs:
                        prediction_diffs['coreml'] = []
                    prediction_diffs['coreml'].append(comparison['coreml_mean_diff'])
        
        for backend, diffs in prediction_diffs.items():
            if diffs:
                summary['accuracy_analysis'][backend] = {
                    'avg_prediction_diff': np.mean(diffs),
                    'max_prediction_diff': np.max(diffs),
                    'num_models': len(diffs)
                }
        
        return summary

def run_r7_experiment(config: Dict[str, Any], dirs: Dict[str, Path], seed: int) -> Dict[str, Any]:
    """Run R7 cross-platform parity experiment."""
    logger.info(f"Running R7 experiment with seed {seed}")
    
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
        
        # Create cross-platform runner
        runner = CrossPlatformRunner(config)
        
        # Create export directory
        export_dir = dirs['models']
        export_dir.mkdir(exist_ok=True)
        
        # Run cross-platform benchmark
        benchmark_results = runner.run_cross_platform_benchmark(
            text_encoder, image_encoder, fusion_head, export_dir
        )
        
        # Compile results
        total_time = time.time() - start_time
        memory_monitor.update_peak()
        
        results = {
            'experiment_type': 'r7_parity',
            'seed': seed,
            'config': config,
            'benchmark_results': benchmark_results,
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
        summary = benchmark_results.get('summary', {})
        latency_analysis = summary.get('latency_analysis', {})
        accuracy_analysis = summary.get('accuracy_analysis', {})
        
        results['metrics'] = {
            'pytorch_avg_latency': latency_analysis.get('pytorch', {}).get('avg_latency_ms', 0),
            'onnx_cpu_avg_latency': latency_analysis.get('onnx_cpu', {}).get('avg_latency_ms', 0),
            'onnx_gpu_avg_latency': latency_analysis.get('onnx_gpu', {}).get('avg_latency_ms', 0),
            'onnx_prediction_diff': accuracy_analysis.get('onnx', {}).get('avg_prediction_diff', 0),
            'coreml_prediction_diff': accuracy_analysis.get('coreml', {}).get('avg_prediction_diff', 0)
        }
        
        logger.info(f"R7 experiment completed successfully in {format_time(total_time)}")
        
        return results
        
    except Exception as e:
        logger.error(f"R7 experiment failed: {e}")
        return {
            'experiment_type': 'r7_parity',
            'seed': seed,
            'error': str(e),
            'status': 'failed'
        }
