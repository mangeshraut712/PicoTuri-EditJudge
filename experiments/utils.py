"""
Experiment Utilities
Common utilities for reproducible experiments
"""

import logging
import random
import numpy as np
import torch
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logging configuration."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_model_hash(model_path: str) -> str:
    """Get hash of model file for verification."""
    if not os.path.exists(model_path):
        return "file_not_found"
    
    hash_md5 = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

def get_config_hash(config: Dict[str, Any]) -> str:
    """Get hash of configuration for tracking."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def save_metrics(metrics: Dict[str, Any], output_path: str):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(input_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)

def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_device(prefer_gpu: bool = True) -> str:
    """Get appropriate device for computation."""
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def format_time(seconds: float) -> str:
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def format_memory(bytes_val: int) -> str:
    """Format memory in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"

def compute_statistics(values: list) -> Dict[str, float]:
    """Compute basic statistics for a list of values."""
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'count': len(values)
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate experiment configuration."""
    required_sections = ['experiment', 'models', 'training', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required section: {section}")
            return False
    
    # Validate experiment section
    exp_config = config['experiment']
    if 'name' not in exp_config:
        logging.error("Experiment name is required")
        return False
    
    # Validate models section
    models_config = config['models']
    if 'text_encoder' not in models_config or 'image_encoder' not in models_config:
        logging.error("Both text_encoder and image_encoder must be specified")
        return False
    
    # Validate training section
    training_config = config['training']
    required_training = ['batch_size', 'learning_rate', 'epochs']
    for field in required_training:
        if field not in training_config:
            logging.error(f"Missing required training field: {field}")
            return False
    
    return True

def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        commit = os.popen('git rev-parse HEAD').read().strip()
        branch = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()
        remote = os.popen('git config --get remote.origin.url').read().strip()
        is_dirty = os.popen('git status --porcelain').read().strip() != ""
        
        return {
            'commit': commit,
            'branch': branch,
            'remote': remote,
            'is_dirty': is_dirty
        }
    except Exception:
        return {
            'commit': 'unknown',
            'branch': 'unknown',
            'remote': 'unknown',
            'is_dirty': False
        }

def get_package_info() -> Dict[str, str]:
    """Get Python package information."""
    info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
    }
    
    # Add optional packages
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import onnx
        info['onnx_version'] = onnx.__version__
    except ImportError:
        pass
    
    try:
        import onnxruntime
        info['onnxruntime_version'] = onnxruntime.__version__
    except ImportError:
        pass
    
    return info

def create_experiment_report(config: Dict[str, Any], results: Dict[str, Any], 
                            output_dir: str) -> str:
    """Create a comprehensive experiment report."""
    report_lines = [
        "# Experiment Report",
        "",
        f"**Experiment**: {config['experiment']['name']}",
        f"**Description**: {config['experiment'].get('description', 'No description')}",
        f"**Status**: {results.get('status', 'unknown')}",
        "",
        "## Configuration",
        "",
        "### Models",
        f"- Text Encoder: {config['models']['text_encoder'].get('name', 'unknown')}",
        f"- Image Encoder: {config['models']['image_encoder'].get('name', 'unknown')}",
        "",
        "### Training",
        f"- Batch Size: {config['training']['batch_size']}",
        f"- Learning Rate: {config['training']['learning_rate']}",
        f"- Epochs: {config['training']['epochs']}",
        "",
        "## Results",
        ""
    ]
    
    if results.get('status') == 'completed':
        metrics = results.get('metrics', {})
        if metrics:
            report_lines.append("### Metrics")
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    report_lines.append(
                        f"- **{metric_name}**: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}"
                    )
    else:
        report_lines.append("Experiment failed to complete successfully.")
        if 'errors' in results:
            report_lines.extend(["Errors:", ""] + [f"- {error}" for error in results['errors']])
    
    report_lines.extend([
        "",
        "## System Information",
        "",
        "### Environment",
    ])
    
    # Add system info
    git_info = get_git_info()
    package_info = get_package_info()
    
    report_lines.extend([
        f"- Python: {package_info['python_version'].split()[0]}",
        f"- PyTorch: {package_info['torch_version']}",
        f"- NumPy: {package_info['numpy_version']}",
        f"- Git Commit: {git_info['commit'][:8]}",
        f"- Branch: {git_info['branch']}",
        "",
        "### Hardware",
        f"- CUDA Available: {torch.cuda.is_available()}",
        f"- MPS Available: {torch.backends.mps.is_available()}",
    ])
    
    if torch.cuda.is_available():
        report_lines.extend([
            f"- GPU: {torch.cuda.get_device_name()}",
            f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        ])
    
    # Write report
    report_path = Path(output_dir) / "experiment_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return str(report_path)

class ExperimentTimer:
    """Context manager for timing experiments."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logging.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logging.info(f"Completed: {self.name} in {format_time(duration)}")

class MemoryMonitor:
    """Monitor memory usage during experiments."""
    
    def __init__(self):
        self.start_memory = None
        self.peak_memory = None
        self.device = get_device()
    
    def start(self):
        """Start monitoring."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
    
    def get_current_memory(self) -> int:
        """Get current memory usage in bytes."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated()
        else:
            return self.peak_memory or self.start_memory
    
    def update_peak(self):
        """Update peak memory usage."""
        current = self.get_current_memory()
        if self.peak_memory is None or current > self.peak_memory:
            self.peak_memory = current

def benchmark_model(model, input_data, num_runs: int = 100, warmup_runs: int = 10):
    """Benchmark model inference latency."""
    device = get_device()
    model = model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            if isinstance(input_data, (list, tuple)):
                input_data_device = [x.to(device) if torch.is_tensor(x) else x for x in input_data]
            else:
                input_data_device = input_data.to(device)
            _ = model(input_data_device)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if isinstance(input_data, (list, tuple)):
                input_data_device = [x.to(device) if torch.is_tensor(x) else x for x in input_data]
            else:
                input_data_device = input_data.to(device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_data_device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return compute_statistics(latencies)
