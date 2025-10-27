#!/usr/bin/env python3
"""
Experiment Runner
Systematic experiment execution with reproducibility and tracking
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import numpy as np
import torch
import random
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.utils import setup_logging, set_seed, get_model_hash
from experiments.r1_embeddings.run import run_r1_experiment
from experiments.r2_fusion.run import run_r2_experiment
from experiments.r3_domain.run import run_r3_experiment
from experiments.r4_preference.run import run_r4_experiment
from experiments.r5_robustness.run import run_r5_experiment
from experiments.r6_batching.run import run_r6_experiment
from experiments.r7_parity.run import run_r7_experiment

logger = logging.getLogger(__name__)

# Experiment registry
EXPERIMENT_REGISTRY = {
    "r1_embeddings": run_r1_experiment,
    "r2_fusion": run_r2_experiment,
    "r3_domain": run_r3_experiment,
    "r4_preference": run_r4_experiment,
    "r5_robustness": run_r5_experiment,
    "r6_batching": run_r6_experiment,
    "r7_parity": run_r7_experiment,
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['experiment', 'models', 'training', 'evaluation']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config")
    
    return config

def setup_experiment_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create experiment directory structure."""
    exp_name = config['experiment']['name']
    base_dir = Path(__file__).parent.parent / "experiments"
    
    dirs = {
        'root': base_dir / exp_name.split('_')[0] / exp_name,
        'configs': base_dir / exp_name.split('_')[0] / "configs",
        'results': base_dir / exp_name.split('_')[0] / exp_name / "results",
        'plots': base_dir / exp_name.split('_')[0] / exp_name / "plots",
        'models': base_dir / exp_name.split('_')[0] / exp_name / "models",
        'logs': base_dir / exp_name.split('_')[0] / exp_name / "logs",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def save_experiment_config(config: Dict[str, Any], dirs: Dict[str, Path]):
    """Save experiment configuration and metadata."""
    # Save full config
    config_path = dirs['results'] / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'git_commit': os.popen('git rev-parse HEAD').read().strip(),
        'experiment_name': config['experiment']['name'],
        'description': config['experiment'].get('description', ''),
    }
    
    metadata_path = dirs['results'] / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def run_single_experiment(config_path: str, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Run a single experiment with specified seeds."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup directories
    dirs = setup_experiment_dirs(config)
    
    # Setup logging
    log_file = dirs['logs'] / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(str(log_file))
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment'].get('description', 'No description')}")
    
    # Save configuration
    save_experiment_config(config, dirs)
    
    # Get experiment type
    exp_name = config['experiment']['name']
    exp_type = exp_name.split('_')[0]
    
    if exp_type not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
    # Get seeds
    if seeds is None:
        seeds = config['training'].get('seeds', [42])
    
    logger.info(f"Running with seeds: {seeds}")
    
    # Run experiment for each seed
    all_results = []
    for i, seed in enumerate(seeds):
        logger.info(f"Running seed {seed} ({i+1}/{len(seeds)})")
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Create seed-specific result directory
        seed_dirs = {
            'results': dirs['results'] / f"seed_{seed}",
            'models': dirs['models'] / f"seed_{seed}",
        }
        for dir_path in seed_dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        try:
            # Run the experiment
            experiment_func = EXPERIMENT_REGISTRY[exp_type]
            result = experiment_func(config, seed_dirs, seed)
            
            # Add seed info
            result['seed'] = seed
            result['config_hash'] = str(hash(str(config)))
            
            all_results.append(result)
            
            # Save individual seed result
            result_path = seed_dirs['results'] / "metrics.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Seed {seed} completed successfully")
            
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            # Save error info
            error_result = {
                'seed': seed,
                'error': str(e),
                'status': 'failed'
            }
            all_results.append(error_result)
    
    # Aggregate results across seeds
    aggregated = aggregate_results(all_results, config)
    
    # Save aggregated results
    agg_path = dirs['results'] / "aggregated_metrics.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Generate plots
    generate_plots(aggregated, dirs, config)
    
    # Generate findings summary
    generate_findings(aggregated, dirs, config)
    
    logger.info(f"Experiment completed: {config['experiment']['name']}")
    logger.info(f"Results saved to: {dirs['results']}")
    
    return aggregated

def aggregate_results(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate results across multiple seeds."""
    # Filter out failed results
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        return {
            'status': 'failed',
            'errors': [r.get('error', 'Unknown error') for r in results]
        }
    
    # Get all metric keys
    metric_keys = set()
    for result in successful_results:
        metric_keys.update(k for k in result.keys() if isinstance(result[k], (int, float)))
    
    # Compute statistics
    aggregated = {
        'status': 'completed',
        'num_seeds': len(successful_results),
        'failed_seeds': len(results) - len(successful_results),
        'metrics': {}
    }
    
    for key in metric_keys:
        values = [r[key] for r in successful_results if key in r]
        if values:
            aggregated['metrics'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
    
    # Add configuration info
    aggregated['config'] = config
    aggregated['timestamp'] = datetime.now().isoformat()
    
    return aggregated

def generate_plots(aggregated: Dict[str, Any], dirs: Dict[str, Path], config: Dict[str, Any]):
    """Generate experiment plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        exp_type = config['experiment']['name'].split('_')[0]
        
        if exp_type == 'r1_embeddings':
            generate_r1_plots(aggregated, dirs, config)
        elif exp_type == 'r2_fusion':
            generate_r2_plots(aggregated, dirs, config)
        elif exp_type == 'r6_batching':
            generate_r6_plots(aggregated, dirs, config)
        else:
            # Generic metrics plot
            generate_generic_plots(aggregated, dirs, config)
        
        logger.info("Plots generated successfully")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plots")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")

def generate_r1_plots(aggregated: Dict[str, Any], dirs: Dict[str, Path], config: Dict[str, Any]):
    """Generate R1 embedding comparison plots."""
    import matplotlib.pyplot as plt
    
    metrics = aggregated['metrics']
    
    # Accuracy vs Latency scatter plot
    if 'auc' in metrics and 'latency_p95' in metrics:
        plt.figure(figsize=(10, 6))
        
        # This would need multiple experiments for proper scatter plot
        # For now, create a simple bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        ax1.bar(['AUC'], [metrics['auc']['mean']], yerr=[metrics['auc']['std']])
        ax1.set_ylabel('AUC')
        ax1.set_title('Model Performance (AUC)')
        ax1.set_ylim(0, 1)
        
        # Latency comparison
        ax2.bar(['P95 Latency (ms)'], [metrics['latency_p95']['mean']], 
                yerr=[metrics['latency_p95']['std']])
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Inference Latency')
        
        plt.tight_layout()
        plt.savefig(dirs['plots'] / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_r2_plots(aggregated: Dict[str, Any], dirs: Dict[str, Path], config: Dict[str, Any]):
    """Generate R2 fusion architecture plots."""
    import matplotlib.pyplot as plt
    
    metrics = aggregated['metrics']
    
    # Architecture comparison
    if 'auc' in metrics and 'ece' in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        ax1.bar(['AUC'], [metrics['auc']['mean']], yerr=[metrics['auc']['std']])
        ax1.set_ylabel('AUC')
        ax1.set_title('Fusion Architecture Performance')
        ax1.set_ylim(0, 1)
        
        # ECE comparison
        ax2.bar(['ECE'], [metrics['ece']['mean']], yerr=[metrics['ece']['std']])
        ax2.set_ylabel('ECE')
        ax2.set_title('Calibration Error')
        
        plt.tight_layout()
        plt.savefig(dirs['plots'] / 'fusion_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_r6_plots(aggregated: Dict[str, Any], dirs: Dict[str, Path], config: Dict[str, Any]):
    """Generate R6 batching and quantization plots."""
    import matplotlib.pyplot as plt
    
    metrics = aggregated['metrics']
    
    # Throughput vs Accuracy
    if 'throughput' in metrics and 'auc' in metrics:
        plt.figure(figsize=(10, 6))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Throughput
        ax1.bar(['Throughput (img/s)'], [metrics['throughput']['mean']], 
                yerr=[metrics['throughput']['std']])
        ax1.set_ylabel('Throughput (images/second)')
        ax1.set_title('Batching Performance')
        
        # AUC
        ax2.bar(['AUC'], [metrics['auc']['mean']], yerr=[metrics['auc']['std']])
        ax2.set_ylabel('AUC')
        ax2.set_title('Model Accuracy')
        
        plt.tight_layout()
        plt.savefig(dirs['plots'] / 'batching_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_generic_plots(aggregated: Dict[str, Any], dirs: Dict[str, Path], config: Dict[str, Any]):
    """Generate generic experiment plots."""
    import matplotlib.pyplot as plt
    
    metrics = aggregated['metrics']
    
    # Create a simple metrics overview
    metric_names = [k for k in metrics.keys() if k in ['auc', 'f1', 'accuracy']]
    if metric_names:
        plt.figure(figsize=(12, 6))
        
        means = [metrics[m]['mean'] for m in metric_names]
        stds = [metrics[m]['std'] for m in metric_names]
        
        plt.bar(metric_names, means, yerr=stds, capsize=5)
        plt.ylabel('Score')
        plt.title('Experiment Metrics Overview')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(dirs['plots'] / 'metrics_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_findings(aggregated: Dict[str, Any], dirs: Dict[str, Path], config: Dict[str, Any]):
    """Generate findings summary."""
    exp_name = config['experiment']['name']
    exp_type = exp_name.split('_')[0]
    
    findings = {
        'experiment': exp_name,
        'description': config['experiment'].get('description', ''),
        'timestamp': datetime.now().isoformat(),
        'status': aggregated['status'],
        'summary': '',
        'key_metrics': {},
        'recommendations': []
    }
    
    if aggregated['status'] == 'completed':
        metrics = aggregated['metrics']
        
        # Extract key metrics
        if 'auc' in metrics:
            findings['key_metrics']['auc'] = f"{metrics['auc']['mean']:.3f} ± {metrics['auc']['std']:.3f}"
        if 'f1' in metrics:
            findings['key_metrics']['f1'] = f"{metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.3f}"
        if 'latency_p95' in metrics:
            findings['key_metrics']['latency_p95_ms'] = f"{metrics['latency_p95']['mean']:.1f} ± {metrics['latency_p95']['std']:.1f}"
        if 'throughput' in metrics:
            findings['key_metrics']['throughput'] = f"{metrics['throughput']['mean']:.1f} ± {metrics['throughput']['std']:.1f} img/s"
        
        # Generate summary based on experiment type
        if exp_type == 'r1_embeddings':
            findings['summary'] = f"Embedding combination achieved AUC of {findings['key_metrics'].get('auc', 'N/A')} with P95 latency of {findings['key_metrics'].get('latency_p95_ms', 'N/A')}."
            if float(metrics['auc']['mean']) > 0.8:
                findings['recommendations'].append("Strong performance achieved, consider for production")
            if float(metrics['latency_p95']['mean']) < 100:
                findings['recommendations'].append("Latency suitable for real-time applications")
        
        elif exp_type == 'r2_fusion':
            findings['summary'] = f"Fusion architecture achieved AUC of {findings['key_metrics'].get('auc', 'N/A')} with ECE of {metrics.get('ece', {}).get('mean', 0):.3f}."
            if metrics.get('ece', {}).get('mean', 1) < 0.1:
                findings['recommendations'].append("Good calibration achieved")
        
        elif exp_type == 'r6_batching':
            findings['summary'] = f"Batching achieved throughput of {findings['key_metrics'].get('throughput', 'N/A')} with AUC of {findings['key_metrics'].get('auc', 'N/A')}."
            if float(metrics['throughput']['mean']) > 20:
                findings['recommendations'].append("High throughput suitable for production workloads")
    
    else:
        findings['summary'] = "Experiment failed to complete successfully."
        findings['recommendations'].append("Review error logs and retry with corrected configuration")
    
    # Save findings
    findings_path = dirs['results'] / "Findings.md"
    with open(findings_path, 'w') as f:
        f.write(f"# Findings: {exp_name}\n\n")
        f.write(f"**Description**: {findings['description']}\n\n")
        f.write(f"**Timestamp**: {findings['timestamp']}\n\n")
        f.write(f"**Status**: {findings['status']}\n\n")
        f.write(f"## Summary\n\n{findings['summary']}\n\n")
        
        if findings['key_metrics']:
            f.write("## Key Metrics\n\n")
            for metric, value in findings['key_metrics'].items():
                f.write(f"- **{metric}**: {value}\n")
            f.write("\n")
        
        if findings['recommendations']:
            f.write("## Recommendations\n\n")
            for rec in findings['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")
        
        f.write("## Detailed Results\n\n")
        f.write("See `aggregated_metrics.json` for complete results.\n\n")
        f.write("## Generated Files\n\n")
        f.write("- `config.yaml`: Experiment configuration\n")
        f.write("- `metadata.json`: Runtime metadata\n")
        f.write("- `aggregated_metrics.json`: Aggregated results across seeds\n")
        f.write("- `plots/`: Generated visualizations\n")
    
    logger.info(f"Findings saved to: {findings_path}")

def run_experiment_suite(suite_names: str, seeds: Optional[List[int]] = None):
    """Run a suite of experiments."""
    suite_list = suite_names.split(',')
    
    for suite_name in suite_list:
        suite_name = suite_name.strip()
        
        # Find all configs for this suite
        exp_dir = Path(__file__).parent.parent / "experiments" / suite_name
        config_dir = exp_dir / "configs"
        
        if not config_dir.exists():
            logger.error(f"Experiment suite not found: {suite_name}")
            continue
        
        # Find all YAML configs
        config_files = list(config_dir.glob("*.yaml"))
        
        if not config_files:
            logger.warning(f"No config files found for suite: {suite_name}")
            continue
        
        logger.info(f"Running {len(config_files)} experiments for suite: {suite_name}")
        
        for config_file in sorted(config_files):
            logger.info(f"Running experiment: {config_file.name}")
            try:
                run_single_experiment(str(config_file), seeds)
            except Exception as e:
                logger.error(f"Failed to run {config_file.name}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Run PicoTuri EditJudge experiments")
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    parser.add_argument("--suite", type=str, help="Experiment suite to run (e.g., r1_embeddings)")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated list of seeds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Setup logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.config:
        # Run single experiment
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        try:
            result = run_single_experiment(args.config, seeds)
            return 0 if result['status'] == 'completed' else 1
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return 1
    
    elif args.suite:
        # Run experiment suite
        try:
            run_experiment_suite(args.suite, seeds)
            return 0
        except Exception as e:
            logger.error(f"Suite execution failed: {e}")
            return 1
    
    else:
        logger.error("Must specify either --config or --suite")
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
