#!/usr/bin/env python3
"""
Experiment Comparison Tool
Compare results across different experiments and configurations
"""

import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.utils import setup_logging

logger = logging.getLogger(__name__)

def load_experiment_results(exp_paths: List[str]) -> Dict[str, Any]:
    """Load results from multiple experiments."""
    results = {}
    
    for exp_path in exp_paths:
        path = Path(exp_path)
        
        if not path.exists():
            logger.warning(f"Experiment path not found: {exp_path}")
            continue
        
        # Look for aggregated_metrics.json
        agg_file = path / "results" / "aggregated_metrics.json"
        if agg_file.exists():
            with open(agg_file, 'r') as f:
                results[path.name] = json.load(f)
        else:
            logger.warning(f"No aggregated results found for {exp_path}")
    
    return results

def extract_metrics_for_comparison(results: Dict[str, Any], 
                                 metrics: List[str]) -> pd.DataFrame:
    """Extract specified metrics from experiment results."""
    data = []
    
    for exp_name, exp_result in results.items():
        if exp_result.get('status') != 'completed':
            continue
        
        row = {'experiment': exp_name}
        
        # Extract metrics
        exp_metrics = exp_result.get('metrics', {})
        for metric in metrics:
            row[metric] = exp_metrics.get(metric, np.nan)
        
        # Add configuration info
        config = exp_result.get('config', {})
        if 'experiment' in config:
            row['description'] = config['experiment'].get('description', '')
            row['type'] = config['experiment'].get('type', '')
        
        data.append(row)
    
    return pd.DataFrame(data)

def create_comparison_table(df: pd.DataFrame, metrics: List[str]) -> str:
    """Create a formatted comparison table."""
    table_lines = [
        "# Experiment Comparison Results",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Metrics Comparison",
        "",
        "| Experiment | " + " | ".join([m.replace('_', ' ').title() for m in metrics]) + " | Description |",
        "|" + "---|" * (len(metrics) + 2)
    ]
    
    for _, row in df.iterrows():
        exp_name = row['experiment']
        metric_values = []
        
        for metric in metrics:
            value = row[metric]
            if pd.isna(value):
                metric_values.append("N/A")
            elif isinstance(value, float):
                if metric in ['auc', 'f1', 'accuracy']:
                    metric_values.append(f"{value:.4f}")
                elif 'latency' in metric.lower():
                    metric_values.append(f"{value:.2f}ms")
                elif 'throughput' in metric.lower():
                    metric_values.append(f"{value:.1f}/s")
                else:
                    metric_values.append(f"{value:.4f}")
            else:
                metric_values.append(str(value))
        
        description = row.get('description', 'No description')
        
        table_lines.append(
            f"| {exp_name} | " + " | ".join(metric_values) + f" | {description} |"
        )
    
    return "\n".join(table_lines)

def create_performance_plots(df: pd.DataFrame, metrics: List[str], 
                            output_dir: Path) -> Dict[str, str]:
    """Create performance comparison plots."""
    plot_files = {}
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Filter out NaN values for plotting
    plot_df = df.dropna(subset=metrics)
    
    if plot_df.empty:
        logger.warning("No valid data for plotting")
        return plot_files
    
    # 1. Metrics bar plot
    if len(metrics) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            if metric in plot_df.columns:
                ax = axes[i]
                
                # Create bar plot
                values = plot_df[metric].values
                exp_names = plot_df['experiment'].values
                
                bars = ax.bar(range(len(values)), values)
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(exp_names, rotation=45, ha='right')
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel(metric.replace('_', ' ').title())
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}' if not np.isnan(value) else 'N/A',
                           ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(metrics), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_file = output_dir / "metrics_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['metrics_comparison'] = str(plot_file)
    
    # 2. Scatter plot: Accuracy vs Latency
    if 'auc' in metrics and any('latency' in m for m in metrics):
        latency_metric = next((m for m in metrics if 'latency' in m), None)
        if latency_metric and latency_metric in plot_df.columns:
            plt.figure(figsize=(10, 6))
            
            x = plot_df[latency_metric]
            y = plot_df['auc']
            exp_names = plot_df['experiment']
            
            plt.scatter(x, y, s=100, alpha=0.7)
            
            # Add labels for each point
            for i, name in enumerate(exp_names):
                plt.annotate(name, (x.iloc[i], y.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            plt.xlabel(latency_metric.replace('_', ' ').title())
            plt.ylabel('AUC')
            plt.title('Accuracy vs Latency Trade-off')
            plt.grid(True, alpha=0.3)
            
            plot_file = output_dir / "accuracy_vs_latency.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['accuracy_vs_latency'] = str(plot_file)
    
    # 3. Pareto front plot
    if 'auc' in metrics and any('latency' in m for m in metrics):
        latency_metric = next((m for m in metrics if 'latency' in m), None)
        if latency_metric and latency_metric in plot_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Calculate Pareto front
            x = plot_df[latency_metric].values
            y = plot_df['auc'].values
            
            # Sort by latency (ascending) for Pareto calculation
            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            
            # Find Pareto optimal points
            pareto_indices = []
            max_auc_so_far = -np.inf
            
            for i in range(len(y_sorted) - 1, -1, -1):
                if y_sorted[i] > max_auc_so_far:
                    pareto_indices.append(i)
                    max_auc_so_far = y_sorted[i]
            
            pareto_indices = sorted(pareto_indices)
            
            # Plot all points
            plt.scatter(x, y, s=100, alpha=0.6, label='All experiments')
            
            # Highlight Pareto front
            plt.scatter(x[pareto_indices], y[pareto_indices], 
                       s=150, alpha=0.8, label='Pareto front', 
                       edgecolors='red', linewidth=2)
            
            # Draw Pareto front line
            if len(pareto_indices) > 1:
                pareto_x = x[pareto_indices]
                pareto_y = y[pareto_indices]
                plt.plot(pareto_x, pareto_y, 'r--', alpha=0.5, label='Pareto frontier')
            
            plt.xlabel(latency_metric.replace('_', ' ').title())
            plt.ylabel('AUC')
            plt.title('Pareto Front: Accuracy vs Latency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = output_dir / "pareto_front.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['pareto_front'] = str(plot_file)
    
    return plot_files

def perform_statistical_tests(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
    """Perform statistical tests between experiments."""
    from scipy.stats import ttest_rel, wilcoxon
    
    results = {}
    
    # Get completed experiments
    completed_exps = df.dropna(subset=metrics)
    
    if len(completed_exps) < 2:
        logger.warning("Need at least 2 completed experiments for statistical tests")
        return results
    
    exp_names = completed_exps['experiment'].tolist()
    
    for metric in metrics:
        if metric not in completed_exps.columns:
            continue
        
        metric_results = {}
        values = completed_exps[metric].values
        
        # Pairwise comparisons
        for i in range(len(exp_names)):
            for j in range(i + 1, len(exp_names)):
                exp1, exp2 = exp_names[i], exp_names[j]
                val1, val2 = values[i], values[j]
                
                if np.isnan(val1) or np.isnan(val2):
                    continue
                
                comparison_key = f"{exp1}_vs_{exp2}"
                
                # Perform t-test (if we had multiple samples per experiment)
                # For now, we'll just report the difference
                diff = val1 - val2
                
                metric_results[comparison_key] = {
                    'difference': diff,
                    'exp1_value': val1,
                    'exp2_value': val2,
                    'relative_improvement': (diff / val2) * 100 if val2 != 0 else 0
                }
        
        if metric_results:
            results[metric] = metric_results
    
    return results

def generate_recommendations(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    """Generate recommendations based on experiment results."""
    recommendations = []
    
    # Find best performing experiment for each metric
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Remove NaN values
        valid_df = df.dropna(subset=[metric])
        
        if valid_df.empty:
            continue
        
        if metric in ['auc', 'f1', 'accuracy']:
            # Higher is better
            best_idx = valid_df[metric].idxmax()
            best_exp = valid_df.loc[best_idx, 'experiment']
            best_val = valid_df.loc[best_idx, metric]
            
            recommendations.append(
                f"**Best {metric.replace('_', ' ').title()}**: {best_exp} ({best_val:.4f})"
            )
        
        elif 'latency' in metric.lower():
            # Lower is better
            best_idx = valid_df[metric].idxmin()
            best_exp = valid_df.loc[best_idx, 'experiment']
            best_val = valid_df.loc[best_idx, metric]
            
            recommendations.append(
                f"**Best {metric.replace('_', ' ').title()}**: {best_exp} ({best_val:.2f}ms)"
            )
        
        elif 'throughput' in metric.lower():
            # Higher is better
            best_idx = valid_df[metric].idxmax()
            best_exp = valid_df.loc[best_idx, 'experiment']
            best_val = valid_df.loc[best_idx, metric]
            
            recommendations.append(
                f"**Best {metric.replace('_', ' ').title()}**: {best_exp} ({best_val:.1f}/s)"
            )
    
    # Trade-off recommendations
    if 'auc' in metrics and any('latency' in m for m in metrics):
        latency_metric = next((m for m in metrics if 'latency' in m), None)
        if latency_metric:
            valid_df = df.dropna(subset=['auc', latency_metric])
            
            if not valid_df.empty:
                # Find experiments with good balance (e.g., AUC > 0.8 and latency < 100ms)
                balanced = valid_df[
                    (valid_df['auc'] > 0.8) & 
                    (valid_df[latency_metric] < 100)
                ]
                
                if not balanced.empty:
                    best_balanced = balanced.loc[balanced['auc'].idxmax(), 'experiment']
                    recommendations.append(
                        f"**Best Balance**: {best_balanced} (good accuracy with acceptable latency)"
                    )
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--experiments", nargs="+", required=True, 
                       help="Paths to experiment directories")
    parser.add_argument("--metrics", nargs="+", 
                       default=["auc", "f1", "latency_p95", "throughput"],
                       help="Metrics to compare")
    parser.add_argument("--output", type=str, default="comparison_results",
                       help="Output directory for results")
    parser.add_argument("--format", choices=["markdown", "html", "json"], 
                       default="markdown", help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load experiment results
    logger.info("Loading experiment results...")
    results = load_experiment_results(args.experiments)
    
    if not results:
        logger.error("No valid experiment results found")
        return 1
    
    logger.info(f"Loaded {len(results)} experiment results")
    
    # Extract metrics for comparison
    df = extract_metrics_for_comparison(results, args.metrics)
    
    if df.empty:
        logger.error("No valid metrics data found")
        return 1
    
    logger.info(f"Extracted metrics for {len(df)} experiments")
    
    # Generate comparison table
    table_content = create_comparison_table(df, args.metrics)
    
    # Create plots
    logger.info("Generating comparison plots...")
    plot_files = create_performance_plots(df, args.metrics, output_dir)
    
    # Perform statistical tests
    logger.info("Performing statistical analysis...")
    statistical_results = perform_statistical_tests(df, args.metrics)
    
    # Generate recommendations
    logger.info("Generating recommendations...")
    recommendations = generate_recommendations(df, args.metrics)
    
    # Save results
    if args.format == "markdown":
        # Save markdown report
        report_lines = [table_content]
        
        if plot_files:
            report_lines.extend([
                "",
                "## Visualizations",
                ""
            ])
            for plot_name, plot_path in plot_files.items():
                plot_rel_path = Path(plot_path).name
                report_lines.append(f"### {plot_name.replace('_', ' ').title()}")
                report_lines.append(f"![{plot_name}]({plot_rel_path})")
                report_lines.append("")
        
        if statistical_results:
            report_lines.extend([
                "",
                "## Statistical Analysis",
                ""
            ])
            for metric, comparisons in statistical_results.items():
                report_lines.append(f"### {metric.replace('_', ' ').title()}")
                for comparison, stats in comparisons.items():
                    report_lines.append(
                        f"- **{comparison}**: Difference = {stats['difference']:.4f}, "
                        f"Relative improvement = {stats['relative_improvement']:.2f}%"
                    )
                report_lines.append("")
        
        if recommendations:
            report_lines.extend([
                "",
                "## Recommendations",
                ""
            ])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
        
        # Write markdown report
        report_file = output_dir / "comparison_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Comparison report saved to {report_file}")
    
    # Save JSON data
    json_file = output_dir / "comparison_data.json"
    comparison_data = {
        'experiments': list(results.keys()),
        'metrics': args.metrics,
        'data': df.to_dict('records'),
        'statistical_tests': statistical_results,
        'recommendations': recommendations,
        'plot_files': plot_files
    }
    
    with open(json_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    logger.info(f"Comparison data saved to {json_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*60)
    print(f"Experiments compared: {len(results)}")
    print(f"Metrics analyzed: {len(args.metrics)}")
    print(f"Output directory: {output_dir}")
    
    if recommendations:
        print("\nTOP RECOMMENDATIONS:")
        for rec in recommendations[:3]:
            print(f"  • {rec}")
    
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
