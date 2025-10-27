#!/usr/bin/env python3
"""
Results Analysis Tool
Statistical analysis and visualization of experiment results
"""

import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.utils import setup_logging

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """Comprehensive results analysis toolkit."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.analysis_results = {}
    
    def load_all_results(self) -> Dict[str, Any]:
        """Load all experiment results from the results directory."""
        logger.info(f"Loading results from {self.results_dir}")
        
        # Find all experiment directories
        exp_dirs = [d for d in self.results_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
        
        for exp_dir in exp_dirs:
            # Look for aggregated results
            agg_file = exp_dir / "results" / "aggregated_metrics.json"
            if agg_file.exists():
                with open(agg_file, 'r') as f:
                    self.results_data[exp_dir.name] = json.load(f)
                logger.info(f"Loaded results for {exp_dir.name}")
        
        logger.info(f"Loaded {len(self.results_data)} experiment results")
        return self.results_data
    
    def extract_metric_data(self, metric: str) -> pd.DataFrame:
        """Extract specific metric data across all experiments."""
        data = []
        
        for exp_name, exp_result in self.results_data.items():
            if exp_result.get('status') != 'completed':
                continue
            
            # Get metric values across seeds
            metrics = exp_result.get('metrics', {})
            if metric in metrics:
                metric_data = metrics[metric]
                
                if isinstance(metric_data, dict) and 'values' in metric_data:
                    # Multiple seeds
                    for i, value in enumerate(metric_data['values']):
                        data.append({
                            'experiment': exp_name,
                            'seed': i,
                            'value': value,
                            'mean': metric_data['mean'],
                            'std': metric_data['std']
                        })
                else:
                    # Single value
                    data.append({
                        'experiment': exp_name,
                        'seed': 0,
                        'value': metric_data,
                        'mean': metric_data,
                        'std': 0
                    })
        
        return pd.DataFrame(data)
    
    def analyze_metric_distributions(self, metrics: List[str]) -> Dict[str, Any]:
        """Analyze distributions of metrics across experiments."""
        logger.info("Analyzing metric distributions")
        
        analysis = {}
        
        for metric in metrics:
            df = self.extract_metric_data(metric)
            
            if df.empty:
                continue
            
            metric_analysis = {
                'experiments': df['experiment'].nunique(),
                'total_samples': len(df),
                'overall_mean': df['value'].mean(),
                'overall_std': df['value'].std(),
                'min_value': df['value'].min(),
                'max_value': df['value'].max(),
                'coefficient_of_variation': df['value'].std() / df['value'].mean() if df['value'].mean() != 0 else np.inf
            }
            
            # Per-experiment statistics
            exp_stats = []
            for exp_name in df['experiment'].unique():
                exp_df = df[df['experiment'] == exp_name]
                exp_stats.append({
                    'experiment': exp_name,
                    'mean': exp_df['value'].mean(),
                    'std': exp_df['value'].std(),
                    'count': len(exp_df),
                    'cv': exp_df['value'].std() / exp_df['value'].mean() if exp_df['value'].mean() != 0 else np.inf
                })
            
            metric_analysis['experiment_stats'] = exp_stats
            
            # Statistical tests
            if len(exp_stats) > 1:
                means = [stat['mean'] for stat in exp_stats]
                stds = [stat['std'] for stat in exp_stats]
                
                # ANOVA test for differences between experiments
                groups = [df[df['experiment'] == exp]['value'].values for exp in df['experiment'].unique()]
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    metric_analysis['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
            
            analysis[metric] = metric_analysis
        
        return analysis
    
    def analyze_correlations(self, metrics: List[str]) -> Dict[str, Any]:
        """Analyze correlations between metrics."""
        logger.info("Analyzing metric correlations")
        
        # Create correlation matrix
        correlation_data = {}
        
        for metric in metrics:
            df = self.extract_metric_data(metric)
            if not df.empty:
                # Use mean values per experiment for correlation
                exp_means = df.groupby('experiment')['value'].mean()
                correlation_data[metric] = exp_means
        
        if len(correlation_data) < 2:
            logger.warning("Need at least 2 metrics for correlation analysis")
            return {}
        
        # Create DataFrame for correlation analysis
        corr_df = pd.DataFrame(correlation_data)
        
        # Calculate correlation matrices
        pearson_corr = corr_df.corr(method='pearson')
        spearman_corr = corr_df.corr(method='spearman')
        
        # Find significant correlations
        significant_correlations = []
        
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                if i < j and metric1 in corr_df.columns and metric2 in corr_df.columns:
                    pearson_r, pearson_p = pearsonr(corr_df[metric1], corr_df[metric2])
                    spearman_r, spearman_p = spearmanr(corr_df[metric1], corr_df[metric2])
                    
                    if pearson_p < 0.05 or spearman_p < 0.05:
                        significant_correlations.append({
                            'metric1': metric1,
                            'metric2': metric2,
                            'pearson_r': pearson_r,
                            'pearson_p': pearson_p,
                            'spearman_r': spearman_r,
                            'spearman_p': spearman_p,
                            'strength': self._interpret_correlation(abs(pearson_r))
                        })
        
        return {
            'pearson_correlation': pearson_corr.to_dict(),
            'spearman_correlation': spearman_corr.to_dict(),
            'significant_correlations': significant_correlations
        }
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        elif abs_r < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def analyze_performance_tradeoffs(self, accuracy_metric: str, 
                                     efficiency_metrics: List[str]) -> Dict[str, Any]:
        """Analyze trade-offs between accuracy and efficiency."""
        logger.info("Analyzing performance trade-offs")
        
        # Get data for analysis
        accuracy_df = self.extract_metric_data(accuracy_metric)
        if accuracy_df.empty:
            logger.warning(f"No data found for accuracy metric: {accuracy_metric}")
            return {}
        
        # Use mean values per experiment
        accuracy_means = accuracy_df.groupby('experiment')['value'].mean()
        
        tradeoff_analysis = {}
        
        for eff_metric in efficiency_metrics:
            eff_df = self.extract_metric_data(eff_metric)
            if eff_df.empty:
                continue
            
            eff_means = eff_df.groupby('experiment')['value'].mean()
            
            # Align experiments
            common_exps = accuracy_means.index.intersection(eff_means.index)
            if len(common_exps) < 2:
                continue
            
            acc_values = accuracy_means[common_exps]
            eff_values = eff_means[common_exps]
            
            # Calculate correlation
            pearson_r, pearson_p = pearsonr(acc_values, eff_values)
            
            # Find Pareto optimal points
            if 'latency' in eff_metric.lower():
                # For latency, lower is better
                pareto_mask = []
                for i in range(len(acc_values)):
                    is_pareto = True
                    for j in range(len(acc_values)):
                        if i != j:
                            if (acc_values.iloc[j] >= acc_values.iloc[i] and 
                                eff_values.iloc[j] <= eff_values.iloc[i] and
                                (acc_values.iloc[j] > acc_values.iloc[i] or 
                                 eff_values.iloc[j] < eff_values.iloc[i])):
                                is_pareto = False
                                break
                    pareto_mask.append(is_pareto)
            else:
                # For throughput, higher is better
                pareto_mask = []
                for i in range(len(acc_values)):
                    is_pareto = True
                    for j in range(len(acc_values)):
                        if i != j:
                            if (acc_values.iloc[j] >= acc_values.iloc[i] and 
                                eff_values.iloc[j] >= eff_values.iloc[i] and
                                (acc_values.iloc[j] > acc_values.iloc[i] or 
                                 eff_values.iloc[j] > eff_values.iloc[i])):
                                is_pareto = False
                                break
                    pareto_mask.append(is_pareto)
            
            pareto_experiments = common_exps[pareto_mask].tolist()
            
            tradeoff_analysis[eff_metric] = {
                'correlation': pearson_r,
                'correlation_p_value': pearson_p,
                'correlation_strength': self._interpret_correlation(abs(pearson_r)),
                'pareto_optimal_experiments': pareto_experiments,
                'num_pareto_optimal': len(pareto_experiments),
                'best_accuracy': acc_values.max(),
                'best_efficiency': eff_values.min() if 'latency' in eff_metric.lower() else eff_values.max()
            }
        
        return tradeoff_analysis
    
    def analyze_reproducibility(self) -> Dict[str, Any]:
        """Analyze reproducibility across seeds."""
        logger.info("Analyzing reproducibility")
        
        reproducibility_analysis = {}
        
        for exp_name, exp_result in self.results_data.items():
            if exp_result.get('status') != 'completed':
                continue
            
            metrics = exp_result.get('metrics', {})
            exp_repro = {}
            
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'values' in metric_data:
                    values = metric_data['values']
                    if len(values) > 1:
                        # Calculate reproducibility metrics
                        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                        range_ratio = (np.max(values) - np.min(values)) / np.mean(values) if np.mean(values) != 0 else np.inf
                        
                        exp_repro[metric_name] = {
                            'num_seeds': len(values),
                            'coefficient_of_variation': cv,
                            'range_ratio': range_ratio,
                            'std_deviation': np.std(values),
                            'reproducibility_score': 1 / (1 + cv) if cv != np.inf else 0
                        }
            
            if exp_repro:
                reproducibility_analysis[exp_name] = exp_repro
        
        return reproducibility_analysis
    
    def generate_summary_report(self, output_dir: Path) -> str:
        """Generate comprehensive summary report."""
        logger.info("Generating summary report")
        
        report_lines = [
            "# Experiment Results Analysis Report",
            "",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Experiments Analyzed**: {len(self.results_data)}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Overall statistics
        total_experiments = len(self.results_data)
        completed_experiments = sum(1 for r in self.results_data.values() if r.get('status') == 'completed')
        
        report_lines.extend([
            f"- **Total Experiments**: {total_experiments}",
            f"- **Completed Experiments**: {completed_experiments}",
            f"- **Success Rate**: {completed_experiments/total_experiments*100:.1f}%",
            ""
        ])
        
        # Key findings from analysis
        if 'metric_distributions' in self.analysis_results:
            report_lines.extend([
                "## Metric Distribution Analysis",
                ""
            ])
            
            for metric, analysis in self.analysis_results['metric_distributions'].items():
                report_lines.extend([
                    f"### {metric.replace('_', ' ').title()}",
                    f"- **Experiments**: {analysis['experiments']}",
                    f"- **Overall Mean**: {analysis['overall_mean']:.4f}",
                    f"- **Overall Std**: {analysis['overall_std']:.4f}",
                    f"- **Range**: [{analysis['min_value']:.4f}, {analysis['max_value']:.4f}]",
                    ""
                ])
                
                if 'anova' in analysis:
                    anova = analysis['anova']
                    significance = "significant" if anova['significant_difference'] else "not significant"
                    report_lines.append(
                        f"- **ANOVA Test**: F={anova['f_statistic']:.3f}, p={anova['p_value']:.3f} ({significance})"
                    )
                    report_lines.append("")
        
        # Correlation analysis
        if 'correlations' in self.analysis_results:
            correlations = self.analysis_results['correlations']
            
            if correlations.get('significant_correlations'):
                report_lines.extend([
                    "## Significant Correlations",
                    ""
                ])
                
                for corr in correlations['significant_correlations']:
                    report_lines.append(
                        f"- **{corr['metric1']} vs {corr['metric2']}**: "
                        f"r={corr['pearson_r']:.3f} ({corr['strength']}), p={corr['pearson_p']:.3f}"
                    )
                
                report_lines.append("")
        
        # Performance trade-offs
        if 'performance_tradeoffs' in self.analysis_results:
            tradeoffs = self.analysis_results['performance_tradeoffs']
            
            report_lines.extend([
                "## Performance Trade-offs",
                ""
            ])
            
            for eff_metric, analysis in tradeoffs.items():
                report_lines.extend([
                    f"### Accuracy vs {eff_metric.replace('_', ' ').title()}",
                    f"- **Correlation**: r={analysis['correlation']:.3f} ({analysis['correlation_strength']})",
                    f"- **Pareto Optimal Experiments**: {analysis['num_pareto_optimal']}",
                    ""
                ])
                
                if analysis['pareto_optimal_experiments']:
                    report_lines.append("**Pareto Optimal Configurations**:")
                    for exp in analysis['pareto_optimal_experiments']:
                        report_lines.append(f"  - {exp}")
                    report_lines.append("")
        
        # Reproducibility analysis
        if 'reproducibility' in self.analysis_results:
            repro = self.analysis_results['reproducibility']
            
            report_lines.extend([
                "## Reproducibility Analysis",
                ""
            ])
            
            # Calculate average reproducibility scores
            all_scores = []
            for exp_name, exp_metrics in repro.items():
                for metric_name, metric_data in exp_metrics.items():
                    all_scores.append(metric_data['reproducibility_score'])
            
            if all_scores:
                avg_repro_score = np.mean(all_scores)
                report_lines.append(
                    f"- **Average Reproducibility Score**: {avg_repro_score:.3f}"
                )
                
                if avg_repro_score > 0.8:
                    report_lines.append("- **Assessment**: Excellent reproducibility")
                elif avg_repro_score > 0.6:
                    report_lines.append("- **Assessment**: Good reproducibility")
                elif avg_repro_score > 0.4:
                    report_lines.append("- **Assessment**: Moderate reproducibility")
                else:
                    report_lines.append("- **Assessment**: Poor reproducibility")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        # Write report
        report_file = output_dir / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Analysis report saved to {report_file}")
        
        return str(report_file)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Check reproducibility
        if 'reproducibility' in self.analysis_results:
            repro = self.analysis_results['reproducibility']
            
            low_repro_exps = []
            for exp_name, exp_metrics in repro.items():
                for metric_name, metric_data in exp_metrics.items():
                    if metric_data['reproducibility_score'] < 0.5:
                        low_repro_exps.append(exp_name)
                        break
            
            if low_repro_exps:
                recommendations.append(
                    f"Consider increasing the number of seeds for experiments: {', '.join(low_repro_exps)} "
                    "to improve reproducibility"
                )
        
        # Check for significant differences
        if 'metric_distributions' in self.analysis_results:
            for metric, analysis in self.analysis_results['metric_distributions'].items():
                if 'anova' in analysis and not analysis['anova']['significant_difference']:
                    recommendations.append(
                        f"No significant differences found for {metric}. "
                        "Consider refining experimental conditions or increasing sample size"
                    )
        
        # Performance trade-off recommendations
        if 'performance_tradeoffs' in self.analysis_results:
            for eff_metric, analysis in self.analysis_results['performance_tradeoffs'].items():
                if abs(analysis['correlation']) > 0.7:
                    recommendations.append(
                        f"Strong trade-off observed between accuracy and {eff_metric}. "
                        f"Consider Pareto optimal configurations: {', '.join(analysis['pareto_optimal_experiments'][:3])}"
                    )
        
        if not recommendations:
            recommendations.append("Experiments show consistent results. Consider exploring new configurations or metrics.")
        
        return recommendations
    
    def run_full_analysis(self, metrics: List[str], output_dir: Path) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        logger.info("Running full analysis pipeline")
        
        # Load data
        self.load_all_results()
        
        if not self.results_data:
            logger.error("No experiment data loaded")
            return {}
        
        # Run analyses
        self.analysis_results['metric_distributions'] = self.analyze_metric_distributions(metrics)
        self.analysis_results['correlations'] = self.analyze_correlations(metrics)
        
        # Performance trade-offs (if accuracy metric specified)
        accuracy_metrics = [m for m in metrics if m in ['auc', 'f1', 'accuracy']]
        efficiency_metrics = [m for m in metrics if any(keyword in m.lower() for keyword in ['latency', 'throughput', 'memory'])]
        
        if accuracy_metrics and efficiency_metrics:
            self.analysis_results['performance_tradeoffs'] = self.analyze_performance_tradeoffs(
                accuracy_metrics[0], efficiency_metrics
            )
        
        # Reproducibility analysis
        self.analysis_results['reproducibility'] = self.analyze_reproducibility()
        
        # Generate report
        report_path = self.generate_summary_report(output_dir)
        
        # Save analysis data
        analysis_file = output_dir / "analysis_data.json"
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        logger.info(f"Full analysis completed. Report saved to {report_path}")
        
        return self.analysis_results

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--metrics", nargs="+", 
                       default=["auc", "f1", "latency_p95", "throughput"],
                       help="Metrics to analyze")
    parser.add_argument("--output", type=str, default="analysis_output",
                       help="Output directory for analysis results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    analyzer = ResultsAnalyzer(args.results_dir)
    results = analyzer.run_full_analysis(args.metrics, output_dir)
    
    if not results:
        logger.error("Analysis failed")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Experiments analyzed: {len(analyzer.results_data)}")
    print(f"Metrics analyzed: {len(args.metrics)}")
    print(f"Output directory: {output_dir}")
    
    # Key findings
    if 'metric_distributions' in results:
        print(f"\nMETRIC ANALYSIS:")
        for metric, analysis in results['metric_distributions'].items():
            if 'anova' in analysis:
                significance = "✓" if analysis['anova']['significant_difference'] else "✗"
                print(f"  • {metric}: {analysis['experiments']} experiments, "
                      f"significant differences: {significance}")
    
    if 'correlations' in results and results['correlations'].get('significant_correlations'):
        print(f"\nSIGNIFICANT CORRELATIONS: {len(results['correlations']['significant_correlations'])}")
        for corr in results['correlations']['significant_correlations'][:3]:
            print(f"  • {corr['metric1']} vs {corr['metric2']}: {corr['strength']} (r={corr['pearson_r']:.3f})")
    
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
