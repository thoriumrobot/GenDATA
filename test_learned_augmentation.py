#!/usr/bin/env python3
"""
Test Learned Augmentation

This script tests the learned augmentation policies and compares their performance
against baseline methods. It validates the recursive augmentation optimization
system with real-world scenarios.
"""

import os
import json
import time
import numpy as np
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from recursive_augmentation_engine import RecursiveAugmentationEngine, TransformationState
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator, EvaluationMetrics
from adaptive_augmentation_pipeline import AdaptiveAugmentationPipeline
from train_augmentation_policy import AugmentationPolicyTrainer
from pipeline_config import AUGMENTATION_POLICY_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentationTester:
    """Tester for learned augmentation policies"""
    
    def __init__(self, config: Dict[str, Any], device: str = 'auto'):
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize components
        self.engine = RecursiveAugmentationEngine(seed=config.get('seed', 42))
        self.evaluator = AugmentationSequenceEvaluator(device=self.device)
        self.pipeline = AdaptiveAugmentationPipeline(config, device=self.device)
        
        # Test results
        self.test_results = {
            'baseline': {},
            'learned_policies': {},
            'comparison': {}
        }
        
        # Statistics
        self.stats = {
            'total_tests': 0,
            'test_start_time': None,
            'test_end_time': None,
            'best_policy': None,
            'improvement_over_baseline': 0.0
        }
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def load_test_cases(self, test_cases_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load test cases for evaluation"""
        if test_cases_path and os.path.exists(test_cases_path):
            try:
                with open(test_cases_path, 'r') as f:
                    test_cases = json.load(f)
                logger.info(f"Loaded {len(test_cases)} test cases from {test_cases_path}")
                return test_cases
            except Exception as e:
                logger.warning(f"Error loading test cases: {e}")
        
        # Generate default test cases
        logger.info("Generating default test cases")
        return self._generate_default_test_cases()
    
    def _generate_default_test_cases(self) -> List[Dict[str, Any]]:
        """Generate default test cases for evaluation"""
        test_cases = [
            {
                'name': 'Simple Calculator',
                'code': '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
}''',
                'complexity': 'low',
                'expected_transformations': ['variable_operation', 'method_extraction']
            },
            {
                'name': 'Array Processing',
                'code': '''
public class ArrayProcessor {
    public int findMax(int[] arr) {
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        return max;
    }
}''',
                'complexity': 'medium',
                'expected_transformations': ['loop_conversion', 'guard_reversal', 'variable_operation']
            },
            {
                'name': 'String Manipulation',
                'code': '''
public class StringUtils {
    public String reverse(String str) {
        StringBuilder sb = new StringBuilder();
        for (int i = str.length() - 1; i >= 0; i--) {
            sb.append(str.charAt(i));
        }
        return sb.toString();
    }
}''',
                'complexity': 'medium',
                'expected_transformations': ['loop_conversion', 'string_concatenation', 'variable_operation']
            },
            {
                'name': 'Recursive Function',
                'code': '''
public class MathUtils {
    public int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
}''',
                'complexity': 'medium',
                'expected_transformations': ['conditional_expression', 'method_extraction']
            },
            {
                'name': 'Complex Algorithm',
                'code': '''
public class SortingAlgorithm {
    public void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}''',
                'complexity': 'high',
                'expected_transformations': ['loop_conversion', 'guard_reversal', 'variable_operation', 'array_access_pattern']
            }
        ]
        
        return test_cases
    
    def test_baseline_augmentation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test baseline (random) augmentation"""
        logger.info("Testing baseline augmentation...")
        
        baseline_results = {}
        
        for test_case in test_cases:
            logger.info(f"Testing baseline on: {test_case['name']}")
            
            # Test random augmentation
            result = self.pipeline.generate_augmented_variants(
                test_case['code'], 
                num_variants=5, 
                policy_method='random'
            )
            
            # Compute metrics
            metrics = self._compute_test_metrics(result, test_case)
            baseline_results[test_case['name']] = metrics
            
            logger.info(f"Baseline {test_case['name']}: score={metrics['overall_score']:.3f}")
        
        # Compute overall baseline performance
        overall_scores = [metrics['overall_score'] for metrics in baseline_results.values()]
        baseline_results['overall'] = {
            'mean_score': np.mean(overall_scores),
            'std_score': np.std(overall_scores),
            'min_score': np.min(overall_scores),
            'max_score': np.max(overall_scores)
        }
        
        self.test_results['baseline'] = baseline_results
        logger.info(f"Baseline overall performance: {baseline_results['overall']['mean_score']:.3f}±{baseline_results['overall']['std_score']:.3f}")
        
        return baseline_results
    
    def test_learned_policies(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test learned augmentation policies"""
        logger.info("Testing learned policies...")
        
        learned_results = {}
        methods = ['rl', 'mcts', 'evolutionary', 'gnn']
        
        for method in methods:
            logger.info(f"Testing {method} policy...")
            method_results = {}
            
            for test_case in test_cases:
                logger.info(f"Testing {method} on: {test_case['name']}")
                
                try:
                    # Test learned policy
                    result = self.pipeline.generate_augmented_variants(
                        test_case['code'], 
                        num_variants=5, 
                        policy_method=method
                    )
                    
                    # Compute metrics
                    metrics = self._compute_test_metrics(result, test_case)
                    method_results[test_case['name']] = metrics
                    
                    logger.info(f"{method} {test_case['name']}: score={metrics['overall_score']:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Error testing {method} on {test_case['name']}: {e}")
                    method_results[test_case['name']] = {
                        'overall_score': 0.0,
                        'error': str(e)
                    }
            
            # Compute overall performance for method
            scores = [m['overall_score'] for m in method_results.values() if 'error' not in m]
            if scores:
                method_results['overall'] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores)
                }
            else:
                method_results['overall'] = {
                    'mean_score': 0.0,
                    'std_score': 0.0,
                    'min_score': 0.0,
                    'max_score': 0.0
                }
            
            learned_results[method] = method_results
            logger.info(f"{method} overall performance: {method_results['overall']['mean_score']:.3f}±{method_results['overall']['std_score']:.3f}")
        
        self.test_results['learned_policies'] = learned_results
        return learned_results
    
    def _compute_test_metrics(self, result, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive test metrics"""
        metrics = {
            'overall_score': 0.0,
            'slicer_resistance': 0.0,
            'model_performance': 0.0,
            'diversity_score': 0.0,
            'compilation_success': 0.0,
            'semantic_preservation': 0.0,
            'processing_time': result.processing_time,
            'num_variants': len(result.augmented_variants),
            'success_rate': result.metadata.get('success_rate', 0.0)
        }
        
        if result.evaluation_metrics:
            # Compute average metrics
            avg_metrics = {
                'slicer_resistance': np.mean([m.slicer_resistance for m in result.evaluation_metrics]),
                'model_performance': np.mean([m.model_performance for m in result.evaluation_metrics]),
                'diversity_score': np.mean([m.diversity_score for m in result.evaluation_metrics]),
                'compilation_success': np.mean([m.compilation_success for m in result.evaluation_metrics]),
                'semantic_preservation': np.mean([m.semantic_preservation for m in result.evaluation_metrics]),
                'overall_score': np.mean([m.overall_score for m in result.evaluation_metrics])
            }
            metrics.update(avg_metrics)
        
        # Add complexity-specific analysis
        metrics['complexity_match'] = self._analyze_complexity_match(result, test_case)
        metrics['transformation_coverage'] = self._analyze_transformation_coverage(result, test_case)
        
        return metrics
    
    def _analyze_complexity_match(self, result, test_case: Dict[str, Any]) -> float:
        """Analyze how well transformations match expected complexity"""
        expected_complexity = test_case.get('complexity', 'medium')
        
        if result.evaluation_metrics:
            avg_complexity = np.mean([m.metadata.get('final_complexity', 0) for m in result.evaluation_metrics])
            
            # Check if complexity is appropriate
            if expected_complexity == 'low' and avg_complexity < 3.0:
                return 1.0
            elif expected_complexity == 'medium' and 3.0 <= avg_complexity <= 7.0:
                return 1.0
            elif expected_complexity == 'high' and avg_complexity > 7.0:
                return 1.0
            else:
                return 0.5  # Partial match
        
        return 0.0
    
    def _analyze_transformation_coverage(self, result, test_case: Dict[str, Any]) -> float:
        """Analyze coverage of expected transformations"""
        expected_transformations = test_case.get('expected_transformations', [])
        
        if not expected_transformations or not result.transformation_sequences:
            return 0.0
        
        # Count how many expected transformations were used
        used_transformations = set()
        for sequence in result.transformation_sequences:
            for transformation in sequence:
                used_transformations.add(transformation.value)
        
        coverage = len(set(expected_transformations).intersection(used_transformations)) / len(expected_transformations)
        return coverage
    
    def compare_methods(self) -> Dict[str, Any]:
        """Compare all methods and identify the best"""
        logger.info("Comparing all methods...")
        
        comparison = {}
        
        # Baseline performance
        baseline_score = self.test_results['baseline']['overall']['mean_score']
        comparison['baseline'] = {
            'score': baseline_score,
            'improvement': 0.0
        }
        
        # Learned policy performance
        best_method = None
        best_score = baseline_score
        
        for method, results in self.test_results['learned_policies'].items():
            method_score = results['overall']['mean_score']
            improvement = ((method_score - baseline_score) / baseline_score) * 100 if baseline_score > 0 else 0
            
            comparison[method] = {
                'score': method_score,
                'improvement': improvement,
                'std_score': results['overall']['std_score']
            }
            
            if method_score > best_score:
                best_score = method_score
                best_method = method
        
        # Update stats
        self.stats['best_policy'] = best_method
        self.stats['improvement_over_baseline'] = comparison[best_method]['improvement'] if best_method else 0.0
        
        comparison['best_method'] = best_method
        comparison['best_score'] = best_score
        comparison['overall_improvement'] = ((best_score - baseline_score) / baseline_score) * 100 if baseline_score > 0 else 0
        
        self.test_results['comparison'] = comparison
        
        logger.info(f"Best method: {best_method} (score: {best_score:.3f}, improvement: {comparison['overall_improvement']:.1f}%)")
        
        return comparison
    
    def run_comprehensive_test(self, test_cases_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive test of all augmentation methods"""
        logger.info("Starting comprehensive augmentation test...")
        
        self.stats['test_start_time'] = time.time()
        self.stats['total_tests'] = 0
        
        # Load test cases
        test_cases = self.load_test_cases(test_cases_path)
        logger.info(f"Running tests on {len(test_cases)} test cases")
        
        # Test baseline
        baseline_results = self.test_baseline_augmentation(test_cases)
        
        # Test learned policies
        learned_results = self.test_learned_policies(test_cases)
        
        # Compare methods
        comparison = self.compare_methods()
        
        # Update stats
        self.stats['test_end_time'] = time.time()
        self.stats['total_tests'] = len(test_cases)
        
        # Save results
        self._save_test_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        logger.info("Comprehensive test completed!")
        logger.info(f"Best policy: {self.stats['best_policy']}")
        logger.info(f"Improvement over baseline: {self.stats['improvement_over_baseline']:.1f}%")
        
        return self.test_results
    
    def _save_test_results(self):
        """Save test results to file"""
        try:
            results_dir = 'results/augmentation_testing'
            os.makedirs(results_dir, exist_ok=True)
            
            # Save test results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(results_dir, f'test_results_{timestamp}.json')
            
            output_data = {
                'test_results': self.test_results,
                'stats': self.stats,
                'config': self.config,
                'timestamp': timestamp
            }
            
            with open(results_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
    
    def _generate_visualizations(self):
        """Generate visualization plots"""
        try:
            plots_dir = 'results/augmentation_testing/plots'
            os.makedirs(plots_dir, exist_ok=True)
            
            # Performance comparison plot
            self._plot_performance_comparison(plots_dir)
            
            # Metric breakdown plot
            self._plot_metric_breakdown(plots_dir)
            
            # Test case analysis plot
            self._plot_test_case_analysis(plots_dir)
            
            logger.info(f"Visualizations saved to {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
    
    def _plot_performance_comparison(self, plots_dir: str):
        """Plot performance comparison across methods"""
        try:
            plt.figure(figsize=(12, 8))
            
            methods = ['baseline'] + list(self.test_results['learned_policies'].keys())
            scores = []
            stds = []
            
            # Baseline
            baseline_score = self.test_results['baseline']['overall']['mean_score']
            baseline_std = self.test_results['baseline']['overall']['std_score']
            scores.append(baseline_score)
            stds.append(baseline_std)
            
            # Learned policies
            for method in self.test_results['learned_policies'].keys():
                method_data = self.test_results['learned_policies'][method]['overall']
                scores.append(method_data['mean_score'])
                stds.append(method_data['std_score'])
            
            # Create bar plot
            x_pos = np.arange(len(methods))
            bars = plt.bar(x_pos, scores, yerr=stds, capsize=5, alpha=0.7)
            
            # Color bars
            colors = ['red' if method == 'baseline' else 'blue' for method in methods]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xlabel('Method')
            plt.ylabel('Overall Score')
            plt.title('Augmentation Policy Performance Comparison')
            plt.xticks(x_pos, methods, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add improvement percentages
            for i, (method, score) in enumerate(zip(methods, scores)):
                if method != 'baseline':
                    improvement = ((score - baseline_score) / baseline_score) * 100
                    plt.text(i, score + stds[i] + 0.01, f'+{improvement:.1f}%', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating performance comparison plot: {e}")
    
    def _plot_metric_breakdown(self, plots_dir: str):
        """Plot metric breakdown for best method"""
        try:
            if not self.stats['best_policy']:
                return
            
            best_method = self.stats['best_policy']
            method_results = self.test_results['learned_policies'][best_method]
            
            # Extract metrics for all test cases
            metrics = ['slicer_resistance', 'model_performance', 'diversity_score', 
                      'compilation_success', 'semantic_preservation']
            
            metric_values = {metric: [] for metric in metrics}
            
            for test_case_name, test_result in method_results.items():
                if test_case_name != 'overall' and 'error' not in test_result:
                    for metric in metrics:
                        if metric in test_result:
                            metric_values[metric].append(test_result[metric])
            
            # Create box plot
            plt.figure(figsize=(12, 8))
            
            data_to_plot = [metric_values[metric] for metric in metrics]
            box_plot = plt.boxplot(data_to_plot, labels=metrics, patch_artist=True)
            
            # Color boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title(f'Metric Breakdown for {best_method.upper()} Policy')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'metric_breakdown_{best_method}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating metric breakdown plot: {e}")
    
    def _plot_test_case_analysis(self, plots_dir: str):
        """Plot test case analysis"""
        try:
            plt.figure(figsize=(14, 10))
            
            # Get test case names
            test_cases = list(self.test_results['baseline'].keys())
            if 'overall' in test_cases:
                test_cases.remove('overall')
            
            methods = ['baseline'] + list(self.test_results['learned_policies'].keys())
            
            # Create heatmap data
            heatmap_data = []
            for method in methods:
                method_scores = []
                for test_case in test_cases:
                    if method == 'baseline':
                        score = self.test_results['baseline'][test_case]['overall_score']
                    else:
                        if test_case in self.test_results['learned_policies'][method]:
                            score = self.test_results['learned_policies'][method][test_case]['overall_score']
                        else:
                            score = 0.0
                    method_scores.append(score)
                heatmap_data.append(method_scores)
            
            # Create heatmap
            sns.heatmap(heatmap_data, 
                       xticklabels=test_cases, 
                       yticklabels=methods,
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Overall Score'})
            
            plt.xlabel('Test Cases')
            plt.ylabel('Methods')
            plt.title('Performance Heatmap: Methods vs Test Cases')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'test_case_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating test case analysis plot: {e}")
    
    def get_summary_report(self) -> str:
        """Generate summary report"""
        report = []
        report.append("=" * 60)
        report.append("AUGMENTATION POLICY TESTING SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Test overview
        report.append(f"Test Duration: {self.stats['test_end_time'] - self.stats['test_start_time']:.2f} seconds")
        report.append(f"Total Test Cases: {self.stats['total_tests']}")
        report.append(f"Best Policy: {self.stats['best_policy']}")
        report.append(f"Improvement over Baseline: {self.stats['improvement_over_baseline']:.1f}%")
        report.append("")
        
        # Method comparison
        report.append("METHOD PERFORMANCE COMPARISON:")
        report.append("-" * 40)
        
        for method, data in self.test_results['comparison'].items():
            if method not in ['best_method', 'best_score', 'overall_improvement']:
                report.append(f"{method.upper():15}: {data['score']:.3f} ({data['improvement']:+.1f}%)")
        
        report.append("")
        
        # Best method details
        if self.stats['best_policy']:
            best_method = self.stats['best_policy']
            best_data = self.test_results['learned_policies'][best_method]['overall']
            report.append(f"BEST METHOD DETAILS ({best_method.upper()}):")
            report.append("-" * 40)
            report.append(f"Mean Score: {best_data['mean_score']:.3f}")
            report.append(f"Std Deviation: {best_data['std_score']:.3f}")
            report.append(f"Min Score: {best_data['min_score']:.3f}")
            report.append(f"Max Score: {best_data['max_score']:.3f}")
            report.append("")
        
        # Test case breakdown
        report.append("TEST CASE BREAKDOWN:")
        report.append("-" * 40)
        
        test_cases = list(self.test_results['baseline'].keys())
        if 'overall' in test_cases:
            test_cases.remove('overall')
        
        for test_case in test_cases:
            report.append(f"\n{test_case}:")
            
            # Baseline
            baseline_score = self.test_results['baseline'][test_case]['overall_score']
            report.append(f"  Baseline: {baseline_score:.3f}")
            
            # Best method
            if self.stats['best_policy']:
                best_score = self.test_results['learned_policies'][self.stats['best_policy']][test_case]['overall_score']
                improvement = ((best_score - baseline_score) / baseline_score) * 100
                report.append(f"  {self.stats['best_policy'].upper()}: {best_score:.3f} ({improvement:+.1f}%)")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test learned augmentation policies')
    parser.add_argument('--test-cases', type=str, default='', 
                       help='Path to test cases file')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use for testing')
    parser.add_argument('--config', type=str, default='', 
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results/augmentation_testing', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = AUGMENTATION_POLICY_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)
    
    # Create tester
    tester = AugmentationTester(config, device=args.device)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(args.test_cases)
    
    # Generate and print summary report
    report = tester.get_summary_report()
    print("\n" + report)
    
    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Summary report saved to {report_path}")


if __name__ == '__main__':
    main()
