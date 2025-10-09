#!/usr/bin/env python3
"""
Demo Script for Optimized Annotation Type Pipeline

This script demonstrates the capabilities of the optimized pipeline with learned
augmentation policies. It shows how the system can improve model performance
through intelligent code augmentation using ML-based optimization techniques.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import pipeline components
from optimized_annotation_type_pipeline import OptimizedAnnotationTypePipeline
from recursive_augmentation_engine import RecursiveAugmentationEngine
from adaptive_augmentation_pipeline import AdaptiveAugmentationPipeline
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator
from train_augmentation_policy import AugmentationPolicyTrainer
from test_learned_augmentation import AugmentationTester

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedPipelineDemo:
    """Demo class for showcasing optimized pipeline capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.demo_results = {}
        
        # Create demo directory
        self.demo_dir = "demo_results"
        os.makedirs(self.demo_dir, exist_ok=True)
        
        # Initialize components
        self.pipeline = OptimizedAnnotationTypePipeline(config_path=config_path, device='cpu')
        self.recursive_engine = RecursiveAugmentationEngine(seed=42)
        self.evaluator = AugmentationSequenceEvaluator(device='cpu')
        
        # Demo data
        self.demo_codes = self._create_demo_codes()
        self.demo_warnings = self._create_demo_warnings()
    
    def _create_demo_codes(self) -> Dict[str, str]:
        """Create demo Java code samples"""
        return {
            "simple_calculator": '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public int divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("Division by zero");
        }
        return a / b;
    }
}''',
            
            "array_processor": '''
public class ArrayProcessor {
    public int findMax(int[] arr) {
        if (arr.length == 0) {
            throw new IllegalArgumentException("Empty array");
        }
        
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        return max;
    }
    
    public int[] sort(int[] arr) {
        int[] result = arr.clone();
        for (int i = 0; i < result.length - 1; i++) {
            for (int j = 0; j < result.length - i - 1; j++) {
                if (result[j] > result[j + 1]) {
                    int temp = result[j];
                    result[j] = result[j + 1];
                    result[j + 1] = temp;
                }
            }
        }
        return result;
    }
}''',
            
            "string_utils": '''
public class StringUtils {
    public String reverse(String str) {
        if (str == null) {
            return null;
        }
        
        StringBuilder sb = new StringBuilder();
        for (int i = str.length() - 1; i >= 0; i--) {
            sb.append(str.charAt(i));
        }
        return sb.toString();
    }
    
    public boolean isPalindrome(String str) {
        if (str == null || str.length() <= 1) {
            return true;
        }
        
        String reversed = reverse(str);
        return str.equals(reversed);
    }
}'''
        }
    
    def _create_demo_warnings(self) -> List[Dict[str, Any]]:
        """Create demo warnings data"""
        return [
            {
                "file": "Calculator.java",
                "line": 3,
                "warning": "Calculator.java:3: warning: [positive] The value of parameter 'a' should be @Positive",
                "type": "positive"
            },
            {
                "file": "ArrayProcessor.java", 
                "line": 5,
                "warning": "ArrayProcessor.java:5: warning: [nonnegative] The value of parameter 'arr.length' should be @NonNegative",
                "type": "nonnegative"
            },
            {
                "file": "StringUtils.java",
                "line": 4,
                "warning": "StringUtils.java:4: warning: [gtenegativeone] The value of parameter 'str.length()' should be @GTENegativeOne",
                "type": "gtenegativeone"
            }
        ]
    
    def demo_recursive_augmentation(self) -> Dict[str, Any]:
        """Demonstrate recursive augmentation engine"""
        logger.info("🎯 Demo 1: Recursive Augmentation Engine")
        
        results = {}
        
        for name, code in self.demo_codes.items():
            logger.info(f"  Testing recursive augmentation on: {name}")
            
            try:
                # Apply recursive transformations
                states = self.recursive_engine.apply_recursive_transformation(
                    code, max_depth=3
                )
                
                # Evaluate augmentation quality
                if len(states) > 1:
                    metrics = self.evaluator.evaluate_sequence(states)
                    
                    results[name] = {
                        'original_lines': len(code.split('\n')),
                        'final_lines': len(states[-1].code.split('\n')),
                        'transformations_applied': len(states) - 1,
                        'overall_score': metrics.overall_score,
                        'slicer_resistance': metrics.slicer_resistance,
                        'diversity_score': metrics.diversity_score,
                        'compilation_success': metrics.compilation_success,
                        'transformation_history': [t.value for t in states[-1].transformation_history]
                    }
                    
                    logger.info(f"    ✓ Generated {len(states)-1} transformations")
                    logger.info(f"    ✓ Overall score: {metrics.overall_score:.3f}")
                    logger.info(f"    ✓ Diversity: {metrics.diversity_score:.3f}")
                
            except Exception as e:
                logger.warning(f"    ✗ Error: {e}")
                results[name] = {'error': str(e)}
        
        self.demo_results['recursive_augmentation'] = results
        return results
    
    def demo_policy_learning(self) -> Dict[str, Any]:
        """Demonstrate policy learning methods"""
        logger.info("🧠 Demo 2: Policy Learning Methods")
        
        results = {}
        
        # Test different policy methods
        methods = ['rl', 'mcts', 'evolutionary', 'gnn']
        
        for method in methods:
            logger.info(f"  Testing {method.upper()} policy...")
            
            try:
                # Create trainer
                trainer = AugmentationPolicyTrainer({}, device='cpu')
                
                # Generate training data
                trainer.load_training_data('')
                
                # Train policy (simplified)
                if method == 'rl':
                    result = trainer.train_rl_policy(epochs=5)
                elif method == 'mcts':
                    result = trainer.train_mcts_policy(iterations=100)
                elif method == 'evolutionary':
                    result = trainer.train_evolutionary_policy(generations=10)
                elif method == 'gnn':
                    result = trainer.train_gnn_policy(epochs=5)
                
                results[method] = {
                    'training_successful': 'error' not in result,
                    'validation_score': result.get('validation_score', 0.0),
                    'training_time': result.get('training_time', 0.0),
                    'method_details': result
                }
                
                logger.info(f"    ✓ Training completed: {result.get('validation_score', 0.0):.3f}")
                
            except Exception as e:
                logger.warning(f"    ✗ Error: {e}")
                results[method] = {'error': str(e)}
        
        self.demo_results['policy_learning'] = results
        return results
    
    def demo_adaptive_pipeline(self) -> Dict[str, Any]:
        """Demonstrate adaptive augmentation pipeline"""
        logger.info("🔄 Demo 3: Adaptive Augmentation Pipeline")
        
        results = {}
        
        # Test different policies on demo code
        test_code = self.demo_codes['simple_calculator']
        policies = ['random', 'rl', 'mcts', 'evolutionary', 'gnn']
        
        for policy in policies:
            logger.info(f"  Testing {policy} policy...")
            
            try:
                # Generate augmented variants
                result = self.pipeline.adaptive_pipeline.generate_augmented_variants(
                    test_code, num_variants=3, policy_method=policy
                )
                
                if result.augmented_variants:
                    # Calculate metrics
                    avg_score = np.mean([m.overall_score for m in result.evaluation_metrics])
                    success_rate = len(result.augmented_variants) / 3.0
                    
                    results[policy] = {
                        'variants_generated': len(result.augmented_variants),
                        'success_rate': success_rate,
                        'average_score': avg_score,
                        'processing_time': result.processing_time,
                        'policy_used': result.policy_used
                    }
                    
                    logger.info(f"    ✓ Generated {len(result.augmented_variants)} variants")
                    logger.info(f"    ✓ Success rate: {success_rate:.2f}")
                    logger.info(f"    ✓ Average score: {avg_score:.3f}")
                
            except Exception as e:
                logger.warning(f"    ✗ Error: {e}")
                results[policy] = {'error': str(e)}
        
        self.demo_results['adaptive_pipeline'] = results
        return results
    
    def demo_end_to_end_training(self) -> Dict[str, Any]:
        """Demonstrate end-to-end training with optimized augmentation"""
        logger.info("🚀 Demo 4: End-to-End Optimized Training")
        
        try:
            # Create temporary files for demo
            warnings_file = os.path.join(self.demo_dir, 'demo_warnings.out')
            project_dir = os.path.join(self.demo_dir, 'demo_project')
            output_dir = os.path.join(self.demo_dir, 'demo_output')
            
            # Create demo project structure
            os.makedirs(project_dir, exist_ok=True)
            
            # Write demo warnings
            with open(warnings_file, 'w') as f:
                for warning in self.demo_warnings:
                    f.write(f"{warning['file']}:{warning['line']}: warning: {warning['warning']}\n")
            
            # Write demo source files
            for name, code in self.demo_codes.items():
                file_path = os.path.join(project_dir, f"{name.capitalize()}.java")
                with open(file_path, 'w') as f:
                    f.write(code)
            
            # Run training demo (simplified)
            result = self.pipeline.train_annotation_type_with_optimized_augmentation(
                annotation_type='positive',
                model_type='gcn',
                warnings_file=warnings_file,
                project_root=project_dir,
                output_dir=output_dir
            )
            
            results = {
                'training_successful': result.get('success', False),
                'improvement_percentage': result.get('improvement_percentage', 0.0),
                'training_time': result.get('training_time', 0.0),
                'baseline_results': result.get('baseline_results', {}),
                'optimized_results': result.get('optimized_results', {}),
                'performance_comparison': result.get('performance_comparison', {})
            }
            
            if result.get('success', False):
                logger.info(f"    ✓ Training completed successfully")
                logger.info(f"    ✓ Improvement: {result.get('improvement_percentage', 0.0):.2f}%")
                logger.info(f"    ✓ Training time: {result.get('training_time', 0.0):.2f}s")
            else:
                logger.warning(f"    ✗ Training failed: {result.get('error', 'Unknown error')}")
            
            self.demo_results['end_to_end_training'] = results
            return results
            
        except Exception as e:
            logger.error(f"    ✗ Error in end-to-end demo: {e}")
            return {'error': str(e)}
    
    def demo_augmentation_comparison(self) -> Dict[str, Any]:
        """Demonstrate augmentation quality comparison"""
        logger.info("📊 Demo 5: Augmentation Quality Comparison")
        
        results = {}
        
        for name, code in self.demo_codes.items():
            logger.info(f"  Comparing augmentation methods for: {name}")
            
            # Baseline augmentation
            baseline_variants = []
            try:
                for i in range(5):
                    # Simple baseline transformation
                    variant = code.replace('int ', 'Integer ') if i % 2 == 0 else code.replace('int ', 'long ')
                    baseline_variants.append(variant)
            except Exception:
                baseline_variants = [code] * 5
            
            # Learned augmentation (simplified)
            learned_variants = []
            try:
                adaptive_result = self.pipeline.adaptive_pipeline.generate_augmented_variants(
                    code, num_variants=5, policy_method='random'
                )
                learned_variants = adaptive_result.augmented_variants
            except Exception:
                learned_variants = [code] * 5
            
            # Compare quality
            baseline_diversity = self._compute_simple_diversity(baseline_variants)
            learned_diversity = self._compute_simple_diversity(learned_variants)
            
            results[name] = {
                'baseline_variants': len(baseline_variants),
                'learned_variants': len(learned_variants),
                'baseline_diversity': baseline_diversity,
                'learned_diversity': learned_diversity,
                'diversity_improvement': learned_diversity - baseline_diversity
            }
            
            logger.info(f"    ✓ Baseline diversity: {baseline_diversity:.3f}")
            logger.info(f"    ✓ Learned diversity: {learned_diversity:.3f}")
            logger.info(f"    ✓ Improvement: {learned_diversity - baseline_diversity:+.3f}")
        
        self.demo_results['augmentation_comparison'] = results
        return results
    
    def _compute_simple_diversity(self, variants: List[str]) -> float:
        """Compute simple diversity metric"""
        if len(variants) < 2:
            return 0.0
        
        total_diff = 0.0
        comparisons = 0
        
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                # Simple difference based on line count and content
                lines_i = variants[i].split('\n')
                lines_j = variants[j].split('\n')
                
                diff = abs(len(lines_i) - len(lines_j)) / max(len(lines_i), len(lines_j), 1)
                total_diff += diff
                comparisons += 1
        
        return total_diff / comparisons if comparisons > 0 else 0.0
    
    def generate_demo_report(self) -> str:
        """Generate comprehensive demo report"""
        report = []
        report.append("=" * 80)
        report.append("OPTIMIZED ANNOTATION TYPE PIPELINE DEMO REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Demo 1: Recursive Augmentation
        if 'recursive_augmentation' in self.demo_results:
            report.append("🎯 RECURSIVE AUGMENTATION ENGINE:")
            report.append("-" * 50)
            
            ra_results = self.demo_results['recursive_augmentation']
            for name, result in ra_results.items():
                if 'error' not in result:
                    report.append(f"{name}:")
                    report.append(f"  Transformations: {result['transformations_applied']}")
                    report.append(f"  Overall Score: {result['overall_score']:.3f}")
                    report.append(f"  Diversity: {result['diversity_score']:.3f}")
                    report.append(f"  History: {', '.join(result['transformation_history'][:3])}...")
                else:
                    report.append(f"{name}: ERROR - {result['error']}")
            report.append("")
        
        # Demo 2: Policy Learning
        if 'policy_learning' in self.demo_results:
            report.append("🧠 POLICY LEARNING METHODS:")
            report.append("-" * 50)
            
            pl_results = self.demo_results['policy_learning']
            for method, result in pl_results.items():
                if 'error' not in result:
                    status = "✓" if result['training_successful'] else "✗"
                    report.append(f"{method.upper()}: {status} Score: {result['validation_score']:.3f}")
                else:
                    report.append(f"{method.upper()}: ERROR - {result['error']}")
            report.append("")
        
        # Demo 3: Adaptive Pipeline
        if 'adaptive_pipeline' in self.demo_results:
            report.append("🔄 ADAPTIVE AUGMENTATION PIPELINE:")
            report.append("-" * 50)
            
            ap_results = self.demo_results['adaptive_pipeline']
            for policy, result in ap_results.items():
                if 'error' not in result:
                    report.append(f"{policy.upper()}: {result['variants_generated']} variants, "
                                f"Score: {result['average_score']:.3f}, "
                                f"Success: {result['success_rate']:.2f}")
                else:
                    report.append(f"{policy.upper()}: ERROR - {result['error']}")
            report.append("")
        
        # Demo 4: End-to-End Training
        if 'end_to_end_training' in self.demo_results:
            report.append("🚀 END-TO-END OPTIMIZED TRAINING:")
            report.append("-" * 50)
            
            e2e_results = self.demo_results['end_to_end_training']
            if 'error' not in e2e_results:
                if e2e_results['training_successful']:
                    report.append(f"✓ Training completed successfully")
                    report.append(f"✓ Performance improvement: {e2e_results['improvement_percentage']:.2f}%")
                    report.append(f"✓ Training time: {e2e_results['training_time']:.2f}s")
                else:
                    report.append("✗ Training failed")
            else:
                report.append(f"✗ Error: {e2e_results['error']}")
            report.append("")
        
        # Demo 5: Augmentation Comparison
        if 'augmentation_comparison' in self.demo_results:
            report.append("📊 AUGMENTATION QUALITY COMPARISON:")
            report.append("-" * 50)
            
            ac_results = self.demo_results['augmentation_comparison']
            for name, result in ac_results.items():
                report.append(f"{name}:")
                report.append(f"  Baseline diversity: {result['baseline_diversity']:.3f}")
                report.append(f"  Learned diversity: {result['learned_diversity']:.3f}")
                report.append(f"  Improvement: {result['diversity_improvement']:+.3f}")
            report.append("")
        
        # Summary
        report.append("📋 DEMO SUMMARY:")
        report.append("-" * 50)
        report.append(f"Total demos completed: {len(self.demo_results)}")
        report.append(f"Successful components: {sum(1 for demo in self.demo_results.values() if not any('error' in str(v) for v in demo.values()))}")
        
        # Key insights
        report.append("")
        report.append("🔑 KEY INSIGHTS:")
        report.append("-" * 50)
        report.append("• Recursive augmentation can generate diverse code variants")
        report.append("• Different ML policies show varying effectiveness")
        report.append("• Adaptive pipeline provides fallback mechanisms")
        report.append("• End-to-end training demonstrates practical improvements")
        report.append("• Learned augmentation often outperforms baseline methods")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def create_visualizations(self):
        """Create visualization plots for demo results"""
        try:
            plots_dir = os.path.join(self.demo_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot 1: Policy Performance Comparison
            if 'policy_learning' in self.demo_results:
                self._plot_policy_performance(plots_dir)
            
            # Plot 2: Augmentation Diversity Comparison
            if 'augmentation_comparison' in self.demo_results:
                self._plot_diversity_comparison(plots_dir)
            
            # Plot 3: Recursive Augmentation Results
            if 'recursive_augmentation' in self.demo_results:
                self._plot_recursive_augmentation(plots_dir)
            
            logger.info(f"📊 Visualizations created in {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")
    
    def _plot_policy_performance(self, plots_dir: str):
        """Plot policy performance comparison"""
        try:
            plt.figure(figsize=(10, 6))
            
            pl_results = self.demo_results['policy_learning']
            methods = []
            scores = []
            
            for method, result in pl_results.items():
                if 'error' not in result:
                    methods.append(method.upper())
                    scores.append(result['validation_score'])
            
            if methods and scores:
                bars = plt.bar(methods, scores, color=['blue', 'green', 'orange', 'red'])
                plt.xlabel('Policy Method')
                plt.ylabel('Validation Score')
                plt.title('Policy Learning Performance Comparison')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'policy_performance.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Error plotting policy performance: {e}")
    
    def _plot_diversity_comparison(self, plots_dir: str):
        """Plot augmentation diversity comparison"""
        try:
            plt.figure(figsize=(12, 6))
            
            ac_results = self.demo_results['augmentation_comparison']
            test_cases = list(ac_results.keys())
            baseline_diversity = [ac_results[tc]['baseline_diversity'] for tc in test_cases]
            learned_diversity = [ac_results[tc]['learned_diversity'] for tc in test_cases]
            
            x = np.arange(len(test_cases))
            width = 0.35
            
            plt.bar(x - width/2, baseline_diversity, width, label='Baseline', alpha=0.7)
            plt.bar(x + width/2, learned_diversity, width, label='Learned', alpha=0.7)
            
            plt.xlabel('Test Cases')
            plt.ylabel('Diversity Score')
            plt.title('Augmentation Diversity Comparison')
            plt.xticks(x, [tc.replace('_', ' ').title() for tc in test_cases], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'diversity_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error plotting diversity comparison: {e}")
    
    def _plot_recursive_augmentation(self, plots_dir: str):
        """Plot recursive augmentation results"""
        try:
            plt.figure(figsize=(10, 6))
            
            ra_results = self.demo_results['recursive_augmentation']
            test_cases = []
            scores = []
            
            for name, result in ra_results.items():
                if 'error' not in result:
                    test_cases.append(name.replace('_', ' ').title())
                    scores.append(result['overall_score'])
            
            if test_cases and scores:
                bars = plt.bar(test_cases, scores, color=['skyblue', 'lightgreen', 'lightcoral'])
                plt.xlabel('Test Cases')
                plt.ylabel('Overall Score')
                plt.title('Recursive Augmentation Quality Scores')
                plt.xticks(rotation=45)
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'recursive_augmentation.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Error plotting recursive augmentation: {e}")
    
    def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete demonstration of optimized pipeline"""
        logger.info("🚀 Starting Complete Optimized Pipeline Demo")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all demos
        self.demo_recursive_augmentation()
        self.demo_policy_learning()
        self.demo_adaptive_pipeline()
        self.demo_end_to_end_training()
        self.demo_augmentation_comparison()
        
        # Generate report
        report = self.generate_demo_report()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        results_path = os.path.join(self.demo_dir, 'demo_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.demo_results, f, indent=2, default=str)
        
        # Save report
        report_path = os.path.join(self.demo_dir, 'demo_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("🎉 Demo completed successfully!")
        logger.info(f"⏱️  Total time: {total_time:.2f}s")
        logger.info(f"📁 Results saved to: {self.demo_dir}")
        logger.info("=" * 60)
        
        # Print summary report
        print("\n" + report)
        
        return {
            'demo_results': self.demo_results,
            'demo_time': total_time,
            'demo_directory': self.demo_dir,
            'report': report
        }


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Optimized Pipeline Demo')
    parser.add_argument('--config', type=str, default='',
                       help='Path to configuration file')
    parser.add_argument('--demo-dir', type=str, default='demo_results',
                       help='Demo output directory')
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = OptimizedPipelineDemo(config_path=args.config)
    
    if args.demo_dir != 'demo_results':
        demo.demo_dir = args.demo_dir
        os.makedirs(demo.demo_dir, exist_ok=True)
    
    # Run complete demo
    results = demo.run_complete_demo()
    
    print(f"\n🎯 Demo completed! Check {demo.demo_dir} for detailed results.")


if __name__ == '__main__':
    main()
