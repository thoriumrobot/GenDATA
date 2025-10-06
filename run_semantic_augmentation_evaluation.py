#!/usr/bin/env python3
"""
Semantic Augmentation Evaluation Runner

This script orchestrates the complete evaluation of semantic augmentation systems:
1. Analyzes which augmentations apply to Checker Framework test cases
2. Runs ablation studies to test individual augmentation contributions
3. Executes comprehensive test cases to verify augmentation behavior
4. Generates comprehensive evaluation reports
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import time

# Import evaluation components
from semantic_augmentation_evaluator import SemanticAugmentationEvaluator
from ablation_study_pipeline import AblationStudyPipeline
from semantic_augmentation_test_suite import SemanticAugmentationTestRunner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticAugmentationEvaluationRunner:
    """Main runner for comprehensive semantic augmentation evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config['output_dir']
        self.checker_framework_dir = config['checker_framework_dir']
        self.cfwr_root = config['cfwr_root']
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize evaluation components
        self.evaluator = SemanticAugmentationEvaluator(
            self.checker_framework_dir, 
            self.output_dir
        )
        
        self.ablation_pipeline = AblationStudyPipeline(
            self.cfwr_root,
            os.path.join(self.output_dir, 'ablation_studies')
        )
        
        self.test_runner = SemanticAugmentationTestRunner(
            os.path.join(self.output_dir, 'test_results')
        )
        
        # Results storage
        self.evaluation_results = {}
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete semantic augmentation evaluation."""
        logger.info("Starting complete semantic augmentation evaluation...")
        
        start_time = time.time()
        
        # Step 1: Analyze Checker Framework coverage
        logger.info("Step 1: Analyzing Checker Framework test case coverage...")
        coverage_results = self.evaluator.analyze_checker_framework_coverage()
        self.evaluation_results['checker_framework_coverage'] = coverage_results
        
        # Step 2: Run ablation studies
        logger.info("Step 2: Running ablation studies...")
        ablation_results = self.ablation_pipeline.run_complete_ablation_study()
        self.evaluation_results['ablation_studies'] = [
            {
                'augmentation_name': r.augmentation_name,
                'baseline_f1': r.baseline_f1,
                'ablated_f1': r.ablated_f1,
                'f1_difference': r.f1_difference,
                'confidence_interval': r.confidence_interval,
                'statistical_significance': r.statistical_significance,
                'training_time': r.training_time
            }
            for r in ablation_results
        ]
        
        # Step 3: Run test cases
        logger.info("Step 3: Running semantic augmentation test cases...")
        test_results = self.test_runner.run_all_tests()
        self.evaluation_results['test_cases'] = test_results
        
        # Step 4: Generate comprehensive report
        logger.info("Step 4: Generating comprehensive evaluation report...")
        final_report = self._generate_comprehensive_report()
        
        # Save complete results
        self._save_complete_results()
        
        total_time = time.time() - start_time
        logger.info(f"Complete evaluation finished in {total_time:.2f} seconds")
        
        return {
            'evaluation_results': self.evaluation_results,
            'final_report': final_report,
            'total_evaluation_time': total_time
        }
    
    def run_checker_framework_analysis_only(self) -> Dict[str, Any]:
        """Run only Checker Framework coverage analysis."""
        logger.info("Running Checker Framework coverage analysis...")
        
        coverage_results = self.evaluator.analyze_checker_framework_coverage()
        
        # Generate focused report
        report = self._generate_coverage_report(coverage_results)
        
        return {
            'coverage_results': coverage_results,
            'report': report
        }
    
    def run_ablation_studies_only(self) -> Dict[str, Any]:
        """Run only ablation studies."""
        logger.info("Running ablation studies...")
        
        ablation_results = self.ablation_pipeline.run_complete_ablation_study()
        
        # Generate focused report
        report = self._generate_ablation_report(ablation_results)
        
        return {
            'ablation_results': ablation_results,
            'report': report
        }
    
    def run_test_cases_only(self) -> Dict[str, Any]:
        """Run only test cases."""
        logger.info("Running semantic augmentation test cases...")
        
        test_results = self.test_runner.run_all_tests()
        
        # Generate focused report
        report = self._generate_test_report(test_results)
        
        return {
            'test_results': test_results,
            'report': report
        }
    
    def _generate_comprehensive_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("# Comprehensive Semantic Augmentation Evaluation Report")
        report.append("=" * 60)
        report.append("")
        report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        # Coverage summary
        if 'checker_framework_coverage' in self.evaluation_results:
            coverage = self.evaluation_results['checker_framework_coverage']
            report.append("### Checker Framework Test Case Coverage")
            report.append(f"- **Total Files Analyzed**: {coverage['total_files']}")
            report.append(f"- **Enhanced System Usage**: {coverage['enhanced_files']} files")
            report.append(f"- **Simple System Usage**: {coverage['simple_files']} files")
            report.append("")
        
        # Ablation summary
        if 'ablation_studies' in self.evaluation_results:
            ablation = self.evaluation_results['ablation_studies']
            significant_degradations = [r for r in ablation if r['statistical_significance'] and r['f1_difference'] > 0.01]
            significant_improvements = [r for r in ablation if r['statistical_significance'] and r['f1_difference'] < -0.01]
            
            report.append("### Ablation Study Results")
            report.append(f"- **Total Augmentation Methods Tested**: {len(ablation)}")
            report.append(f"- **Significant Performance Degradations**: {len(significant_degradations)}")
            report.append(f"- **Significant Performance Improvements**: {len(significant_improvements)}")
            report.append("")
        
        # Test results summary
        if 'test_cases' in self.evaluation_results:
            test_results = self.evaluation_results['test_cases']
            report.append("### Test Case Results")
            report.append(f"- **Total Tests**: {test_results['total_tests']}")
            report.append(f"- **Passed Tests**: {test_results['total_tests'] - test_results['failures'] - test_results['errors']}")
            report.append(f"- **Failed Tests**: {test_results['failures']}")
            report.append(f"- **Error Tests**: {test_results['errors']}")
            report.append(f"- **Success Rate**: {test_results['success_rate']:.2%}")
            report.append("")
        
        # Detailed Analysis
        report.append("## Detailed Analysis")
        report.append("")
        
        # Most impactful augmentations
        if 'ablation_studies' in self.evaluation_results:
            report.append("### Most Impactful Augmentations")
            report.append("")
            
            ablation_results = self.evaluation_results['ablation_studies']
            sorted_results = sorted(ablation_results, key=lambda x: x['f1_difference'], reverse=True)
            
            report.append("#### Top 10 Performance Impacts")
            report.append("")
            report.append("| Rank | Augmentation | F1 Difference | Significance |")
            report.append("|------|--------------|---------------|--------------|")
            
            for i, result in enumerate(sorted_results[:10], 1):
                significance = "✓" if result['statistical_significance'] else "✗"
                report.append(f"| {i} | {result['augmentation_name']} | {result['f1_difference']:.4f} | {significance} |")
            
            report.append("")
        
        # Transformation coverage analysis
        if 'checker_framework_coverage' in self.evaluation_results:
            report.append("### Transformation Coverage Analysis")
            report.append("")
            
            coverage = self.evaluation_results['checker_framework_coverage']
            transformation_coverage = coverage.get('transformation_coverage', {})
            
            # Sort by coverage percentage
            sorted_coverage = sorted(transformation_coverage.items(), 
                                   key=lambda x: x[1]['percentage'], reverse=True)
            
            report.append("#### Top 10 Most Applied Transformations")
            report.append("")
            report.append("| Rank | Transformation | Files | Coverage % |")
            report.append("|------|----------------|-------|------------|")
            
            for i, (transformation, data) in enumerate(sorted_coverage[:10], 1):
                report.append(f"| {i} | {transformation} | {data['count']} | {data['percentage']:.1f}% |")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        # Based on ablation results
        if 'ablation_studies' in self.evaluation_results:
            ablation_results = self.evaluation_results['ablation_studies']
            
            # Critical augmentations (high degradation)
            critical_augmentations = [r for r in ablation_results 
                                    if r['statistical_significance'] and r['f1_difference'] > 0.02]
            
            if critical_augmentations:
                report.append("### Critical Augmentations (High Performance Impact)")
                report.append("")
                report.append("These augmentations show significant positive impact on model performance:")
                report.append("")
                for result in critical_augmentations:
                    report.append(f"- **{result['augmentation_name']}**: {result['f1_difference']:.4f} F1 improvement")
                report.append("")
            
            # Beneficial augmentations (improvements)
            beneficial_augmentations = [r for r in ablation_results 
                                      if r['statistical_significance'] and r['f1_difference'] < -0.02]
            
            if beneficial_augmentations:
                report.append("### Augmentations with Negative Impact")
                report.append("")
                report.append("These augmentations show significant negative impact on model performance:")
                report.append("")
                for result in beneficial_augmentations:
                    report.append(f"- **{result['augmentation_name']}**: {abs(result['f1_difference']):.4f} F1 degradation")
                report.append("")
        
        # System recommendations
        if 'checker_framework_coverage' in self.evaluation_results:
            coverage = self.evaluation_results['checker_framework_coverage']
            
            enhanced_ratio = coverage['enhanced_files'] / coverage['total_files'] if coverage['total_files'] > 0 else 0
            simple_ratio = coverage['simple_files'] / coverage['total_files'] if coverage['total_files'] > 0 else 0
            
            report.append("### System Usage Recommendations")
            report.append("")
            report.append(f"- **Enhanced System**: Used in {enhanced_ratio:.1%} of Checker Framework test cases")
            report.append(f"- **Simple System**: Used in {simple_ratio:.1%} of Checker Framework test cases")
            report.append("")
            
            if simple_ratio > 0.8:
                report.append("**Recommendation**: The high usage of the Simple system suggests that Checker Framework test cases are primarily simple code. Consider optimizing the Simple augmentation system for better coverage.")
            elif enhanced_ratio > 0.5:
                report.append("**Recommendation**: The significant usage of the Enhanced system suggests complex code patterns in Checker Framework tests. Both systems are well-utilized.")
            else:
                report.append("**Recommendation**: Balanced usage of both systems indicates good complexity-based selection.")
        
        # Save report
        report_content = "\n".join(report)
        report_file = os.path.join(self.output_dir, 'comprehensive_evaluation_report.md')
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Comprehensive report saved to {report_file}")
        return report_content
    
    def _generate_coverage_report(self, coverage_results: Dict[str, Any]) -> str:
        """Generate focused coverage report."""
        report = []
        report.append("# Checker Framework Coverage Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        report.append(f"- **Total Files**: {coverage_results['total_files']}")
        report.append(f"- **Enhanced System**: {coverage_results['enhanced_files']} files")
        report.append(f"- **Simple System**: {coverage_results['simple_files']} files")
        
        return "\n".join(report)
    
    def _generate_ablation_report(self, ablation_results: List[Any]) -> str:
        """Generate focused ablation report."""
        report = []
        report.append("# Ablation Study Report")
        report.append("=" * 30)
        report.append("")
        
        significant_count = sum(1 for r in ablation_results if r.statistical_significance)
        report.append(f"- **Total Methods**: {len(ablation_results)}")
        report.append(f"- **Significant Results**: {significant_count}")
        
        return "\n".join(report)
    
    def _generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate focused test report."""
        report = []
        report.append("# Test Case Report")
        report.append("=" * 20)
        report.append("")
        
        report.append(f"- **Total Tests**: {test_results['total_tests']}")
        report.append(f"- **Success Rate**: {test_results['success_rate']:.2%}")
        
        return "\n".join(report)
    
    def _save_complete_results(self):
        """Save complete evaluation results to JSON."""
        output_file = os.path.join(self.output_dir, 'complete_evaluation_results.json')
        
        with open(output_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Complete results saved to {output_file}")


def main():
    """Main function to run semantic augmentation evaluation."""
    parser = argparse.ArgumentParser(description='Run semantic augmentation evaluation')
    
    # Configuration
    parser.add_argument('--checker_framework_dir',
                       default='/home/ubuntu/checker-framework/checker/tests/index/',
                       help='Directory containing Checker Framework test cases')
    parser.add_argument('--cfwr_root',
                       default='/home/ubuntu/GenDATA',
                       help='CFWR root directory')
    parser.add_argument('--output_dir',
                       default='/home/ubuntu/GenDATA/evaluation_results/',
                       help='Output directory for evaluation results')
    
    # Evaluation options
    parser.add_argument('--run_all', action='store_true',
                       help='Run complete evaluation (all components)')
    parser.add_argument('--run_coverage', action='store_true',
                       help='Run only Checker Framework coverage analysis')
    parser.add_argument('--run_ablation', action='store_true',
                       help='Run only ablation studies')
    parser.add_argument('--run_tests', action='store_true',
                       help='Run only test cases')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'checker_framework_dir': args.checker_framework_dir,
        'cfwr_root': args.cfwr_root,
        'output_dir': args.output_dir
    }
    
    # Create evaluation runner
    runner = SemanticAugmentationEvaluationRunner(config)
    
    # Run evaluation based on arguments
    if args.run_all or (not any([args.run_coverage, args.run_ablation, args.run_tests])):
        # Run complete evaluation
        results = runner.run_complete_evaluation()
        print("\n" + "="*60)
        print("COMPLETE SEMANTIC AUGMENTATION EVALUATION SUMMARY")
        print("="*60)
        print(f"Total evaluation time: {results['total_evaluation_time']:.2f} seconds")
        print(f"Results saved to: {args.output_dir}")
        
    elif args.run_coverage:
        results = runner.run_checker_framework_analysis_only()
        print("Checker Framework coverage analysis complete")
        
    elif args.run_ablation:
        results = runner.run_ablation_studies_only()
        print("Ablation studies complete")
        
    elif args.run_tests:
        results = runner.run_test_cases_only()
        print("Test cases complete")


if __name__ == '__main__':
    main()
