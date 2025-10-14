#!/usr/bin/env python3
"""
Ablation Study Evaluator

Evaluates and compares performance metrics across different ablation studies.
Calculates performance loss and generates statistical analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AblationStudyEvaluator:
    """Evaluates ablation study results and calculates performance metrics"""
    
    def __init__(self, ablation_results_dir: str):
        self.results_dir = Path(ablation_results_dir)
        self.baseline_metrics = None
        self.ablation_metrics = {}
        
        # Load results
        self._load_results()
        
        logger.info(f"Initialized AblationStudyEvaluator with {len(self.ablation_metrics)} ablation cases")
    
    def _load_results(self):
        """Load all ablation study results"""
        try:
            # Load comprehensive results
            summary_file = self.results_dir / 'ablation_results_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.comprehensive_results = json.load(f)
                
                # Extract baseline and ablation metrics
                individual_results = self.comprehensive_results.get('individual_results', {})
                self.baseline_metrics = individual_results.get('baseline', {}).get('metrics', {})
                
                for case_name, results in individual_results.items():
                    if case_name != 'baseline':
                        self.ablation_metrics[case_name] = results.get('metrics', {})
                
                logger.info(f"Loaded results for {len(self.ablation_metrics)} ablation cases")
            else:
                # Load individual result files
                self._load_individual_results()
                
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            self.comprehensive_results = {}
    
    def _load_individual_results(self):
        """Load results from individual case directories"""
        for case_dir in self.results_dir.iterdir():
            if case_dir.is_dir():
                results_file = case_dir / 'results.json'
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        
                        case_name = results.get('case_name', case_dir.name)
                        metrics = results.get('metrics', {})
                        
                        if case_name == 'baseline':
                            self.baseline_metrics = metrics
                        else:
                            self.ablation_metrics[case_name] = metrics
                            
                    except Exception as e:
                        logger.warning(f"Could not load results from {results_file}: {e}")
    
    def calculate_performance_loss(self) -> Dict[str, Any]:
        """Calculate performance loss for each ablation case compared to baseline"""
        if not self.baseline_metrics:
            logger.error("No baseline metrics found")
            return {}
        
        performance_analysis = {
            'baseline': self.baseline_metrics,
            'ablation_cases': {},
            'summary_statistics': {}
        }
        
        # Primary metric: warning reduction percentage
        baseline_reduction = self.baseline_metrics.get('reduction_percentage', 0)
        baseline_warnings = self.baseline_metrics.get('baseline_warnings', 0)
        baseline_models = self.baseline_metrics.get('models_trained', 0)
        
        performance_losses = []
        
        for case_name, metrics in self.ablation_metrics.items():
            case_analysis = {
                'metrics': metrics,
                'performance_loss': {}
            }
            
            # Calculate performance loss metrics
            case_reduction = metrics.get('reduction_percentage', 0)
            case_warnings = metrics.get('baseline_warnings', 0)
            case_models = metrics.get('models_trained', 0)
            
            # Warning reduction loss (primary metric)
            warning_reduction_loss = max(0, baseline_reduction - case_reduction)
            case_analysis['performance_loss']['warning_reduction_loss'] = warning_reduction_loss
            
            # Model training loss (secondary metric)
            if baseline_models > 0:
                models_loss = max(0, (baseline_models - case_models) / baseline_models * 100)
                case_analysis['performance_loss']['models_loss_percentage'] = models_loss
            else:
                models_loss = 0
                case_analysis['performance_loss']['models_loss_percentage'] = models_loss
            
            # Overall performance loss (weighted: 80% warning reduction, 20% models)
            overall_loss = (warning_reduction_loss * 0.8) + (models_loss * 0.2)
            case_analysis['performance_loss']['overall_loss_percentage'] = overall_loss
            
            performance_losses.append(overall_loss)
            performance_analysis['ablation_cases'][case_name] = case_analysis
        
        # Calculate summary statistics
        if performance_losses:
            performance_analysis['summary_statistics'] = {
                'mean_performance_loss': np.mean(performance_losses),
                'median_performance_loss': np.median(performance_losses),
                'std_performance_loss': np.std(performance_losses),
                'min_performance_loss': np.min(performance_losses),
                'max_performance_loss': np.max(performance_losses),
                'total_ablations': len(performance_losses)
            }
        
        return performance_analysis
    
    def analyze_transformation_impact(self) -> Dict[str, Any]:
        """Analyze impact of removing individual transformations"""
        transformation_analysis = {
            'enhanced_transformations': {},
            'simple_transformations': {},
            'summary': {}
        }
        
        enhanced_transforms = [
            'loop_conversion', 'guard_reversal', 'math_commutativity', 'math_associativity',
            'math_identity', 'math_strength_reduction', 'de_morgans_laws', 'ternary_if_else',
            'switch_if_else', 'variable_inlining', 'variable_extraction', 'method_extraction',
            'method_inlining', 'conditional_restructuring', 'array_access_patterns',
            'string_concatenation', 'numeric_literals', 'exception_handling', 'lambda_conversion',
            'stream_conversion', 'builder_pattern', 'functional_programming'
        ]
        
        simple_transforms = [
            'simple_method_calls', 'simple_assignments', 'simple_conditionals',
            'simple_array_access', 'simple_return_statements', 'simple_variable_declarations',
            'simple_constructor_calls', 'simple_field_access', 'simple_string_operations',
            'simple_numeric_operations'
        ]
        
        # Analyze enhanced transformations
        enhanced_impacts = []
        for transform in enhanced_transforms:
            case_name = f'ablate_{transform}'
            if case_name in self.ablation_metrics:
                metrics = self.ablation_metrics[case_name]
                overall_loss = self._calculate_case_performance_loss(case_name)
                transformation_analysis['enhanced_transformations'][transform] = {
                    'performance_loss': overall_loss,
                    'metrics': metrics
                }
                enhanced_impacts.append(overall_loss)
        
        # Analyze simple transformations
        simple_impacts = []
        for transform in simple_transforms:
            case_name = f'ablate_{transform}'
            if case_name in self.ablation_metrics:
                metrics = self.ablation_metrics[case_name]
                overall_loss = self._calculate_case_performance_loss(case_name)
                transformation_analysis['simple_transformations'][transform] = {
                    'performance_loss': overall_loss,
                    'metrics': metrics
                }
                simple_impacts.append(overall_loss)
        
        # Calculate summary statistics
        transformation_analysis['summary'] = {
            'enhanced_transforms': {
                'count': len(enhanced_impacts),
                'mean_impact': np.mean(enhanced_impacts) if enhanced_impacts else 0,
                'std_impact': np.std(enhanced_impacts) if enhanced_impacts else 0
            },
            'simple_transforms': {
                'count': len(simple_impacts),
                'mean_impact': np.mean(simple_impacts) if simple_impacts else 0,
                'std_impact': np.std(simple_impacts) if simple_impacts else 0
            }
        }
        
        return transformation_analysis
    
    def _calculate_case_performance_loss(self, case_name: str) -> float:
        """Calculate overall performance loss for a specific case"""
        if case_name not in self.ablation_metrics or not self.baseline_metrics:
            return 0.0
        
        metrics = self.ablation_metrics[case_name]
        baseline_reduction = self.baseline_metrics.get('reduction_percentage', 0)
        baseline_models = self.baseline_metrics.get('models_trained', 0)
        
        case_reduction = metrics.get('reduction_percentage', 0)
        case_models = metrics.get('models_trained', 0)
        
        # Warning reduction loss (primary metric)
        warning_reduction_loss = max(0, baseline_reduction - case_reduction)
        
        # Model training loss (secondary metric)
        if baseline_models > 0:
            models_loss = max(0, (baseline_models - case_models) / baseline_models * 100)
        else:
            models_loss = 0
        
        # Overall performance loss (weighted: 80% warning reduction, 20% models)
        overall_loss = (warning_reduction_loss * 0.8) + (models_loss * 0.2)
        
        return overall_loss
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'ablation_study_overview': self._get_study_overview(),
            'performance_analysis': self.calculate_performance_loss(),
            'transformation_analysis': self.analyze_transformation_impact(),
            'key_findings': self._extract_key_findings(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _get_study_overview(self) -> Dict[str, Any]:
        """Get overview of the ablation study"""
        return {
            'total_ablation_cases': len(self.ablation_metrics),
            'baseline_case': 'baseline' if self.baseline_metrics else None,
            'transformation_ablations': len([k for k in self.ablation_metrics.keys() if k.startswith('ablate_')]),
            'special_cases': [k for k in self.ablation_metrics.keys() if not k.startswith('ablate_')],
            'study_timestamp': self.comprehensive_results.get('ablation_study_summary', {}).get('timestamp', 'Unknown')
        }
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from the analysis"""
        findings = []
        
        performance_analysis = self.calculate_performance_loss()
        summary_stats = performance_analysis.get('summary_statistics', {})
        
        if summary_stats:
            mean_loss = summary_stats.get('mean_performance_loss', 0)
            max_loss = summary_stats.get('max_performance_loss', 0)
            
            findings.append(f"Average performance loss across all ablations: {mean_loss:.2f}%")
            findings.append(f"Maximum performance loss observed: {max_loss:.2f}%")
            
            if mean_loss > 10:
                findings.append("Data augmentation has significant impact on model performance")
            elif mean_loss > 5:
                findings.append("Data augmentation has moderate impact on model performance")
            else:
                findings.append("Data augmentation has minimal impact on model performance")
        
        # Transformation-specific findings
        transform_analysis = self.analyze_transformation_impact()
        enhanced_summary = transform_analysis.get('summary', {}).get('enhanced_transforms', {})
        simple_summary = transform_analysis.get('summary', {}).get('simple_transforms', {})
        
        if enhanced_summary and simple_summary:
            enhanced_mean = enhanced_summary.get('mean_impact', 0)
            simple_mean = simple_summary.get('mean_impact', 0)
            
            if enhanced_mean > simple_mean:
                findings.append("Enhanced transformations have greater impact than simple transformations")
            elif simple_mean > enhanced_mean:
                findings.append("Simple transformations have greater impact than enhanced transformations")
            else:
                findings.append("Enhanced and simple transformations have similar impact")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on the analysis"""
        recommendations = []
        
        performance_analysis = self.calculate_performance_loss()
        summary_stats = performance_analysis.get('summary_statistics', {})
        
        if summary_stats:
            mean_loss = summary_stats.get('mean_performance_loss', 0)
            
            if mean_loss > 15:
                recommendations.append("Consider optimizing augmentation pipeline - high performance loss detected")
            elif mean_loss > 5:
                recommendations.append("Monitor augmentation effectiveness - moderate performance impact")
            else:
                recommendations.append("Augmentation pipeline is performing well - low performance loss")
        
        # Transformation-specific recommendations
        transform_analysis = self.analyze_transformation_impact()
        
        # Find most impactful transformations
        all_transforms = {}
        all_transforms.update(transform_analysis.get('enhanced_transformations', {}))
        all_transforms.update(transform_analysis.get('simple_transformations', {}))
        
        if all_transforms:
            # Sort by performance loss
            sorted_transforms = sorted(all_transforms.items(), 
                                     key=lambda x: x[1].get('performance_loss', 0), 
                                     reverse=True)
            
            if sorted_transforms:
                top_transform = sorted_transforms[0]
                recommendations.append(f"Most critical transformation: {top_transform[0]} "
                                     f"(loss: {top_transform[1].get('performance_loss', 0):.2f}%)")
        
        return recommendations
    
    def save_analysis_report(self, output_file: Optional[str] = None) -> str:
        """Save analysis report to file"""
        if output_file is None:
            output_file = self.results_dir / 'ablation_analysis_report.json'
        else:
            output_file = Path(output_file)
        
        report = self.generate_comparison_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to {output_file}")
        return str(output_file)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ablation study results')
    parser.add_argument('--results_dir', required=True, help='Directory containing ablation study results')
    parser.add_argument('--output_file', help='Output file for analysis report')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AblationStudyEvaluator(args.results_dir)
    
    # Generate and save report
    output_file = evaluator.save_analysis_report(args.output_file)
    
    # Print summary
    report = evaluator.generate_comparison_report()
    key_findings = report.get('key_findings', [])
    
    print("\n=== Ablation Study Analysis Summary ===")
    for finding in key_findings:
        print(f"• {finding}")
    
    print(f"\nDetailed analysis report saved to: {output_file}")
    
    return 0

if __name__ == '__main__':
    exit(main())
