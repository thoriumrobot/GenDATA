#!/usr/bin/env python3
"""
Evaluation Script for Checker Value Emphasis

Analyzes learned attention weights to see which values are being emphasized
for each checker. Visualizes emphasis patterns and compares with manual features.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from checker_config import CheckerType, get_all_checker_types, get_checker_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckerEmphasisEvaluator:
    """Evaluate learned value emphasis for checker-specific models"""
    
    def __init__(self, results_file: str):
        """
        Initialize evaluator
        
        Args:
            results_file: Path to checker training results JSON file
        """
        self.results_file = Path(results_file)
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        self.output_dir = self.results_file.parent / 'emphasis_analysis'
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_attention_weights(self) -> Dict[str, Any]:
        """Analyze attention weights across all checkers and models"""
        logger.info("Analyzing learned attention weights...")
        
        analysis = {
            'checkers': {},
            'summary': {}
        }
        
        training_results = self.results.get('training_results', {})
        
        for checker_name, checker_results in training_results.items():
            checker_type_name = checker_results.get('checker_type', checker_name)
            target_values = checker_results.get('target_values', [])
            
            logger.info(f"\nAnalyzing {checker_results.get('checker_name', checker_name)}:")
            logger.info(f"  Target values: {target_values}")
            
            checker_analysis = {
                'checker_name': checker_results.get('checker_name', checker_name),
                'target_values': target_values,
                'models': {}
            }
            
            # Analyze each model type
            for model_type, models in checker_results.get('models', {}).items():
                model_analysis = {
                    'annotation_types': {}
                }
                
                for ann_type, result in models.items():
                    if not result.get('success'):
                        continue
                    
                    attention_summary = result.get('attention_summary')
                    if attention_summary:
                        # Sort patterns by emphasis weight
                        sorted_patterns = sorted(
                            attention_summary.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        
                        model_analysis['annotation_types'][ann_type] = {
                            'top_patterns': sorted_patterns[:5],
                            'all_patterns': attention_summary,
                            'max_emphasis': max(attention_summary.values()) if attention_summary else 0,
                            'min_emphasis': min(attention_summary.values()) if attention_summary else 0,
                            'avg_emphasis': sum(attention_summary.values()) / len(attention_summary) if attention_summary else 0
                        }
                        
                        logger.info(f"    {model_type} - {ann_type}:")
                        logger.info(f"      Top 3 patterns: {sorted_patterns[:3]}")
                
                checker_analysis['models'][model_type] = model_analysis
            
            analysis['checkers'][checker_name] = checker_analysis
        
        # Generate summary statistics
        self._generate_summary(analysis)
        analysis['summary'] = self.summary_stats
        
        return analysis
    
    def _generate_summary(self, analysis: Dict[str, Any]):
        """Generate summary statistics"""
        self.summary_stats = {
            'total_checkers': len(analysis['checkers']),
            'checkers_analyzed': [],
            'average_emphasis_by_checker': {},
            'most_emphasized_patterns': {}
        }
        
        for checker_name, checker_analysis in analysis['checkers'].items():
            self.summary_stats['checkers_analyzed'].append(checker_name)
            
            # Calculate average emphasis across all models and annotation types
            all_emphases = []
            for model_type, model_analysis in checker_analysis.get('models', {}).items():
                for ann_type, ann_analysis in model_analysis.get('annotation_types', {}).items():
                    all_emphases.extend(ann_analysis.get('all_patterns', {}).values())
            
            if all_emphases:
                self.summary_stats['average_emphasis_by_checker'][checker_name] = {
                    'mean': np.mean(all_emphases),
                    'std': np.std(all_emphases),
                    'max': np.max(all_emphases),
                    'min': np.min(all_emphases)
                }
            
            # Find most emphasized patterns
            pattern_emphases = {}
            for model_type, model_analysis in checker_analysis.get('models', {}).items():
                for ann_type, ann_analysis in model_analysis.get('annotation_types', {}).items():
                    for pattern, emphasis in ann_analysis.get('all_patterns', {}).items():
                        if pattern not in pattern_emphases:
                            pattern_emphases[pattern] = []
                        pattern_emphases[pattern].append(emphasis)
            
            if pattern_emphases:
                avg_pattern_emphases = {
                    pattern: np.mean(emphases)
                    for pattern, emphases in pattern_emphases.items()
                }
                top_patterns = sorted(
                    avg_pattern_emphases.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                self.summary_stats['most_emphasized_patterns'][checker_name] = top_patterns
    
    def visualize_emphasis(self, analysis: Dict[str, Any]):
        """Create visualizations of learned emphasis"""
        logger.info("Creating visualizations...")
        
        for checker_name, checker_analysis in analysis['checkers'].items():
            # Create bar chart for each model type
            for model_type, model_analysis in checker_analysis.get('models', {}).items():
                for ann_type, ann_analysis in model_analysis.get('annotation_types', {}).items():
                    patterns = ann_analysis.get('all_patterns', {})
                    if not patterns:
                        continue
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
                    pattern_names = [p[0] for p in sorted_patterns]
                    emphasis_values = [p[1] for p in sorted_patterns]
                    
                    ax.barh(pattern_names, emphasis_values)
                    ax.set_xlabel('Emphasis Weight')
                    ax.set_title(f'{checker_analysis["checker_name"]} - {model_type} - {ann_type}\nLearned Value Emphasis')
                    ax.grid(axis='x', alpha=0.3)
                    
                    plt.tight_layout()
                    output_file = self.output_dir / f'{checker_name}_{model_type}_{ann_type}_emphasis.png'
                    plt.savefig(output_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"  Saved visualization: {output_file}")
    
    def compare_with_manual_features(self, analysis: Dict[str, Any]):
        """Compare learned emphasis with manual 'could be zero' features"""
        logger.info("\nComparing learned emphasis with manual features...")
        
        comparison = {
            'checkers': {}
        }
        
        # Manual feature scaling (from "could be zero" implementation)
        manual_scaling = {
            'is_used_as_array_index': 2.0,
            'is_loop_variable': 2.0,
            'is_subtraction_result': 1.5,
            'is_param_in_array_context': 2.0,
            'compared_with_length': 1.5,
            'initialized_to_zero': 2.0,
            'used_in_nonnegative_check': 2.0,
            'is_offset_or_position': 1.5,
            'could_be_zero_score': 3.0
        }
        
        for checker_name, checker_analysis in analysis['checkers'].items():
            checker_comparison = {
                'manual_scaling': manual_scaling,
                'learned_emphasis': {}
            }
            
            # Extract learned emphasis for common patterns
            for model_type, model_analysis in checker_analysis.get('models', {}).items():
                for ann_type, ann_analysis in model_analysis.get('annotation_types', {}).items():
                    learned = ann_analysis.get('all_patterns', {})
                    
                    # Compare overlapping patterns
                    overlapping = {}
                    for pattern, learned_weight in learned.items():
                        # Try to match with manual patterns
                        for manual_pattern, manual_weight in manual_scaling.items():
                            if pattern in manual_pattern or manual_pattern in pattern:
                                overlapping[pattern] = {
                                    'learned': learned_weight,
                                    'manual': manual_weight,
                                    'ratio': learned_weight / manual_weight if manual_weight > 0 else 0
                                }
                                break
                    
                    if overlapping:
                        checker_comparison['learned_emphasis'][f'{model_type}_{ann_type}'] = overlapping
            
            comparison['checkers'][checker_name] = checker_comparison
        
        # Save comparison
        comparison_file = self.output_dir / 'emphasis_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to: {comparison_file}")
        
        return comparison
    
    def generate_report(self, analysis: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        report_lines = [
            "# Checker Value Emphasis Evaluation Report",
            "",
            f"Generated: {self.results.get('timestamp', 'Unknown')}",
            "",
            "## Summary",
            "",
            f"Total checkers analyzed: {len(analysis['checkers'])}",
            "",
            "## Results by Checker",
            ""
        ]
        
        for checker_name, checker_analysis in analysis['checkers'].items():
            report_lines.extend([
                f"### {checker_analysis['checker_name']}",
                "",
                f"**Target Values**: {', '.join(checker_analysis.get('target_values', []))}",
                ""
            ])
            
            # Summary stats
            if checker_name in self.summary_stats.get('average_emphasis_by_checker', {}):
                stats = self.summary_stats['average_emphasis_by_checker'][checker_name]
                report_lines.extend([
                    f"**Average Emphasis**: {stats['mean']:.3f} ± {stats['std']:.3f}",
                    f"**Range**: {stats['min']:.3f} - {stats['max']:.3f}",
                    ""
                ])
            
            # Top patterns
            if checker_name in self.summary_stats.get('most_emphasized_patterns', {}):
                top_patterns = self.summary_stats['most_emphasized_patterns'][checker_name]
                report_lines.append("**Most Emphasized Patterns**:")
                for pattern, emphasis in top_patterns:
                    report_lines.append(f"  - {pattern}: {emphasis:.3f}")
                report_lines.append("")
            
            # Model-specific results
            for model_type, model_analysis in checker_analysis.get('models', {}).items():
                report_lines.append(f"#### {model_type}")
                for ann_type, ann_analysis in model_analysis.get('annotation_types', {}).items():
                    top_patterns = ann_analysis.get('top_patterns', [])
                    if top_patterns:
                        report_lines.append(f"**{ann_type}**:")
                        for pattern, emphasis in top_patterns[:3]:
                            report_lines.append(f"  - {pattern}: {emphasis:.3f}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / 'evaluation_report.md'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to: {report_file}")
        
        return str(report_file)
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("=" * 80)
        logger.info("CHECKER VALUE EMPHASIS EVALUATION")
        logger.info("=" * 80)
        
        # Analyze attention weights
        analysis = self.analyze_attention_weights()
        
        # Visualize emphasis
        try:
            self.visualize_emphasis(analysis)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
        
        # Compare with manual features
        comparison = self.compare_with_manual_features(analysis)
        
        # Generate report
        report_file = self.generate_report(analysis, comparison)
        
        # Save full analysis
        analysis_file = self.output_dir / 'full_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump({
                'analysis': analysis,
                'comparison': comparison,
                'summary': self.summary_stats
            }, f, indent=2)
        
        logger.info(f"\nFull analysis saved to: {analysis_file}")
        logger.info(f"Evaluation report: {report_file}")
        
        return analysis, comparison


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate learned value emphasis for checker-specific models'
    )
    parser.add_argument(
        '--results_file',
        type=str,
        default='checker_specific_models/checker_training_results.json',
        help='Path to checker training results JSON file'
    )
    
    args = parser.parse_args()
    
    evaluator = CheckerEmphasisEvaluator(args.results_file)
    analysis, comparison = evaluator.run_full_evaluation()
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation completed!")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

