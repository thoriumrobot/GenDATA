#!/usr/bin/env python3
"""
Ablation Study Report Generator

Generates comprehensive markdown reports and visualizations for ablation study results.
Creates tables, charts, and statistical summaries.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AblationStudyReportGenerator:
    """Generates comprehensive reports and visualizations for ablation studies"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.analysis_data = None
        self.plots_dir = self.results_dir / 'ablation_results_plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load analysis data
        self._load_analysis_data()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"Initialized AblationStudyReportGenerator for {self.results_dir}")
    
    def _load_analysis_data(self):
        """Load analysis data from evaluation results"""
        try:
            # Try to load from analysis report first
            analysis_file = self.results_dir / 'ablation_analysis_report.json'
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    self.analysis_data = json.load(f)
                logger.info("Loaded analysis data from evaluation report")
                return
            
            # Fall back to summary results
            summary_file = self.results_dir / 'ablation_results_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.comprehensive_results = json.load(f)
                
                # Convert to analysis format
                self.analysis_data = self._convert_summary_to_analysis()
                logger.info("Converted summary data to analysis format")
            else:
                logger.error("No analysis data found")
                self.analysis_data = {}
                
        except Exception as e:
            logger.error(f"Error loading analysis data: {e}")
            self.analysis_data = {}
    
    def _convert_summary_to_analysis(self) -> Dict[str, Any]:
        """Convert summary results to analysis format"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'ablation_study_overview': {
                'total_ablation_cases': len(self.comprehensive_results.get('individual_results', {})) - 1,
                'baseline_case': 'baseline',
                'study_timestamp': self.comprehensive_results.get('ablation_study_summary', {}).get('timestamp', 'Unknown')
            },
            'performance_analysis': {
                'baseline': {},
                'ablation_cases': {},
                'summary_statistics': {}
            },
            'transformation_analysis': {
                'enhanced_transformations': {},
                'simple_transformations': {},
                'summary': {}
            }
        }
        
        # Extract performance data
        individual_results = self.comprehensive_results.get('individual_results', {})
        performance_comparison = self.comprehensive_results.get('performance_comparison', {})
        
        baseline_metrics = individual_results.get('baseline', {}).get('metrics', {})
        analysis['performance_analysis']['baseline'] = baseline_metrics
        
        for case_name, results in individual_results.items():
            if case_name != 'baseline':
                metrics = results.get('metrics', {})
                performance_loss = performance_comparison.get('performance_loss', {}).get(case_name, {})
                
                analysis['performance_analysis']['ablation_cases'][case_name] = {
                    'metrics': metrics,
                    'performance_loss': performance_loss.get('performance_loss_percentage', 0)
                }
        
        return analysis
    
    def generate_markdown_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive markdown report"""
        if output_file is None:
            output_file = self.results_dir / 'ablation_results_report.md'
        else:
            output_file = Path(output_file)
        
        report_content = self._create_markdown_content()
        
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Markdown report saved to {output_file}")
        return str(output_file)
    
    def _create_markdown_content(self) -> str:
        """Create markdown report content"""
        if not self.analysis_data:
            return "# Ablation Study Report\n\nNo analysis data available."
        
        content = []
        
        # Header
        content.append("# Ablation Study Results Report")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Overview
        overview = self.analysis_data.get('ablation_study_overview', {})
        content.append("## Study Overview")
        content.append("")
        content.append(f"- **Total Ablation Cases:** {overview.get('total_ablation_cases', 0)}")
        content.append(f"- **Baseline Case:** {overview.get('baseline_case', 'N/A')}")
        content.append(f"- **Study Timestamp:** {overview.get('study_timestamp', 'Unknown')}")
        content.append("")
        
        # Key Findings
        key_findings = self.analysis_data.get('key_findings', [])
        if key_findings:
            content.append("## Key Findings")
            content.append("")
            for finding in key_findings:
                content.append(f"- {finding}")
            content.append("")
        
        # Performance Analysis
        content.append("## Performance Analysis")
        content.append("")
        
        performance_analysis = self.analysis_data.get('performance_analysis', {})
        summary_stats = performance_analysis.get('summary_statistics', {})
        
        if summary_stats:
            content.append("### Summary Statistics")
            content.append("")
            content.append(f"- **Mean Performance Loss:** {summary_stats.get('mean_performance_loss', 0):.2f}%")
            content.append(f"- **Median Performance Loss:** {summary_stats.get('median_performance_loss', 0):.2f}%")
            content.append(f"- **Standard Deviation:** {summary_stats.get('std_performance_loss', 0):.2f}%")
            content.append(f"- **Min Performance Loss:** {summary_stats.get('min_performance_loss', 0):.2f}%")
            content.append(f"- **Max Performance Loss:** {summary_stats.get('max_performance_loss', 0):.2f}%")
            content.append("")
        
        # Baseline vs No Augmentation
        content.append("### No Augmentation Impact")
        content.append("")
        no_aug_data = performance_analysis.get('ablation_cases', {}).get('no_augmentation', {})
        if no_aug_data:
            loss = no_aug_data.get('performance_loss', {})
            content.append(f"- **Baseline Performance:** {performance_analysis.get('baseline', {}).get('training_time_seconds', 0):.2f}s")
            content.append(f"- **No Augmentation Performance:** {no_aug_data.get('metrics', {}).get('training_time_seconds', 0):.2f}s")
            content.append(f"- **Performance Loss:** {loss:.2f}%")
            content.append("")
        
        # Random Walk Impact
        content.append("### Random Walk Optimization Impact")
        content.append("")
        no_rw_data = performance_analysis.get('ablation_cases', {}).get('no_random_walk', {})
        if no_rw_data:
            loss = no_rw_data.get('performance_loss', {})
            content.append(f"- **With Random Walk:** {performance_analysis.get('baseline', {}).get('training_time_seconds', 0):.2f}s")
            content.append(f"- **Without Random Walk:** {no_rw_data.get('metrics', {}).get('training_time_seconds', 0):.2f}s")
            content.append(f"- **Performance Loss:** {loss:.2f}%")
            content.append("")
        
        # Transformation Impact
        content.append("## Individual Transformation Impact")
        content.append("")
        
        transform_analysis = self.analysis_data.get('transformation_analysis', {})
        
        # Enhanced Transformations
        enhanced_transforms = transform_analysis.get('enhanced_transformations', {})
        if enhanced_transforms:
            content.append("### Enhanced Transformations (17 methods)")
            content.append("")
            content.append("| Transformation | Performance Loss (%) |")
            content.append("|----------------|---------------------|")
            
            # Sort by performance loss
            sorted_enhanced = sorted(enhanced_transforms.items(), 
                                   key=lambda x: x[1].get('performance_loss', 0), 
                                   reverse=True)
            
            for transform, data in sorted_enhanced:
                loss = data.get('performance_loss', 0)
                content.append(f"| {transform} | {loss:.2f} |")
            content.append("")
        
        # Simple Transformations
        simple_transforms = transform_analysis.get('simple_transformations', {})
        if simple_transforms:
            content.append("### Simple Transformations (10 methods)")
            content.append("")
            content.append("| Transformation | Performance Loss (%) |")
            content.append("|----------------|---------------------|")
            
            # Sort by performance loss
            sorted_simple = sorted(simple_transforms.items(), 
                                 key=lambda x: x[1].get('performance_loss', 0), 
                                 reverse=True)
            
            for transform, data in sorted_simple:
                loss = data.get('performance_loss', 0)
                content.append(f"| {transform} | {loss:.2f} |")
            content.append("")
        
        # Summary Statistics
        transform_summary = transform_analysis.get('summary', {})
        if transform_summary:
            enhanced_summary = transform_summary.get('enhanced_transforms', {})
            simple_summary = transform_summary.get('simple_transforms', {})
            
            content.append("### Transformation Summary")
            content.append("")
            content.append(f"- **Enhanced Transformations:**")
            content.append(f"  - Count: {enhanced_summary.get('count', 0)}")
            content.append(f"  - Mean Impact: {enhanced_summary.get('mean_impact', 0):.2f}%")
            content.append(f"  - Std Deviation: {enhanced_summary.get('std_impact', 0):.2f}%")
            content.append("")
            content.append(f"- **Simple Transformations:**")
            content.append(f"  - Count: {simple_summary.get('count', 0)}")
            content.append(f"  - Mean Impact: {simple_summary.get('mean_impact', 0):.2f}%")
            content.append(f"  - Std Deviation: {simple_summary.get('std_impact', 0):.2f}%")
            content.append("")
        
        # Recommendations
        recommendations = self.analysis_data.get('recommendations', [])
        if recommendations:
            content.append("## Recommendations")
            content.append("")
            for rec in recommendations:
                content.append(f"- {rec}")
            content.append("")
        
        # Visualizations
        content.append("## Visualizations")
        content.append("")
        content.append("The following visualizations are available in the `ablation_results_plots/` directory:")
        content.append("")
        content.append("- `performance_loss_bar.png` - Bar chart showing performance loss per ablation case")
        content.append("- `transformation_impact_heatmap.png` - Heatmap of transformation impact")
        content.append("- `ablation_comparison.png` - Overall ablation comparison")
        content.append("")
        
        return "\n".join(content)
    
    def create_performance_loss_bar_chart(self) -> str:
        """Create bar chart showing performance loss per ablation case"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        performance_analysis = self.analysis_data.get('performance_analysis', {})
        ablation_cases = performance_analysis.get('ablation_cases', {})
        
        if not ablation_cases:
            logger.warning("No ablation cases data for bar chart")
            return ""
        
        # Prepare data
        case_names = []
        performance_losses = []
        colors = []
        
        # Color coding
        no_aug_color = 'red'
        no_rw_color = 'orange'
        transform_color = 'blue'
        
        for case_name, data in ablation_cases.items():
            loss = data.get('performance_loss', 0)
            case_names.append(case_name.replace('ablate_', '').replace('_', ' ').title())
            performance_losses.append(loss)
            
            # Color coding
            if 'no_augmentation' in case_name:
                colors.append(no_aug_color)
            elif 'no_random_walk' in case_name:
                colors.append(no_rw_color)
            else:
                colors.append(transform_color)
        
        # Create bar chart
        bars = ax.bar(range(len(case_names)), performance_losses, color=colors, alpha=0.7)
        
        # Customize chart
        ax.set_xlabel('Ablation Cases')
        ax.set_ylabel('Performance Loss (%)')
        ax.set_title('Performance Loss by Ablation Case')
        ax.set_xticks(range(len(case_names)))
        ax.set_xticklabels(case_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, loss in zip(bars, performance_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{loss:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=no_aug_color, label='No Augmentation'),
            mpatches.Patch(color=no_rw_color, label='No Random Walk'),
            mpatches.Patch(color=transform_color, label='Transform Removal')
        ]
        ax.legend(handles=legend_elements)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        output_file = self.plots_dir / 'performance_loss_bar.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance loss bar chart saved to {output_file}")
        return str(output_file)
    
    def create_transformation_impact_heatmap(self) -> str:
        """Create heatmap showing transformation impact"""
        transform_analysis = self.analysis_data.get('transformation_analysis', {})
        
        enhanced_transforms = transform_analysis.get('enhanced_transformations', {})
        simple_transforms = transform_analysis.get('simple_transformations', {})
        
        if not enhanced_transforms and not simple_transforms:
            logger.warning("No transformation data for heatmap")
            return ""
        
        # Prepare data for heatmap
        all_transforms = []
        all_impacts = []
        
        for transform, data in enhanced_transforms.items():
            all_transforms.append(f"E: {transform}")
            all_impacts.append(data.get('performance_loss', 0))
        
        for transform, data in simple_transforms.items():
            all_transforms.append(f"S: {transform}")
            all_impacts.append(data.get('performance_loss', 0))
        
        if not all_transforms:
            return ""
        
        # Create heatmap data
        n_transforms = len(all_transforms)
        n_cols = min(5, n_transforms)  # Max 5 columns
        n_rows = (n_transforms + n_cols - 1) // n_cols  # Ceiling division
        
        # Pad with zeros if needed
        while len(all_impacts) < n_rows * n_cols:
            all_impacts.append(0)
            all_transforms.append('')
        
        # Reshape data
        impact_matrix = np.array(all_impacts[:n_rows * n_cols]).reshape(n_rows, n_cols)
        transform_matrix = np.array(all_transforms[:n_rows * n_cols]).reshape(n_rows, n_cols)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, n_rows * 0.8)))
        
        im = ax.imshow(impact_matrix, cmap='YlOrRd', aspect='auto')
        
        # Add text annotations
        for i in range(n_rows):
            for j in range(n_cols):
                if i * n_cols + j < n_transforms:
                    text = ax.text(j, i, f'{impact_matrix[i, j]:.1f}%\n{transform_matrix[i, j]}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        # Customize chart
        ax.set_title('Transformation Impact Heatmap\n(E = Enhanced, S = Simple)')
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Loss (%)')
        
        plt.tight_layout()
        
        # Save chart
        output_file = self.plots_dir / 'transformation_impact_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Transformation impact heatmap saved to {output_file}")
        return str(output_file)
    
    def create_ablation_comparison_chart(self) -> str:
        """Create overall ablation comparison chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        performance_analysis = self.analysis_data.get('performance_analysis', {})
        summary_stats = performance_analysis.get('summary_statistics', {})
        
        # Chart 1: Overall performance loss distribution
        if summary_stats:
            metrics = ['Mean', 'Median', 'Min', 'Max']
            values = [
                summary_stats.get('mean_performance_loss', 0),
                summary_stats.get('median_performance_loss', 0),
                summary_stats.get('min_performance_loss', 0),
                summary_stats.get('max_performance_loss', 0)
            ]
            
            bars1 = ax1.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            ax1.set_ylabel('Performance Loss (%)')
            ax1.set_title('Performance Loss Statistics')
            ax1.set_ylim(0, max(values) * 1.1 if values else 10)
            
            # Add value labels
            for bar, value in zip(bars1, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        # Chart 2: Case type comparison
        ablation_cases = performance_analysis.get('ablation_cases', {})
        
        case_types = {'No Augmentation': [], 'No Random Walk': [], 'Transform Removal': []}
        
        for case_name, data in ablation_cases.items():
            loss = data.get('performance_loss', 0)
            if 'no_augmentation' in case_name:
                case_types['No Augmentation'].append(loss)
            elif 'no_random_walk' in case_name:
                case_types['No Random Walk'].append(loss)
            else:
                case_types['Transform Removal'].append(loss)
        
        # Calculate means for each type
        type_names = []
        type_means = []
        type_counts = []
        
        for type_name, losses in case_types.items():
            if losses:
                type_names.append(type_name)
                type_means.append(np.mean(losses))
                type_counts.append(len(losses))
        
        if type_names:
            bars2 = ax2.bar(type_names, type_means, color=['red', 'orange', 'blue'], alpha=0.7)
            ax2.set_ylabel('Mean Performance Loss (%)')
            ax2.set_title('Performance Loss by Case Type')
            
            # Add count labels
            for bar, mean_val, count in zip(bars2, type_means, type_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{mean_val:.1f}%\n(n={count})', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save chart
        output_file = self.plots_dir / 'ablation_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Ablation comparison chart saved to {output_file}")
        return str(output_file)
    
    def generate_all_visualizations(self) -> List[str]:
        """Generate all visualizations"""
        generated_files = []
        
        try:
            # Performance loss bar chart
            bar_file = self.create_performance_loss_bar_chart()
            if bar_file:
                generated_files.append(bar_file)
            
            # Transformation impact heatmap
            heatmap_file = self.create_transformation_impact_heatmap()
            if heatmap_file:
                generated_files.append(heatmap_file)
            
            # Ablation comparison chart
            comparison_file = self.create_ablation_comparison_chart()
            if comparison_file:
                generated_files.append(comparison_file)
            
            logger.info(f"Generated {len(generated_files)} visualization files")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return generated_files
    
    def generate_complete_report(self, output_file: Optional[str] = None) -> str:
        """Generate complete report with markdown and visualizations"""
        # Generate markdown report
        markdown_file = self.generate_markdown_report(output_file)
        
        # Generate all visualizations
        visualization_files = self.generate_all_visualizations()
        
        logger.info(f"Complete report generated: {markdown_file}")
        logger.info(f"Visualizations generated: {len(visualization_files)} files")
        
        return markdown_file

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ablation study reports')
    parser.add_argument('--results_dir', required=True, help='Directory containing ablation study results')
    parser.add_argument('--output_file', help='Output markdown file')
    parser.add_argument('--visualizations_only', action='store_true', help='Generate only visualizations')
    parser.add_argument('--report_only', action='store_true', help='Generate only markdown report')
    
    args = parser.parse_args()
    
    # Initialize report generator
    generator = AblationStudyReportGenerator(args.results_dir)
    
    if args.visualizations_only:
        # Generate only visualizations
        visualization_files = generator.generate_all_visualizations()
        print(f"Generated {len(visualization_files)} visualization files")
    elif args.report_only:
        # Generate only markdown report
        report_file = generator.generate_markdown_report(args.output_file)
        print(f"Generated markdown report: {report_file}")
    else:
        # Generate complete report
        report_file = generator.generate_complete_report(args.output_file)
        print(f"Generated complete report: {report_file}")
    
    return 0

if __name__ == '__main__':
    exit(main())
