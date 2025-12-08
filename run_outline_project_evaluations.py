#!/usr/bin/env python3
"""
Complete Evaluation Orchestrator for Outline Projects

This script orchestrates the complete evaluation pipeline for projects mentioned
in GenDATA outline.md (Agrona, Hipparchus, Eclipse Collections).

Pipeline Steps:
1. Prepare Projects (clone/download if needed)
2. Generate Warnings (run Lower Bound Checker)
3. Generate Slices (use Soot slicer)
4. Generate CFGs (convert slices to CFGs)
5. Generate Predictions (all available models)
6. Compute Metrics (precision, recall, F1, warning reduction)
7. Generate Report (comprehensive evaluation report)
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'

# Projects to evaluate
OUTLINE_PROJECTS = ['agrona', 'hipparchus', 'eclipse-collections']


def prepare_projects() -> bool:
    """Prepare projects (clone/download if needed)."""
    logger.info("=" * 80)
    logger.info("Step 1: Preparing Projects")
    logger.info("=" * 80)
    
    try:
        from prepare_outline_projects import main as prepare_main
        result = prepare_main()
        return result == 0
    except Exception as e:
        logger.error(f"Error preparing projects: {e}")
        return False


def run_evaluation() -> Dict[str, dict]:
    """Run evaluation on all projects."""
    logger.info("=" * 80)
    logger.info("Running Evaluation Pipeline")
    logger.info("=" * 80)
    
    try:
        from evaluate_outline_projects import evaluate_project
        
        results = {}
        for project_name in OUTLINE_PROJECTS:
            evaluation_status = evaluate_project(project_name)
            results[project_name] = evaluation_status
        
        return results
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return {}


def generate_comprehensive_report(results: Dict[str, dict]) -> Path:
    """Generate comprehensive evaluation report."""
    logger.info("=" * 80)
    logger.info("Generating Comprehensive Report")
    logger.info("=" * 80)
    
    report_file = GEN_DATA_ROOT / 'OUTLINE_PROJECTS_EVALUATION_RESULTS.md'
    
    # Collect metrics from all projects
    all_metrics = {}
    for project_name in OUTLINE_PROJECTS:
        metrics_file = CASE_STUDIES_DIR / project_name / 'evaluation_metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    all_metrics[project_name] = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metrics for {project_name}: {e}")
    
    # Count statuses
    successful = len([p for p, s in results.items() if s.get('status') == 'success'])
    no_warnings = len([p for p, s in results.items() if s.get('status') == 'no_warnings'])
    failed = len([p for p, s in results.items() if s.get('status') not in ['success', 'no_warnings']])
    
    # Generate report
    with open(report_file, 'w') as f:
        f.write("# Outline Projects Evaluation Results\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Projects Successfully Evaluated: {successful}/{len(OUTLINE_PROJECTS)}\n")
        f.write(f"- Projects with No Warnings: {no_warnings}/{len(OUTLINE_PROJECTS)}\n")
        f.write(f"- Projects Failed: {failed}/{len(OUTLINE_PROJECTS)}\n\n")
        
        f.write("## Results by Project\n\n")
        for project_name in OUTLINE_PROJECTS:
            status_dict = results.get(project_name, {})
            status = status_dict.get('status', 'unknown')
            warning_count = status_dict.get('warning_count', 0)
            steps_completed = status_dict.get('steps_completed', [])
            steps_failed = status_dict.get('steps_failed', [])
            
            if status == 'success':
                f.write(f"### {project_name}: ✅ Success ({warning_count} warnings)\n\n")
            elif status == 'no_warnings':
                f.write(f"### {project_name}: ⚠️ No Warnings Found\n\n")
                f.write(f"*This project has no Lower Bound Checker warnings. This is not a failure - the project may be well-annotated or not use array indexing in ways that trigger warnings.*\n\n")
            else:
                f.write(f"### {project_name}: ❌ Failed ({status})\n\n")
            
            # Status details
            f.write(f"**Status Details:**\n")
            f.write(f"- Warning Count: {warning_count}\n")
            f.write(f"- Steps Completed: {', '.join(steps_completed) if steps_completed else 'None'}\n")
            if steps_failed:
                f.write(f"- Steps Failed: {', '.join(steps_failed)}\n")
            f.write("\n")
            
            # Metrics if available
            if project_name in all_metrics:
                project_metrics = all_metrics[project_name]
                
                f.write("#### Summary Statistics\n\n")
                f.write("| Model | F1 Score | Warning Reduction |\n")
                f.write("|-------|----------|-------------------|\n")
                
                for model_name, metrics in project_metrics.items():
                    f1 = metrics.get('f1_weighted', 'N/A')
                    wr = metrics.get('warning_reduction', 'N/A')
                    if isinstance(f1, float):
                        f1 = f"{f1:.3f}"
                    if isinstance(wr, float):
                        wr = f"{wr:.1f}%"
                    f.write(f"| {model_name} | {f1} | {wr} |\n")
                
                f.write("\n")
            elif status != 'no_warnings':
                f.write("*No metrics available.*\n\n")
        
        # Cross-project summary
        if all_metrics:
            f.write("## Cross-Project Summary\n\n")
            
            # Aggregate by model
            model_aggregates = {}
            for project_metrics in all_metrics.values():
                for model_name, metrics in project_metrics.items():
                    if model_name not in model_aggregates:
                        model_aggregates[model_name] = {
                            'f1_scores': [],
                            'warning_reductions': []
                        }
                    
                    f1 = metrics.get('f1_weighted')
                    wr = metrics.get('warning_reduction')
                    
                    if f1 is not None:
                        model_aggregates[model_name]['f1_scores'].append(f1)
                    if wr is not None:
                        model_aggregates[model_name]['warning_reductions'].append(wr)
            
            f.write("### Average Metrics by Model\n\n")
            f.write("| Model | Avg F1 Score | Avg Warning Reduction |\n")
            f.write("|-------|--------------|---------------------|\n")
            
            for model_name, aggregates in sorted(model_aggregates.items()):
                avg_f1 = sum(aggregates['f1_scores']) / len(aggregates['f1_scores']) if aggregates['f1_scores'] else None
                avg_wr = sum(aggregates['warning_reductions']) / len(aggregates['warning_reductions']) if aggregates['warning_reductions'] else None
                
                f1_str = f"{avg_f1:.3f}" if avg_f1 is not None else "N/A"
                wr_str = f"{avg_wr:.1f}%" if avg_wr is not None else "N/A"
                
                f.write(f"| {model_name} | {f1_str} | {wr_str} |\n")
            
            f.write("\n")
    
    logger.info(f"Comprehensive report saved to {report_file}")
    return report_file


def main():
    """Main function."""
    logger.info("Starting complete evaluation pipeline for outline projects...")
    
    # Step 1: Prepare projects
    if not prepare_projects():
        logger.warning("Project preparation had issues, continuing anyway...")
    
    # Step 2-6: Run evaluation (includes warnings, slices, CFGs, predictions, metrics)
    results = run_evaluation()
    
    # Step 7: Generate comprehensive report
    report_file = generate_comprehensive_report(results)
    
    # Summary
    logger.info("=" * 80)
    logger.info("Evaluation Pipeline Complete")
    logger.info("=" * 80)
    
    successful = len([p for p, s in results.items() if s.get('status') == 'success'])
    no_warnings = len([p for p, s in results.items() if s.get('status') == 'no_warnings'])
    failed = len([p for p, s in results.items() if s.get('status') not in ['success', 'no_warnings']])
    
    logger.info(f"Projects Successfully Evaluated: {successful}/{len(OUTLINE_PROJECTS)}")
    logger.info(f"Projects with No Warnings: {no_warnings}/{len(OUTLINE_PROJECTS)}")
    logger.info(f"Projects Failed: {failed}/{len(OUTLINE_PROJECTS)}")
    logger.info(f"Report: {report_file}")
    
    # Return 0 if at least some projects were processed (even if no warnings)
    return 0 if (successful > 0 or no_warnings > 0) else 1


if __name__ == '__main__':
    exit(main())

