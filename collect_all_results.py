#!/usr/bin/env python3
"""
Collect All Results

Aggregates evaluation results from:
- Model-based annotation placement (evaluate_all_checkers.py)
- WPI inference (run_wpi_all_checkers.py)
- Legacy evaluation scripts (evaluate_annotation_placement.py, run_placement_pipeline.py)

Generates unified comparison reports.

Usage:
    python collect_all_results.py
    python collect_all_results.py --output results_summary.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
ANNOTATION_EVAL_DIR = GEN_DATA_ROOT / 'annotation_evaluation'
WPI_OUTPUT_DIR = GEN_DATA_ROOT / 'wpi_output'


# Result file locations
RESULT_FILES = {
    'model_based': {
        'all_checkers': ANNOTATION_EVAL_DIR / 'evaluation_report_all_checkers.json',
        'lower_bound': ANNOTATION_EVAL_DIR / 'evaluation_report.json',
        'placement_pipeline': GEN_DATA_ROOT / 'placement_pipeline_results.json',
    },
    'wpi': {
        'all_checkers': WPI_OUTPUT_DIR / 'wpi_all_checkers_report.json',
        'comparison': WPI_OUTPUT_DIR / 'wpi_comparison_report.json',
    }
}


@dataclass
class ProjectResult:
    """Unified result for a project"""
    project_name: str
    checker_name: str
    method: str  # 'model_based' or 'wpi'
    baseline_warnings: int
    final_warnings: int
    reduction: int
    reduction_percentage: float
    source_file: str
    additional_info: Optional[Dict] = None


def load_json_file(path: Path) -> Optional[Dict]:
    """Load a JSON file if it exists"""
    if not path.exists():
        logger.debug(f"File not found: {path}")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None


def collect_model_based_results() -> List[ProjectResult]:
    """Collect model-based evaluation results"""
    results = []
    
    # Try all_checkers unified report first
    all_checkers_data = load_json_file(RESULT_FILES['model_based']['all_checkers'])
    if all_checkers_data and 'results' in all_checkers_data:
        for checker_name, checker_results in all_checkers_data['results'].items():
            for proj in checker_results:
                # Find best model result
                best_reduction = 0.0
                best_model = None
                model_results = proj.get('model_results', [])
                
                for mr in model_results:
                    reduction = mr.get('reduction_percentage', 0.0)
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_model = mr.get('base_model', 'unknown')
                
                results.append(ProjectResult(
                    project_name=proj.get('project_name', 'unknown'),
                    checker_name=checker_name,
                    method='model_based',
                    baseline_warnings=proj.get('baseline_warnings', 0),
                    final_warnings=int(proj.get('baseline_warnings', 0) * (1 - best_reduction / 100)),
                    reduction=int(proj.get('baseline_warnings', 0) * best_reduction / 100),
                    reduction_percentage=best_reduction,
                    source_file=str(RESULT_FILES['model_based']['all_checkers']),
                    additional_info={'best_model': best_model, 'model_count': len(model_results)}
                ))
    
    # Also check lower_bound legacy report
    lb_data = load_json_file(RESULT_FILES['model_based']['lower_bound'])
    if lb_data:
        # Handle different report formats
        if 'results' in lb_data:
            for proj_result in lb_data['results']:
                proj_name = proj_result.get('project_name', 'unknown')
                
                # Check if we already have this from all_checkers
                already_exists = any(
                    r.project_name == proj_name and r.checker_name == 'lower_bound' and r.method == 'model_based'
                    for r in results
                )
                
                if not already_exists:
                    best = proj_result.get('best_model', {})
                    results.append(ProjectResult(
                        project_name=proj_name,
                        checker_name='lower_bound',
                        method='model_based',
                        baseline_warnings=proj_result.get('baseline_warnings', 0),
                        final_warnings=best.get('warnings_after', 0),
                        reduction=best.get('warning_reduction', 0),
                        reduction_percentage=best.get('reduction_percentage', 0.0),
                        source_file=str(RESULT_FILES['model_based']['lower_bound']),
                        additional_info={'best_model': best.get('base_model', 'unknown')}
                    ))
    
    # Check placement pipeline results
    pipeline_data = load_json_file(RESULT_FILES['model_based']['placement_pipeline'])
    if pipeline_data:
        for checker_name, checker_results in pipeline_data.items():
            if isinstance(checker_results, dict) and 'projects' in checker_results:
                for proj_name, proj_data in checker_results['projects'].items():
                    # Check if we already have this
                    already_exists = any(
                        r.project_name == proj_name and r.checker_name == checker_name and r.method == 'model_based'
                        for r in results
                    )
                    
                    if not already_exists:
                        baseline = proj_data.get('baseline_warnings', 0)
                        after = proj_data.get('after_warnings', baseline)
                        reduction = baseline - after
                        reduction_pct = (reduction / baseline * 100) if baseline > 0 else 0.0
                        
                        results.append(ProjectResult(
                            project_name=proj_name,
                            checker_name=checker_name,
                            method='model_based',
                            baseline_warnings=baseline,
                            final_warnings=after,
                            reduction=reduction,
                            reduction_percentage=reduction_pct,
                            source_file=str(RESULT_FILES['model_based']['placement_pipeline']),
                            additional_info={'placement_method': proj_data.get('placement_method', 'heuristic')}
                        ))
    
    return results


def collect_wpi_results() -> List[ProjectResult]:
    """Collect WPI evaluation results"""
    results = []
    
    # Try all_checkers WPI report
    wpi_data = load_json_file(RESULT_FILES['wpi']['all_checkers'])
    if wpi_data and 'results' in wpi_data:
        for checker_name, checker_results in wpi_data['results'].items():
            for proj in checker_results:
                results.append(ProjectResult(
                    project_name=proj.get('project_name', 'unknown'),
                    checker_name=proj.get('checker_name', checker_name),
                    method='wpi',
                    baseline_warnings=proj.get('baseline_warnings', 0),
                    final_warnings=proj.get('after_wpi_warnings', 0),
                    reduction=proj.get('baseline_warnings', 0) - proj.get('after_wpi_warnings', 0),
                    reduction_percentage=proj.get('reduction_percentage', 0.0),
                    source_file=str(RESULT_FILES['wpi']['all_checkers']),
                    additional_info={
                        'iterations': proj.get('iterations', 0),
                        'ajava_files': proj.get('ajava_files_count', 0)
                    }
                ))
    
    # Check comparison report for additional results
    comparison_data = load_json_file(RESULT_FILES['wpi']['comparison'])
    if comparison_data and 'wpi_results' in comparison_data:
        for proj in comparison_data['wpi_results']:
            proj_name = proj.get('project', 'unknown')
            
            # Check if we already have this
            already_exists = any(
                r.project_name == proj_name and r.method == 'wpi'
                for r in results
            )
            
            if not already_exists:
                results.append(ProjectResult(
                    project_name=proj_name,
                    checker_name='lower_bound',  # Assume lower_bound for legacy comparison
                    method='wpi',
                    baseline_warnings=proj.get('baseline_warnings', 0),
                    final_warnings=proj.get('wpi_warnings', 0),
                    reduction=proj.get('baseline_warnings', 0) - proj.get('wpi_warnings', 0),
                    reduction_percentage=proj.get('wpi_reduction_pct', 0.0),
                    source_file=str(RESULT_FILES['wpi']['comparison'])
                ))
    
    return results


def generate_comparison_table(model_results: List[ProjectResult], 
                             wpi_results: List[ProjectResult]) -> Dict[str, List[Dict]]:
    """Generate comparison table between model-based and WPI"""
    comparison = {}
    
    # Group by project
    all_projects = set()
    for r in model_results + wpi_results:
        all_projects.add((r.project_name, r.checker_name))
    
    for project_name, checker_name in sorted(all_projects):
        key = f"{checker_name}"
        if key not in comparison:
            comparison[key] = []
        
        model_result = next(
            (r for r in model_results if r.project_name == project_name and r.checker_name == checker_name),
            None
        )
        wpi_result = next(
            (r for r in wpi_results if r.project_name == project_name and r.checker_name == checker_name),
            None
        )
        
        comparison[key].append({
            'project': project_name,
            'baseline_warnings': model_result.baseline_warnings if model_result else (wpi_result.baseline_warnings if wpi_result else 0),
            'model_reduction_pct': model_result.reduction_percentage if model_result else None,
            'model_best_model': model_result.additional_info.get('best_model') if model_result and model_result.additional_info else None,
            'wpi_reduction_pct': wpi_result.reduction_percentage if wpi_result else None,
            'wpi_iterations': wpi_result.additional_info.get('iterations') if wpi_result and wpi_result.additional_info else None,
            'winner': determine_winner(model_result, wpi_result),
        })
    
    return comparison


def determine_winner(model_result: Optional[ProjectResult], 
                    wpi_result: Optional[ProjectResult]) -> str:
    """Determine which method performed better"""
    if model_result is None and wpi_result is None:
        return 'no_data'
    if model_result is None:
        return 'wpi'
    if wpi_result is None:
        return 'model'
    
    if model_result.reduction_percentage > wpi_result.reduction_percentage + 1:
        return 'model'
    elif wpi_result.reduction_percentage > model_result.reduction_percentage + 1:
        return 'wpi'
    else:
        return 'tie'


def print_summary(model_results: List[ProjectResult], 
                 wpi_results: List[ProjectResult],
                 comparison: Dict[str, List[Dict]]) -> None:
    """Print human-readable summary"""
    print("\n" + "="*80)
    print("WARNING REDUCTION RESULTS SUMMARY")
    print("="*80)
    
    # Summary by checker
    for checker_name in sorted(comparison.keys()):
        print(f"\n{checker_name.upper()} CHECKER:")
        print("-" * 60)
        print(f"{'Project':<40} {'Model %':>10} {'WPI %':>10} {'Winner':>10}")
        print("-" * 60)
        
        for proj in comparison[checker_name]:
            model_pct = f"{proj['model_reduction_pct']:.1f}" if proj['model_reduction_pct'] is not None else "N/A"
            wpi_pct = f"{proj['wpi_reduction_pct']:.1f}" if proj['wpi_reduction_pct'] is not None else "N/A"
            print(f"{proj['project']:<40} {model_pct:>10} {wpi_pct:>10} {proj['winner']:>10}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    model_wins = sum(1 for c in comparison.values() for p in c if p['winner'] == 'model')
    wpi_wins = sum(1 for c in comparison.values() for p in c if p['winner'] == 'wpi')
    ties = sum(1 for c in comparison.values() for p in c if p['winner'] == 'tie')
    
    print(f"Model-based wins: {model_wins}")
    print(f"WPI wins: {wpi_wins}")
    print(f"Ties: {ties}")
    
    if model_results:
        avg_model = sum(r.reduction_percentage for r in model_results) / len(model_results)
        print(f"\nAverage model-based reduction: {avg_model:.1f}%")
    
    if wpi_results:
        avg_wpi = sum(r.reduction_percentage for r in wpi_results) / len(wpi_results)
        print(f"Average WPI reduction: {avg_wpi:.1f}%")


def save_results(model_results: List[ProjectResult],
                wpi_results: List[ProjectResult],
                comparison: Dict[str, List[Dict]],
                output_file: Path) -> None:
    """Save collected results to JSON"""
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'sources': {
                'model_based': [str(f) for f in RESULT_FILES['model_based'].values() if f.exists()],
                'wpi': [str(f) for f in RESULT_FILES['wpi'].values() if f.exists()],
            }
        },
        'model_based_results': [asdict(r) for r in model_results],
        'wpi_results': [asdict(r) for r in wpi_results],
        'comparison': comparison,
        'summary': {
            'model_based_count': len(model_results),
            'wpi_count': len(wpi_results),
            'avg_model_reduction': sum(r.reduction_percentage for r in model_results) / len(model_results) if model_results else 0,
            'avg_wpi_reduction': sum(r.reduction_percentage for r in wpi_results) / len(wpi_results) if wpi_results else 0,
        }
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Collect all evaluation results')
    parser.add_argument('--output', default='annotation_evaluation/all_results_summary.json',
                       help='Output file for collected results')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    logger.info("Collecting evaluation results...")
    
    # Collect results
    model_results = collect_model_based_results()
    logger.info(f"Found {len(model_results)} model-based results")
    
    wpi_results = collect_wpi_results()
    logger.info(f"Found {len(wpi_results)} WPI results")
    
    # Generate comparison
    comparison = generate_comparison_table(model_results, wpi_results)
    
    # Print summary
    if not args.quiet:
        print_summary(model_results, wpi_results, comparison)
    
    # Save results
    output_path = GEN_DATA_ROOT / args.output
    save_results(model_results, wpi_results, comparison, output_path)
    
    logger.info("\nResults collection complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
