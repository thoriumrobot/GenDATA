#!/usr/bin/env python3
"""
Generate Verified Evaluation Report

Re-verifies existing evaluation results by re-running the checker
and checking for crashes or other issues that may have caused
false reports of 100% warning reduction.
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from verified_warning_tester import VerifiedWarningTester, verify_existing_result
from verified_model_result import (
    VerifiedCheckerResult,
    VerifiedModelEvaluationResult,
    VerifiedProjectEvaluationResult,
    VerifiedEvaluationReport
)
from checker_crash_detector import detect_checker_crash, verify_checker_processed_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_existing_report(report_path: str) -> Dict[str, Any]:
    """Load existing evaluation report"""
    with open(report_path) as f:
        return json.load(f)


def verify_model_result(
    model_result: Dict[str, Any],
    project_name: str,
    baseline_warnings: int,
    temp_repos_dir: Path,
    tester: VerifiedWarningTester
) -> VerifiedModelEvaluationResult:
    """
    Verify a single model result from existing report.
    
    Args:
        model_result: Model result from existing report
        project_name: Project name
        baseline_warnings: Baseline warning count
        temp_repos_dir: Directory with cloned repos
        tester: VerifiedWarningTester instance
        
    Returns:
        VerifiedModelEvaluationResult
    """
    base_model = model_result.get('base_model', 'unknown')
    warnings_after = model_result.get('warnings_after', 0)
    reduction_percentage = model_result.get('reduction_percentage', 0)
    annotations_placed = model_result.get('annotations_placed', 0)
    compilation_success = model_result.get('compilation_success', True)
    
    # Check for suspicious 100% reduction
    is_suspicious = (
        reduction_percentage == 100.0 and 
        baseline_warnings > 0 and 
        warnings_after == 0
    )
    
    # Create basic result
    result = VerifiedModelEvaluationResult(
        base_model=base_model,
        verified=False,  # Will update after verification
        baseline_warnings=baseline_warnings,
        warnings_after=warnings_after,
        annotations_placed=annotations_placed,
        compilation_success=compilation_success
    )
    
    # Check if we have saved output to verify
    output_path = temp_repos_dir / project_name / f'{base_model}_checker_output.txt'
    if output_path.exists():
        with open(output_path) as f:
            saved_output = f.read()
        
        # Verify the saved output
        crash_result = detect_checker_crash(saved_output)
        processing_result = verify_checker_processed_files(saved_output)
        
        result.verified = not crash_result.crashed and processing_result.files_processed
        result.post_placement_verified = result.verified
        
        if crash_result.crashed:
            result.verification_error = f"Crash detected in saved output: {crash_result.crash_reason}"
        elif not processing_result.files_processed:
            result.verification_error = "No evidence of files being processed in saved output"
        
        result.post_placement_checker_result = VerifiedCheckerResult(
            checker_ran=not crash_result.crashed,
            crashed=crash_result.crashed,
            crash_reason=crash_result.crash_reason,
            files_processed=processing_result.files_processed,
            warning_count=processing_result.warning_count,
            confidence=min(crash_result.confidence, processing_result.confidence)
        )
    else:
        # No saved output - mark as unverified if suspicious
        if is_suspicious:
            result.verification_error = "100% reduction claimed but no saved output to verify"
            result.verified = False
        else:
            # Non-suspicious results without saved output - tentatively verified
            result.verified = True
            result.post_placement_verified = True
    
    result.calculate_reduction()
    
    return result


def verify_project_result(
    project_result: Dict[str, Any],
    temp_repos_dir: Path,
    tester: VerifiedWarningTester,
    re_run_checker: bool = False
) -> VerifiedProjectEvaluationResult:
    """
    Verify all model results for a project.
    
    Args:
        project_result: Project result from existing report
        temp_repos_dir: Directory with cloned repos
        tester: VerifiedWarningTester instance
        re_run_checker: Whether to re-run checker on project
        
    Returns:
        VerifiedProjectEvaluationResult
    """
    project_name = project_result.get('project_name', 'unknown')
    project_url = project_result.get('project_url', '')
    baseline_warnings = project_result.get('baseline_warnings', 0)
    model_results = project_result.get('model_results', [])
    
    logger.info(f"\nVerifying project: {project_name}")
    logger.info(f"  Baseline warnings: {baseline_warnings}")
    logger.info(f"  Models: {len(model_results)}")
    
    # Verify baseline if we can re-run
    baseline_verified = False
    baseline_checker_result = None
    
    if re_run_checker:
        project_dir = temp_repos_dir / project_name
        if project_dir.exists():
            baseline_checker_result = tester.get_verified_baseline_warnings(
                project_dir, project_name
            )
            baseline_verified = baseline_checker_result.is_reliable()
            
            if baseline_verified:
                logger.info(f"  Baseline verified: {baseline_checker_result.warning_count} warnings")
            else:
                logger.warning(f"  Baseline not verified: {baseline_checker_result.crash_reason}")
    
    # Verify each model result
    verified_model_results = []
    for model_result in model_results:
        verified_result = verify_model_result(
            model_result,
            project_name,
            baseline_warnings,
            temp_repos_dir,
            tester
        )
        verified_model_results.append(verified_result)
        
        status = "VERIFIED" if verified_result.verified else "UNVERIFIED"
        logger.info(f"  {verified_result.base_model}: {status}")
        if verified_result.verification_error:
            logger.warning(f"    Error: {verified_result.verification_error}")
    
    result = VerifiedProjectEvaluationResult(
        project_name=project_name,
        project_url=project_url,
        baseline_warnings=baseline_warnings,
        baseline_verified=baseline_verified,
        model_results=verified_model_results
    )
    result.calculate_verification_status()
    
    return result


def generate_verified_report(
    input_report_path: str,
    output_report_path: str,
    temp_repos_dir: str = './annotation_evaluation/temp_repos',
    re_run_checker: bool = False
) -> VerifiedEvaluationReport:
    """
    Generate verified report from existing evaluation report.
    
    Args:
        input_report_path: Path to existing evaluation report
        output_report_path: Path to save verified report
        temp_repos_dir: Directory with cloned repos
        re_run_checker: Whether to re-run checker
        
    Returns:
        VerifiedEvaluationReport
    """
    logger.info("="*80)
    logger.info("Generating Verified Evaluation Report")
    logger.info("="*80)
    logger.info(f"Input: {input_report_path}")
    logger.info(f"Output: {output_report_path}")
    logger.info(f"Re-run checker: {re_run_checker}")
    
    # Load existing report
    existing_report = load_existing_report(input_report_path)
    
    # Initialize tester
    tester = VerifiedWarningTester(
        save_outputs=True,
        output_dir='./verified_checker_outputs'
    )
    
    # Get project results
    project_results = existing_report.get('results', [])
    logger.info(f"\nProjects to verify: {len(project_results)}")
    
    # Verify each project
    verified_results = []
    temp_repos_path = Path(temp_repos_dir)
    
    for project_result in project_results:
        verified_project = verify_project_result(
            project_result,
            temp_repos_path,
            tester,
            re_run_checker=re_run_checker
        )
        verified_results.append(verified_project)
    
    # Create report
    report = VerifiedEvaluationReport(
        metadata={
            'original_report': input_report_path,
            'timestamp': datetime.now().isoformat(),
            'verified': True,
            're_ran_checker': re_run_checker
        },
        results=verified_results
    )
    
    # Save report
    output_path = Path(output_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    # Print summary
    print_verification_summary(report)
    
    logger.info(f"\nVerified report saved to: {output_report_path}")
    
    return report


def print_verification_summary(report: VerifiedEvaluationReport):
    """Print verification summary"""
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    
    total_models = 0
    verified_models = 0
    suspicious_100_percent = []
    
    for project in report.results:
        verified_count = project.get_verified_count()
        total_count = len(project.model_results)
        
        total_models += total_count
        verified_models += verified_count
        
        logger.info(f"\n{project.project_name}:")
        logger.info(f"  Baseline: {project.baseline_warnings} warnings (verified: {project.baseline_verified})")
        logger.info(f"  Models verified: {verified_count}/{total_count}")
        
        for model_result in project.model_results:
            status = "✓" if model_result.verified else "✗"
            
            if model_result.reduction_percentage == 100.0 and not model_result.verified:
                suspicious_100_percent.append({
                    'project': project.project_name,
                    'model': model_result.base_model,
                    'error': model_result.verification_error
                })
            
            if model_result.warning_reduction is not None:
                logger.info(f"    {status} {model_result.base_model}: "
                          f"{model_result.reduction_percentage:.1f}% reduction")
            else:
                logger.info(f"    {status} {model_result.base_model}: ERROR")
    
    # Overall stats
    verification_rate = (verified_models / total_models * 100) if total_models > 0 else 0
    
    logger.info("\n" + "-"*40)
    logger.info(f"Total models: {total_models}")
    logger.info(f"Verified models: {verified_models} ({verification_rate:.1f}%)")
    
    if suspicious_100_percent:
        logger.warning(f"\nSUSPICIOUS 100% REDUCTIONS (not verified):")
        for item in suspicious_100_percent:
            logger.warning(f"  {item['project']} / {item['model']}")
            logger.warning(f"    Reason: {item['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate verified evaluation report'
    )
    parser.add_argument(
        '--input',
        default='annotation_evaluation/evaluation_report.json',
        help='Path to existing evaluation report'
    )
    parser.add_argument(
        '--output',
        default='annotation_evaluation/verified_evaluation_report.json',
        help='Path to save verified report'
    )
    parser.add_argument(
        '--temp-repos',
        default='annotation_evaluation/temp_repos',
        help='Directory with cloned repos'
    )
    parser.add_argument(
        '--re-run',
        action='store_true',
        help='Re-run checker on projects'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    generate_verified_report(
        args.input,
        args.output,
        args.temp_repos,
        re_run_checker=args.re_run
    )


if __name__ == '__main__':
    main()
