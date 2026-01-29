#!/usr/bin/env python3
"""
Verified Evaluation Wrapper

Wrapper around AnnotationPlacementEvaluator that adds verification
of checker results before claiming warning reduction.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from evaluate_annotation_placement import (
    AnnotationPlacementEvaluator,
    ModelEvaluationResult,
    ProjectEvaluationResult,
    EvaluationReport
)
from verified_warning_tester import VerifiedWarningTester
from verified_model_result import (
    VerifiedCheckerResult,
    VerifiedModelEvaluationResult,
    VerifiedProjectEvaluationResult,
    VerifiedEvaluationReport
)
from checker_evaluation_config import get_checker_config

# Set up logging
logger = logging.getLogger(__name__)


class VerifiedEvaluationWrapper:
    """
    Wrapper that adds verification to annotation placement evaluation.
    
    This ensures that warning reduction claims are verified by checking
    that the checker actually ran successfully (didn't crash) both
    before and after annotation placement.
    """
    
    def __init__(self,
                 work_dir: str = './annotation_evaluation_verified',
                 checker_cp: Optional[str] = None,
                 cfg_dir: Optional[str] = None,
                 timeout: int = 600,
                 save_checker_outputs: bool = True):
        """
        Initialize verified evaluation wrapper.
        
        Args:
            work_dir: Working directory for evaluation
            checker_cp: Checker Framework classpath
            cfg_dir: CFG directory
            timeout: Timeout for checker runs
            save_checker_outputs: Whether to save checker outputs
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base evaluator
        self.base_evaluator = AnnotationPlacementEvaluator(
            work_dir=str(self.work_dir / 'base'),
            checker_cp=checker_cp,
            cfg_dir=cfg_dir,
            timeout=timeout
        )
        
        # Initialize verified tester
        self.verified_tester = VerifiedWarningTester(
            checker_cp=checker_cp,
            temp_dir=str(self.work_dir / 'temp'),
            timeout=timeout,
            save_outputs=save_checker_outputs,
            output_dir=str(self.work_dir / 'checker_outputs')
        )
        
        # Get base models from config
        config = get_checker_config('lower_bound')
        self.base_models = config.get('base_models', 
            ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n'])
        
        logger.info(f"Initialized VerifiedEvaluationWrapper")
        logger.info(f"  Work directory: {self.work_dir}")
        logger.info(f"  Save outputs: {save_checker_outputs}")
    
    def evaluate_model_verified(self,
                               project_dir: Path,
                               backup_dir: Path,
                               project_name: str,
                               base_model: str,
                               baseline_result: VerifiedCheckerResult,
                               cfg_dir: Path) -> VerifiedModelEvaluationResult:
        """
        Evaluate a single model with verification.
        
        Args:
            project_dir: Project directory
            backup_dir: Backup directory
            project_name: Project name
            base_model: Base model to evaluate
            baseline_result: Verified baseline result
            cfg_dir: CFG directory
            
        Returns:
            VerifiedModelEvaluationResult
        """
        start_time = time.time()
        
        # Check baseline is valid
        if not baseline_result.is_reliable():
            return VerifiedModelEvaluationResult(
                base_model=base_model,
                verified=False,
                verification_error=f"Baseline not reliable: {baseline_result.crash_reason or 'low confidence'}",
                baseline_verified=False,
                baseline_checker_result=baseline_result,
                baseline_warnings=baseline_result.warning_count,
                evaluation_time_seconds=time.time() - start_time
            )
        
        baseline_warnings = baseline_result.warning_count
        
        # Restore project from backup
        try:
            self.base_evaluator.restore_project(project_dir, backup_dir)
        except Exception as e:
            return VerifiedModelEvaluationResult(
                base_model=base_model,
                verified=False,
                verification_error=f"Failed to restore project: {e}",
                baseline_verified=True,
                baseline_checker_result=baseline_result,
                baseline_warnings=baseline_warnings,
                error_message=str(e),
                evaluation_time_seconds=time.time() - start_time
            )
        
        # Generate predictions using base evaluator
        predictions_file = self.work_dir / 'predictions' / project_name / f'{base_model}_predictions.json'
        if not self.base_evaluator.generate_predictions(project_dir, cfg_dir, base_model, predictions_file):
            return VerifiedModelEvaluationResult(
                base_model=base_model,
                verified=False,
                verification_error="Failed to generate predictions",
                baseline_verified=True,
                baseline_checker_result=baseline_result,
                baseline_warnings=baseline_warnings,
                placement_success=False,
                error_message="Failed to generate predictions",
                evaluation_time_seconds=time.time() - start_time
            )
        
        # Place annotations
        placement_output = self.work_dir / 'placement_output' / project_name / base_model
        placement_stats = self.base_evaluator.place_annotations(project_dir, predictions_file, placement_output)
        annotations_placed = placement_stats.get('successful', 0)
        
        if annotations_placed == 0:
            return VerifiedModelEvaluationResult(
                base_model=base_model,
                verified=False,
                verification_error="No annotations placed",
                baseline_verified=True,
                baseline_checker_result=baseline_result,
                baseline_warnings=baseline_warnings,
                annotations_placed=0,
                placement_success=False,
                error_message="No annotations placed",
                evaluation_time_seconds=time.time() - start_time
            )
        
        # Run verified checker after placement
        post_result = self.verified_tester.get_verified_post_placement_warnings(
            project_dir,
            project_name,
            base_model
        )
        
        # Check if post-placement result is reliable
        if post_result.crashed:
            return VerifiedModelEvaluationResult(
                base_model=base_model,
                verified=False,
                verification_error=f"Post-placement checker crashed: {post_result.crash_reason}",
                baseline_verified=True,
                baseline_checker_result=baseline_result,
                post_placement_verified=False,
                post_placement_checker_result=post_result,
                baseline_warnings=baseline_warnings,
                warnings_after=None,  # Unknown - don't claim 0
                annotations_placed=annotations_placed,
                placement_success=True,
                compilation_success=False,
                evaluation_time_seconds=time.time() - start_time
            )
        
        if not post_result.is_reliable():
            return VerifiedModelEvaluationResult(
                base_model=base_model,
                verified=False,
                verification_error=f"Post-placement result unreliable (confidence: {post_result.confidence:.2f})",
                baseline_verified=True,
                baseline_checker_result=baseline_result,
                post_placement_verified=False,
                post_placement_checker_result=post_result,
                baseline_warnings=baseline_warnings,
                warnings_after=post_result.warning_count,  # Include but mark unverified
                annotations_placed=annotations_placed,
                placement_success=True,
                evaluation_time_seconds=time.time() - start_time
            )
        
        # Both baseline and post-placement are verified
        result = VerifiedModelEvaluationResult(
            base_model=base_model,
            verified=True,
            baseline_verified=True,
            baseline_checker_result=baseline_result,
            post_placement_verified=True,
            post_placement_checker_result=post_result,
            baseline_warnings=baseline_warnings,
            warnings_after=post_result.warning_count,
            annotations_placed=annotations_placed,
            placement_success=True,
            compilation_success=post_result.compilation_success,
            evaluation_time_seconds=time.time() - start_time
        )
        
        # Calculate reduction
        result.calculate_reduction()
        
        logger.info(f"Verified result for {base_model}: "
                   f"baseline={baseline_warnings}, after={post_result.warning_count}, "
                   f"reduction={result.warning_reduction} ({result.reduction_percentage:.1f}%)")
        
        return result
    
    def evaluate_project_verified(self, 
                                  project: Dict[str, Any]) -> VerifiedProjectEvaluationResult:
        """
        Evaluate all models for a project with verification.
        
        Args:
            project: Project dictionary
            
        Returns:
            VerifiedProjectEvaluationResult
        """
        start_time = time.time()
        project_name = project['project_name']
        project_url = project['project_url']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating project (verified): {project_name}")
        logger.info(f"{'='*80}")
        
        # Clone project
        project_dir = self.verified_tester.clone_repository(project_url, project_name)
        if not project_dir:
            return VerifiedProjectEvaluationResult(
                project_name=project_name,
                project_url=project_url,
                error_message="Failed to clone repository",
                evaluation_time_seconds=time.time() - start_time
            )
        
        # Get verified baseline warnings
        fallback_count = project.get('warning_count', 0)
        baseline_result = self.verified_tester.get_verified_baseline_warnings(
            project_dir,
            project_name,
            fallback_count=fallback_count
        )
        
        if baseline_result.crashed:
            return VerifiedProjectEvaluationResult(
                project_name=project_name,
                project_url=project_url,
                baseline_verified=False,
                baseline_verification_error=f"Baseline checker crashed: {baseline_result.crash_reason}",
                error_message="Baseline checker crashed",
                evaluation_time_seconds=time.time() - start_time
            )
        
        baseline_warnings = baseline_result.get_warning_count()
        if baseline_warnings is None:
            baseline_warnings = fallback_count
            baseline_verified = False
        else:
            baseline_verified = True
        
        # Backup project
        try:
            backup_dir = self.base_evaluator.backup_project(project_dir, project_name)
        except Exception as e:
            return VerifiedProjectEvaluationResult(
                project_name=project_name,
                project_url=project_url,
                baseline_warnings=baseline_warnings,
                baseline_verified=baseline_verified,
                error_message=f"Backup failed: {e}",
                evaluation_time_seconds=time.time() - start_time
            )
        
        # Find CFG directory
        cfg_dir = self.base_evaluator.find_cfg_directory(project_dir, project_name)
        if not cfg_dir:
            return VerifiedProjectEvaluationResult(
                project_name=project_name,
                project_url=project_url,
                baseline_warnings=baseline_warnings,
                baseline_verified=baseline_verified,
                error_message="CFG directory not found",
                evaluation_time_seconds=time.time() - start_time
            )
        
        # Evaluate each model
        model_results = []
        for base_model in self.base_models:
            logger.info(f"\nEvaluating {base_model} model (verified)")
            
            try:
                result = self.evaluate_model_verified(
                    project_dir,
                    backup_dir,
                    project_name,
                    base_model,
                    baseline_result,
                    cfg_dir
                )
                model_results.append(result)
                
                logger.info(f"  Verified: {result.verified}")
                logger.info(f"  Annotations: {result.annotations_placed}")
                logger.info(f"  Warnings after: {result.warnings_after}")
                if result.verified and result.warning_reduction is not None:
                    logger.info(f"  Reduction: {result.warning_reduction} ({result.reduction_percentage:.1f}%)")
                if result.verification_error:
                    logger.warning(f"  Verification error: {result.verification_error}")
                    
            except Exception as e:
                logger.error(f"Error evaluating {base_model}: {e}")
                model_results.append(VerifiedModelEvaluationResult(
                    base_model=base_model,
                    verified=False,
                    verification_error=f"Exception: {e}",
                    baseline_verified=baseline_verified,
                    baseline_warnings=baseline_warnings,
                    error_message=str(e)
                ))
        
        result = VerifiedProjectEvaluationResult(
            project_name=project_name,
            project_url=project_url,
            baseline_warnings=baseline_warnings,
            baseline_verified=baseline_verified,
            model_results=model_results,
            evaluation_time_seconds=time.time() - start_time
        )
        result.calculate_verification_status()
        
        return result
    
    def run_verified_evaluation(self,
                               candidates_file: str,
                               output_file: str) -> VerifiedEvaluationReport:
        """
        Run complete verified evaluation.
        
        Args:
            candidates_file: Path to project candidates JSON
            output_file: Path to save report
            
        Returns:
            VerifiedEvaluationReport
        """
        # Load projects
        projects = self.base_evaluator.load_qualifying_projects(candidates_file)
        
        if not projects:
            logger.error("No qualifying projects found")
            return VerifiedEvaluationReport(
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'verified': True,
                    'projects_evaluated': 0
                },
                results=[]
            )
        
        # Evaluate each project
        results = []
        for project in projects:
            result = self.evaluate_project_verified(project)
            results.append(result)
        
        # Create report
        report = VerifiedEvaluationReport(
            metadata={
                'timestamp': datetime.now().isoformat(),
                'verified': True,
                'projects_evaluated': len(results),
                'base_models_tested': len(self.base_models),
                'base_models': self.base_models
            },
            results=results
        )
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Also save legacy format
        legacy_path = output_path.with_suffix('.legacy.json')
        with open(legacy_path, 'w') as f:
            json.dump(report.to_legacy_dict(), f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Verified evaluation complete")
        logger.info(f"Report saved to: {output_file}")
        logger.info(f"Legacy format saved to: {legacy_path}")
        logger.info(f"{'='*80}")
        
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: VerifiedEvaluationReport):
        """Print evaluation summary"""
        logger.info("\nVerified Evaluation Summary:")
        
        total_verified = 0
        total_models = 0
        
        for result in report.results:
            logger.info(f"\n  {result.project_name}:")
            logger.info(f"    Baseline warnings: {result.baseline_warnings} (verified: {result.baseline_verified})")
            
            verified_count = result.get_verified_count()
            total_count = len(result.model_results)
            total_verified += verified_count
            total_models += total_count
            
            logger.info(f"    Verified models: {verified_count}/{total_count}")
            
            for model_result in result.model_results:
                status = "VERIFIED" if model_result.verified else "UNVERIFIED"
                if model_result.verified and model_result.reduction_percentage is not None:
                    logger.info(f"      {model_result.base_model}: {status} - "
                              f"{model_result.reduction_percentage:.1f}% reduction")
                else:
                    error = model_result.verification_error or "unknown error"
                    logger.info(f"      {model_result.base_model}: {status} - {error}")
        
        logger.info(f"\n  Total verified: {total_verified}/{total_models} "
                   f"({100*total_verified/total_models:.1f}% if total_models else 0}%)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run verified annotation placement evaluation')
    parser.add_argument('--candidates-file',
                       default='project_discovery_manual/lower_bound_project_candidates.json',
                       help='Path to project candidates JSON')
    parser.add_argument('--output-file',
                       default='annotation_evaluation_verified/verified_evaluation_report.json',
                       help='Path to save report')
    parser.add_argument('--work-dir',
                       default='./annotation_evaluation_verified',
                       help='Working directory')
    parser.add_argument('--checker-cp',
                       help='Checker Framework classpath')
    parser.add_argument('--cfg-dir',
                       help='CFG directory')
    parser.add_argument('--timeout',
                       type=int,
                       default=600,
                       help='Timeout for checker runs')
    parser.add_argument('--no-save-outputs',
                       action='store_true',
                       help='Do not save checker outputs')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    wrapper = VerifiedEvaluationWrapper(
        work_dir=args.work_dir,
        checker_cp=args.checker_cp,
        cfg_dir=args.cfg_dir,
        timeout=args.timeout,
        save_checker_outputs=not args.no_save_outputs
    )
    
    report = wrapper.run_verified_evaluation(args.candidates_file, args.output_file)
    
    return 0 if report.results else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
