#!/usr/bin/env python3
"""
Verified Warning Tester

Wrapper around LowerBoundWarningTester that adds crash detection
and verification of checker execution results.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from test_lower_bound_warnings import LowerBoundWarningTester, WarningStats
from checker_crash_detector import (
    detect_checker_crash, 
    verify_checker_processed_files,
    analyze_checker_output,
    CheckerTimeoutError,
    CheckerCrashError
)
from verified_model_result import VerifiedCheckerResult

# Set up logging
logger = logging.getLogger(__name__)


class VerifiedWarningTester:
    """
    Wrapper that adds crash detection and verification to LowerBoundWarningTester.
    
    This class ensures that checker results are verified before being used,
    preventing false claims of 0 warnings when the checker actually crashed.
    """
    
    def __init__(self, 
                 checker_cp: Optional[str] = None,
                 temp_dir: Optional[str] = None,
                 timeout: int = 600,
                 save_outputs: bool = True,
                 output_dir: Optional[str] = None):
        """
        Initialize verified warning tester.
        
        Args:
            checker_cp: Checker Framework classpath
            temp_dir: Temporary directory for operations
            timeout: Timeout in seconds
            save_outputs: Whether to save checker outputs for debugging
            output_dir: Directory to save outputs (if save_outputs is True)
        """
        self.base_tester = LowerBoundWarningTester(
            checker_cp=checker_cp,
            temp_dir=temp_dir,
            timeout=timeout
        )
        
        self.save_outputs = save_outputs
        self.output_dir = Path(output_dir) if output_dir else Path('./verified_checker_outputs')
        if self.save_outputs:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized VerifiedWarningTester")
    
    def run_verified_checker(self, 
                            repo_dir: Path, 
                            java_files: List[str],
                            label: str = "checker_run") -> VerifiedCheckerResult:
        """
        Run the checker with verification of the result.
        
        Args:
            repo_dir: Repository/project directory
            java_files: List of Java files to check
            label: Label for output file (if saving)
            
        Returns:
            VerifiedCheckerResult with verification details
        """
        start_time = time.time()
        
        if not java_files:
            return VerifiedCheckerResult(
                checker_ran=False,
                crashed=False,
                crash_reason="No Java files provided",
                files_processed=False,
                warnings_verified=False,
                confidence=0.0,
                execution_time_seconds=0.0
            )
        
        try:
            # Run the checker using base tester
            success, output = self.base_tester.run_lower_bound_checker(repo_dir, java_files)
            returncode = 0 if success else 1  # Base tester doesn't expose returncode
            
        except Exception as e:
            logger.error(f"Exception running checker: {e}")
            return VerifiedCheckerResult(
                checker_ran=False,
                crashed=True,
                crash_reason=f"Exception during execution: {str(e)}",
                execution_time_seconds=time.time() - start_time,
                confidence=0.0
            )
        
        execution_time = time.time() - start_time
        
        # Save output if requested
        output_path = None
        if self.save_outputs and output:
            output_path = self._save_output(output, label)
        
        # Analyze the output - pass returncode to properly handle empty output
        crash_result, processing_result = analyze_checker_output(output, len(java_files), returncode=returncode)
        
        # Parse warnings if we think the checker ran successfully
        warning_count = 0
        error_count = 0
        if not crash_result.crashed and output:
            stats = self.base_tester.parse_warnings(output)
            warning_count = stats.total_warnings
            # Count actual checker warnings vs compilation errors
            checker_warnings = 0
            for line in stats.warning_lines:
                if '[index' in line.lower() or 'lowerbound' in line.lower():
                    checker_warnings += 1
            if checker_warnings > 0:
                warning_count = checker_warnings
            error_count = processing_result.error_count
        
        # Build verified result
        result = VerifiedCheckerResult(
            checker_ran=success and not crash_result.crashed,
            crashed=crash_result.crashed,
            crash_reason=crash_result.crash_reason,
            crash_indicators_found=crash_result.crash_indicators_found,
            has_stack_trace=crash_result.has_stack_trace,
            compilation_success=not crash_result.has_compilation_errors or crash_result.has_success_indicators,
            has_compilation_errors=crash_result.has_compilation_errors,
            files_processed=processing_result.files_processed,
            files_count=processing_result.files_mentioned,
            warnings_verified=processing_result.warnings_parsed or warning_count > 0,
            warning_count=warning_count,
            error_count=error_count,
            raw_output=output if len(output) < 100000 else output[:100000] + "\n... (truncated)",
            returncode=returncode,
            confidence=min(crash_result.confidence, processing_result.confidence),
            execution_time_seconds=execution_time
        )
        
        # Log the result
        if result.crashed:
            logger.warning(f"Checker CRASHED: {result.crash_reason}")
        elif not result.checker_ran:
            logger.warning(f"Checker did not run successfully")
        else:
            logger.info(f"Checker ran successfully: {result.warning_count} warnings, confidence={result.confidence:.2f}")
        
        return result
    
    def run_verified_checker_with_fallback(self,
                                          repo_dir: Path,
                                          java_files: List[str],
                                          label: str = "checker_run",
                                          fallback_warning_count: Optional[int] = None) -> VerifiedCheckerResult:
        """
        Run checker with verification, with optional fallback for unreliable results.
        
        Args:
            repo_dir: Repository directory
            java_files: Java files to check
            label: Label for output file
            fallback_warning_count: Fallback count if result is unreliable (still marked unverified)
            
        Returns:
            VerifiedCheckerResult
        """
        result = self.run_verified_checker(repo_dir, java_files, label)
        
        # If unreliable and we have a fallback, use it but mark as unverified
        if not result.is_reliable() and fallback_warning_count is not None:
            logger.warning(f"Using fallback warning count {fallback_warning_count} due to unreliable result")
            result.warning_count = fallback_warning_count
            # Keep verified as False and low confidence
            result.warnings_verified = False
        
        return result
    
    def get_verified_baseline_warnings(self,
                                       project_dir: Path,
                                       project_name: str,
                                       max_files: int = 50,
                                       fallback_count: Optional[int] = None) -> VerifiedCheckerResult:
        """
        Get verified baseline warnings for a project.
        
        Args:
            project_dir: Project directory
            project_name: Project name for labeling
            max_files: Maximum files to check
            fallback_count: Fallback count if unreliable
            
        Returns:
            VerifiedCheckerResult with baseline warnings
        """
        # Find Java files
        java_files = self.base_tester.find_java_files(project_dir, max_files=max_files)
        
        if not java_files:
            logger.warning(f"No Java files found in {project_dir}")
            result = VerifiedCheckerResult(
                checker_ran=False,
                crashed=False,
                crash_reason="No Java files found",
                files_processed=False,
                confidence=0.0
            )
            if fallback_count is not None:
                result.warning_count = fallback_count
            return result
        
        return self.run_verified_checker_with_fallback(
            project_dir, 
            java_files, 
            label=f"{project_name}_baseline",
            fallback_warning_count=fallback_count
        )
    
    def get_verified_post_placement_warnings(self,
                                            project_dir: Path,
                                            project_name: str,
                                            model_name: str,
                                            max_files: int = 50) -> VerifiedCheckerResult:
        """
        Get verified warnings after annotation placement.
        
        Args:
            project_dir: Project directory (with annotations placed)
            project_name: Project name
            model_name: Model name for labeling
            max_files: Maximum files to check
            
        Returns:
            VerifiedCheckerResult with post-placement warnings
        """
        # Find Java files
        java_files = self.base_tester.find_java_files(project_dir, max_files=max_files)
        
        if not java_files:
            return VerifiedCheckerResult(
                checker_ran=False,
                crashed=False,
                crash_reason="No Java files found after placement",
                files_processed=False,
                confidence=0.0
            )
        
        return self.run_verified_checker(
            project_dir, 
            java_files, 
            label=f"{project_name}_{model_name}_post_placement"
        )
    
    def _save_output(self, output: str, label: str) -> Path:
        """
        Save checker output to file.
        
        Args:
            output: Checker output string
            label: Label for the file
            
        Returns:
            Path to saved file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.txt"
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w') as f:
                f.write(output)
            return output_path
        except Exception as e:
            logger.warning(f"Failed to save output: {e}")
            return None
    
    # Delegate methods to base tester
    def find_java_files(self, repo_dir: Path, max_files: Optional[int] = None) -> List[str]:
        """Find Java files in repository"""
        return self.base_tester.find_java_files(repo_dir, max_files)
    
    def clone_repository(self, clone_url: str, project_name: str) -> Optional[Path]:
        """Clone a repository"""
        return self.base_tester.clone_repository(clone_url, project_name)


def verify_existing_result(output: str) -> VerifiedCheckerResult:
    """
    Verify an existing checker output string.
    
    Useful for re-verifying results from saved outputs.
    
    Args:
        output: Checker output string
        
    Returns:
        VerifiedCheckerResult
    """
    if not output:
        return VerifiedCheckerResult(
            checker_ran=False,
            crashed=True,
            crash_reason="Empty output",
            confidence=0.0
        )
    
    crash_result, processing_result = analyze_checker_output(output)
    
    return VerifiedCheckerResult(
        checker_ran=not crash_result.crashed,
        crashed=crash_result.crashed,
        crash_reason=crash_result.crash_reason,
        crash_indicators_found=crash_result.crash_indicators_found,
        has_stack_trace=crash_result.has_stack_trace,
        has_compilation_errors=crash_result.has_compilation_errors,
        files_processed=processing_result.files_processed,
        files_count=processing_result.files_mentioned,
        warnings_verified=processing_result.warnings_parsed,
        warning_count=processing_result.warning_count,
        error_count=processing_result.error_count,
        raw_output=output if len(output) < 100000 else output[:100000],
        confidence=min(crash_result.confidence, processing_result.confidence)
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run verified checker on a project')
    parser.add_argument('project_dir', help='Project directory')
    parser.add_argument('--max-files', type=int, default=50, help='Maximum files to check')
    parser.add_argument('--output-dir', default='./verified_checker_outputs', help='Output directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    tester = VerifiedWarningTester(output_dir=args.output_dir)
    
    project_dir = Path(args.project_dir)
    project_name = project_dir.name
    
    result = tester.get_verified_baseline_warnings(
        project_dir, 
        project_name, 
        max_files=args.max_files
    )
    
    print(f"\n{'='*60}")
    print(f"Project: {project_name}")
    print(f"{'='*60}")
    print(f"Checker ran: {result.checker_ran}")
    print(f"Crashed: {result.crashed}")
    if result.crash_reason:
        print(f"Crash reason: {result.crash_reason}")
    print(f"Files processed: {result.files_count}")
    print(f"Warning count: {result.warning_count}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Is reliable: {result.is_reliable()}")
    print(f"{'='*60}")
