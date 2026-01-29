#!/usr/bin/env python3
"""
WPI Comparison Script

Runs Checker Framework's Whole Program Inference (WPI) on test projects
and compares results with model-based annotation placement.

This script:
1. Manages backups of unannotated project copies
2. Runs WPI on each test project with the Index Checker
3. Collects and parses WPI results
4. Compares WPI results with model-based annotation placement results
"""

import os
import sys
import json
import shutil
import subprocess
import logging
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WPIResult:
    """Result of running WPI on a project"""
    project_name: str
    success: bool
    baseline_warnings: int
    after_wpi_warnings: int
    reduction_percentage: float
    annotations_inferred: int
    execution_time_seconds: float
    error_message: Optional[str] = None
    ajava_files: List[str] = None
    
    def __post_init__(self):
        if self.ajava_files is None:
            self.ajava_files = []


@dataclass
class ComparisonResult:
    """Comparison between WPI and model-based results"""
    project_name: str
    wpi_result: WPIResult
    model_baseline_warnings: int
    model_after_warnings: int
    model_reduction_percentage: float
    wpi_better: bool
    difference: float  # WPI reduction - Model reduction


class WPIComparison:
    """Runs WPI and compares with model-based annotation placement"""
    
    def __init__(self, 
                 base_dir: str = '/home/ubuntu/GenDATA/annotation_evaluation',
                 checker_framework: str = None,
                 timeout: int = 3600):
        """
        Initialize WPI comparison runner.
        
        Args:
            base_dir: Base directory for annotation evaluation
            checker_framework: Path to Checker Framework installation
            timeout: Timeout for WPI execution in seconds
        """
        self.base_dir = Path(base_dir)
        self.backups_dir = self.base_dir / 'backups'
        self.wpi_backups_dir = self.base_dir / 'wpi_backups'
        self.wpi_results_dir = self.base_dir / 'wpi_results'
        self.temp_repos_dir = self.base_dir / 'temp_repos'
        
        # Checker Framework configuration
        self.checker_framework = checker_framework or os.environ.get(
            'CHECKERFRAMEWORK', '/home/ubuntu/checker-framework'
        )
        self.wpi_script = Path(self.checker_framework) / 'checker' / 'bin' / 'wpi.sh'
        
        self.timeout = timeout
        
        # Projects to test
        self.projects = [
            'sortpom',
            'eclipse-external-annotations-m2e-plugin', 
            'pom-tuner'
        ]
        
        # Ensure directories exist
        self.wpi_backups_dir.mkdir(parents=True, exist_ok=True)
        self.wpi_results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"WPI Comparison initialized")
        logger.info(f"  Checker Framework: {self.checker_framework}")
        logger.info(f"  WPI Script: {self.wpi_script}")
        logger.info(f"  Timeout: {self.timeout}s")
    
    def verify_environment(self) -> Tuple[bool, str]:
        """Verify required environment is set up correctly"""
        issues = []
        
        # Check CHECKERFRAMEWORK
        if not Path(self.checker_framework).exists():
            issues.append(f"CHECKERFRAMEWORK directory not found: {self.checker_framework}")
        
        # Check wpi.sh exists
        if not self.wpi_script.exists():
            issues.append(f"wpi.sh not found: {self.wpi_script}")
        
        # Check JAVA_HOME
        java_home = os.environ.get('JAVA_HOME')
        if not java_home:
            issues.append("JAVA_HOME is not set")
        elif not Path(java_home).exists():
            issues.append(f"JAVA_HOME directory not found: {java_home}")
        
        # Check backups exist
        if not self.backups_dir.exists():
            issues.append(f"Backups directory not found: {self.backups_dir}")
        
        if issues:
            return False, "\n".join(issues)
        return True, "Environment verified successfully"
    
    def create_wpi_backup(self, project_name: str, force: bool = False) -> bool:
        """
        Create a clean backup copy for WPI testing.
        
        Args:
            project_name: Name of the project
            force: If True, overwrite existing backup
            
        Returns:
            True if backup was created successfully
        """
        source = self.backups_dir / project_name
        dest = self.wpi_backups_dir / project_name
        
        if not source.exists():
            logger.error(f"Source backup not found: {source}")
            return False
        
        if dest.exists():
            if force:
                logger.info(f"Removing existing WPI backup: {dest}")
                shutil.rmtree(dest)
            else:
                logger.info(f"WPI backup already exists: {dest}")
                return True
        
        logger.info(f"Creating WPI backup: {source} -> {dest}")
        shutil.copytree(source, dest)
        return True
    
    def run_wpi(self, project_name: str) -> WPIResult:
        """
        Run Whole Program Inference on a project.
        
        Args:
            project_name: Name of the project to run WPI on
            
        Returns:
            WPIResult with execution details
        """
        project_dir = self.wpi_backups_dir / project_name
        
        if not project_dir.exists():
            return WPIResult(
                project_name=project_name,
                success=False,
                baseline_warnings=0,
                after_wpi_warnings=0,
                reduction_percentage=0.0,
                annotations_inferred=0,
                execution_time_seconds=0.0,
                error_message=f"Project directory not found: {project_dir}"
            )
        
        logger.info(f"Running WPI on {project_name}...")
        logger.info(f"  Project directory: {project_dir}")
        
        # Build the WPI command
        # Use Index Checker for Lower Bound warnings
        # Add -b flag to skip style checks that may fail (editorconfig, spotless, etc.)
        extra_build_args = '-DskipTests -Deditorconfig.skip=true -Dspotless.check.skip=true -Denforcer.skip=true -Dcheckstyle.skip=true'
        
        cmd = [
            'bash',
            str(self.wpi_script),
            '-d', str(project_dir),
            '-t', str(self.timeout),
            '-b', extra_build_args,
            '--',
            '--checker', 'org.checkerframework.checker.index.IndexChecker'
        ]
        
        logger.info(f"  Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Set up environment
            env = os.environ.copy()
            env['CHECKERFRAMEWORK'] = self.checker_framework
            
            # Run WPI
            result = subprocess.run(
                cmd,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout + 60,  # Extra buffer for cleanup
                env=env
            )
            
            execution_time = time.time() - start_time
            
            # Save raw output
            output_file = self.wpi_results_dir / f"{project_name}_wpi_output.txt"
            with open(output_file, 'w') as f:
                f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                f.write(f"=== STDERR ===\n{result.stderr}\n\n")
                f.write(f"=== RETURN CODE: {result.returncode} ===\n")
            
            logger.info(f"  WPI completed in {execution_time:.1f}s")
            logger.info(f"  Return code: {result.returncode}")
            logger.info(f"  Output saved to: {output_file}")
            
            # Parse results
            wpi_result = self.parse_wpi_results(project_name, project_dir, execution_time)
            return wpi_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"  WPI timed out after {execution_time:.1f}s")
            return WPIResult(
                project_name=project_name,
                success=False,
                baseline_warnings=0,
                after_wpi_warnings=0,
                reduction_percentage=0.0,
                annotations_inferred=0,
                execution_time_seconds=execution_time,
                error_message=f"Timeout after {self.timeout}s"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"  WPI failed with error: {e}")
            return WPIResult(
                project_name=project_name,
                success=False,
                baseline_warnings=0,
                after_wpi_warnings=0,
                reduction_percentage=0.0,
                annotations_inferred=0,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
    def parse_wpi_results(self, project_name: str, project_dir: Path, 
                          execution_time: float) -> WPIResult:
        """
        Parse WPI results from output files.
        
        Args:
            project_name: Name of the project
            project_dir: Path to project directory
            execution_time: Time taken to run WPI
            
        Returns:
            WPIResult with parsed information
        """
        dljc_out = project_dir / 'dljc-out'
        typecheck_out = dljc_out / 'typecheck.out'
        wpi_stdout_log = dljc_out / 'wpi-stdout.log'
        
        baseline_warnings = 0
        after_wpi_warnings = 0
        annotations_inferred = 0
        ajava_files = []
        success = False
        error_message = None
        
        # Check for DLJC/Maven compatibility issues
        if wpi_stdout_log.exists():
            try:
                with open(wpi_stdout_log, 'r') as f:
                    log_content = f.read()
                
                # Check for "no source files" error - indicates DLJC couldn't capture files
                if "no source files" in log_content or "'java_files': []" in log_content:
                    error_message = "DLJC could not capture Java source files from Maven build. This is a known compatibility issue."
                    logger.warning(f"  {error_message}")
                    return WPIResult(
                        project_name=project_name,
                        success=False,
                        baseline_warnings=0,
                        after_wpi_warnings=0,
                        reduction_percentage=0.0,
                        annotations_inferred=0,
                        execution_time_seconds=execution_time,
                        error_message=error_message
                    )
            except Exception as e:
                logger.warning(f"  Error checking WPI log: {e}")
        
        # Check if dljc-out exists
        if not dljc_out.exists():
            error_message = "dljc-out directory not found - WPI may have failed"
            logger.warning(f"  {error_message}")
        else:
            # Parse typecheck.out for final warning count
            if typecheck_out.exists():
                try:
                    with open(typecheck_out, 'r') as f:
                        content = f.read()
                    
                    # Count warnings in final typecheck output
                    warning_pattern = re.compile(r'\.java:\d+: warning:', re.IGNORECASE)
                    warnings = warning_pattern.findall(content)
                    after_wpi_warnings = len(warnings)
                    
                    # Also check for summary line
                    summary_match = re.search(r'(\d+)\s+warning', content)
                    if summary_match:
                        after_wpi_warnings = int(summary_match.group(1))
                    
                    success = True
                    logger.info(f"  Parsed typecheck.out: {after_wpi_warnings} warnings")
                except Exception as e:
                    logger.warning(f"  Error parsing typecheck.out: {e}")
            
            # Parse wpi-stdout.log for more details
            if wpi_stdout_log.exists():
                try:
                    with open(wpi_stdout_log, 'r') as f:
                        content = f.read()
                    
                    # Look for iteration info and ajava file locations
                    ajava_pattern = re.compile(r'-Aajava=([^\s]+)')
                    ajava_matches = ajava_pattern.findall(content)
                    if ajava_matches:
                        for ajava_dir in ajava_matches:
                            if Path(ajava_dir).exists():
                                for ajava_file in Path(ajava_dir).rglob('*.ajava'):
                                    ajava_files.append(str(ajava_file))
                    
                    # Count annotations in ajava files
                    annotation_count = 0
                    for ajava_file in ajava_files:
                        try:
                            with open(ajava_file, 'r') as f:
                                ajava_content = f.read()
                            # Count annotation insertions
                            annotation_count += len(re.findall(
                                r'@(Positive|NonNegative|GTENegativeOne|IndexFor|LTLengthOf)',
                                ajava_content
                            ))
                        except:
                            pass
                    annotations_inferred = annotation_count
                    
                    logger.info(f"  Found {len(ajava_files)} ajava files with {annotations_inferred} annotations")
                except Exception as e:
                    logger.warning(f"  Error parsing wpi-stdout.log: {e}")
            
            # Try to find baseline warning count from first iteration
            iteration0_dir = dljc_out / 'iteration0'
            if iteration0_dir.exists():
                for log_file in iteration0_dir.glob('*.log'):
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                        warning_pattern = re.compile(r'\.java:\d+: warning:', re.IGNORECASE)
                        warnings = warning_pattern.findall(content)
                        if warnings:
                            baseline_warnings = max(baseline_warnings, len(warnings))
                    except:
                        pass
        
        # Calculate reduction
        if baseline_warnings > 0:
            reduction = ((baseline_warnings - after_wpi_warnings) / baseline_warnings) * 100
        else:
            reduction = 0.0
        
        return WPIResult(
            project_name=project_name,
            success=success,
            baseline_warnings=baseline_warnings,
            after_wpi_warnings=after_wpi_warnings,
            reduction_percentage=reduction,
            annotations_inferred=annotations_inferred,
            execution_time_seconds=execution_time,
            error_message=error_message,
            ajava_files=ajava_files
        )
    
    def run_direct_checker(self, project_name: str) -> Tuple[int, str]:
        """
        Run Checker Framework directly on project files (bypassing DLJC/WPI).
        
        This provides a baseline comparison when WPI can't run due to DLJC issues.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Tuple of (warning_count, output)
        """
        project_dir = self.wpi_backups_dir / project_name
        
        if not project_dir.exists():
            return 0, f"Project directory not found: {project_dir}"
        
        # Find Java files (excluding test directories)
        java_files = []
        exclude_dirs = {'test', 'tests', 'target', '.git', 'build'}
        
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        
        if not java_files:
            return 0, "No Java files found"
        
        logger.info(f"  Running direct checker on {len(java_files)} files...")
        
        # Build classpath
        checker_jar = Path(self.checker_framework) / 'checker' / 'dist' / 'checker.jar'
        if not checker_jar.exists():
            checker_jar = Path(self.checker_framework) / 'checker' / 'build' / 'libs' / 'checker.jar'
        
        # Run checker directly
        cmd = [
            'java',
            '-cp', str(checker_jar),
            'org.checkerframework.checker.index.IndexChecker',
        ]
        cmd.extend(java_files[:50])  # Limit to first 50 files for speed
        
        # Alternative: use javac directly
        javac_cmd = [
            str(Path(self.checker_framework) / 'checker' / 'bin' / 'javac'),
            '-processor', 'org.checkerframework.checker.index.IndexChecker',
            '-Xmaxwarns', '10000',
            '-proc:only',
        ]
        javac_cmd.extend(java_files[:50])
        
        try:
            result = subprocess.run(
                javac_cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(project_dir)
            )
            
            output = result.stdout + result.stderr
            
            # Count warnings
            warning_pattern = re.compile(r'\.java:\d+: warning:', re.IGNORECASE)
            warnings = warning_pattern.findall(output)
            
            return len(warnings), output[:5000]
            
        except subprocess.TimeoutExpired:
            return 0, "Timeout running checker"
        except Exception as e:
            return 0, f"Error: {e}"

    def load_model_results(self) -> Dict[str, Dict[str, Any]]:
        """Load model-based annotation placement results"""
        evaluation_report = self.base_dir.parent / 'annotation_evaluation' / 'evaluation_report.json'
        
        if not evaluation_report.exists():
            evaluation_report = self.base_dir / 'evaluation_report.json'
        
        if not evaluation_report.exists():
            logger.warning(f"Model evaluation report not found")
            return {}
        
        with open(evaluation_report, 'r') as f:
            data = json.load(f)
        
        results = {}
        for project in data.get('results', []):
            project_name = project.get('project_name')
            baseline = project.get('baseline_warnings', 0)
            
            # Get best model result (excluding GBT which failed)
            best_reduction = 0
            best_after = baseline
            for model_result in project.get('model_results', []):
                if model_result.get('placement_success'):
                    reduction = model_result.get('reduction_percentage', 0)
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_after = model_result.get('warnings_after', baseline)
            
            results[project_name] = {
                'baseline_warnings': baseline,
                'after_warnings': best_after,
                'reduction_percentage': best_reduction
            }
        
        return results
    
    def compare_results(self, wpi_results: List[WPIResult]) -> List[ComparisonResult]:
        """Compare WPI results with model results"""
        model_results = self.load_model_results()
        comparisons = []
        
        for wpi_result in wpi_results:
            project_name = wpi_result.project_name
            model_data = model_results.get(project_name, {})
            
            model_baseline = model_data.get('baseline_warnings', 0)
            model_after = model_data.get('after_warnings', 0)
            model_reduction = model_data.get('reduction_percentage', 0)
            
            # Use model baseline if WPI couldn't determine it
            if wpi_result.baseline_warnings == 0 and model_baseline > 0:
                wpi_result.baseline_warnings = model_baseline
                if wpi_result.success:
                    wpi_result.reduction_percentage = (
                        (model_baseline - wpi_result.after_wpi_warnings) / model_baseline * 100
                    )
            
            wpi_better = wpi_result.reduction_percentage > model_reduction
            difference = wpi_result.reduction_percentage - model_reduction
            
            comparison = ComparisonResult(
                project_name=project_name,
                wpi_result=wpi_result,
                model_baseline_warnings=model_baseline,
                model_after_warnings=model_after,
                model_reduction_percentage=model_reduction,
                wpi_better=wpi_better,
                difference=difference
            )
            comparisons.append(comparison)
        
        return comparisons
    
    def generate_report(self, comparisons: List[ComparisonResult]) -> None:
        """Generate comparison report files"""
        timestamp = datetime.now().isoformat()
        
        # JSON report
        json_report = {
            'metadata': {
                'timestamp': timestamp,
                'checker_framework': self.checker_framework,
                'timeout_seconds': self.timeout
            },
            'projects': []
        }
        
        for comparison in comparisons:
            project_data = {
                'name': comparison.project_name,
                'wpi_results': {
                    'success': comparison.wpi_result.success,
                    'baseline_warnings': comparison.wpi_result.baseline_warnings,
                    'after_wpi_warnings': comparison.wpi_result.after_wpi_warnings,
                    'reduction_percentage': comparison.wpi_result.reduction_percentage,
                    'annotations_inferred': comparison.wpi_result.annotations_inferred,
                    'execution_time_seconds': comparison.wpi_result.execution_time_seconds,
                    'error_message': comparison.wpi_result.error_message,
                    'ajava_files_count': len(comparison.wpi_result.ajava_files)
                },
                'model_results': {
                    'baseline_warnings': comparison.model_baseline_warnings,
                    'after_model_warnings': comparison.model_after_warnings,
                    'reduction_percentage': comparison.model_reduction_percentage
                },
                'comparison': {
                    'wpi_better': comparison.wpi_better,
                    'difference': comparison.difference
                }
            }
            json_report['projects'].append(project_data)
        
        json_path = self.wpi_results_dir / 'wpi_comparison_report.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved to: {json_path}")
        
        # Markdown report
        md_lines = [
            "# WPI vs Model-Based Annotation Placement Comparison",
            "",
            f"**Generated**: {timestamp}",
            "",
            "## Summary",
            "",
            "| Project | WPI Reduction | Model Reduction | Winner | Difference |",
            "|---------|---------------|-----------------|--------|------------|"
        ]
        
        for comparison in comparisons:
            winner = "WPI" if comparison.wpi_better else "Model"
            if comparison.wpi_result.error_message:
                winner = "N/A (WPI failed)"
            
            md_lines.append(
                f"| {comparison.project_name} | "
                f"{comparison.wpi_result.reduction_percentage:.1f}% | "
                f"{comparison.model_reduction_percentage:.1f}% | "
                f"{winner} | "
                f"{comparison.difference:+.1f}% |"
            )
        
        md_lines.extend([
            "",
            "## Detailed Results",
            ""
        ])
        
        for comparison in comparisons:
            md_lines.extend([
                f"### {comparison.project_name}",
                "",
                "**WPI Results**:",
                f"- Success: {comparison.wpi_result.success}",
                f"- Baseline warnings: {comparison.wpi_result.baseline_warnings}",
                f"- After WPI warnings: {comparison.wpi_result.after_wpi_warnings}",
                f"- Reduction: {comparison.wpi_result.reduction_percentage:.1f}%",
                f"- Annotations inferred: {comparison.wpi_result.annotations_inferred}",
                f"- Execution time: {comparison.wpi_result.execution_time_seconds:.1f}s",
            ])
            
            if comparison.wpi_result.error_message:
                md_lines.append(f"- Error: {comparison.wpi_result.error_message}")
            
            md_lines.extend([
                "",
                "**Model Results**:",
                f"- Baseline warnings: {comparison.model_baseline_warnings}",
                f"- After model warnings: {comparison.model_after_warnings}",
                f"- Reduction: {comparison.model_reduction_percentage:.1f}%",
                ""
            ])
        
        md_path = self.wpi_results_dir / 'wpi_comparison_report.md'
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        logger.info(f"Markdown report saved to: {md_path}")
    
    def run_all(self, force_backup: bool = False) -> List[ComparisonResult]:
        """
        Run WPI on all projects and generate comparison.
        
        Args:
            force_backup: If True, recreate WPI backups even if they exist
            
        Returns:
            List of comparison results
        """
        # Verify environment
        ok, message = self.verify_environment()
        if not ok:
            logger.error(f"Environment verification failed:\n{message}")
            return []
        logger.info(message)
        
        # Create backups
        logger.info("Creating WPI backups...")
        for project in self.projects:
            if not self.create_wpi_backup(project, force=force_backup):
                logger.error(f"Failed to create backup for {project}")
                return []
        
        # Run WPI on each project
        wpi_results = []
        for project in self.projects:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {project}")
            logger.info('='*60)
            
            result = self.run_wpi(project)
            wpi_results.append(result)
            
            logger.info(f"Result: success={result.success}, "
                       f"warnings={result.after_wpi_warnings}, "
                       f"annotations={result.annotations_inferred}")
        
        # Compare with model results
        logger.info(f"\n{'='*60}")
        logger.info("Comparing WPI and model results...")
        logger.info('='*60)
        
        comparisons = self.compare_results(wpi_results)
        
        # Generate reports
        self.generate_report(comparisons)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        for comparison in comparisons:
            status = "SUCCESS" if comparison.wpi_result.success else "FAILED"
            logger.info(f"{comparison.project_name}: {status}")
            if comparison.wpi_result.success:
                logger.info(f"  WPI: {comparison.wpi_result.reduction_percentage:.1f}% reduction")
                logger.info(f"  Model: {comparison.model_reduction_percentage:.1f}% reduction")
                winner = "WPI" if comparison.wpi_better else "Model"
                logger.info(f"  Winner: {winner} (diff: {comparison.difference:+.1f}%)")
        
        return comparisons


def main():
    parser = argparse.ArgumentParser(
        description='Run WPI comparison on test projects'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=3600,
        help='Timeout for WPI execution per project in seconds (default: 3600)'
    )
    parser.add_argument(
        '--force-backup', '-f',
        action='store_true',
        help='Force recreation of WPI backups'
    )
    parser.add_argument(
        '--project', '-p',
        type=str,
        help='Run on specific project only'
    )
    parser.add_argument(
        '--checker-framework',
        type=str,
        help='Path to Checker Framework installation'
    )
    
    args = parser.parse_args()
    
    # Create comparison runner
    wpi = WPIComparison(
        checker_framework=args.checker_framework,
        timeout=args.timeout
    )
    
    # Filter projects if specified
    if args.project:
        if args.project in wpi.projects:
            wpi.projects = [args.project]
        else:
            logger.error(f"Unknown project: {args.project}")
            logger.error(f"Available: {wpi.projects}")
            sys.exit(1)
    
    # Run comparison
    comparisons = wpi.run_all(force_backup=args.force_backup)
    
    if not comparisons:
        logger.error("No results generated")
        sys.exit(1)
    
    logger.info(f"\nResults saved to: {wpi.wpi_results_dir}")


if __name__ == '__main__':
    main()
