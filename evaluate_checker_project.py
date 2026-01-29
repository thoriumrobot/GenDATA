#!/usr/bin/env python3
"""
Evaluate Checker Project

This module provides shared functionality for evaluating GitHub Java projects
with different Checker Framework checkers (SQL Quotes, Signature String, etc.).
"""

import os
import subprocess
import logging
import shutil
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Checker Framework configuration
CHECKER_FRAMEWORK_HOME = '/home/ubuntu/checker-framework'
CHECKER_CP = f'{CHECKER_FRAMEWORK_HOME}/checker/dist/checker-qual.jar:{CHECKER_FRAMEWORK_HOME}/checker/dist/checker.jar'
CHECKER_JAVAC = f'{CHECKER_FRAMEWORK_HOME}/checker/bin/javac'

# Processor class names for each checker
CHECKER_PROCESSORS = {
    'lower_bound': 'org.checkerframework.checker.index.IndexChecker',
    'sql_quotes': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
    'signature_string': 'org.checkerframework.checker.signature.SignatureChecker',
}

@dataclass
class ProjectEvaluationResult:
    """Result of evaluating a project with a checker"""
    project_name: str
    project_url: str
    checker_name: str
    clone_success: bool
    build_system: Optional[str]
    compile_success: bool
    checker_success: bool
    warning_count: int
    compilation_error_count: int
    warnings_file: Optional[str]
    error_message: Optional[str]
    java_file_count: int


def clone_repository(repo_url: str, target_dir: Path, shallow: bool = True) -> bool:
    """Clone a GitHub repository"""
    try:
        cmd = ['git', 'clone']
        if shallow:
            cmd.extend(['--depth', '1'])
        cmd.extend([repo_url, str(target_dir)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info(f"Successfully cloned {repo_url}")
            return True
        else:
            logger.warning(f"Failed to clone {repo_url}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout cloning {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Error cloning {repo_url}: {e}")
        return False


def detect_build_system(project_dir: Path) -> Optional[str]:
    """Detect the build system used by the project"""
    if (project_dir / 'pom.xml').exists():
        return 'maven'
    elif (project_dir / 'build.gradle').exists() or (project_dir / 'build.gradle.kts').exists():
        return 'gradle'
    elif (project_dir / 'build.xml').exists():
        return 'ant'
    return None


def compile_project(project_dir: Path, build_system: str, timeout: int = 300) -> Tuple[bool, str]:
    """Attempt to compile the project"""
    try:
        if build_system == 'maven':
            cmd = ['mvn', 'compile', '-DskipTests', '-q']
        elif build_system == 'gradle':
            # Check for gradlew
            gradlew = project_dir / 'gradlew'
            if gradlew.exists():
                cmd = ['./gradlew', 'compileJava', '-x', 'test', '-q']
            else:
                cmd = ['gradle', 'compileJava', '-x', 'test', '-q']
        elif build_system == 'ant':
            cmd = ['ant', 'compile']
        else:
            return False, "Unknown build system"
        
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully compiled {project_dir.name}")
            return True, ""
        else:
            error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
            logger.warning(f"Compilation failed for {project_dir.name}: {error_msg}")
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except Exception as e:
        return False, str(e)


def find_java_files(project_dir: Path, exclude_dirs: Optional[List[str]] = None, max_files: int = 500) -> List[str]:
    """Find Java files in the project"""
    if exclude_dirs is None:
        exclude_dirs = [
            'test', 'tests', 'test-src', 'test-sources',
            'target', 'build', '.git', '.gradle', '.mvn',
            'benchmark', 'benchmarks', 'jmh',
            'example', 'examples', 'demo', 'demos',
            'generated', 'generated-sources'
        ]
    
    java_files = []
    
    for root, dirs, files in os.walk(project_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        # Skip paths containing excluded patterns
        root_lower = root.lower()
        if any(excluded in root_lower for excluded in ['benchmark', 'jmh', 'test']):
            continue
        
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
                if len(java_files) >= max_files:
                    return java_files
    
    return java_files


def run_checker(project_dir: Path, checker_name: str, output_file: Path, 
                max_files: int = 500, timeout: int = 300) -> Tuple[bool, int, int]:
    """
    Run a Checker Framework checker on the project
    
    Returns:
        Tuple of (success, warning_count, compilation_error_count)
    """
    processor = CHECKER_PROCESSORS.get(checker_name)
    if not processor:
        logger.error(f"Unknown checker: {checker_name}")
        return False, 0, 0
    
    java_files = find_java_files(project_dir, max_files=max_files)
    if not java_files:
        logger.warning(f"No Java files found in {project_dir}")
        return False, 0, 0
    
    logger.info(f"Running {checker_name} checker on {len(java_files)} Java files")
    
    # Build javac command using Checker Framework's javac wrapper
    # The wrapper handles Java module system restrictions automatically
    cmd = [
        CHECKER_JAVAC,
        '-processor', processor,
        '-Xmaxwarns', '1000',
        '-d', '/tmp/checker_output',
        '-sourcepath', str(project_dir),
        '-proc:only',  # Only run annotation processing, don't compile
    ]
    cmd.extend(java_files)
    
    try:
        os.makedirs('/tmp/checker_output', exist_ok=True)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_dir),
            timeout=timeout
        )
        
        # Parse warnings from stderr
        all_output = result.stderr + result.stdout
        
        # Count actual checker warnings vs compilation errors
        warning_count = 0
        compilation_error_count = 0
        warning_lines = []
        
        # Checker-specific warning patterns for each checker type
        checker_warning_patterns = {
            'sql_quotes': [
                '[assignment]', '[compound.assignment]', '[argument]', '[return]',
                '[SqlEvenQuotes]', '[SqlOddQuotes]', '[SqlQuotes'
            ],
            'signature_string': [
                '[argument]', '[assignment]', '[return]', '[override.return]',
                '[FullyQualifiedName]', '[BinaryName]', '[FieldDescriptor]',
                '[InternalForm]', '[ClassGetName]', '[SignatureName'
            ],
            'lower_bound': [
                '[array.access.unsafe.low]', '[array.length.negative]',
                '[assignment]', '[argument]', '[return]', '[Positive]', 
                '[NonNegative]', '[GTENegativeOne]', '[LowerBound'
            ]
        }
        
        # Get patterns for the current checker
        patterns_for_checker = checker_warning_patterns.get(checker_name, [])
        
        for line in all_output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a compilation error
            is_compilation_error = (
                'cannot find symbol' in line.lower() or
                ('package' in line.lower() and 'does not exist' in line.lower()) or
                'symbol:' in line.lower() or
                ('Annotation processor' in line and 'not found' in line)
            )
            
            # Check if this is a checker-specific warning
            is_checker_warning = False
            if ('error:' in line.lower() or 'warning:' in line.lower()) and '[' in line:
                # Check for generic warnings to exclude
                is_generic_warning = (
                    '[removal]' in line or '[deprecation]' in line or 
                    '[unchecked]' in line or '[rawtypes]' in line or
                    '[serial]' in line or '[cast]' in line or
                    '[path]' in line or '[options]' in line
                )
                
                if not is_generic_warning and not is_compilation_error:
                    # Check if it matches a checker-specific pattern
                    if patterns_for_checker:
                        for pattern in patterns_for_checker:
                            if pattern.lower() in line.lower():
                                is_checker_warning = True
                                break
                    else:
                        # If no specific patterns defined, count any non-generic warning
                        is_checker_warning = True
            
            if is_compilation_error:
                compilation_error_count += 1
            elif is_checker_warning:
                warning_count += 1
                warning_lines.append(line)
        
        # Write warnings to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(f"# {checker_name} Checker Warnings\n")
            f.write(f"# Project: {project_dir}\n")
            f.write(f"# Total Warnings: {warning_count}\n")
            f.write(f"# Compilation Errors: {compilation_error_count}\n\n")
            for line in warning_lines:
                f.write(line + '\n')
        
        logger.info(f"Found {warning_count} checker warnings, {compilation_error_count} compilation errors")
        return True, warning_count, compilation_error_count
        
    except subprocess.TimeoutExpired:
        logger.warning(f"Checker timeout for {project_dir}")
        return False, 0, 0
    except Exception as e:
        logger.error(f"Error running checker: {e}")
        return False, 0, 0


def evaluate_project(
    project_url: str,
    checker_name: str,
    work_dir: Path,
    case_studies_dir: Path,
    min_warnings: int = 5
) -> ProjectEvaluationResult:
    """
    Evaluate a GitHub project with a checker
    
    Args:
        project_url: GitHub repository URL
        checker_name: Name of checker to use
        work_dir: Temporary working directory
        case_studies_dir: Directory to save successful projects
        min_warnings: Minimum warnings required to consider project suitable
        
    Returns:
        ProjectEvaluationResult with evaluation details
    """
    # Extract project name from URL
    project_name = project_url.rstrip('/').split('/')[-1].replace('.git', '')
    
    result = ProjectEvaluationResult(
        project_name=project_name,
        project_url=project_url,
        checker_name=checker_name,
        clone_success=False,
        build_system=None,
        compile_success=False,
        checker_success=False,
        warning_count=0,
        compilation_error_count=0,
        warnings_file=None,
        error_message=None,
        java_file_count=0
    )
    
    project_dir = work_dir / project_name
    
    try:
        # Step 1: Clone repository
        logger.info(f"Evaluating {project_name} for {checker_name} checker...")
        
        if not clone_repository(project_url, project_dir):
            result.error_message = "Failed to clone repository"
            return result
        result.clone_success = True
        
        # Step 2: Detect build system
        result.build_system = detect_build_system(project_dir)
        if not result.build_system:
            result.error_message = "No supported build system found"
            return result
        
        # Step 3: Count Java files
        java_files = find_java_files(project_dir)
        result.java_file_count = len(java_files)
        
        if result.java_file_count == 0:
            result.error_message = "No Java files found"
            return result
        
        # Step 4: Compile project (optional - some projects may work without full compilation)
        compile_success, compile_error = compile_project(project_dir, result.build_system)
        result.compile_success = compile_success
        
        # Step 5: Run checker
        warnings_file = work_dir / f"{project_name}_{checker_name}_warnings.out"
        checker_success, warning_count, compilation_errors = run_checker(
            project_dir, checker_name, warnings_file
        )
        
        result.checker_success = checker_success
        result.warning_count = warning_count
        result.compilation_error_count = compilation_errors
        
        if checker_success and warning_count >= min_warnings:
            result.warnings_file = str(warnings_file)
            logger.info(f"SUCCESS: {project_name} has {warning_count} {checker_name} warnings")
        else:
            if warning_count < min_warnings:
                result.error_message = f"Only {warning_count} warnings (need {min_warnings}+)"
            
    except Exception as e:
        result.error_message = str(e)
        logger.error(f"Error evaluating {project_name}: {e}")
    
    return result


def save_successful_project(
    result: ProjectEvaluationResult,
    work_dir: Path,
    case_studies_dir: Path
) -> bool:
    """Save a successful project to the case studies directory"""
    if not result.warnings_file or result.warning_count < 5:
        return False
    
    project_name = result.project_name
    source_dir = work_dir / project_name
    target_dir = case_studies_dir / project_name
    
    try:
        # Copy project to case studies
        if target_dir.exists():
            logger.info(f"Project {project_name} already exists in case_studies")
        else:
            shutil.copytree(source_dir, target_dir)
            logger.info(f"Copied {project_name} to case_studies")
        
        # Copy warnings file
        source_warnings = Path(result.warnings_file)
        target_warnings = target_dir / f"{project_name}_{result.checker_name}_warnings.out"
        shutil.copy(source_warnings, target_warnings)
        
        logger.info(f"Saved {project_name} with {result.warning_count} warnings to {target_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving project {project_name}: {e}")
        return False


def main():
    """Test the evaluation module"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a project with a checker')
    parser.add_argument('project_url', help='GitHub repository URL')
    parser.add_argument('--checker', default='sql_quotes', 
                       choices=['sql_quotes', 'signature_string', 'lower_bound'],
                       help='Checker to use')
    parser.add_argument('--work-dir', default='/tmp/checker_eval',
                       help='Working directory')
    parser.add_argument('--min-warnings', type=int, default=5,
                       help='Minimum warnings required')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    case_studies_dir = Path('/home/ubuntu/GenDATA/case_studies')
    
    result = evaluate_project(
        args.project_url,
        args.checker,
        work_dir,
        case_studies_dir,
        args.min_warnings
    )
    
    print(f"\n{'='*60}")
    print(f"Project: {result.project_name}")
    print(f"Checker: {result.checker_name}")
    print(f"Clone Success: {result.clone_success}")
    print(f"Build System: {result.build_system}")
    print(f"Compile Success: {result.compile_success}")
    print(f"Checker Success: {result.checker_success}")
    print(f"Warning Count: {result.warning_count}")
    print(f"Compilation Errors: {result.compilation_error_count}")
    print(f"Java Files: {result.java_file_count}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    print(f"{'='*60}")
    
    if result.warning_count >= args.min_warnings:
        save_successful_project(result, work_dir, case_studies_dir)


if __name__ == '__main__':
    main()
