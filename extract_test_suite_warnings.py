#!/usr/bin/env python3
"""
Extract Warnings from Checker Framework Test Suites

This script extracts warnings from Checker Framework test suites by:
1. Temporarily removing checker-specific annotations from test files
2. Running the checker to generate warnings
3. Restoring original annotations
4. Aggregating warnings into output file

This is necessary because test suites are often fully annotated, preventing
warnings from being generated during normal checker execution.
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Set, Dict
import re

# Add GenDATA root to path
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
sys.path.insert(0, str(GEN_DATA_ROOT))

from checker_framework_runner import CheckerFrameworkRunner
from checker_evaluation_config import get_checker_config
from studies.remove_annotations_for_evaluation import (
    get_annotations_for_checker,
    remove_annotations_from_file
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_java_files_in_test_suite(test_suite_path: Path, 
                                  exclude_backups: bool = True) -> List[Path]:
    """
    Find all Java files in a test suite directory.
    
    Args:
        test_suite_path: Root directory of test suite
        exclude_backups: Whether to exclude backup files (.backup, .orig)
        
    Returns:
        List of Java file paths
    """
    java_files = []
    
    if not test_suite_path.exists():
        logger.warning(f"Test suite path does not exist: {test_suite_path}")
        return java_files
    
    for java_file in test_suite_path.rglob('*.java'):
        # Skip backup files
        if exclude_backups and (java_file.name.endswith('.backup') or 
                                 java_file.name.endswith('.orig')):
            continue
        
        java_files.append(java_file)
    
    return sorted(java_files)


def extract_warnings_from_test_suite(checker_name: str,
                                    test_suite_path: Path,
                                    output_file: Path,
                                    use_temp_dir: bool = True) -> bool:
    """
    Extract warnings from a test suite by removing annotations and running checker.
    
    Args:
        checker_name: Name of checker (lower_bound, sql_quotes, signature_string)
        test_suite_path: Root directory of test suite
        output_file: Path to output warning file
        use_temp_dir: Whether to use temporary directory for modified files
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Extracting warnings from test suite: {test_suite_path}")
    logger.info(f"Checker: {checker_name}")
    logger.info(f"Output: {output_file}")
    
    # Get annotations to remove
    annotations_to_remove = set(get_annotations_for_checker(checker_name))
    if not annotations_to_remove:
        logger.error(f"No annotations found for checker: {checker_name}")
        return False
    
    logger.info(f"Removing annotations: {sorted(annotations_to_remove)}")
    
    # Find Java files
    java_files = find_java_files_in_test_suite(test_suite_path)
    if not java_files:
        logger.warning(f"No Java files found in test suite: {test_suite_path}")
        return False
    
    logger.info(f"Found {len(java_files)} Java files")
    
    # Create temporary directory for modified files
    if use_temp_dir:
        temp_dir = tempfile.mkdtemp(prefix='gendata_test_extract_')
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Copy test suite structure to temp directory
        temp_test_suite = Path(temp_dir) / 'test_suite'
        shutil.copytree(test_suite_path, temp_test_suite, dirs_exist_ok=True)
        
        # Remove annotations from all files in temp directory
        temp_java_files = find_java_files_in_test_suite(temp_test_suite)
        modified_count = 0
        
        for java_file in temp_java_files:
            result = remove_annotations_from_file(
                java_file, 
                annotations_to_remove, 
                java_file
            )
            if result:
                modified_count += 1
        
        logger.info(f"Removed annotations from {modified_count}/{len(temp_java_files)} files")
        
        # Run checker on modified test suite
        project_root = str(temp_test_suite)
        cleanup_temp = True
    else:
        # Modify files in place (not recommended for test suites)
        logger.warning("Modifying test suite files in place - not recommended!")
        project_root = str(test_suite_path)
        cleanup_temp = False
        temp_dir = None
    
    try:
        # Initialize checker runner
        runner = CheckerFrameworkRunner(checker_name=checker_name)
        
        # Create temporary output file
        temp_output = Path(tempfile.mktemp(suffix='.out')) if use_temp_dir else output_file
        
        # Run checker
        logger.info(f"Running checker on modified test suite...")
        success = runner.run_checker_on_project(
            project_root=project_root,
            output_file=str(temp_output),
            max_files=None
        )
        
        if not success:
            logger.error("Failed to run checker on test suite")
            return False
        
        # Count warnings
        try:
            from checker_framework_runner import count_checker_warnings
            warning_count = count_checker_warnings(str(temp_output))
        except (AttributeError, ImportError):
            # Fallback: count lines that look like warnings
            warning_count = 0
            if temp_output.exists():
                with open(temp_output, 'r') as f:
                    for line in f:
                        if ':' in line and ('error:' in line or 'warning:' in line or 'compiler.' in line):
                            warning_count += 1
        logger.info(f"Generated {warning_count} warnings")
        
        # Convert paths in warnings file to be relative to original test suite
        if use_temp_dir and temp_output.exists():
            # Read warnings and adjust paths
            with open(temp_output, 'r') as f:
                warning_lines = f.readlines()
            
            # Adjust paths to be relative to original test suite
            adjusted_warnings = []
            test_suite_str = str(test_suite_path.resolve())
            
            for line in warning_lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    adjusted_warnings.append(line)
                    continue
                
                # Replace temp directory path with original test suite path
                if temp_dir in line:
                    line = line.replace(str(temp_test_suite.resolve()), test_suite_str)
                    # Make path relative to test suite root
                    line = re.sub(
                        rf'^{re.escape(test_suite_str)}/',
                        '',
                        line
                    )
                
                adjusted_warnings.append(line)
            
            # Write adjusted warnings to output file
            with open(output_file, 'w') as f:
                f.write(f"# Checker Framework {checker_name} Test Suite Warnings\n")
                f.write(f"# Test Suite: {test_suite_path}\n")
                f.write(f"# Extraction Method: Annotation Removal\n")
                f.write(f"# Total Warnings: {warning_count}\n")
                f.write(f"\n")
                for line in adjusted_warnings:
                    f.write(line + '\n')
        else:
            # Just copy temp output to final output
            if temp_output.exists():
                shutil.copy2(temp_output, output_file)
        
        return True
        
    except Exception as e:
        logger.error(f"Error extracting warnings: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
        
    finally:
        # Clean up temporary directory
        if cleanup_temp and temp_dir and Path(temp_dir).exists():
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract warnings from Checker Framework test suites'
    )
    parser.add_argument(
        '--checker',
        required=True,
        choices=['lower_bound', 'sql_quotes', 'signature_string'],
        help='Checker name'
    )
    parser.add_argument(
        '--test-suite',
        type=Path,
        required=True,
        help='Path to test suite directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output warning file path'
    )
    parser.add_argument(
        '--no-temp-dir',
        action='store_true',
        help='Modify files in place (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Validate test suite exists
    if not args.test_suite.exists():
        logger.error(f"Test suite not found: {args.test_suite}")
        return 1
    
    # Extract warnings
    success = extract_warnings_from_test_suite(
        checker_name=args.checker,
        test_suite_path=args.test_suite,
        output_file=args.output,
        use_temp_dir=not args.no_temp_dir
    )
    
    if success:
        logger.info(f"✅ Successfully extracted warnings to {args.output}")
        return 0
    else:
        logger.error(f"❌ Failed to extract warnings")
        return 1


if __name__ == '__main__':
    sys.exit(main())

