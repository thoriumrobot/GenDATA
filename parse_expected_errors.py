#!/usr/bin/env python3
"""
Parse Expected Error Comments from Checker Framework Test Files

This script parses `// :: error:` comments from Checker Framework test files
and generates warning entries in the standard format used by GenDATA.

The format is: file:line:column: level: [checker.message] message
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pattern to match error comments: // :: error: (error_type)
ERROR_COMMENT_PATTERN = re.compile(
    r'//\s*::\s*error:\s*\(([^)]+)\)'
)

# Map Checker Framework error types to checker message keys
ERROR_TYPE_MAPPING = {
    'assignment': 'assignment',
    'compound.assignment': 'compound.assignment',
    'array.length.negative': 'array.length.negative',
    'type.anno.before.modifier': 'type.anno.before.modifier',
    # Add more mappings as needed
}


def parse_error_comment(line: str, line_number: int) -> Optional[Tuple[str, int]]:
    """
    Parse an error comment from a line.
    
    Args:
        line: Line of code containing error comment
        line_number: Line number in file
        
    Returns:
        Tuple of (error_type, line_number) if found, None otherwise
    """
    match = ERROR_COMMENT_PATTERN.search(line)
    if match:
        error_type = match.group(1).strip()
        return (error_type, line_number)
    return None


def parse_expected_errors_from_file(java_file: Path, 
                                   test_suite_root: Path) -> List[Dict[str, any]]:
    """
    Parse expected error comments from a Java file.
    
    Args:
        java_file: Path to Java file
        test_suite_root: Root directory of test suite (for relative paths)
        
    Returns:
        List of warning dictionaries with file, line, column, level, message
    """
    warnings = []
    
    if not java_file.exists():
        logger.warning(f"File not found: {java_file}")
        return warnings
    
    try:
        with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Get relative path from test suite root
        try:
            rel_path = java_file.relative_to(test_suite_root)
        except ValueError:
            # File not under test suite root, use filename
            rel_path = Path(java_file.name)
        
        # Parse each line for error comments
        for line_num, line in enumerate(lines, start=1):
            error_info = parse_error_comment(line, line_num)
            if error_info:
                error_type, error_line = error_info
                
                # Map error type to checker message
                checker_message = ERROR_TYPE_MAPPING.get(
                    error_type, 
                    error_type.replace('.', '.')
                )
                
                # Generate warning entry
                warning = {
                    'file': str(rel_path),
                    'line': error_line,
                    'column': 0,  # Column not available from comments
                    'level': 'error',
                    'checker_message': checker_message,
                    'message': f'[{checker_message}] Expected error: {error_type}'
                }
                warnings.append(warning)
    
    except Exception as e:
        logger.error(f"Error parsing file {java_file}: {e}")
    
    return warnings


def format_warning_entry(warning: Dict[str, any]) -> str:
    """
    Format a warning dictionary into the standard warning file format.
    
    The slicer expects either:
    1. file:line:col: compiler.err.proc.messager: [checker] message (CF format)
    2. file:line: error/warning: message (simple format, no column)
    
    We'll use the simple format since we don't have accurate column numbers.
    """
    file_path = warning['file']
    line = warning['line']
    level = warning['level']  # 'error' or 'warning'
    checker_msg = warning['checker_message']
    message = warning.get('message', f'Expected error: {checker_msg}')
    
    # Use simple format: file:line: error/warning: message
    # This matches the slicer's simplePattern
    return f"{file_path}:{line}: {level}: [{checker_msg}] {message}"


def parse_expected_errors_from_test_suite(test_suite_path: Path,
                                        output_file: Path,
                                        checker_name: str) -> bool:
    """
    Parse expected errors from all Java files in a test suite.
    
    Args:
        test_suite_path: Root directory of test suite
        output_file: Path to output warning file
        checker_name: Name of checker (for header)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Parsing expected errors from test suite: {test_suite_path}")
    
    if not test_suite_path.exists():
        logger.error(f"Test suite not found: {test_suite_path}")
        return False
    
    # Find all Java files
    java_files = list(test_suite_path.rglob('*.java'))
    
    # Exclude backup files
    java_files = [f for f in java_files 
                  if not f.name.endswith('.backup') and 
                     not f.name.endswith('.orig')]
    
    if not java_files:
        logger.warning(f"No Java files found in test suite")
        return False
    
    logger.info(f"Found {len(java_files)} Java files")
    
    # Parse errors from all files
    all_warnings = []
    files_with_errors = 0
    
    for java_file in java_files:
        warnings = parse_expected_errors_from_file(java_file, test_suite_path)
        if warnings:
            all_warnings.extend(warnings)
            files_with_errors += 1
            logger.debug(f"Found {len(warnings)} expected errors in {java_file.name}")
    
    logger.info(f"Found {len(all_warnings)} expected errors in {files_with_errors} files")
    
    # Write warnings to output file
    try:
        with open(output_file, 'w') as f:
            f.write(f"# Checker Framework {checker_name} Test Suite Expected Errors\n")
            f.write(f"# Test Suite: {test_suite_path}\n")
            f.write(f"# Extraction Method: Error Comment Parsing\n")
            f.write(f"# Total Expected Errors: {len(all_warnings)}\n")
            f.write(f"\n")
            
            for warning in all_warnings:
                warning_line = format_warning_entry(warning)
                f.write(warning_line + '\n')
        
        logger.info(f"✅ Wrote {len(all_warnings)} expected errors to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse expected error comments from Checker Framework test files'
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
        '--checker',
        type=str,
        default='unknown',
        help='Checker name (for header)'
    )
    
    args = parser.parse_args()
    
    # Validate test suite exists
    if not args.test_suite.exists():
        logger.error(f"Test suite not found: {args.test_suite}")
        return 1
    
    # Parse expected errors
    success = parse_expected_errors_from_test_suite(
        test_suite_path=args.test_suite,
        output_file=args.output,
        checker_name=args.checker
    )
    
    if success:
        logger.info(f"✅ Successfully parsed expected errors")
        return 0
    else:
        logger.error(f"❌ Failed to parse expected errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())

