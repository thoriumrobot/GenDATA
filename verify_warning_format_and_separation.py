#!/usr/bin/env python3
"""
Verify Warning Format and Dataset Separation

This script verifies that:
1. Warning files generated match index1.out format
2. File paths in warnings are relative (not absolute)
3. Training pipeline can locate original Java files correctly
4. Datasets are stored in checker-specific directories
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')

# Expected warning format: file:line:column: level: [checker.message] message
WARNING_PATTERN = re.compile(r'^(.+?):(\d+):(\d+):\s*(.+?):\s*\[(.+?)\]\s*(.+)$')
WARNING_PATTERN_NO_COL = re.compile(r'^(.+?):(\d+):\s*(.+?):\s*\[(.+?)\]\s*(.+)$')


def verify_warning_format(warnings_file: Path, project_root: Path) -> Tuple[bool, List[str]]:
    """
    Verify that warning file format matches index1.out format.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    if not warnings_file.exists():
        issues.append(f"Warning file does not exist: {warnings_file}")
        return False, issues
    
    logger.info(f"Verifying warning format: {warnings_file}")
    
    with open(warnings_file, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        issues.append("Warning file is empty")
        return False, issues
    
    valid_warnings = 0
    invalid_lines = []
    absolute_paths = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Skip comments
        if line.startswith('#'):
            continue
        
        # Check if line matches warning format
        match = WARNING_PATTERN.match(line) or WARNING_PATTERN_NO_COL.match(line)
        if match:
            valid_warnings += 1
            file_path = match.group(1)
            
            # Check if path is absolute
            if os.path.isabs(file_path):
                absolute_paths.append((i, file_path))
            
            # Check if path exists relative to project_root
            relative_path = project_root / file_path
            if not relative_path.exists() and not os.path.isabs(file_path):
                # Try to find the file
                found = False
                for root, dirs, files in os.walk(project_root):
                    if file_path in files or file_path.endswith('.java') and os.path.basename(file_path) in files:
                        found = True
                        break
                if not found:
                    issues.append(f"Line {i}: File not found: {file_path} (relative to {project_root})")
        else:
            invalid_lines.append((i, line[:80]))  # First 80 chars
    
    if valid_warnings == 0:
        issues.append("No valid warnings found in file")
        return False, issues
    
    if absolute_paths:
        issues.append(f"Found {len(absolute_paths)} warnings with absolute paths (should be relative)")
        for line_num, path in absolute_paths[:5]:  # Show first 5
            issues.append(f"  Line {line_num}: {path}")
    
    if invalid_lines:
        issues.append(f"Found {len(invalid_lines)} lines that don't match warning format")
        for line_num, line_content in invalid_lines[:5]:  # Show first 5
            issues.append(f"  Line {line_num}: {line_content}")
    
    logger.info(f"Found {valid_warnings} valid warnings")
    
    if issues:
        return False, issues
    else:
        return True, []


def verify_dataset_separation() -> Tuple[bool, List[str]]:
    """
    Verify that datasets for different checkers are in separate directories.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    logger.info("Verifying dataset directory separation...")
    
    checkers = ['lower_bound', 'sql_quotes', 'signature_string']
    expected_dirs = {
        'slices': [f'slices_adaptive_specimin_{c}' for c in checkers],
        'cfg': [f'cfg_output_adaptive_specimin_{c}' for c in checkers],
        'augmented': [f'augmented_code_adaptive_{c}' for c in checkers],
    }
    
    # Check that directories exist and are separate
    for dir_type, dirs in expected_dirs.items():
        existing_dirs = []
        for dir_name in dirs:
            dir_path = GEN_DATA_ROOT / dir_name
            if dir_path.exists():
                existing_dirs.append(dir_name)
        
        if len(existing_dirs) > 1:
            logger.info(f"Found {len(existing_dirs)} {dir_type} directories: {existing_dirs}")
        elif len(existing_dirs) == 1:
            logger.info(f"Found 1 {dir_type} directory: {existing_dirs[0]}")
        else:
            logger.info(f"No {dir_type} directories found yet (this is OK if training hasn't started)")
    
    # Check for conflicts (shared directories)
    shared_dirs = [
        'slices_adaptive_specimin',
        'cfg_output_adaptive_specimin',
        'augmented_code_adaptive'
    ]
    
    conflicts = []
    for shared_dir in shared_dirs:
        shared_path = GEN_DATA_ROOT / shared_dir
        if shared_path.exists():
            conflicts.append(shared_dir)
            issues.append(f"Found shared directory: {shared_dir} (should be checker-specific)")
    
    if conflicts:
        logger.warning(f"Found {len(conflicts)} shared directories that may cause conflicts")
    else:
        logger.info("No shared directories found - good separation")
    
    if issues:
        return False, issues
    else:
        return True, []


def test_path_resolution(warnings_file: Path, project_root: Path) -> Tuple[bool, List[str]]:
    """
    Test that file paths in warnings can be resolved to actual Java files.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    logger.info(f"Testing path resolution for {warnings_file}")
    
    try:
        from lower_bound_checker import LowerBoundChecker
        checker = LowerBoundChecker()
        warnings = checker.parse_warnings(str(warnings_file))
    except Exception as e:
        issues.append(f"Failed to parse warnings: {e}")
        return False, issues
    
    if not warnings:
        issues.append("No warnings parsed from file")
        return False, issues
    
    logger.info(f"Parsed {len(warnings)} warnings")
    
    resolved_count = 0
    unresolved_count = 0
    
    for warning in warnings[:10]:  # Test first 10 warnings
        file_path = warning.get('file', '')
        line_num = warning.get('line', 0)
        
        # Try to resolve file path
        if os.path.isabs(file_path):
            full_path = Path(file_path)
        else:
            full_path = project_root / file_path
        
        if full_path.exists():
            resolved_count += 1
            logger.debug(f"✓ Resolved: {file_path} -> {full_path}")
        else:
            unresolved_count += 1
            issues.append(f"Cannot resolve: {file_path} (tried {full_path})")
    
    logger.info(f"Resolved {resolved_count}/{resolved_count + unresolved_count} test warnings")
    
    if unresolved_count > 0:
        return False, issues
    else:
        return True, []


def main():
    """Main verification function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify warning format and dataset separation')
    parser.add_argument('--warnings-file', type=Path, help='Warning file to verify (default: index1.out)')
    parser.add_argument('--project-root', type=Path, default='/home/ubuntu/checker-framework/checker/tests/index',
                       help='Project root directory')
    parser.add_argument('--check-format-only', action='store_true', help='Only check warning format')
    parser.add_argument('--check-separation-only', action='store_true', help='Only check dataset separation')
    
    args = parser.parse_args()
    
    warnings_file = args.warnings_file or (GEN_DATA_ROOT / 'index1.out')
    project_root = Path(args.project_root)
    
    logger.info("=" * 80)
    logger.info("Warning Format and Dataset Separation Verification")
    logger.info("=" * 80)
    
    all_passed = True
    
    if not args.check_separation_only:
        logger.info("\n1. Verifying Warning Format")
        logger.info("-" * 80)
        format_valid, format_issues = verify_warning_format(warnings_file, project_root)
        if format_valid:
            logger.info("✅ Warning format is valid")
        else:
            logger.error("❌ Warning format has issues:")
            for issue in format_issues:
                logger.error(f"  - {issue}")
            all_passed = False
        
        logger.info("\n2. Testing Path Resolution")
        logger.info("-" * 80)
        path_valid, path_issues = test_path_resolution(warnings_file, project_root)
        if path_valid:
            logger.info("✅ Path resolution works correctly")
        else:
            logger.error("❌ Path resolution has issues:")
            for issue in path_issues:
                logger.error(f"  - {issue}")
            all_passed = False
    
    if not args.check_format_only:
        logger.info("\n3. Verifying Dataset Separation")
        logger.info("-" * 80)
        separation_valid, separation_issues = verify_dataset_separation()
        if separation_valid:
            logger.info("✅ Dataset separation is correct")
        else:
            logger.warning("⚠️ Dataset separation has issues:")
            for issue in separation_issues:
                logger.warning(f"  - {issue}")
            # Don't fail on this, just warn
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ All verifications passed!")
        return 0
    else:
        logger.error("❌ Some verifications failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

