#!/usr/bin/env python3
"""
Remove checker annotations from Java source files for evaluation purposes.

This utility removes checker-specific annotations from Java files to generate
warnings for evaluation when projects have no warnings. This enables testing
the evaluation pipeline on projects that would otherwise have zero warnings.

Usage:
    python3 remove_annotations_for_evaluation.py \
        --project_path case_studies/guava \
        --checker lower_bound \
        --output_dir case_studies/guava_no_annotations
"""

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Optional
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Annotation sets for each checker
CHECKER_ANNOTATIONS = {
    'lower_bound': ['@Positive', '@NonNegative', '@GTENegativeOne'],
    'sql_quotes': ['@SqlEvenQuotes', '@SqlOddQuotes'],
    'signature_string': ['@FullyQualifiedName', '@BinaryName', '@FieldDescriptor'],
}

# Pattern to match annotations (with optional package prefix)
ANNOTATION_PATTERN = re.compile(
    r'@(?:[\w.]+\.)?(Positive|NonNegative|GTENegativeOne|SqlEvenQuotes|SqlOddQuotes|FullyQualifiedName|BinaryName|FieldDescriptor)\b'
)


def get_annotations_for_checker(checker_name: str) -> List[str]:
    """Get list of annotations for a specific checker."""
    checker_lower = checker_name.lower().replace(' ', '_')
    
    # Map common checker name variations
    if checker_lower in ['lower_bound', 'lowerbound', 'index']:
        return CHECKER_ANNOTATIONS['lower_bound']
    elif checker_lower in ['sql_quotes', 'sqlquotes', 'quotes']:
        return CHECKER_ANNOTATIONS['sql_quotes']
    elif checker_lower in ['signature_string', 'signaturestring', 'signature']:
        return CHECKER_ANNOTATIONS['signature_string']
    else:
        # Default to all annotations
        all_annotations = []
        for ann_list in CHECKER_ANNOTATIONS.values():
            all_annotations.extend(ann_list)
        return all_annotations


def remove_annotations_from_line(line: str, annotations_to_remove: Set[str]) -> tuple[str, bool]:
    """
    Remove annotations from a single line.
    
    Returns:
        (modified_line, was_modified)
    """
    original_line = line
    
    # Remove standalone annotation lines (e.g., "@Positive\n")
    stripped = line.strip()
    for annotation in annotations_to_remove:
        # Check if entire line is just the annotation (with optional whitespace)
        if stripped == annotation or stripped == f'{annotation}':
            return '', True
        
        # Remove annotation from line (with optional package prefix)
        # Pattern: @package.Annotation or @Annotation
        pattern = re.compile(r'@(?:[\w.]+\.)?' + re.escape(annotation.replace('@', '')) + r'\b\s*')
        line = pattern.sub('', line)
        
        # Also handle comment-style annotations: /*@Annotation*/
        comment_pattern = re.compile(r'/\*@' + re.escape(annotation.replace('@', '')) + r'\*/\s*')
        line = comment_pattern.sub('', line)
    
    # Clean up extra whitespace
    line = re.sub(r'\s+', ' ', line)
    line = line.strip()
    
    # If line became empty, return empty string
    if not line.strip():
        return '', True
    
    # Preserve original line ending
    if original_line.endswith('\n') and not line.endswith('\n'):
        line += '\n'
    
    was_modified = (line != original_line)
    return line, was_modified


def remove_annotations_from_file(java_file: Path, annotations_to_remove: Set[str], 
                                 output_file: Optional[Path] = None) -> Optional[Path]:
    """
    Remove checker annotations from a Java file.
    
    Args:
        java_file: Path to Java source file
        annotations_to_remove: Set of annotation names to remove (without @)
        output_file: Optional output file path (default: overwrite original)
    
    Returns:
        Path to modified file, or None if error
    """
    if not java_file.exists():
        logger.warning(f"Java file not found: {java_file}")
        return None
    
    try:
        # Read file
        with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Process lines (in reverse to handle line removal)
        modified_lines = []
        removed_count = 0
        
        for line in lines:
            modified_line, was_modified = remove_annotations_from_line(line, annotations_to_remove)
            
            # Only add non-empty lines
            if modified_line.strip() or not was_modified:
                modified_lines.append(modified_line)
            
            if was_modified:
                removed_count += 1
        
        # Write to output file
        if output_file is None:
            output_file = java_file
        
        # Create backup if overwriting original
        if output_file == java_file:
            backup_file = java_file.with_suffix('.java.backup')
            shutil.copy2(java_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} annotation(s) from {java_file.name}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error removing annotations from {java_file}: {e}")
        return None


def remove_annotations_from_project(project_path: Path, checker_name: str,
                                   output_dir: Optional[Path] = None,
                                   create_backup: bool = True) -> Dict[str, Path]:
    """
    Remove checker annotations from all Java files in a project.
    
    Args:
        project_path: Root directory of Java project
        checker_name: Name of checker (lower_bound, sql_quotes, signature_string)
        output_dir: Optional output directory (default: modify in place)
        create_backup: Whether to create backup of original files
    
    Returns:
        Dictionary mapping original file paths to modified file paths
    """
    annotations_to_remove = set(get_annotations_for_checker(checker_name))
    
    if not annotations_to_remove:
        logger.warning(f"No annotations found for checker: {checker_name}")
        return {}
    
    logger.info(f"Removing annotations for checker '{checker_name}': {sorted(annotations_to_remove)}")
    
    # Find all Java files
    java_files = list(project_path.rglob('*.java'))
    
    if not java_files:
        logger.warning(f"No Java files found in {project_path}")
        return {}
    
    logger.info(f"Found {len(java_files)} Java files")
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    modified_files = {}
    total_removed = 0
    
    for java_file in java_files:
        # Skip backup files
        if java_file.name.endswith('.backup'):
            continue
        
        # Determine output path
        if output_dir:
            # Preserve directory structure
            rel_path = java_file.relative_to(project_path)
            output_file = output_dir / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_file = java_file
        
        # Remove annotations
        result = remove_annotations_from_file(java_file, annotations_to_remove, output_file)
        
        if result:
            modified_files[str(java_file)] = result
            # Count removed annotations
            with open(java_file, 'r') as f:
                content = f.read()
                for ann in annotations_to_remove:
                    total_removed += content.count(f'@{ann}')
                    total_removed += content.count(f'/*@{ann}*/')
    
    logger.info(f"Processed {len(modified_files)} files, removed approximately {total_removed} annotations")
    
    return modified_files


def test_annotation_removal(test_file: Path, checker_name: str) -> bool:
    """
    Test annotation removal on a sample file and verify warnings are generated.
    
    Returns:
        True if test successful, False otherwise
    """
    logger.info(f"Testing annotation removal on {test_file}")
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / test_file.name
        
        # Copy file
        shutil.copy2(test_file, temp_file)
        
        # Remove annotations
        annotations_to_remove = set(get_annotations_for_checker(checker_name))
        result = remove_annotations_from_file(temp_file, annotations_to_remove, temp_file)
        
        if not result:
            logger.error("Failed to remove annotations")
            return False
        
        # Read original and modified files
        with open(test_file, 'r') as f:
            original_content = f.read()
        
        with open(temp_file, 'r') as f:
            modified_content = f.read()
        
        # Verify annotations were removed
        removed_count = 0
        for ann in annotations_to_remove:
            original_count = original_content.count(f'@{ann}')
            modified_count = modified_content.count(f'@{ann}')
            removed = original_count - modified_count
            if removed > 0:
                logger.info(f"  Removed {removed} occurrence(s) of @{ann}")
                removed_count += removed
        
        if removed_count == 0:
            logger.warning("No annotations were removed - file may not contain target annotations")
            return False
        
        logger.info(f"Successfully removed {removed_count} annotation(s)")
        logger.info("Test passed: Annotation removal works correctly")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Remove checker annotations from Java files for evaluation'
    )
    parser.add_argument('--project_path', type=Path, required=True,
                       help='Root directory of Java project')
    parser.add_argument('--checker', type=str, required=True,
                       choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       help='Checker name')
    parser.add_argument('--output_dir', type=Path, default=None,
                       help='Output directory (default: modify in place)')
    parser.add_argument('--test', action='store_true',
                       help='Test annotation removal on a sample file')
    parser.add_argument('--test_file', type=Path, default=None,
                       help='Specific file to test (if --test)')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        if args.test_file:
            test_file = args.test_file
        else:
            # Find a file with annotations
            java_files = list(args.project_path.rglob('*.java'))
            test_file = None
            for java_file in java_files[:10]:  # Check first 10 files
                with open(java_file, 'r') as f:
                    content = f.read()
                    annotations = get_annotations_for_checker(args.checker)
                    if any(f'@{ann}' in content for ann in annotations):
                        test_file = java_file
                        break
            
            if not test_file:
                logger.error("No files with annotations found for testing")
                return 1
        
        success = test_annotation_removal(test_file, args.checker)
        return 0 if success else 1
    
    # Remove annotations from project
    modified_files = remove_annotations_from_project(
        args.project_path,
        args.checker,
        args.output_dir
    )
    
    logger.info(f"Annotation removal complete. Modified {len(modified_files)} files.")
    
    if args.output_dir:
        logger.info(f"Modified files saved to: {args.output_dir}")
        logger.info("Original files unchanged.")
    else:
        logger.info("Original files modified. Backups created with .backup extension.")
    
    return 0


if __name__ == '__main__':
    exit(main())

