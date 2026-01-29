#!/usr/bin/env python3
"""
Annotate Project for Training

This script injects Checker Framework type annotations into Java projects
to create warnings that can be reduced by adding more annotations.

The key constraint is that warnings must be solvable by ADDING annotations,
not removing them. This is achieved by:
1. Adding annotations to "entry point" method parameters
2. Leaving internal variables/fields unannotated
3. The model learns to add annotations to the unannotated locations
"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Checker Framework configuration
CHECKER_FRAMEWORK_HOME = '/home/ubuntu/checker-framework'
CHECKER_JAVAC = f'{CHECKER_FRAMEWORK_HOME}/checker/bin/javac'

# Annotation imports for each checker
CHECKER_IMPORTS = {
    'sql_quotes': [
        'org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes',
        'org.checkerframework.checker.sqlquotes.qual.SqlOddQuotes',
        'org.checkerframework.checker.sqlquotes.qual.SqlQuotesUnknown',
    ],
    'signature_string': [
        'org.checkerframework.checker.signature.qual.BinaryName',
        'org.checkerframework.checker.signature.qual.FullyQualifiedName',
        'org.checkerframework.checker.signature.qual.ClassGetName',
        'org.checkerframework.checker.signature.qual.FieldDescriptor',
        'org.checkerframework.checker.signature.qual.InternalForm',
    ],
}

# Checker processors
CHECKER_PROCESSORS = {
    'sql_quotes': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
    'signature_string': 'org.checkerframework.checker.signature.SignatureChecker',
}


@dataclass
class AnnotationPattern:
    """Pattern for where to inject annotations"""
    file_pattern: str  # Regex for file paths
    method_pattern: str  # Regex for method signatures
    param_name: str  # Parameter name to annotate
    annotation: str  # Annotation to add (without @)


@dataclass
class InjectionResult:
    """Result of annotation injection"""
    file_path: str
    line_number: int
    original_line: str
    modified_line: str
    annotation: str


def load_patterns(config_file: Path) -> Dict[str, List[AnnotationPattern]]:
    """Load annotation patterns from configuration file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    patterns = {}
    for checker, pattern_list in config.items():
        patterns[checker] = [
            AnnotationPattern(**p) for p in pattern_list
        ]
    
    return patterns


def add_import_to_file(file_path: Path, imports: List[str]) -> bool:
    """Add import statements to a Java file if not already present"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check which imports are missing
    missing_imports = []
    for imp in imports:
        import_stmt = f'import {imp};'
        if import_stmt not in content:
            missing_imports.append(import_stmt)
    
    if not missing_imports:
        return False  # No changes needed
    
    # Find the position to insert imports (after package statement, before first import or class)
    lines = content.split('\n')
    insert_pos = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('package '):
            insert_pos = i + 1
            # Skip empty lines after package
            while insert_pos < len(lines) and not lines[insert_pos].strip():
                insert_pos += 1
            break
    
    # If there are existing imports, insert before them
    for i, line in enumerate(lines[insert_pos:], insert_pos):
        stripped = line.strip()
        if stripped.startswith('import '):
            insert_pos = i
            break
        elif stripped.startswith('public ') or stripped.startswith('class ') or stripped.startswith('@'):
            # Insert before class definition
            insert_pos = i
            break
    
    # Insert the missing imports
    for imp in missing_imports:
        lines.insert(insert_pos, imp)
        insert_pos += 1
    
    # Add a blank line after imports if needed
    if insert_pos < len(lines) and lines[insert_pos].strip() and not lines[insert_pos].strip().startswith('import'):
        lines.insert(insert_pos, '')
    
    # Write back
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Added {len(missing_imports)} imports to {file_path}")
    return True


def find_method_and_annotate_param(
    file_path: Path,
    method_pattern: str,
    param_name: str,
    annotation: str
) -> List[InjectionResult]:
    """Find methods matching pattern and annotate specified parameter"""
    results = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    method_regex = re.compile(method_pattern)
    
    # Pattern to match parameter: type name (with optional generics)
    # Matches: String sql, final String sql, Connection conn
    param_pattern = re.compile(
        rf'(\b(?:final\s+)?[\w<>\[\],\s]+\s+)({re.escape(param_name)})(\s*[,)])'
    )
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line contains a method matching the pattern
        if method_regex.search(line):
            # Look for the parameter in this line or continuation lines
            method_lines = [line]
            j = i
            
            # Collect all lines of the method signature (until we find the opening brace or semicolon)
            while j < len(lines) and '{' not in line and ';' not in line:
                j += 1
                if j < len(lines):
                    method_lines.append(lines[j])
                    line = lines[j]
            
            # Join method lines and look for parameter
            method_text = ''.join(method_lines)
            
            # Check if parameter exists and is not already annotated
            param_match = param_pattern.search(method_text)
            if param_match:
                # Check if already annotated
                prefix = param_match.group(1)
                if f'@{annotation}' not in prefix:
                    # Find which line contains the parameter
                    for k, method_line in enumerate(method_lines):
                        if param_name in method_line:
                            original = lines[i + k]
                            # Add annotation before the parameter type
                            modified_line = param_pattern.sub(
                                rf'@{annotation} \1\2\3',
                                original
                            )
                            if modified_line != original:
                                results.append(InjectionResult(
                                    file_path=str(file_path),
                                    line_number=i + k + 1,
                                    original_line=original.rstrip(),
                                    modified_line=modified_line.rstrip(),
                                    annotation=annotation
                                ))
                                lines[i + k] = modified_line
                                modified = True
                            break
        i += 1
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
    
    return results


def annotate_sql_param(file_path: Path, annotation: str = 'SqlEvenQuotes') -> List[InjectionResult]:
    """Annotate SQL string parameters in a file"""
    results = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern for method parameters named 'sql' or 'query'
    # This is a simplified approach - finds String sql or String query parameters
    patterns = [
        (r'String\s+sql\s*[,)]', 'sql'),
        (r'String\s+query\s*[,)]', 'query'),
        (r'final\s+String\s+sql\s*[,)]', 'sql'),
        (r'final\s+String\s+query\s*[,)]', 'query'),
    ]
    
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        for pattern, param_name in patterns:
            if re.search(pattern, line):
                # Check if already annotated
                if f'@{annotation}' not in line:
                    # Add annotation before String
                    new_line = re.sub(
                        rf'(\bString\s+{param_name}\s*)([,)])',
                        rf'@{annotation} \1\2',
                        line
                    )
                    # Also handle final String
                    new_line = re.sub(
                        rf'(final\s+)(String\s+{param_name}\s*)([,)])',
                        rf'\1@{annotation} \2\3',
                        new_line
                    )
                    
                    if new_line != line:
                        results.append(InjectionResult(
                            file_path=str(file_path),
                            line_number=i + 1,
                            original_line=line.rstrip(),
                            modified_line=new_line.rstrip(),
                            annotation=annotation
                        ))
                        lines[i] = new_line
                        modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
    
    return results


def annotate_classname_param(file_path: Path, annotation: str = 'BinaryName') -> List[InjectionResult]:
    """Annotate class name string parameters in a file"""
    results = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern for method parameters named 'className', 'name', 'typeName'
    patterns = [
        (r'String\s+className\s*[,)]', 'className'),
        (r'String\s+typeName\s*[,)]', 'typeName'),
        (r'final\s+String\s+className\s*[,)]', 'className'),
        (r'final\s+String\s+typeName\s*[,)]', 'typeName'),
    ]
    
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        for pattern, param_name in patterns:
            if re.search(pattern, line):
                # Check if already annotated
                if f'@{annotation}' not in line:
                    # Add annotation before String
                    new_line = re.sub(
                        rf'(\bString\s+{param_name}\s*)([,)])',
                        rf'@{annotation} \1\2',
                        line
                    )
                    # Also handle final String
                    new_line = re.sub(
                        rf'(final\s+)(String\s+{param_name}\s*)([,)])',
                        rf'\1@{annotation} \2\3',
                        new_line
                    )
                    
                    if new_line != line:
                        results.append(InjectionResult(
                            file_path=str(file_path),
                            line_number=i + 1,
                            original_line=line.rstrip(),
                            modified_line=new_line.rstrip(),
                            annotation=annotation
                        ))
                        lines[i] = new_line
                        modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
    
    return results


def annotate_project(
    project_path: Path,
    checker_type: str,
    patterns_config: Optional[Path] = None
) -> Tuple[List[InjectionResult], int]:
    """
    Annotate a project for training with a specific checker
    
    Args:
        project_path: Path to project root
        checker_type: 'sql_quotes' or 'signature_string'
        patterns_config: Optional path to patterns configuration file
        
    Returns:
        Tuple of (list of injection results, number of files modified)
    """
    all_results = []
    files_modified = 0
    
    # Get imports for this checker
    imports = CHECKER_IMPORTS.get(checker_type, [])
    
    # Find all Java files
    java_files = list(project_path.rglob('*.java'))
    
    # Filter out test files
    java_files = [f for f in java_files if '/test/' not in str(f) and '/tests/' not in str(f)]
    
    logger.info(f"Found {len(java_files)} Java source files in {project_path}")
    
    for java_file in java_files:
        file_results = []
        
        if checker_type == 'sql_quotes':
            file_results = annotate_sql_param(java_file)
        elif checker_type == 'signature_string':
            file_results = annotate_classname_param(java_file)
        
        if file_results:
            # Add imports to this file
            add_import_to_file(java_file, imports)
            all_results.extend(file_results)
            files_modified += 1
    
    return all_results, files_modified


def run_checker(project_path: Path, checker_type: str, output_file: Path) -> Tuple[int, int]:
    """
    Run the checker on the project and count warnings
    
    Returns:
        Tuple of (checker_warning_count, compilation_error_count)
    """
    processor = CHECKER_PROCESSORS.get(checker_type)
    if not processor:
        raise ValueError(f"Unknown checker type: {checker_type}")
    
    # Find Java files
    java_files = list(project_path.rglob('*.java'))
    java_files = [str(f) for f in java_files if '/test/' not in str(f)]
    
    if not java_files:
        return 0, 0
    
    # Build command
    cmd = [
        CHECKER_JAVAC,
        '-processor', processor,
        '-Xmaxerrs', '1000',
        '-Xmaxwarns', '1000',
        '-d', '/tmp/checker_output',
        '-proc:only',
    ]
    cmd.extend(java_files[:500])  # Limit files
    
    try:
        os.makedirs('/tmp/checker_output', exist_ok=True)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_path),
            timeout=300
        )
        
        output = result.stderr + result.stdout
        
        # Count checker-specific warnings
        warning_count = 0
        error_count = 0
        
        checker_patterns = {
            'sql_quotes': ['[assignment]', '[compound.assignment]', '[argument]', '[return]'],
            'signature_string': ['[assignment]', '[argument]', '[return]', '[override.return]'],
        }
        
        patterns = checker_patterns.get(checker_type, [])
        
        for line in output.split('\n'):
            if any(p in line for p in patterns):
                warning_count += 1
            elif 'cannot find symbol' in line.lower() or 'package' in line.lower() and 'does not exist' in line.lower():
                error_count += 1
        
        # Write output to file
        with open(output_file, 'w') as f:
            f.write(f"# {checker_type} Checker Warnings\n")
            f.write(f"# Project: {project_path}\n")
            f.write(f"# Checker Warnings: {warning_count}\n")
            f.write(f"# Compilation Errors: {error_count}\n\n")
            f.write(output)
        
        return warning_count, error_count
        
    except Exception as e:
        logger.error(f"Error running checker: {e}")
        return 0, 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotate project for checker training')
    parser.add_argument('project_path', help='Path to project')
    parser.add_argument('--checker', required=True,
                       choices=['sql_quotes', 'signature_string'],
                       help='Checker type')
    parser.add_argument('--output', help='Output file for warnings')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    
    args = parser.parse_args()
    
    project_path = Path(args.project_path)
    
    if not project_path.exists():
        print(f"Project path does not exist: {project_path}")
        return 1
    
    print(f"Annotating {project_path} for {args.checker} checker...")
    
    if args.dry_run:
        print("DRY RUN - no files will be modified")
    
    # Annotate the project
    results, files_modified = annotate_project(project_path, args.checker)
    
    print(f"\nAnnotation Results:")
    print(f"  Files modified: {files_modified}")
    print(f"  Annotations added: {len(results)}")
    
    for result in results[:10]:  # Show first 10
        print(f"\n  {result.file_path}:{result.line_number}")
        print(f"    - {result.original_line}")
        print(f"    + {result.modified_line}")
    
    if len(results) > 10:
        print(f"\n  ... and {len(results) - 10} more")
    
    # Run checker if output file specified
    if args.output and not args.dry_run:
        output_file = Path(args.output)
        print(f"\nRunning {args.checker} checker...")
        warnings, errors = run_checker(project_path, args.checker, output_file)
        print(f"  Checker warnings: {warnings}")
        print(f"  Compilation errors: {errors}")
        print(f"  Output saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())
