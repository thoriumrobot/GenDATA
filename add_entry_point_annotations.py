#!/usr/bin/env python3
"""
Add Entry-Point Annotations to Java Files

This script adds checker-specific entry-point annotations to Java methods
to create fixable warnings for evaluation.

Entry-point annotations work like this:
1. A method parameter is annotated (e.g., @BinaryName String className)
2. Callers that pass unannotated strings generate warnings
3. The model learns to add annotations to fix these warnings

Usage:
    python add_entry_point_annotations.py --checker sql_quotes --project commons-dbcp
    python add_entry_point_annotations.py --checker signature_string --project guice
    python add_entry_point_annotations.py --all
"""

import os
import re
import sys
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
EVALUATION_READY_DIR = GEN_DATA_ROOT / 'annotation_evaluation' / 'evaluation_ready'
CHECKER_FRAMEWORK_HOME = Path('/home/ubuntu/checker-framework')

# Checker configurations
CHECKER_CONFIGS = {
    'sql_quotes': {
        'processor': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
        'annotation': '@SqlEvenQuotes',
        'import': 'import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;',
        'projects': ['commons-dbcp', 'mybatis-3'],
        # Patterns to find methods that should have annotated parameters
        'method_patterns': [
            # Methods with String sql parameter
            r'(public|protected)\s+\w+[\w<>,\s]*\s+(execute|query|update|prepareStatement|prepareCall)\s*\([^)]*String\s+(\w+)',
            r'(public|protected)\s+\w+[\w<>,\s]*\s+\w+\s*\([^)]*String\s+sql\b',
        ],
        # Parameter names that indicate SQL
        'param_names': ['sql', 'query', 'statement', 'sqlQuery', 'sqlStatement'],
    },
    'signature_string': {
        'processor': 'org.checkerframework.checker.signature.SignatureChecker',
        'annotation': '@BinaryName',
        'import': 'import org.checkerframework.checker.signature.qual.BinaryName;',
        'projects': ['guice', 'cglib'],
        # Patterns to find methods that should have annotated parameters  
        'method_patterns': [
            r'(public|protected)\s+Class\s*<[^>]*>\s+\w+\s*\([^)]*String\s+(\w+)',
            r'(public|protected)\s+\w+[\w<>,\s]*\s+(loadClass|forName|findClass|defineClass)\s*\([^)]*String\s+(\w+)',
        ],
        # Parameter names that indicate class names
        'param_names': ['className', 'name', 'typeName', 'clazz', 'type', 'classname'],
    },
}

# Target files for each project
TARGET_FILES = {
    'commons-dbcp': [
        'src/main/java/org/apache/commons/dbcp2/DelegatingStatement.java',
        'src/main/java/org/apache/commons/dbcp2/DelegatingConnection.java',
        'src/main/java/org/apache/commons/dbcp2/DelegatingPreparedStatement.java',
        'src/main/java/org/apache/commons/dbcp2/PoolingConnection.java',
        'src/main/java/org/apache/commons/dbcp2/BasicDataSource.java',
    ],
    'mybatis-3': [
        'src/main/java/org/apache/ibatis/executor/SimpleExecutor.java',
        'src/main/java/org/apache/ibatis/executor/BaseExecutor.java',
        'src/main/java/org/apache/ibatis/session/SqlSession.java',
        'src/main/java/org/apache/ibatis/jdbc/SqlRunner.java',
        'src/main/java/org/apache/ibatis/scripting/defaults/DefaultParameterHandler.java',
    ],
    'guice': [
        'core/src/com/google/inject/internal/util/SourceProvider.java',
        'core/src/com/google/inject/internal/aop/ClassBuilding.java',
        'core/src/com/google/inject/spi/TypeLiteral.java',
        'core/src/com/google/inject/internal/MoreTypes.java',
        'core/src/com/google/inject/internal/util/Classes.java',
    ],
    'cglib': [
        'cglib/src/main/java/net/sf/cglib/core/AbstractClassGenerator.java',
        'cglib/src/main/java/net/sf/cglib/core/ReflectUtils.java',
        'cglib/src/main/java/net/sf/cglib/proxy/Enhancer.java',
        'cglib/src/main/java/net/sf/cglib/core/ClassNameReader.java',
        'cglib/src/main/java/net/sf/cglib/core/DefaultNamingPolicy.java',
    ],
}


@dataclass
class AnnotationResult:
    """Result of adding annotations to a file"""
    file_path: str
    annotations_added: int
    imports_added: bool
    success: bool
    error: Optional[str] = None


def add_import_if_missing(content: str, import_statement: str) -> Tuple[str, bool]:
    """Add import statement if not already present"""
    if import_statement in content:
        return content, False
    
    # Find the best place to add the import
    # Look for existing Checker Framework imports first
    cf_import_match = re.search(r'(import\s+org\.checkerframework[^;]+;)', content)
    if cf_import_match:
        # Add after existing CF import
        insert_pos = cf_import_match.end()
        return content[:insert_pos] + '\n' + import_statement + content[insert_pos:], True
    
    # Look for package declaration
    package_match = re.search(r'^package\s+[\w.]+;', content, re.MULTILINE)
    if package_match:
        # Find the first import after package
        import_match = re.search(r'\nimport\s+', content[package_match.end():])
        if import_match:
            insert_pos = package_match.end() + import_match.start() + 1
            return content[:insert_pos] + import_statement + '\n' + content[insert_pos:], True
        else:
            # No imports, add after package
            insert_pos = package_match.end()
            return content[:insert_pos] + '\n\n' + import_statement + content[insert_pos:], True
    
    # Fallback: add at the beginning
    return import_statement + '\n' + content, True


def find_method_parameters_to_annotate(content: str, checker_name: str) -> List[Tuple[int, str, str]]:
    """
    Find method parameters that should be annotated.
    
    Returns list of (line_number, original_line, parameter_name)
    """
    config = CHECKER_CONFIGS.get(checker_name, {})
    param_names = config.get('param_names', [])
    annotation = config.get('annotation', '')
    
    results = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # Skip if already annotated
        if annotation in line:
            continue
        
        # Look for method declarations with String parameters
        # Match patterns like: public void execute(String sql, ...)
        method_match = re.search(
            r'(public|protected)\s+[\w<>,\s]+\s+\w+\s*\([^)]*\bString\s+(\w+)',
            line
        )
        
        if method_match:
            param_name = method_match.group(2)
            # Check if parameter name suggests it's a target
            if any(pn.lower() in param_name.lower() for pn in param_names):
                results.append((i, line, param_name))
                continue
        
        # Also check for specific method names
        for pattern in config.get('method_patterns', []):
            if re.search(pattern, line):
                # Extract parameter name
                param_match = re.search(r'String\s+(\w+)', line)
                if param_match:
                    param_name = param_match.group(1)
                    if (i, line, param_name) not in results:
                        results.append((i, line, param_name))
                break
    
    return results


def add_annotation_to_parameter(line: str, param_name: str, annotation: str) -> str:
    """Add annotation to a specific parameter in a method declaration"""
    # Pattern to match String param_name
    pattern = rf'(\bString\s+)({param_name}\b)'
    replacement = rf'{annotation} \1\2'
    
    # Check if already annotated
    if f'{annotation} String {param_name}' in line:
        return line
    
    return re.sub(pattern, replacement, line)


def add_annotations_to_file(file_path: Path, checker_name: str, 
                            max_annotations: int = 5) -> AnnotationResult:
    """Add entry-point annotations to a Java file"""
    config = CHECKER_CONFIGS.get(checker_name)
    if not config:
        return AnnotationResult(str(file_path), 0, False, False, f"Unknown checker: {checker_name}")
    
    if not file_path.exists():
        return AnnotationResult(str(file_path), 0, False, False, "File not found")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        annotation = config['annotation']
        import_stmt = config['import']
        
        # Find parameters to annotate
        targets = find_method_parameters_to_annotate(content, checker_name)
        
        if not targets:
            return AnnotationResult(str(file_path), 0, False, True, "No suitable parameters found")
        
        # Limit number of annotations
        targets = targets[:max_annotations]
        
        # Add annotations (process from bottom to top to maintain line numbers)
        lines = content.split('\n')
        annotations_added = 0
        
        for line_num, original_line, param_name in reversed(targets):
            idx = line_num - 1
            if idx < len(lines):
                new_line = add_annotation_to_parameter(lines[idx], param_name, annotation)
                if new_line != lines[idx]:
                    lines[idx] = new_line
                    annotations_added += 1
        
        content = '\n'.join(lines)
        
        # Add import if we added any annotations
        imports_added = False
        if annotations_added > 0:
            content, imports_added = add_import_if_missing(content, import_stmt)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"  Added {annotations_added} annotations to {file_path.name}")
        
        return AnnotationResult(str(file_path), annotations_added, imports_added, True)
        
    except Exception as e:
        return AnnotationResult(str(file_path), 0, False, False, str(e))


def run_checker(project_dir: Path, checker_name: str) -> Tuple[int, str]:
    """Run checker on project and count warnings"""
    config = CHECKER_CONFIGS.get(checker_name)
    if not config:
        return -1, f"Unknown checker: {checker_name}"
    
    processor = config['processor']
    
    # Find Java files
    java_files = []
    exclude_patterns = ['/test/', '/tests/', '/target/', '/build/', '/.git/']
    
    for java_file in project_dir.rglob('*.java'):
        path_str = str(java_file)
        if not any(p in path_str for p in exclude_patterns):
            java_files.append(java_file)
            if len(java_files) >= 50:
                break
    
    if not java_files:
        return 0, "No Java files found"
    
    checker_javac = CHECKER_FRAMEWORK_HOME / 'checker' / 'bin' / 'javac'
    checker_cp = f"{CHECKER_FRAMEWORK_HOME}/checker/dist/checker-qual.jar"
    
    cmd = [
        str(checker_javac),
        '-processor', processor,
        '-cp', checker_cp,
        '-Xlint:-processing',
        '-Awarns',
    ] + [str(f) for f in java_files[:30]]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(project_dir)
        )
        output = result.stdout + result.stderr
        
        # Count checker-specific warnings
        warning_count = 0
        for line in output.split('\n'):
            if 'error:' in line.lower() or 'warning:' in line.lower():
                if '[' in line and ']' in line:
                    # Skip generic warnings
                    if not any(w in line for w in ['[deprecation]', '[removal]', '[unchecked]', '[rawtypes]', '[path]']):
                        warning_count += 1
        
        return warning_count, output
        
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)


def annotate_project(project_name: str, checker_name: str, 
                     max_annotations_per_file: int = 5) -> Dict:
    """Add entry-point annotations to a project"""
    logger.info(f"Annotating {project_name} for {checker_name} checker")
    
    project_dir = EVALUATION_READY_DIR / checker_name / project_name
    
    result = {
        'project': project_name,
        'checker': checker_name,
        'files_modified': 0,
        'total_annotations': 0,
        'baseline_before': 0,
        'baseline_after': 0,
        'success': False,
    }
    
    if not project_dir.exists():
        result['error'] = f"Project directory not found: {project_dir}"
        return result
    
    # Check baseline before
    baseline_before, _ = run_checker(project_dir, checker_name)
    result['baseline_before'] = baseline_before
    logger.info(f"  Baseline before: {baseline_before} warnings")
    
    if baseline_before > 0:
        logger.info(f"  Project already has {baseline_before} warnings - skipping")
        result['baseline_after'] = baseline_before
        result['success'] = True
        return result
    
    # Get target files for this project
    target_files = TARGET_FILES.get(project_name, [])
    
    if not target_files:
        # Find files automatically if no targets specified
        target_files = find_candidate_files(project_dir, checker_name)
    
    # Add annotations to each target file
    for rel_path in target_files:
        file_path = project_dir / rel_path
        if file_path.exists():
            file_result = add_annotations_to_file(file_path, checker_name, max_annotations_per_file)
            if file_result.annotations_added > 0:
                result['files_modified'] += 1
                result['total_annotations'] += file_result.annotations_added
    
    # Check baseline after
    baseline_after, _ = run_checker(project_dir, checker_name)
    result['baseline_after'] = baseline_after
    logger.info(f"  Baseline after: {baseline_after} warnings")
    
    if baseline_after > 0:
        result['success'] = True
        logger.info(f"  Successfully created {baseline_after} warnings")
    else:
        logger.warning(f"  Still 0 warnings - may need more annotations")
        result['error'] = "Could not create warnings"
    
    return result


def find_candidate_files(project_dir: Path, checker_name: str) -> List[str]:
    """Find candidate files for annotation based on checker type"""
    candidates = []
    
    config = CHECKER_CONFIGS.get(checker_name, {})
    param_names = config.get('param_names', [])
    
    # Keywords to look for in file content
    if checker_name == 'sql_quotes':
        keywords = ['executeQuery', 'executeUpdate', 'prepareStatement', 'prepareCall', 'Statement']
    else:  # signature_string
        keywords = ['Class.forName', 'loadClass', 'findClass', 'ClassLoader', 'getClass']
    
    exclude_patterns = ['/test/', '/tests/', '/target/', '/build/', '/.git/']
    
    for java_file in project_dir.rglob('*.java'):
        path_str = str(java_file)
        if any(p in path_str for p in exclude_patterns):
            continue
        
        try:
            with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if file contains relevant keywords
            if any(kw in content for kw in keywords):
                # Check if it has String parameters with target names
                if any(f'String {pn}' in content for pn in param_names):
                    rel_path = str(java_file.relative_to(project_dir))
                    candidates.append(rel_path)
                    if len(candidates) >= 10:
                        break
        except:
            continue
    
    return candidates


def annotate_all_projects() -> Dict[str, List[Dict]]:
    """Annotate all projects with 0 baseline warnings"""
    results = {}
    
    for checker_name, config in CHECKER_CONFIGS.items():
        results[checker_name] = []
        
        for project_name in config['projects']:
            result = annotate_project(project_name, checker_name)
            results[checker_name].append(result)
    
    return results


def print_summary(results: Dict[str, List[Dict]]) -> None:
    """Print summary of annotation results"""
    print("\n" + "="*60)
    print("ENTRY-POINT ANNOTATION SUMMARY")
    print("="*60)
    
    all_success = True
    
    for checker, checker_results in results.items():
        print(f"\n{checker}:")
        print("-" * 40)
        
        for r in checker_results:
            status = "OK" if r.get('success') else "FAILED"
            before = r.get('baseline_before', 0)
            after = r.get('baseline_after', 0)
            annotations = r.get('total_annotations', 0)
            
            print(f"  {r['project']}: {status}")
            print(f"    Before: {before} warnings, After: {after} warnings")
            print(f"    Annotations added: {annotations}")
            
            if not r.get('success'):
                all_success = False
    
    print("\n" + "="*60)
    if all_success:
        print("All projects now have positive baseline warnings!")
    else:
        print("Some projects still have 0 warnings - manual intervention may be needed")


def main():
    parser = argparse.ArgumentParser(description='Add entry-point annotations to Java files')
    parser.add_argument('--checker', choices=['sql_quotes', 'signature_string'],
                       help='Specific checker to annotate for')
    parser.add_argument('--project', help='Specific project to annotate')
    parser.add_argument('--all', action='store_true', help='Annotate all projects')
    parser.add_argument('--max-annotations', type=int, default=5,
                       help='Maximum annotations per file')
    
    args = parser.parse_args()
    
    if args.all:
        results = annotate_all_projects()
        print_summary(results)
    elif args.checker and args.project:
        result = annotate_project(args.project, args.checker, args.max_annotations)
        print_summary({args.checker: [result]})
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
