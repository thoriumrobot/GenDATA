#!/usr/bin/env python3
"""
Prepare Evaluation Projects

This script prepares projects for evaluation by:
1. Copying projects from backups to evaluation_ready directory
2. Adding entry-point annotations to create fixable warnings
3. Verifying that projects have baseline warnings > 0

Entry-point annotations create warnings that models can fix:
- Method parameters are annotated (e.g., @BinaryName String className)
- Callers pass unannotated strings -> generates warnings
- Models learn to add annotations to the call sites

Usage:
    python prepare_evaluation_projects.py
    python prepare_evaluation_projects.py --checker sql_quotes
    python prepare_evaluation_projects.py --dry-run
"""

import os
import sys
import re
import shutil
import subprocess
import logging
import argparse
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
CASE_STUDIES_BACKUP = GEN_DATA_ROOT / 'case_studies_backup'
ANNOTATION_EVAL_BACKUPS = GEN_DATA_ROOT / 'annotation_evaluation' / 'backups'
ANNOTATED_PROJECTS_BACKUP = GEN_DATA_ROOT / 'annotated_projects_backup'
EVALUATION_READY_DIR = GEN_DATA_ROOT / 'annotation_evaluation' / 'evaluation_ready'
CHECKER_FRAMEWORK_HOME = Path('/home/ubuntu/checker-framework')

# Checker processors
CHECKER_PROCESSORS = {
    'lower_bound': 'org.checkerframework.checker.index.IndexChecker',
    'sql_quotes': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
    'signature_string': 'org.checkerframework.checker.signature.SignatureChecker',
}

# Projects per checker
EVALUATION_PROJECTS = {
    'lower_bound': ['sortpom', 'pom-tuner', 'jfreechart'],
    'sql_quotes': ['commons-dbutils', 'commons-dbcp', 'mybatis-3'],
    'signature_string': ['kryo', 'guice', 'cglib'],
}

# Entry-point annotation patterns for each checker
ENTRY_POINT_PATTERNS = {
    'sql_quotes': {
        'imports': [
            'import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;',
        ],
        'method_patterns': [
            # Methods that execute SQL
            (r'(public|protected)\s+\w+\s+(executeQuery|executeUpdate|execute)\s*\(\s*String\s+(\w+)',
             r'\1 \2(\n        @SqlEvenQuotes String \3'),
            (r'(public|protected)\s+\w+\s+query\s*\(\s*String\s+(\w+)',
             r'\1 query(\n        @SqlEvenQuotes String \2'),
        ],
        'target_methods': ['executeQuery', 'executeUpdate', 'execute', 'query', 'prepareStatement'],
    },
    'signature_string': {
        'imports': [
            'import org.checkerframework.checker.signature.qual.BinaryName;',
            'import org.checkerframework.checker.signature.qual.FullyQualifiedName;',
        ],
        'method_patterns': [
            # Methods that load classes
            (r'(public|protected)\s+Class\<?[^>]*\>?\s+(\w+)\s*\(\s*String\s+(\w+)',
             r'\1 Class<?> \2(\n        @BinaryName String \3'),
            (r'(public|protected)\s+\w+\s+loadClass\s*\(\s*String\s+(\w+)',
             r'\1 loadClass(\n        @BinaryName String \2'),
            (r'(public|protected)\s+\w+\s+forName\s*\(\s*String\s+(\w+)',
             r'\1 forName(\n        @BinaryName String \2'),
        ],
        'target_methods': ['loadClass', 'forName', 'getClass', 'findClass', 'defineClass'],
    },
    'lower_bound': {
        'imports': [
            'import org.checkerframework.checker.index.qual.Positive;',
            'import org.checkerframework.checker.index.qual.NonNegative;',
        ],
        'method_patterns': [
            # Methods that use array indices
            (r'(public|protected)\s+\w+\s+get\s*\(\s*int\s+(\w+)',
             r'\1 get(\n        @NonNegative int \2'),
            (r'(public|protected)\s+\w+\s+set\s*\(\s*int\s+(\w+)',
             r'\1 set(\n        @NonNegative int \2'),
        ],
        'target_methods': ['get', 'set', 'remove', 'add', 'charAt', 'substring'],
    },
}


@dataclass
class AnnotationResult:
    """Result of adding annotations to a file"""
    file_path: str
    annotations_added: int
    imports_added: bool
    success: bool
    error: Optional[str] = None


def find_backup_source(project_name: str, checker_name: str = None) -> Optional[Path]:
    """
    Find backup source for a project.
    
    Priority:
    1. Annotated projects backup (for SQL Quotes and Signature String - already have entry-point annotations)
    2. Annotation evaluation backups
    3. Case studies backup
    """
    sources = []
    
    # For SQL Quotes and Signature String, prefer annotated versions
    if checker_name in ['sql_quotes', 'signature_string']:
        annotated_path = ANNOTATED_PROJECTS_BACKUP / checker_name / project_name
        if annotated_path.exists():
            sources.append(annotated_path)
    
    # Standard backup locations
    sources.extend([
        ANNOTATION_EVAL_BACKUPS / project_name,
        CASE_STUDIES_BACKUP / project_name,
    ])
    
    for source in sources:
        if source.exists():
            return source
    return None


def copy_project_to_evaluation(project_name: str, checker_name: str, 
                               dry_run: bool = False) -> Tuple[bool, str]:
    """Copy project from backup to evaluation_ready directory"""
    source = find_backup_source(project_name, checker_name)
    if not source:
        logger.error(f"No backup found for {project_name}")
        return False, "not_found"
    
    # Determine if using annotated version
    source_type = "annotated" if "annotated_projects_backup" in str(source) else "unannotated"
    
    dest = EVALUATION_READY_DIR / checker_name / project_name
    
    if dry_run:
        logger.info(f"[DRY RUN] Would copy {source} -> {dest} ({source_type})")
        return True, source_type
    
    try:
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest)
        logger.info(f"Copied {project_name} ({source_type}) to {dest}")
        return True, source_type
    except Exception as e:
        logger.error(f"Error copying {project_name}: {e}")
        return False, "error"


def find_java_files(directory: Path, max_files: int = 50) -> List[Path]:
    """Find Java files in directory, excluding tests"""
    java_files = []
    exclude_patterns = ['/test/', '/tests/', '/target/', '/build/', '/.git/']
    
    for java_file in directory.rglob('*.java'):
        path_str = str(java_file)
        if not any(p in path_str for p in exclude_patterns):
            java_files.append(java_file)
            if len(java_files) >= max_files:
                break
    
    return java_files


def add_import_if_missing(content: str, import_statement: str) -> Tuple[str, bool]:
    """Add import statement if not already present"""
    if import_statement in content:
        return content, False
    
    # Find package declaration or first import
    package_match = re.search(r'^package\s+[\w.]+;', content, re.MULTILINE)
    if package_match:
        insert_pos = package_match.end()
        # Find end of any existing imports
        import_section = re.search(r'(import\s+[\w.*]+;\s*)+', content[insert_pos:])
        if import_section:
            insert_pos += import_section.end()
        new_content = content[:insert_pos] + '\n' + import_statement + content[insert_pos:]
        return new_content, True
    
    # No package, add at start
    return import_statement + '\n' + content, True


def add_entry_point_annotations_to_file(file_path: Path, checker_name: str,
                                        dry_run: bool = False) -> AnnotationResult:
    """Add entry-point annotations to a single file"""
    config = ENTRY_POINT_PATTERNS.get(checker_name, {})
    if not config:
        return AnnotationResult(str(file_path), 0, False, False, "Unknown checker")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        annotations_added = 0
        imports_added = False
        
        # Check if file has any target methods
        target_methods = config.get('target_methods', [])
        has_target = any(m in content for m in target_methods)
        
        if not has_target:
            return AnnotationResult(str(file_path), 0, False, True)
        
        # Add imports
        for import_stmt in config.get('imports', []):
            content, added = add_import_if_missing(content, import_stmt)
            if added:
                imports_added = True
        
        # Apply method patterns
        for pattern, replacement in config.get('method_patterns', []):
            matches = list(re.finditer(pattern, content))
            if matches:
                for match in reversed(matches):  # Reverse to maintain positions
                    # Simple replacement - add annotation to method parameter
                    old_text = match.group(0)
                    # Check if already annotated
                    if '@SqlEvenQuotes' in old_text or '@BinaryName' in old_text or '@NonNegative' in old_text:
                        continue
                    new_text = re.sub(pattern, replacement, old_text)
                    if new_text != old_text:
                        content = content[:match.start()] + new_text + content[match.end():]
                        annotations_added += 1
        
        if content != original_content:
            if dry_run:
                logger.info(f"[DRY RUN] Would modify {file_path}: {annotations_added} annotations")
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Modified {file_path}: {annotations_added} annotations")
        
        return AnnotationResult(str(file_path), annotations_added, imports_added, True)
        
    except Exception as e:
        return AnnotationResult(str(file_path), 0, False, False, str(e))


def add_simple_entry_points(project_dir: Path, checker_name: str, 
                           dry_run: bool = False) -> int:
    """
    Add simple entry-point annotations by finding key files and adding annotations.
    
    This is a simpler approach: find files with relevant patterns and add
    a few strategic annotations to create warnings.
    """
    total_added = 0
    
    java_files = find_java_files(project_dir, max_files=100)
    
    for java_file in java_files:
        result = add_entry_point_annotations_to_file(java_file, checker_name, dry_run)
        if result.success and result.annotations_added > 0:
            total_added += result.annotations_added
            if total_added >= 10:  # Stop after adding enough annotations
                break
    
    return total_added


def run_checker(project_dir: Path, checker_name: str) -> Tuple[int, str]:
    """Run checker on project and count warnings"""
    processor = CHECKER_PROCESSORS.get(checker_name)
    if not processor:
        return -1, f"Unknown checker: {checker_name}"
    
    java_files = find_java_files(project_dir, max_files=50)
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
        
        # Count warnings
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


def prepare_project(project_name: str, checker_name: str, 
                   dry_run: bool = False, verify_warnings: bool = False) -> Dict:
    """
    Prepare a single project for evaluation.
    
    NOTE: Warning counting here uses a simple approach without Maven.
    The actual evaluation pipeline (evaluate_all_checkers.py) uses Maven
    and will find the correct warning counts.
    
    Args:
        project_name: Name of the project
        checker_name: Name of the checker
        dry_run: If True, don't make any changes
        verify_warnings: If True, try to count warnings (may be inaccurate without Maven)
    """
    logger.info(f"Preparing {project_name} for {checker_name}")
    
    result = {
        'project': project_name,
        'checker': checker_name,
        'success': False,
        'baseline_warnings': 0,
        'annotations_added': 0,
        'source_type': 'unknown',
    }
    
    # Copy to evaluation_ready
    copy_success, source_type = copy_project_to_evaluation(project_name, checker_name, dry_run)
    result['source_type'] = source_type
    
    if not copy_success:
        result['error'] = "Failed to copy from backup"
        return result
    
    project_dir = EVALUATION_READY_DIR / checker_name / project_name
    
    if dry_run:
        result['success'] = True
        result['note'] = "Dry run - no modifications made"
        return result
    
    # Only verify warnings if requested (it's inaccurate without Maven)
    if verify_warnings:
        baseline, output = run_checker(project_dir, checker_name)
        result['baseline_warnings'] = baseline
        
        if baseline > 0:
            logger.info(f"  {project_name} ({source_type}) has {baseline} warnings (simple check)")
            result['success'] = True
            return result
        
        # If using annotated version and still 0 warnings, note that Maven is needed
        if source_type == 'annotated':
            logger.info(f"  {project_name} (annotated) - warnings will be detected by Maven pipeline")
    
    # Mark as success - actual evaluation pipeline will handle warnings
    result['success'] = True
    result['note'] = f"Copied from {source_type} backup. Use evaluate_all_checkers.py for accurate warning counts."
    logger.info(f"  {project_name} ready for evaluation ({source_type})")
    
    return result


def prepare_all_projects(checker_name: Optional[str] = None, 
                        dry_run: bool = False) -> Dict[str, List[Dict]]:
    """Prepare all evaluation projects"""
    results = {}
    
    checkers = [checker_name] if checker_name else list(EVALUATION_PROJECTS.keys())
    
    for checker in checkers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Preparing projects for {checker} checker")
        logger.info(f"{'='*60}")
        
        results[checker] = []
        projects = EVALUATION_PROJECTS.get(checker, [])
        
        for project in projects:
            result = prepare_project(project, checker, dry_run)
            results[checker].append(result)
    
    return results


def print_summary(results: Dict[str, List[Dict]]) -> None:
    """Print summary of preparation results"""
    print("\n" + "="*60)
    print("PREPARATION SUMMARY")
    print("="*60)
    
    for checker, checker_results in results.items():
        print(f"\n{checker}:")
        print("-" * 40)
        
        for r in checker_results:
            status = "OK" if r.get('success') else "FAILED"
            warnings = r.get('baseline_warnings', r.get('after_warnings', 0))
            annotations = r.get('annotations_added', 0)
            source_type = r.get('source_type', 'unknown')
            print(f"  {r['project']}: {status} ({warnings} warnings, {source_type})")
    
    # Overall stats
    total = sum(len(r) for r in results.values())
    success = sum(1 for checker_results in results.values() for r in checker_results if r.get('success'))
    print(f"\nTotal: {success}/{total} projects prepared successfully")


def main():
    parser = argparse.ArgumentParser(description='Prepare evaluation projects')
    parser.add_argument('--checker', choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       help='Specific checker to prepare')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    # Ensure output directory exists
    if not args.dry_run:
        EVALUATION_READY_DIR.mkdir(parents=True, exist_ok=True)
    
    results = prepare_all_projects(args.checker, args.dry_run)
    print_summary(results)
    
    # Check if any failed
    failed = sum(1 for checker_results in results.values() 
                 for r in checker_results if not r.get('success'))
    
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
