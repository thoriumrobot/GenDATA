#!/usr/bin/env python3
"""
Verify Baseline Warnings for All Evaluation Projects

This script runs the checkers on all projects and reports baseline warnings.
It uses the Checker Framework to count warnings without Maven integration.
"""

import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
    'lower_bound': {
        'processor': 'org.checkerframework.checker.index.IndexChecker',
        'projects': ['sortpom', 'pom-tuner', 'jfreechart'],
    },
    'sql_quotes': {
        'processor': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
        'projects': ['commons-dbutils', 'commons-dbcp', 'mybatis-3'],
    },
    'signature_string': {
        'processor': 'org.checkerframework.checker.signature.SignatureChecker',
        'projects': ['kryo', 'guice', 'cglib'],
    },
}


def find_java_files(directory: Path, max_files: int = 50) -> List[Path]:
    """Find Java files in a directory, excluding tests and build directories"""
    java_files = []
    
    exclude_patterns = [
        '/test/', '/tests/', '/target/', '/build/', 
        '/generated/', '/.git/', '/benchmark/'
    ]
    
    for java_file in directory.rglob('*.java'):
        path_str = str(java_file)
        if not any(pattern in path_str for pattern in exclude_patterns):
            java_files.append(java_file)
            if len(java_files) >= max_files:
                break
    
    return java_files


def run_checker(project_dir: Path, checker_name: str) -> Tuple[int, str]:
    """Run checker on project and count warnings"""
    config = CHECKER_CONFIGS.get(checker_name)
    if not config:
        return -1, f"Unknown checker: {checker_name}"
    
    processor = config['processor']
    
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
        
        # Count checker-specific warnings
        warning_count = 0
        warning_lines = []
        for line in output.split('\n'):
            if 'error:' in line.lower() or 'warning:' in line.lower():
                if '[' in line and ']' in line:
                    # Skip generic warnings
                    if not any(w in line for w in ['[deprecation]', '[removal]', '[unchecked]', '[rawtypes]', '[path]']):
                        warning_count += 1
                        warning_lines.append(line.strip()[:100])
        
        return warning_count, '\n'.join(warning_lines[:5])
        
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)


def verify_all_baselines() -> Dict[str, Dict]:
    """Verify baseline warnings for all projects"""
    results = {}
    
    for checker_name, config in CHECKER_CONFIGS.items():
        results[checker_name] = {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Checking {checker_name}")
        logger.info(f"{'='*60}")
        
        for project_name in config['projects']:
            project_dir = EVALUATION_READY_DIR / checker_name / project_name
            
            if not project_dir.exists():
                logger.warning(f"  {project_name}: NOT FOUND")
                results[checker_name][project_name] = {
                    'warnings': -1,
                    'status': 'NOT_FOUND',
                    'error': f"Directory not found: {project_dir}"
                }
                continue
            
            warnings, output = run_checker(project_dir, checker_name)
            
            status = 'OK' if warnings > 0 else 'ZERO_WARNINGS'
            logger.info(f"  {project_name}: {warnings} warnings ({status})")
            
            results[checker_name][project_name] = {
                'warnings': warnings,
                'status': status,
                'sample_output': output[:500] if output else ''
            }
    
    return results


def print_summary(results: Dict[str, Dict]) -> bool:
    """Print summary and return True if all projects have positive warnings"""
    print("\n" + "="*60)
    print("BASELINE WARNINGS SUMMARY")
    print("="*60)
    
    all_positive = True
    
    for checker_name, projects in results.items():
        print(f"\n{checker_name}:")
        print("-" * 40)
        
        for project_name, data in projects.items():
            warnings = data['warnings']
            status = data['status']
            marker = "OK" if warnings > 0 else "NEEDS WORK"
            print(f"  {project_name}: {warnings} warnings [{marker}]")
            
            if warnings <= 0:
                all_positive = False
    
    print("\n" + "="*60)
    if all_positive:
        print("SUCCESS: All projects have positive baseline warnings!")
    else:
        print("WARNING: Some projects still have 0 warnings")
    print("="*60)
    
    return all_positive


def main():
    results = verify_all_baselines()
    success = print_summary(results)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
