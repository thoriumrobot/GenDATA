#!/usr/bin/env python3
"""
Find Signature String Checker Projects

This script finds GitHub Java projects that are suitable for Signature String Checker evaluation.
It searches for projects with reflection/class loading code and evaluates them until 3 suitable
projects are found.
"""

import os
import sys
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import evaluation module
from evaluate_checker_project import (
    evaluate_project,
    save_successful_project,
    ProjectEvaluationResult
)

# Known projects that are likely to have reflection/class loading code
# These are curated based on their known usage of Class.forName, reflection, bytecode manipulation
SIGNATURE_PROJECT_CANDIDATES = [
    # Bytecode manipulation libraries
    'https://github.com/asm-ow2/asm.git',
    'https://github.com/jboss-javassist/javassist.git',
    'https://github.com/cglib/cglib.git',
    'https://github.com/raphw/byte-buddy.git',
    
    # Reflection utilities
    'https://github.com/ronmamo/reflections.git',
    'https://github.com/classgraph/classgraph.git',
    
    # Serialization frameworks (use class names)
    'https://github.com/google/gson.git',
    'https://github.com/FasterXML/jackson-core.git',
    'https://github.com/FasterXML/jackson-databind.git',
    'https://github.com/EsotericSoftware/kryo.git',
    
    # Dependency injection (heavy reflection usage)
    'https://github.com/google/guice.git',
    'https://github.com/google/dagger.git',
    
    # Testing frameworks (reflection for test discovery)
    'https://github.com/junit-team/junit4.git',
    'https://github.com/mockito/mockito.git',
    'https://github.com/powermock/powermock.git',
    
    # Plugin systems
    'https://github.com/pf4j/pf4j.git',
    
    # Code generation / reflection
    'https://github.com/square/javapoet.git',
    'https://github.com/derive4j/derive4j.git',
    
    # Scripting / dynamic languages on JVM
    'https://github.com/beanshell/beanshell.git',
    'https://github.com/jruby/jruby.git',
]


def find_signature_string_projects(
    target_count: int = 3,
    work_dir: Path = None,
    case_studies_dir: Path = None,
    min_warnings: int = 5
) -> List[ProjectEvaluationResult]:
    """
    Find projects suitable for Signature String Checker evaluation
    
    Args:
        target_count: Number of suitable projects to find
        work_dir: Working directory for cloning
        case_studies_dir: Directory to save successful projects
        min_warnings: Minimum warnings required
        
    Returns:
        List of successful ProjectEvaluationResults
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix='signature_string_'))
    if case_studies_dir is None:
        case_studies_dir = Path('/home/ubuntu/GenDATA/case_studies')
    
    successful_projects = []
    all_results = []
    
    logger.info(f"Searching for {target_count} projects suitable for Signature String Checker...")
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Case studies directory: {case_studies_dir}")
    
    for project_url in SIGNATURE_PROJECT_CANDIDATES:
        if len(successful_projects) >= target_count:
            break
        
        project_name = project_url.rstrip('/').split('/')[-1].replace('.git', '')
        
        # Skip if already in case_studies with valid warnings
        existing_warnings = case_studies_dir / project_name / f"{project_name}_signature_string_warnings.out"
        if existing_warnings.exists():
            # Check if it has enough warnings
            try:
                with open(existing_warnings, 'r') as f:
                    content = f.read()
                    if 'Annotation processor' not in content and 'not found' not in content:
                        warning_count = content.count('error:') + content.count('warning:')
                        if warning_count >= min_warnings:
                            logger.info(f"Skipping {project_name} - already evaluated with {warning_count} warnings")
                            continue
            except:
                pass
        
        # Clean up previous clone if exists
        project_dir = work_dir / project_name
        if project_dir.exists():
            shutil.rmtree(project_dir)
        
        result = evaluate_project(
            project_url=project_url,
            checker_name='signature_string',
            work_dir=work_dir,
            case_studies_dir=case_studies_dir,
            min_warnings=min_warnings
        )
        
        all_results.append(result)
        
        if result.warning_count >= min_warnings:
            successful_projects.append(result)
            save_successful_project(result, work_dir, case_studies_dir)
            logger.info(f"Found suitable project {len(successful_projects)}/{target_count}: {result.project_name} ({result.warning_count} warnings)")
        else:
            logger.info(f"Project {result.project_name}: {result.warning_count} warnings (not enough)")
    
    # Save summary
    summary = {
        'checker': 'signature_string',
        'target_count': target_count,
        'found_count': len(successful_projects),
        'total_evaluated': len(all_results),
        'timestamp': datetime.now().isoformat(),
        'successful_projects': [
            {
                'name': r.project_name,
                'url': r.project_url,
                'warning_count': r.warning_count,
                'java_files': r.java_file_count,
                'build_system': r.build_system
            }
            for r in successful_projects
        ],
        'all_results': [
            {
                'name': r.project_name,
                'url': r.project_url,
                'clone_success': r.clone_success,
                'compile_success': r.compile_success,
                'checker_success': r.checker_success,
                'warning_count': r.warning_count,
                'error': r.error_message
            }
            for r in all_results
        ]
    }
    
    summary_file = case_studies_dir / 'signature_string_project_discovery.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    return successful_projects


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find projects for Signature String Checker')
    parser.add_argument('--target', type=int, default=3,
                       help='Number of projects to find')
    parser.add_argument('--work-dir', default='/tmp/signature_string_discovery',
                       help='Working directory')
    parser.add_argument('--min-warnings', type=int, default=5,
                       help='Minimum warnings required')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    case_studies_dir = Path('/home/ubuntu/GenDATA/case_studies')
    
    results = find_signature_string_projects(
        target_count=args.target,
        work_dir=work_dir,
        case_studies_dir=case_studies_dir,
        min_warnings=args.min_warnings
    )
    
    print(f"\n{'='*60}")
    print(f"Signature String Checker Project Discovery Results")
    print(f"{'='*60}")
    print(f"Found {len(results)}/{args.target} suitable projects:")
    for r in results:
        print(f"  - {r.project_name}: {r.warning_count} warnings")
    print(f"{'='*60}")
    
    if len(results) < args.target:
        print(f"\nWARNING: Only found {len(results)} projects, need {args.target}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
