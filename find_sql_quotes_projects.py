#!/usr/bin/env python3
"""
Find SQL Quotes Checker Projects

This script finds GitHub Java projects that are suitable for SQL Quotes Checker evaluation.
It searches for projects with SQL/database code and evaluates them until 3 suitable projects
are found.
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

# Known projects that are likely to have SQL-related code
# These are curated based on their known SQL usage
SQL_PROJECT_CANDIDATES = [
    # JDBC/Database libraries
    'https://github.com/brettwooldridge/HikariCP.git',
    'https://github.com/mybatis/mybatis-3.git',
    'https://github.com/jdbi/jdbi.git',
    'https://github.com/jOOQ/jOOQ.git',
    'https://github.com/querydsl/querydsl.git',
    'https://github.com/apache/commons-dbcp.git',
    'https://github.com/apache/commons-dbutils.git',
    
    # Database tools
    'https://github.com/flyway/flyway.git',
    'https://github.com/liquibase/liquibase.git',
    'https://github.com/h2database/h2database.git',
    
    # Web frameworks with DB access
    'https://github.com/dropwizard/dropwizard.git',
    'https://github.com/micronaut-projects/micronaut-data.git',
    
    # SQL utilities
    'https://github.com/jsqlparser/jsqlparser.git',
    'https://github.com/calcite-rs/calcite.git',
    'https://github.com/apache/shardingsphere.git',
    
    # Smaller database utilities
    'https://github.com/yasserg/crawler4j.git',
    'https://github.com/alibaba/druid.git',
    'https://github.com/p6spy/p6spy.git',
    'https://github.com/prestodb/presto.git',
]


def find_sql_quotes_projects(
    target_count: int = 3,
    work_dir: Path = None,
    case_studies_dir: Path = None,
    min_warnings: int = 5
) -> List[ProjectEvaluationResult]:
    """
    Find projects suitable for SQL Quotes Checker evaluation
    
    Args:
        target_count: Number of suitable projects to find
        work_dir: Working directory for cloning
        case_studies_dir: Directory to save successful projects
        min_warnings: Minimum warnings required
        
    Returns:
        List of successful ProjectEvaluationResults
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix='sql_quotes_'))
    if case_studies_dir is None:
        case_studies_dir = Path('/home/ubuntu/GenDATA/case_studies')
    
    successful_projects = []
    all_results = []
    
    logger.info(f"Searching for {target_count} projects suitable for SQL Quotes Checker...")
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Case studies directory: {case_studies_dir}")
    
    for project_url in SQL_PROJECT_CANDIDATES:
        if len(successful_projects) >= target_count:
            break
        
        project_name = project_url.rstrip('/').split('/')[-1].replace('.git', '')
        
        # Skip if already in case_studies with valid warnings
        existing_warnings = case_studies_dir / project_name / f"{project_name}_sql_quotes_warnings.out"
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
            checker_name='sql_quotes',
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
        'checker': 'sql_quotes',
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
    
    summary_file = case_studies_dir / 'sql_quotes_project_discovery.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    return successful_projects


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find projects for SQL Quotes Checker')
    parser.add_argument('--target', type=int, default=3,
                       help='Number of projects to find')
    parser.add_argument('--work-dir', default='/tmp/sql_quotes_discovery',
                       help='Working directory')
    parser.add_argument('--min-warnings', type=int, default=5,
                       help='Minimum warnings required')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    case_studies_dir = Path('/home/ubuntu/GenDATA/case_studies')
    
    results = find_sql_quotes_projects(
        target_count=args.target,
        work_dir=work_dir,
        case_studies_dir=case_studies_dir,
        min_warnings=args.min_warnings
    )
    
    print(f"\n{'='*60}")
    print(f"SQL Quotes Checker Project Discovery Results")
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
