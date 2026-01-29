#!/usr/bin/env python3
"""
Ensure Project Backups Exist

This script ensures that all required project backups exist in the
annotation_evaluation/backups/ directory by copying from case_studies_backup/.

It NEVER modifies existing backups - only copies missing ones.

Usage:
    python ensure_project_backups.py
    python ensure_project_backups.py --dry-run
"""

import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_BACKUP = GEN_DATA_ROOT / 'case_studies_backup'
ANNOTATION_EVAL_BACKUPS = GEN_DATA_ROOT / 'annotation_evaluation' / 'backups'

# Required projects per checker for evaluation (3 real GitHub projects each, no training sets)
REQUIRED_PROJECTS = {
    'lower_bound': [
        'sortpom',
        'pom-tuner',
        'jfreechart',
    ],
    'sql_quotes': [
        'commons-dbutils',
        'commons-dbcp',
        'mybatis-3',
    ],
    'signature_string': [
        'kryo',
        'guice',
        'cglib',
    ],
}


def get_all_required_projects() -> List[str]:
    """Get list of all required projects"""
    projects = set()
    for checker_projects in REQUIRED_PROJECTS.values():
        projects.update(checker_projects)
    return sorted(list(projects))


def check_backup_exists(project_name: str) -> Dict[str, bool]:
    """
    Check if backup exists in either backup location.
    
    Returns:
        Dict with 'case_studies' and 'annotation_eval' keys
    """
    return {
        'case_studies': (CASE_STUDIES_BACKUP / project_name).exists(),
        'annotation_eval': (ANNOTATION_EVAL_BACKUPS / project_name).exists(),
    }


def copy_backup(project_name: str, dry_run: bool = False) -> bool:
    """
    Copy a project from case_studies_backup to annotation_evaluation/backups.
    
    Args:
        project_name: Name of the project
        dry_run: If True, only print what would be done
        
    Returns:
        True if copy successful (or would be successful in dry run)
    """
    source = CASE_STUDIES_BACKUP / project_name
    dest = ANNOTATION_EVAL_BACKUPS / project_name
    
    if not source.exists():
        logger.error(f"Source backup not found: {source}")
        return False
    
    if dest.exists():
        logger.info(f"Backup already exists: {dest}")
        return True
    
    if dry_run:
        logger.info(f"[DRY RUN] Would copy {source} -> {dest}")
        return True
    
    try:
        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the project
        shutil.copytree(source, dest)
        logger.info(f"Copied backup: {source} -> {dest}")
        return True
        
    except Exception as e:
        logger.error(f"Error copying {project_name}: {e}")
        return False


def ensure_all_backups(dry_run: bool = False) -> Dict[str, Dict]:
    """
    Ensure all required backups exist.
    
    Args:
        dry_run: If True, only print what would be done
        
    Returns:
        Dict with status for each project
    """
    results = {}
    
    # Ensure backup directory exists
    if not dry_run:
        ANNOTATION_EVAL_BACKUPS.mkdir(parents=True, exist_ok=True)
    
    all_projects = get_all_required_projects()
    
    logger.info(f"Checking {len(all_projects)} required projects...")
    
    for project_name in all_projects:
        status = check_backup_exists(project_name)
        
        if status['annotation_eval']:
            results[project_name] = {
                'status': 'exists',
                'location': str(ANNOTATION_EVAL_BACKUPS / project_name),
            }
        elif status['case_studies']:
            # Need to copy
            success = copy_backup(project_name, dry_run)
            results[project_name] = {
                'status': 'copied' if success else 'copy_failed',
                'source': str(CASE_STUDIES_BACKUP / project_name),
                'dest': str(ANNOTATION_EVAL_BACKUPS / project_name),
            }
        else:
            results[project_name] = {
                'status': 'not_found',
                'error': 'No backup found in either location',
            }
    
    return results


def print_summary(results: Dict[str, Dict]) -> None:
    """Print summary of results"""
    logger.info("\n" + "="*60)
    logger.info("BACKUP STATUS SUMMARY")
    logger.info("="*60)
    
    for checker, projects in REQUIRED_PROJECTS.items():
        logger.info(f"\n{checker}:")
        for project in projects:
            if project in results:
                status = results[project]['status']
                if status == 'exists':
                    logger.info(f"  [OK] {project}")
                elif status == 'copied':
                    logger.info(f"  [COPIED] {project}")
                elif status == 'copy_failed':
                    logger.info(f"  [FAILED] {project}")
                else:
                    logger.info(f"  [MISSING] {project}")
            else:
                logger.info(f"  [UNKNOWN] {project}")
    
    # Overall stats
    total = len(results)
    exists = sum(1 for r in results.values() if r['status'] == 'exists')
    copied = sum(1 for r in results.values() if r['status'] == 'copied')
    failed = sum(1 for r in results.values() if r['status'] in ['copy_failed', 'not_found'])
    
    logger.info(f"\nTotal: {total} projects")
    logger.info(f"  Already existed: {exists}")
    logger.info(f"  Newly copied: {copied}")
    logger.info(f"  Failed/Missing: {failed}")


def main():
    parser = argparse.ArgumentParser(description='Ensure all project backups exist')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only print what would be done, do not copy')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    logger.info(f"Source: {CASE_STUDIES_BACKUP}")
    logger.info(f"Target: {ANNOTATION_EVAL_BACKUPS}")
    
    results = ensure_all_backups(dry_run=args.dry_run)
    print_summary(results)
    
    # Check if any failed
    failed_count = sum(1 for r in results.values() if r['status'] in ['copy_failed', 'not_found'])
    
    if failed_count > 0:
        logger.warning(f"\n{failed_count} projects could not be backed up!")
        return 1
    
    logger.info("\nAll backups are in place!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
