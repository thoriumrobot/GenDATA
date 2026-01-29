#!/usr/bin/env python3
"""
Backup Case Studies

This script creates comprehensive backups of all case study projects
used for training and evaluation of annotation placement models.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'
BACKUP_DIR = GEN_DATA_ROOT / 'case_studies_backup'

# Projects to backup by category
PROJECTS_BY_CATEGORY = {
    'annotated': [
        'commons-dbutils',      # SQL Quotes - 51 annotations
        'kryo',                 # Signature String - 3 annotations
        'guice',                # Signature String - 3 annotations
    ],
    'training_examples': [
        'training_sql_quotes',  # SQL Quotes training examples
        'training_signature',   # Signature String training examples
    ],
    'cf_test_suites': [
        'cf_sqlquotes_tests',   # Checker Framework SQL Quotes tests
        'cf_signature_tests',   # Checker Framework Signature tests
    ],
    'evaluation': [
        'agrona',
        'eclipse-collections',
        'guava',
        'hipparchus',
        'jfreechart',
        'plume-lib',
    ],
    'other': [
        'cglib',
        'commons-dbcp',
        'mybatis-3',
    ]
}


def get_all_projects() -> List[str]:
    """Get list of all projects to backup"""
    projects = []
    for category_projects in PROJECTS_BY_CATEGORY.values():
        projects.extend(category_projects)
    return projects


def backup_project(project_name: str) -> bool:
    """
    Backup a single project
    
    Args:
        project_name: Name of the project directory
        
    Returns:
        True if backup successful, False otherwise
    """
    source_dir = CASE_STUDIES_DIR / project_name
    target_dir = BACKUP_DIR / project_name
    
    if not source_dir.exists():
        logger.warning(f"Source directory not found: {source_dir}")
        return False
    
    try:
        # Remove existing backup if present
        if target_dir.exists():
            shutil.rmtree(target_dir)
            logger.debug(f"Removed existing backup: {target_dir}")
        
        # Copy the project
        shutil.copytree(source_dir, target_dir)
        logger.info(f"Backed up {project_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error backing up {project_name}: {e}")
        return False


def backup_all_projects() -> Dict[str, List[str]]:
    """
    Backup all case study projects
    
    Returns:
        Dictionary with 'success' and 'failed' lists
    """
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all projects
    all_projects = get_all_projects()
    
    # Backup each project
    for project_name in all_projects:
        source_dir = CASE_STUDIES_DIR / project_name
        if not source_dir.exists():
            results['skipped'].append(project_name)
            logger.warning(f"Skipping {project_name} (not found)")
            continue
            
        if backup_project(project_name):
            results['success'].append(project_name)
        else:
            results['failed'].append(project_name)
    
    return results


def create_backup_manifest(results: Dict[str, List[str]]):
    """Create a manifest file with backup information"""
    manifest_path = BACKUP_DIR / 'MANIFEST.md'
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"""# Case Studies Backup Manifest

**Created**: {timestamp}

## Purpose

This directory contains backups of all case study projects used for training
and evaluating annotation placement models for the Checker Framework.

## Projects by Category

### Annotated Projects (with injected annotations for training)
"""
    
    for project in PROJECTS_BY_CATEGORY['annotated']:
        status = "✓" if project in results['success'] else "✗"
        content += f"- {status} `{project}/`\n"
    
    content += """
### Training Examples (hand-crafted examples)
"""
    for project in PROJECTS_BY_CATEGORY['training_examples']:
        status = "✓" if project in results['success'] else "✗"
        content += f"- {status} `{project}/`\n"
    
    content += """
### Checker Framework Test Suites
"""
    for project in PROJECTS_BY_CATEGORY['cf_test_suites']:
        status = "✓" if project in results['success'] else "✗"
        content += f"- {status} `{project}/`\n"
    
    content += """
### Evaluation Projects
"""
    for project in PROJECTS_BY_CATEGORY['evaluation']:
        status = "✓" if project in results['success'] else "✗"
        content += f"- {status} `{project}/`\n"
    
    content += """
### Other Projects
"""
    for project in PROJECTS_BY_CATEGORY['other']:
        status = "✓" if project in results['success'] else "✗"
        content += f"- {status} `{project}/`\n"
    
    content += f"""
## Backup Summary

- **Successful**: {len(results['success'])}
- **Failed**: {len(results['failed'])}
- **Skipped**: {len(results['skipped'])}

## Checkers Covered

- **Lower Bound Checker**: `@Positive`, `@NonNegative`, `@GTENegativeOne`
- **SQL Quotes Checker**: `@SqlEvenQuotes`, `@SqlOddQuotes`
- **Signature String Checker**: `@BinaryName`, `@FullyQualifiedName`, `@FieldDescriptor`, `@ClassGetName`, `@InternalForm`

## Usage

To restore a project:
```bash
cp -r case_studies_backup/PROJECT_NAME case_studies/
```

To restore all projects:
```bash
cp -r case_studies_backup/* case_studies/
```
"""
    
    with open(manifest_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Created manifest: {manifest_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup case study projects')
    parser.add_argument('--list', action='store_true',
                       help='List projects to backup without actually backing up')
    parser.add_argument('--category', choices=list(PROJECTS_BY_CATEGORY.keys()),
                       help='Backup only projects in a specific category')
    
    args = parser.parse_args()
    
    if args.list:
        print("Projects to backup:")
        for category, projects in PROJECTS_BY_CATEGORY.items():
            print(f"\n{category}:")
            for project in projects:
                source = CASE_STUDIES_DIR / project
                exists = "EXISTS" if source.exists() else "MISSING"
                print(f"  - {project} [{exists}]")
        return 0
    
    print("="*60)
    print("Backing up Case Study Projects")
    print("="*60)
    
    results = backup_all_projects()
    
    # Create manifest
    create_backup_manifest(results)
    
    print("\n" + "="*60)
    print("Backup Summary")
    print("="*60)
    print(f"Successful: {len(results['success'])}")
    for project in results['success']:
        print(f"  ✓ {project}")
    
    if results['skipped']:
        print(f"\nSkipped (not found): {len(results['skipped'])}")
        for project in results['skipped']:
            print(f"  - {project}")
    
    if results['failed']:
        print(f"\nFailed: {len(results['failed'])}")
        for project in results['failed']:
            print(f"  ✗ {project}")
    
    print(f"\nBackup location: {BACKUP_DIR}")
    
    return 0 if not results['failed'] else 1


if __name__ == '__main__':
    exit(main())
