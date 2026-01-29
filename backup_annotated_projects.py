#!/usr/bin/env python3
"""
Backup Annotated Projects

This script creates backups of projects that have been annotated with 
Checker Framework type annotations for training purposes.

Projects are organized by checker type:
- sql_quotes/: Projects with @SqlEvenQuotes, @SqlOddQuotes annotations
- signature_string/: Projects with @BinaryName, @FullyQualifiedName, etc.
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
BACKUP_DIR = GEN_DATA_ROOT / 'annotated_projects_backup'

# Projects to backup, organized by checker type
ANNOTATED_PROJECTS = {
    'sql_quotes': [
        'commons-dbutils',      # 51 @SqlEvenQuotes annotations
        'training_sql_quotes',  # Training examples
        'cf_sqlquotes_tests',   # Checker Framework test suite
    ],
    'signature_string': [
        'kryo',                 # 3 @BinaryName annotations
        'guice',                # 3 @BinaryName annotations
        'training_signature',   # Training examples
        'cf_signature_tests',   # Checker Framework test suite
    ]
}


def create_backup_structure():
    """Create the backup directory structure"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    for checker_type in ANNOTATED_PROJECTS.keys():
        checker_dir = BACKUP_DIR / checker_type
        checker_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created backup directory: {checker_dir}")


def backup_project(project_name: str, checker_type: str) -> bool:
    """
    Backup a single project
    
    Args:
        project_name: Name of the project directory
        checker_type: Type of checker (sql_quotes or signature_string)
        
    Returns:
        True if backup successful, False otherwise
    """
    source_dir = CASE_STUDIES_DIR / project_name
    target_dir = BACKUP_DIR / checker_type / project_name
    
    if not source_dir.exists():
        logger.warning(f"Source directory not found: {source_dir}")
        return False
    
    try:
        # Remove existing backup if present
        if target_dir.exists():
            shutil.rmtree(target_dir)
            logger.info(f"Removed existing backup: {target_dir}")
        
        # Copy the project
        shutil.copytree(source_dir, target_dir)
        logger.info(f"Backed up {project_name} to {target_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error backing up {project_name}: {e}")
        return False


def backup_all_projects() -> Dict[str, List[str]]:
    """
    Backup all annotated projects
    
    Returns:
        Dictionary with 'success' and 'failed' lists
    """
    results = {
        'success': [],
        'failed': []
    }
    
    # Create directory structure
    create_backup_structure()
    
    # Backup each project
    for checker_type, projects in ANNOTATED_PROJECTS.items():
        logger.info(f"\nBacking up {checker_type} projects...")
        
        for project_name in projects:
            if backup_project(project_name, checker_type):
                results['success'].append(f"{checker_type}/{project_name}")
            else:
                results['failed'].append(f"{checker_type}/{project_name}")
    
    return results


def create_backup_manifest():
    """Create a manifest file with backup information"""
    manifest_path = BACKUP_DIR / 'MANIFEST.md'
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = f"""# Annotated Projects Backup Manifest

**Created**: {timestamp}

## Purpose

This directory contains backups of Java projects that have been annotated with
Checker Framework type annotations for training annotation placement models.

## Directory Structure

```
annotated_projects_backup/
├── sql_quotes/
│   ├── commons-dbutils/     # 51 @SqlEvenQuotes annotations
│   ├── training_sql_quotes/ # Training examples with warnings
│   └── cf_sqlquotes_tests/  # Checker Framework test suite
└── signature_string/
    ├── kryo/                # 3 @BinaryName annotations  
    ├── guice/               # 3 @BinaryName annotations
    ├── training_signature/  # Training examples with warnings
    └── cf_signature_tests/  # Checker Framework test suite
```

## Annotations by Checker

### SQL Quotes Checker
- `@SqlEvenQuotes` - String with even number of SQL quotes
- `@SqlOddQuotes` - String with odd number of SQL quotes
- Import: `org.checkerframework.checker.sqlquotes.qual.*`

### Signature String Checker
- `@BinaryName` - Binary class name (e.g., "java.lang.String")
- `@FullyQualifiedName` - Fully qualified name (e.g., "java.lang.String")  
- `@FieldDescriptor` - Field descriptor (e.g., "Ljava/lang/String;")
- `@ClassGetName` - Result of Class.getName()
- `@InternalForm` - Internal form (e.g., "java/lang/String")
- Import: `org.checkerframework.checker.signature.qual.*`

## Usage

These projects can be used to:
1. Train annotation placement models
2. Test checker warnings before/after annotation
3. Verify warning reduction with correct annotations

## Running Checkers

```bash
# SQL Quotes Checker
/home/ubuntu/checker-framework/checker/bin/javac \\
  -processor org.checkerframework.checker.sqlquotes.SqlQuotesChecker \\
  YourFile.java

# Signature String Checker
/home/ubuntu/checker-framework/checker/bin/javac \\
  -processor org.checkerframework.checker.signature.SignatureChecker \\
  YourFile.java
```
"""
    
    with open(manifest_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Created manifest: {manifest_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup annotated projects')
    parser.add_argument('--list', action='store_true',
                       help='List projects to backup without actually backing up')
    
    args = parser.parse_args()
    
    if args.list:
        print("Projects to backup:")
        for checker_type, projects in ANNOTATED_PROJECTS.items():
            print(f"\n{checker_type}:")
            for project in projects:
                source = CASE_STUDIES_DIR / project
                exists = "EXISTS" if source.exists() else "MISSING"
                print(f"  - {project} [{exists}]")
        return 0
    
    print("="*60)
    print("Backing up Annotated Projects")
    print("="*60)
    
    results = backup_all_projects()
    
    # Create manifest
    create_backup_manifest()
    
    print("\n" + "="*60)
    print("Backup Summary")
    print("="*60)
    print(f"Successful: {len(results['success'])}")
    for project in results['success']:
        print(f"  - {project}")
    
    if results['failed']:
        print(f"\nFailed: {len(results['failed'])}")
        for project in results['failed']:
            print(f"  - {project}")
    
    print(f"\nBackup location: {BACKUP_DIR}")
    
    return 0 if not results['failed'] else 1


if __name__ == '__main__':
    exit(main())
