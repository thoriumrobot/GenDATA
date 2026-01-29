#!/usr/bin/env python3
"""
Backup Safety Module

This module provides safety checks to ensure backup directories are never modified.
All evaluation scripts should use these functions before any write operations.

Usage:
    from backup_safety import verify_not_backup_dir, restore_from_backup

    # Before any write operation:
    if not verify_not_backup_dir(path_to_modify):
        raise ValueError("Cannot modify backup directory!")
        
    # To restore a project (copies from backup, never modifies backup):
    restore_from_backup(project_name, target_dir)
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')

# Backup directories that should NEVER be modified
BACKUP_DIRECTORIES = [
    GEN_DATA_ROOT / 'case_studies_backup',
    GEN_DATA_ROOT / 'annotation_evaluation' / 'backups',
    GEN_DATA_ROOT / 'annotated_projects_backup',
]


def get_backup_directories() -> List[Path]:
    """Get list of all backup directories"""
    return BACKUP_DIRECTORIES


def is_backup_directory(path: Path) -> bool:
    """
    Check if a path is inside a backup directory.
    
    Args:
        path: Path to check
        
    Returns:
        True if path is inside a backup directory
    """
    path = Path(path).resolve()
    
    for backup_dir in BACKUP_DIRECTORIES:
        if not backup_dir.exists():
            continue
        backup_dir = backup_dir.resolve()
        try:
            path.relative_to(backup_dir)
            return True  # Path is under this backup directory
        except ValueError:
            continue  # Path is not under this backup directory
    
    return False


def verify_not_backup_dir(path: Path, raise_error: bool = False) -> bool:
    """
    Verify that a path is NOT inside a backup directory.
    
    This should be called before any write operation to ensure backups
    are never accidentally modified.
    
    Args:
        path: Path to check
        raise_error: If True, raise ValueError instead of returning False
        
    Returns:
        True if path is safe to modify (NOT a backup directory)
        
    Raises:
        ValueError: If raise_error=True and path is a backup directory
    """
    if is_backup_directory(path):
        msg = f"SAFETY CHECK FAILED: Attempted to write to backup directory: {path}"
        logger.error(msg)
        if raise_error:
            raise ValueError(msg)
        return False
    return True


def find_backup_source(project_name: str, checker_name: str = None) -> Optional[Path]:
    """
    Find a backup source for a project.
    
    Searches backup directories in order of preference:
    1. For SQL Quotes and Signature String: evaluation_ready (has entry-point annotations)
    2. Annotated projects backup
    3. Annotation evaluation backups
    4. Case studies backup
    
    Args:
        project_name: Name of the project
        checker_name: Name of the checker (affects search order)
        
    Returns:
        Path to backup directory if found, None otherwise
    """
    backup_sources = []
    
    # For SQL Quotes and Signature String, prefer evaluation_ready (has entry-point annotations)
    if checker_name in ['sql_quotes', 'signature_string']:
        eval_ready = GEN_DATA_ROOT / 'annotation_evaluation' / 'evaluation_ready' / checker_name / project_name
        if eval_ready.exists():
            backup_sources.append(eval_ready)
        
        # Also check annotated projects backup
        annotated = GEN_DATA_ROOT / 'annotated_projects_backup' / checker_name / project_name
        if annotated.exists():
            backup_sources.append(annotated)
    
    # Standard backup locations
    backup_sources.extend([
        GEN_DATA_ROOT / 'annotation_evaluation' / 'backups' / project_name,
        GEN_DATA_ROOT / 'case_studies_backup' / project_name,
    ])
    
    for source in backup_sources:
        if source.exists():
            return source
    
    return None


def restore_from_backup(project_name: str, target_dir: Path, 
                       force: bool = False, checker_name: str = None) -> bool:
    """
    Restore a project from backup to target directory.
    
    This function:
    1. Verifies target_dir is NOT a backup directory (safety check)
    2. Finds the backup source
    3. Removes existing target_dir if it exists
    4. Copies from backup to target
    
    Args:
        project_name: Name of the project to restore
        target_dir: Target directory (must NOT be a backup directory!)
        force: If True, overwrite existing target_dir
        checker_name: Checker name (affects backup source priority)
        
    Returns:
        True if restore successful
        
    Raises:
        ValueError: If target_dir is a backup directory
    """
    target_dir = Path(target_dir)
    
    # SAFETY CHECK: Never write to backup directories
    if not verify_not_backup_dir(target_dir, raise_error=True):
        return False
    
    # Find backup source
    source = find_backup_source(project_name, checker_name)
    if not source:
        logger.error(f"No backup found for {project_name}")
        logger.error(f"  Checked: evaluation_ready/, annotated_projects_backup/, annotation_evaluation/backups/, case_studies_backup/")
        return False
    
    try:
        # Remove existing target if it exists
        if target_dir.exists():
            if not force and not is_safe_to_overwrite(target_dir):
                logger.warning(f"Target exists and may contain important data: {target_dir}")
                logger.warning(f"  Use force=True to overwrite")
                return False
            shutil.rmtree(target_dir)
            logger.debug(f"Removed existing: {target_dir}")
        
        # Create parent directory
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy from backup
        shutil.copytree(source, target_dir)
        logger.info(f"Restored {project_name}: {source} -> {target_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error restoring {project_name}: {e}")
        return False


def is_safe_to_overwrite(path: Path) -> bool:
    """
    Check if a directory is safe to overwrite.
    
    Safe to overwrite if:
    - It's in a temporary directory (temp_repos, wpi_work, etc.)
    - It's not a backup directory
    """
    path = Path(path).resolve()
    
    # Always safe if in known working directories
    safe_parent_names = [
        'temp_repos', 'wpi_work', 'wpi_projects', 'temp', 
        'working', 'evaluation_work'
    ]
    
    for parent in path.parents:
        if parent.name in safe_parent_names:
            return True
    
    return True  # Default to allowing for flexibility


def protect_backup_decorator(func):
    """
    Decorator that ensures a function doesn't write to backup directories.
    
    Use on functions that take a 'path' or 'project_dir' argument.
    
    Usage:
        @protect_backup_decorator
        def my_function(project_dir: Path, ...):
            ...
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check for path-like arguments
        path_args = ['path', 'project_dir', 'target_dir', 'output_dir', 'dest']
        
        for arg_name in path_args:
            if arg_name in kwargs:
                path = kwargs[arg_name]
                if path is not None:
                    verify_not_backup_dir(Path(path), raise_error=True)
        
        return func(*args, **kwargs)
    
    return wrapper


def verify_backups_unchanged(before_checksums: dict, after_checksums: dict) -> bool:
    """
    Verify that backup directories were not modified.
    
    Args:
        before_checksums: Dict of {path: checksum} before operation
        after_checksums: Dict of {path: checksum} after operation
        
    Returns:
        True if all checksums match (backups unchanged)
    """
    for path, checksum in before_checksums.items():
        if path in after_checksums and after_checksums[path] != checksum:
            logger.error(f"Backup was modified: {path}")
            logger.error(f"  Before: {checksum}")
            logger.error(f"  After: {after_checksums[path]}")
            return False
    return True


def compute_directory_checksum(directory: Path) -> str:
    """
    Compute a simple checksum for a directory (based on file count and size).
    
    Args:
        directory: Directory to checksum
        
    Returns:
        Checksum string
    """
    import hashlib
    
    directory = Path(directory)
    if not directory.exists():
        return "DOES_NOT_EXIST"
    
    file_count = 0
    total_size = 0
    
    for f in directory.rglob('*'):
        if f.is_file():
            file_count += 1
            total_size += f.stat().st_size
    
    checksum_data = f"{file_count}:{total_size}"
    return hashlib.md5(checksum_data.encode()).hexdigest()[:16]


def snapshot_backups() -> dict:
    """
    Take a snapshot of backup directories for later verification.
    
    Returns:
        Dict mapping backup paths to checksums
    """
    snapshots = {}
    
    for backup_dir in BACKUP_DIRECTORIES:
        if backup_dir.exists():
            for project_dir in backup_dir.iterdir():
                if project_dir.is_dir():
                    key = str(project_dir)
                    snapshots[key] = compute_directory_checksum(project_dir)
    
    return snapshots


# Export for convenience
__all__ = [
    'verify_not_backup_dir',
    'restore_from_backup',
    'find_backup_source',
    'is_backup_directory',
    'get_backup_directories',
    'protect_backup_decorator',
    'snapshot_backups',
    'verify_backups_unchanged',
    'compute_directory_checksum',
]
