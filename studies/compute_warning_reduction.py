#!/usr/bin/env python3
"""
Compute warning reduction after applying predicted annotations.

This module calculates warning reduction by:
1. Counting baseline warnings from original warnings file
2. Applying predicted annotations to Java files
3. Re-running checker on annotated files
4. Counting remaining warnings
5. Calculating reduction percentage
"""

import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import annotation application utility
from studies.apply_predictions_to_files import apply_annotations_to_project

# Import checker framework runner
try:
    from checker_framework_runner import CheckerFrameworkRunner
except ImportError:
    logger.warning("checker_framework_runner not available, will use subprocess directly")
    CheckerFrameworkRunner = None


def count_warnings_in_file(warnings_file: Path) -> int:
    """Count warnings in a warnings file."""
    if not warnings_file.exists():
        return 0
    
    try:
        with open(warnings_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count warning lines (typically start with file path or error message)
        # Format varies, but usually one warning per line or block
        lines = content.strip().split('\n')
        
        # Count non-empty lines that look like warnings
        # Warnings typically contain file paths or error messages
        warning_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains warning indicators
            if any(indicator in line.lower() for indicator in ['error:', 'warning:', '.java:', 'indexchecker']):
                warning_count += 1
            # Also count lines that look like file paths followed by line numbers
            elif '.java:' in line and any(char.isdigit() for char in line):
                warning_count += 1
        
        # If no clear pattern, count non-empty lines
        if warning_count == 0:
            warning_count = len([l for l in lines if l.strip()])
        
        return warning_count
    
    except Exception as e:
        logger.error(f"Error counting warnings in {warnings_file}: {e}")
        return 0


def count_baseline_warnings(warnings_file: Path) -> int:
    """Count baseline warnings from original warnings file."""
    return count_warnings_in_file(warnings_file)


def run_checker_on_annotated_files(project_path: Path, annotated_files: Dict[str, Path],
                                   output_file: Path) -> int:
    """Run checker on annotated files and return warning count."""
    if CheckerFrameworkRunner:
        # Use CheckerFrameworkRunner if available
        runner = CheckerFrameworkRunner()
        
        # Create temporary directory with annotated files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_project = Path(temp_dir) / project_path.name
            temp_project.mkdir(parents=True)
            
            # Copy annotated files to temp directory
            for source_file, annotated_file in annotated_files.items():
                # Preserve directory structure
                rel_path = Path(source_file).relative_to(project_path)
                target_file = temp_project / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(annotated_file, target_file)
            
            # Run checker on temp project
            success = runner.run_checker_on_project(
                str(temp_project),
                str(output_file),
                max_files=1000  # Process all annotated files
            )
            
            if success:
                return count_warnings_in_file(output_file)
            else:
                logger.warning("Checker failed to run on annotated files")
                return -1
    
    else:
        # Fallback: use subprocess directly
        logger.warning("Using subprocess fallback for checker execution")
        return -1


def calculate_warning_reduction(baseline_count: int, remaining_count: int) -> float:
    """Calculate warning reduction percentage."""
    if baseline_count == 0:
        # No baseline warnings - return 0% reduction (or 100% if no remaining warnings)
        return 100.0 if remaining_count == 0 else 0.0
    
    reduction = ((baseline_count - remaining_count) / baseline_count) * 100.0
    
    # Ensure reduction is between 0 and 100
    return max(0.0, min(100.0, reduction))


def compute_warning_reduction_for_model(project_name: str, model_name: str,
                                       baseline_warnings_file: Path,
                                       project_path: Path) -> Dict:
    """Compute warning reduction for a specific model."""
    logger.info(f"Computing warning reduction for {project_name} with model {model_name}")
    
    # Load predictions
    predictions_file = project_path / f'predictions_{model_name}.json'
    
    if not predictions_file.exists():
        logger.warning(f"Predictions file not found: {predictions_file}")
        return {
            'baseline_warnings': 0,
            'remaining_warnings': 0,
            'reduction_percentage': 0.0,
            'error': 'predictions_file_not_found'
        }
    
    # Count baseline warnings
    baseline_count = count_baseline_warnings(baseline_warnings_file)
    logger.info(f"Baseline warnings: {baseline_count}")
    
    if baseline_count == 0:
        logger.warning("No baseline warnings found")
        return {
            'baseline_warnings': 0,
            'remaining_warnings': 0,
            'reduction_percentage': 0.0,
            'note': 'no_baseline_warnings'
        }
    
    # Apply annotations
    logger.info(f"Applying annotations from {predictions_file}")
    annotated_files = apply_annotations_to_project(project_path, predictions_file)
    
    if not annotated_files:
        logger.warning("No annotated files created")
        return {
            'baseline_warnings': baseline_count,
            'remaining_warnings': baseline_count,
            'reduction_percentage': 0.0,
            'error': 'no_annotated_files'
        }
    
    logger.info(f"Applied annotations to {len(annotated_files)} files")
    
    # Run checker on annotated files
    temp_warnings_file = project_path / f'temp_warnings_{model_name}.out'
    remaining_count = run_checker_on_annotated_files(
        project_path,
        annotated_files,
        temp_warnings_file
    )
    
    # Clean up temp warnings file
    if temp_warnings_file.exists():
        temp_warnings_file.unlink()
    
    if remaining_count < 0:
        logger.error("Failed to run checker on annotated files")
        return {
            'baseline_warnings': baseline_count,
            'remaining_warnings': baseline_count,
            'reduction_percentage': 0.0,
            'error': 'checker_execution_failed'
        }
    
    logger.info(f"Remaining warnings: {remaining_count}")
    
    # Calculate reduction
    reduction_percentage = calculate_warning_reduction(baseline_count, remaining_count)
    
    logger.info(f"Warning reduction: {reduction_percentage:.2f}%")
    
    return {
        'baseline_warnings': baseline_count,
        'remaining_warnings': remaining_count,
        'reduction_percentage': reduction_percentage,
        'num_annotated_files': len(annotated_files)
    }


def compute_warning_reduction_for_all_models(project_name: str,
                                            baseline_warnings_file: Path,
                                            project_path: Path,
                                            model_names: List[str]) -> Dict[str, Dict]:
    """Compute warning reduction for all models."""
    results = {}
    
    for model_name in model_names:
        try:
            result = compute_warning_reduction_for_model(
                project_name,
                model_name,
                baseline_warnings_file,
                project_path
            )
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error computing warning reduction for {model_name}: {e}")
            results[model_name] = {
                'baseline_warnings': 0,
                'remaining_warnings': 0,
                'reduction_percentage': 0.0,
                'error': str(e)
            }
    
    return results


def main():
    """Main function for testing."""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python compute_warning_reduction.py <project_name> <model_name> <baseline_warnings_file>")
        print("Example: python compute_warning_reduction.py agrona gbt case_studies/agrona/agrona_warnings.out")
        sys.exit(1)
    
    project_name = sys.argv[1]
    model_name = sys.argv[2]
    baseline_warnings_file = Path(sys.argv[3])
    project_path = Path('/home/ubuntu/GenDATA/case_studies') / project_name
    
    if not project_path.exists():
        print(f"Error: Project path not found: {project_path}")
        sys.exit(1)
    
    if not baseline_warnings_file.exists():
        print(f"Error: Baseline warnings file not found: {baseline_warnings_file}")
        sys.exit(1)
    
    result = compute_warning_reduction_for_model(
        project_name,
        model_name,
        baseline_warnings_file,
        project_path
    )
    
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

