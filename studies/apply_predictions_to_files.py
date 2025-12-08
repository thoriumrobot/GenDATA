#!/usr/bin/env python3
"""
Apply predicted annotations to Java source files.

This utility applies annotations from predictions_{model}.json files to Java
source files, creating annotated copies for warning reduction evaluation.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lower Bound Checker annotations
LOWER_BOUND_ANNOTATIONS = ['@Positive', '@NonNegative', '@GTENegativeOne']


def load_predictions(predictions_file: Path) -> List[Dict]:
    """Load predictions from predictions_{model}.json file."""
    try:
        with open(predictions_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load predictions from {predictions_file}: {e}")
        return []


def group_predictions_by_file(predictions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group predictions by file path."""
    grouped = {}
    
    for entry in predictions:
        file_path = entry.get('file_path', '')
        preds = entry.get('predictions', [])
        
        if file_path and preds:
            if file_path not in grouped:
                grouped[file_path] = []
            grouped[file_path].extend(preds)
    
    return grouped


def find_line_in_code(lines: List[str], target_line: int, annotation: str) -> Optional[int]:
    """Find the best line to place annotation, accounting for context."""
    if target_line < 1 or target_line > len(lines):
        return None
    
    # Try exact line first
    line_idx = target_line - 1
    line_content = lines[line_idx].strip()
    
    # Check if line already has this annotation
    if annotation in line_content:
        return None  # Already annotated
    
    # Check if this is a variable declaration, parameter, or return type
    # Look for patterns that indicate where annotation should go
    
    # For variable declarations: look for type name pattern
    if re.search(r'\b(int|long|short|byte|float|double|char|boolean|String|Integer|Long)\s+\w+', line_content):
        return line_idx
    
    # For method parameters: look for parameter in method signature
    if '(' in line_content and ')' in line_content:
        return line_idx
    
    # For return types: look for method signature
    if re.search(r'\b(public|private|protected|static)?\s*\w+\s+\w+\s*\(', line_content):
        return line_idx
    
    # Default: place before the line
    return line_idx


def apply_annotation_to_line(lines: List[str], line_idx: int, annotation: str) -> bool:
    """Apply annotation to a specific line."""
    if line_idx < 0 or line_idx >= len(lines):
        return False
    
    line = lines[line_idx]
    
    # Check if annotation already exists
    if annotation in line:
        return False
    
    # Get indentation
    indent_match = re.match(r'^(\s*)', line)
    indent = indent_match.group(1) if indent_match else ''
    
    # Try to insert annotation before the line
    # For variable declarations and parameters, try inline annotation
    if re.search(r'\b(int|long|short|byte|float|double|char|boolean|String|Integer|Long)\s+\w+', line):
        # Try to insert before type
        annotated_line = re.sub(
            r'(\b(int|long|short|byte|float|double|char|boolean|String|Integer|Long)\s+)',
            f'{annotation} \\1',
            line,
            count=1
        )
        if annotated_line != line:
            lines[line_idx] = annotated_line
            return True
    
    # For method parameters, try to insert before parameter type
    if '(' in line and ')' in line:
        # Try to insert annotation before first parameter type
        annotated_line = re.sub(
            r'(\([^)]*?)(\b(int|long|short|byte|float|double|char|boolean|String|Integer|Long)\s+\w+)',
            f'\\1{annotation} \\2',
            line,
            count=1
        )
        if annotated_line != line:
            lines[line_idx] = annotated_line
            return True
    
    # Fallback: insert annotation line before this line
    annotation_line = f"{indent}{annotation}\n"
    lines.insert(line_idx, annotation_line)
    return True


def apply_annotations_to_file(java_file: Path, predictions: List[Dict], output_file: Optional[Path] = None) -> Optional[Path]:
    """Apply annotations to a single Java file."""
    if not java_file.exists():
        logger.warning(f"Java file not found: {java_file}")
        return None
    
    try:
        # Read file
        with open(java_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Group predictions by line number (process in reverse to avoid line shifts)
        predictions_by_line = {}
        for pred in predictions:
            line_num = pred.get('line')
            ann_type = pred.get('type')
            
            if line_num and ann_type and ann_type in LOWER_BOUND_ANNOTATIONS:
                if line_num not in predictions_by_line:
                    predictions_by_line[line_num] = []
                predictions_by_line[line_num].append(ann_type)
        
        # Sort line numbers in reverse order
        sorted_lines = sorted(predictions_by_line.keys(), reverse=True)
        
        # Apply annotations
        applied_count = 0
        for line_num in sorted_lines:
            annotations = predictions_by_line[line_num]
            
            # Find target line
            line_idx = find_line_in_code(lines, line_num, annotations[0])
            
            if line_idx is not None:
                # Apply each annotation
                for annotation in annotations:
                    if apply_annotation_to_line(lines, line_idx, annotation):
                        applied_count += 1
        
        # Write to output file
        if output_file is None:
            output_file = java_file.parent / f"{java_file.stem}.annotated{java_file.suffix}"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        logger.info(f"Applied {applied_count} annotations to {java_file.name}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error applying annotations to {java_file}: {e}")
        return None


def apply_annotations_to_project(project_path: Path, predictions_file: Path, 
                                 output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Apply annotations to all files in a project."""
    # Load predictions
    predictions = load_predictions(predictions_file)
    
    if not predictions:
        logger.warning(f"No predictions found in {predictions_file}")
        return {}
    
    # Group by file
    predictions_by_file = group_predictions_by_file(predictions)
    
    # Create output directory
    if output_dir is None:
        output_dir = project_path / 'annotated_files'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply annotations to each file
    annotated_files = {}
    
    for file_path_str, file_predictions in predictions_by_file.items():
        # Resolve file path
        file_path = Path(file_path_str)
        
        # Try to find file in project
        if not file_path.exists():
            # Try relative to project path
            rel_path = file_path
            if 'case_studies' in str(file_path):
                # Extract relative path
                parts = str(file_path).split('case_studies/')
                if len(parts) > 1:
                    rel_path = project_path.parent / 'case_studies' / parts[1]
                else:
                    rel_path = project_path / file_path.name
            
            if rel_path.exists():
                file_path = rel_path
            else:
                # Try to find file by name
                found_files = list(project_path.rglob(file_path.name))
                if found_files:
                    file_path = found_files[0]
                else:
                    logger.warning(f"Could not find file: {file_path_str}")
                    continue
        
        # Apply annotations
        output_file = output_dir / file_path.name
        annotated_file = apply_annotations_to_file(file_path, file_predictions, output_file)
        
        if annotated_file:
            annotated_files[str(file_path)] = annotated_file
    
    logger.info(f"Applied annotations to {len(annotated_files)} files")
    return annotated_files


def create_annotated_copy(source_file: Path, annotations: List[Dict]) -> Optional[Path]:
    """Create an annotated copy of a source file."""
    return apply_annotations_to_file(source_file, annotations)


def main():
    """Main function for testing."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python apply_predictions_to_files.py <project_path> <predictions_file> [output_dir]")
        print("Example: python apply_predictions_to_files.py case_studies/agrona case_studies/agrona/predictions_gbt.json")
        sys.exit(1)
    
    project_path = Path(sys.argv[1])
    predictions_file = Path(sys.argv[2])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not project_path.exists():
        print(f"Error: Project path not found: {project_path}")
        sys.exit(1)
    
    if not predictions_file.exists():
        print(f"Error: Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    annotated_files = apply_annotations_to_project(project_path, predictions_file, output_dir)
    
    print(f"Successfully annotated {len(annotated_files)} files")
    for source, annotated in annotated_files.items():
        print(f"  {source} -> {annotated}")


if __name__ == '__main__':
    main()

