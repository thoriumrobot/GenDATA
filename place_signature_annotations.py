#!/usr/bin/env python3
"""
Signature String Annotation Placement System

This module places Signature String Checker annotations on Java code
based on model predictions or heuristic analysis.

Annotation Types:
- @BinaryName: Binary class name (e.g., "java.lang.String$Inner")
- @FullyQualifiedName: Fully qualified name (e.g., "java.lang.String")
- @FieldDescriptor: Field descriptor (e.g., "Ljava/lang/String;")
- @ClassGetName: Result of Class.getName()
- @InternalForm: Internal form (e.g., "java/lang/String")

Import: org.checkerframework.checker.signature.qual.*
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignatureAnnotationType(Enum):
    """Signature String Checker annotation types"""
    BINARY_NAME = "@BinaryName"
    FULLY_QUALIFIED_NAME = "@FullyQualifiedName"
    FIELD_DESCRIPTOR = "@FieldDescriptor"
    CLASS_GET_NAME = "@ClassGetName"
    INTERNAL_FORM = "@InternalForm"


# Import statements for Signature String annotations
SIGNATURE_IMPORTS = [
    "import org.checkerframework.checker.signature.qual.BinaryName;",
    "import org.checkerframework.checker.signature.qual.FullyQualifiedName;",
    "import org.checkerframework.checker.signature.qual.FieldDescriptor;",
    "import org.checkerframework.checker.signature.qual.ClassGetName;",
    "import org.checkerframework.checker.signature.qual.InternalForm;",
]


@dataclass
class SignaturePlacement:
    """Represents a Signature String annotation placement"""
    file_path: str
    line_number: int
    annotation: SignatureAnnotationType
    target_element: str
    confidence: float = 1.0
    reason: str = ""


class SignatureAnnotationPlacer:
    """Places Signature String Checker annotations on Java code"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            self.lines = []
        self.placements: List[SignaturePlacement] = []
    
    def detect_signature_format(self, text: str) -> Optional[SignatureAnnotationType]:
        """
        Detect the signature format of a string
        
        Returns:
            The appropriate annotation type, or None if not a signature string
        """
        text = text.strip().strip('"').strip("'")
        
        if not text:
            return None
        
        # Field descriptor: L...;
        if text.startswith('L') and text.endswith(';') and '/' in text:
            return SignatureAnnotationType.FIELD_DESCRIPTOR
        
        # Array field descriptor: [L...; or [I, etc.
        if text.startswith('[') and ('L' in text or any(c in text for c in 'IJBZCSFD')):
            return SignatureAnnotationType.FIELD_DESCRIPTOR
        
        # Internal form: uses / as separator
        if '/' in text and '.' not in text and not text.startswith('http'):
            return SignatureAnnotationType.INTERNAL_FORM
        
        # Fully qualified name: uses . as separator, no $ for inner classes
        if '.' in text and '$' not in text and '/' not in text:
            # Check if it looks like a class name (starts with uppercase segments)
            parts = text.split('.')
            if len(parts) >= 2 and parts[-1][0:1].isupper():
                return SignatureAnnotationType.FULLY_QUALIFIED_NAME
        
        # Binary name: uses . as separator with $ for inner classes
        if '.' in text and '$' in text:
            return SignatureAnnotationType.BINARY_NAME
        
        # Simple class name pattern (e.g., "java.lang.String")
        if '.' in text:
            parts = text.split('.')
            if all(part.isidentifier() for part in parts if part):
                return SignatureAnnotationType.BINARY_NAME
        
        return None
    
    def is_class_forname_pattern(self, line: str) -> bool:
        """Check if line contains Class.forName pattern"""
        return bool(re.search(r'Class\s*\.\s*forName\s*\(', line))
    
    def is_getname_pattern(self, line: str) -> bool:
        """Check if line contains .getName() pattern"""
        return bool(re.search(r'\.\s*getName\s*\(\s*\)', line))
    
    def is_getcanonicalname_pattern(self, line: str) -> bool:
        """Check if line contains .getCanonicalName() pattern"""
        return bool(re.search(r'\.\s*getCanonicalName\s*\(\s*\)', line))
    
    def is_signature_related(self, line: str) -> bool:
        """Check if a line contains signature-related patterns"""
        patterns = [
            r'Class\s*\.\s*forName',
            r'\.getName\s*\(\)',
            r'\.getCanonicalName\s*\(\)',
            r'\.getTypeName\s*\(\)',
            r'ClassLoader',
            r'\.loadClass\s*\(',
            r'className',
            r'typeName',
            r'TypeLiteral',
        ]
        return any(re.search(p, line) for p in patterns)
    
    def is_valid_annotation_target(self, line_number: int) -> bool:
        """Check if the line is a valid annotation target"""
        if line_number < 1 or line_number > len(self.lines):
            return False
        
        line = self.lines[line_number - 1].strip()
        
        # Skip empty lines or comments
        if not line or line.startswith('//') or line.startswith('/*') or line.startswith('*'):
            return False
        
        # Invalid targets
        invalid_patterns = [
            r'^new\s+', r'^return\s*', r'^if\s*\(', r'^else\s*',
            r'^for\s*\(', r'^while\s*\(', r'^throw\s+', r'^try\s*\{',
            r'^\}\s*$', r'^\{\s*$',
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, line):
                return False
        
        # Valid targets - declarations with String type
        valid_patterns = [
            r'String\s+\w+\s*=',           # String var =
            r'final\s+String\s+\w+\s*=',   # final String var =
            r'String\s+\w+\s*[,)]',        # String parameter
            r'String\s+\w+\s*;',           # String declaration
            r'Class\s*<\s*\?\s*>\s+\w+',   # Class<?> var
        ]
        
        for pattern in valid_patterns:
            if re.search(pattern, line):
                return True
        
        return False
    
    def infer_annotation_from_context(self, line: str) -> SignatureAnnotationType:
        """Infer the best annotation type from code context"""
        # Class.forName expects BinaryName
        if self.is_class_forname_pattern(line):
            return SignatureAnnotationType.BINARY_NAME
        
        # .getName() returns ClassGetName
        if self.is_getname_pattern(line):
            return SignatureAnnotationType.CLASS_GET_NAME
        
        # .getCanonicalName() returns FullyQualifiedName
        if self.is_getcanonicalname_pattern(line):
            return SignatureAnnotationType.FULLY_QUALIFIED_NAME
        
        # Look for string literals and analyze format
        string_pattern = r'"([^"\\]|\\.)*"'
        matches = re.findall(string_pattern, line)
        
        for match in matches:
            annotation = self.detect_signature_format(match)
            if annotation:
                return annotation
        
        # Default to BinaryName (most common for class name strings)
        return SignatureAnnotationType.BINARY_NAME
    
    def place_annotation(self, line_number: int, annotation: SignatureAnnotationType,
                        target_element: str = "") -> bool:
        """Place annotation at the specified line"""
        try:
            if line_number < 1 or line_number > len(self.lines):
                return False
            
            if not self.is_valid_annotation_target(line_number):
                logger.warning(f"Invalid annotation target at line {line_number}")
                return False
            
            # Get indentation
            original_line = self.lines[line_number - 1]
            indent = len(original_line) - len(original_line.lstrip())
            indent_str = ' ' * indent
            
            # Insert annotation
            annotation_line = f"{indent_str}{annotation.value}\n"
            self.lines.insert(line_number - 1, annotation_line)
            
            self.placements.append(SignaturePlacement(
                file_path=self.file_path,
                line_number=line_number,
                annotation=annotation,
                target_element=target_element
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing annotation at line {line_number}: {e}")
            return False
    
    def add_imports(self, annotations_used: Optional[List[SignatureAnnotationType]] = None) -> bool:
        """Add Signature String import statements to the file"""
        # Find insertion point (after package, before first import or class)
        insert_pos = 0
        
        for i, line in enumerate(self.lines):
            stripped = line.strip()
            if stripped.startswith('package '):
                insert_pos = i + 1
                # Skip empty lines after package
                while insert_pos < len(self.lines) and not self.lines[insert_pos].strip():
                    insert_pos += 1
                break
        
        # Find existing imports
        for i, line in enumerate(self.lines[insert_pos:], insert_pos):
            stripped = line.strip()
            if stripped.startswith('import '):
                insert_pos = i
                break
            elif stripped.startswith('public ') or stripped.startswith('class '):
                break
        
        # Determine which imports to add
        if annotations_used:
            imports_to_add = []
            for ann in annotations_used:
                if ann == SignatureAnnotationType.BINARY_NAME:
                    imports_to_add.append(SIGNATURE_IMPORTS[0])
                elif ann == SignatureAnnotationType.FULLY_QUALIFIED_NAME:
                    imports_to_add.append(SIGNATURE_IMPORTS[1])
                elif ann == SignatureAnnotationType.FIELD_DESCRIPTOR:
                    imports_to_add.append(SIGNATURE_IMPORTS[2])
                elif ann == SignatureAnnotationType.CLASS_GET_NAME:
                    imports_to_add.append(SIGNATURE_IMPORTS[3])
                elif ann == SignatureAnnotationType.INTERNAL_FORM:
                    imports_to_add.append(SIGNATURE_IMPORTS[4])
        else:
            imports_to_add = SIGNATURE_IMPORTS
        
        # Add imports that are not already present
        file_content = ''.join(self.lines)
        imports_added = 0
        
        for imp in imports_to_add:
            if imp not in file_content:
                self.lines.insert(insert_pos, imp + '\n')
                insert_pos += 1
                imports_added += 1
        
        if imports_added > 0:
            # Add blank line after imports if needed
            if insert_pos < len(self.lines) and self.lines[insert_pos].strip():
                if not self.lines[insert_pos].strip().startswith('import'):
                    self.lines.insert(insert_pos, '\n')
            
            logger.info(f"Added {imports_added} Signature imports to {self.file_path}")
        
        return imports_added > 0
    
    def save_file(self):
        """Save the modified file"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.writelines(self.lines)
            logger.info(f"Saved {self.file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
    
    def analyze_and_place(self, target_lines: Optional[List[int]] = None) -> List[SignaturePlacement]:
        """
        Analyze the file and place annotations
        
        Args:
            target_lines: Optional list of line numbers to annotate
                         If provided, annotations are placed at these lines
                         If not, heuristic analysis is used
        
        Returns:
            List of placed annotations
        """
        # First pass: collect all lines to annotate
        lines_to_annotate = []
        
        if target_lines:
            # Use specified lines
            for line_num in target_lines:
                if line_num > 0 and line_num <= len(self.lines):
                    if self.is_valid_annotation_target(line_num):
                        line = self.lines[line_num - 1]
                        annotation = self.infer_annotation_from_context(line)
                        lines_to_annotate.append((line_num, annotation))
        else:
            # Heuristic analysis - find signature-related String declarations
            for i, line in enumerate(self.lines, 1):
                if self.is_signature_related(line) and self.is_valid_annotation_target(i):
                    annotation = self.infer_annotation_from_context(line)
                    lines_to_annotate.append((i, annotation))
        
        # Second pass: place annotations in reverse order to maintain line numbers
        for line_num, annotation in sorted(lines_to_annotate, reverse=True):
            self.place_annotation(line_num, annotation)
        
        return self.placements


def place_signature_annotations_from_predictions(
    predictions_file: str,
    project_root: str,
    output_dir: Optional[str] = None
) -> Dict[str, List[SignaturePlacement]]:
    """
    Place Signature String annotations based on model predictions
    
    Args:
        predictions_file: JSON file with predictions (file, line, annotation_type, confidence)
        project_root: Root directory of the project
        output_dir: Optional output directory (if None, modifies files in place)
    
    Returns:
        Dictionary mapping file paths to their placements
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Group predictions by file
    file_predictions = {}
    for pred in predictions:
        file_path = pred['file_path']
        if file_path not in file_predictions:
            file_predictions[file_path] = []
        file_predictions[file_path].append(pred)
    
    results = {}
    
    for file_path, preds in file_predictions.items():
        full_path = os.path.join(project_root, file_path)
        
        if output_dir:
            # Copy to output directory
            rel_path = os.path.relpath(full_path, project_root)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            import shutil
            shutil.copy(full_path, output_path)
            full_path = output_path
        
        placer = SignatureAnnotationPlacer(full_path)
        
        # Map annotation type strings to enums
        annotation_map = {
            '@BinaryName': SignatureAnnotationType.BINARY_NAME,
            '@FullyQualifiedName': SignatureAnnotationType.FULLY_QUALIFIED_NAME,
            '@FieldDescriptor': SignatureAnnotationType.FIELD_DESCRIPTOR,
            '@ClassGetName': SignatureAnnotationType.CLASS_GET_NAME,
            '@InternalForm': SignatureAnnotationType.INTERNAL_FORM,
        }
        
        # Place annotations based on predictions (reverse order to maintain line numbers)
        for pred in sorted(preds, key=lambda x: -x['line_number']):
            annotation_type = pred.get('annotation_type', '@BinaryName')
            annotation = annotation_map.get(annotation_type, SignatureAnnotationType.BINARY_NAME)
            
            placer.place_annotation(
                pred['line_number'],
                annotation,
                pred.get('target_element', '')
            )
        
        # Add imports and save
        annotations_used = list(set(p.annotation for p in placer.placements))
        placer.add_imports(annotations_used)
        placer.save_file()
        
        results[file_path] = placer.placements
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Place Signature String annotations')
    parser.add_argument('file_or_dir', help='Java file or directory to annotate')
    parser.add_argument('--predictions', help='JSON file with predictions')
    parser.add_argument('--output', help='Output directory (default: modify in place)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    
    args = parser.parse_args()
    
    path = Path(args.file_or_dir)
    
    if args.predictions:
        results = place_signature_annotations_from_predictions(
            args.predictions,
            str(path),
            args.output
        )
        print(f"Placed annotations in {len(results)} files")
        for file_path, placements in results.items():
            print(f"  {file_path}: {len(placements)} annotations")
    
    elif path.is_file():
        placer = SignatureAnnotationPlacer(str(path))
        placements = placer.analyze_and_place()
        
        if not args.dry_run:
            annotations_used = list(set(p.annotation for p in placements))
            placer.add_imports(annotations_used)
            placer.save_file()
        
        print(f"Placed {len(placements)} annotations in {path}")
        for p in placements:
            print(f"  Line {p.line_number}: {p.annotation.value}")
    
    elif path.is_dir():
        total = 0
        for java_file in path.rglob('*.java'):
            if '/test/' in str(java_file):
                continue
            
            placer = SignatureAnnotationPlacer(str(java_file))
            placements = placer.analyze_and_place()
            
            if placements:
                if not args.dry_run:
                    annotations_used = list(set(p.annotation for p in placements))
                    placer.add_imports(annotations_used)
                    placer.save_file()
                
                total += len(placements)
                print(f"{java_file}: {len(placements)} annotations")
        
        print(f"\nTotal: {total} annotations")
    
    else:
        print(f"Path not found: {path}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
