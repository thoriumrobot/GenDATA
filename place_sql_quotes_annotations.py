#!/usr/bin/env python3
"""
SQL Quotes Annotation Placement System

This module places SQL Quotes Checker annotations (@SqlEvenQuotes, @SqlOddQuotes)
on Java code based on model predictions or heuristic analysis.

Annotation Types:
- @SqlEvenQuotes: String with even number of single quotes (valid SQL)
- @SqlOddQuotes: String with odd number of single quotes (potential SQL injection)

Import: org.checkerframework.checker.sqlquotes.qual.*
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


class SqlQuotesAnnotationType(Enum):
    """SQL Quotes Checker annotation types"""
    SQL_EVEN_QUOTES = "@SqlEvenQuotes"
    SQL_ODD_QUOTES = "@SqlOddQuotes"


# Import statements for SQL Quotes annotations
SQL_QUOTES_IMPORTS = [
    "import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;",
    "import org.checkerframework.checker.sqlquotes.qual.SqlOddQuotes;",
]


@dataclass
class SqlQuotesPlacement:
    """Represents a SQL Quotes annotation placement"""
    file_path: str
    line_number: int
    annotation: SqlQuotesAnnotationType
    target_element: str
    confidence: float = 1.0
    reason: str = ""


class SqlQuotesAnnotationPlacer:
    """Places SQL Quotes Checker annotations on Java code"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            self.lines = []
        self.placements: List[SqlQuotesPlacement] = []
    
    def count_single_quotes(self, text: str) -> int:
        """Count single quotes in a string, handling escaped quotes"""
        # Remove escaped quotes first
        cleaned = text.replace("\\'", "").replace("''", "")
        return cleaned.count("'")
    
    def is_sql_related(self, line: str) -> bool:
        """Check if a line contains SQL-related patterns"""
        sql_patterns = [
            r'\bSELECT\b', r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b',
            r'\bFROM\b', r'\bWHERE\b', r'\bJOIN\b', r'\bCREATE\b',
            r'\bDROP\b', r'\bALTER\b', r'\bTABLE\b', r'\bINTO\b',
            r'executeQuery', r'executeUpdate', r'prepareStatement',
            r'PreparedStatement', r'Statement\s*\.\s*execute',
        ]
        line_upper = line.upper()
        return any(re.search(p, line_upper, re.IGNORECASE) for p in sql_patterns)
    
    def analyze_string_literal(self, line: str) -> Optional[SqlQuotesAnnotationType]:
        """
        Analyze a line for SQL string literals and determine annotation type
        
        Returns:
            Annotation type based on quote parity, or None if not applicable
        """
        # Find string literals in the line - simple pattern for quoted strings
        string_pattern = r'"[^"]*"'
        matches = re.findall(string_pattern, line)
        
        if not matches:
            return None
        
        # Analyze quote parity in the combined strings
        total_quotes = 0
        for match in matches:
            total_quotes += self.count_single_quotes(match)
        
        if total_quotes == 0:
            # No single quotes - even quotes (valid)
            return SqlQuotesAnnotationType.SQL_EVEN_QUOTES
        elif total_quotes % 2 == 0:
            return SqlQuotesAnnotationType.SQL_EVEN_QUOTES
        else:
            return SqlQuotesAnnotationType.SQL_ODD_QUOTES
    
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
        ]
        
        for pattern in valid_patterns:
            if re.search(pattern, line):
                return True
        
        return False
    
    def place_annotation(self, line_number: int, annotation: SqlQuotesAnnotationType, 
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
            
            self.placements.append(SqlQuotesPlacement(
                file_path=self.file_path,
                line_number=line_number,
                annotation=annotation,
                target_element=target_element
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing annotation at line {line_number}: {e}")
            return False
    
    def add_imports(self) -> bool:
        """Add SQL Quotes import statements to the file"""
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
        
        # Add imports that are not already present
        file_content = ''.join(self.lines)
        imports_added = 0
        
        for imp in SQL_QUOTES_IMPORTS:
            if imp not in file_content:
                self.lines.insert(insert_pos, imp + '\n')
                insert_pos += 1
                imports_added += 1
        
        if imports_added > 0:
            # Add blank line after imports if needed
            if insert_pos < len(self.lines) and self.lines[insert_pos].strip():
                if not self.lines[insert_pos].strip().startswith('import'):
                    self.lines.insert(insert_pos, '\n')
            
            logger.info(f"Added {imports_added} SQL Quotes imports to {self.file_path}")
        
        return imports_added > 0
    
    def save_file(self):
        """Save the modified file"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.writelines(self.lines)
            logger.info(f"Saved {self.file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
    
    def analyze_and_place(self, sql_param_lines: Optional[List[int]] = None) -> List[SqlQuotesPlacement]:
        """
        Analyze the file and place annotations
        
        Args:
            sql_param_lines: Optional list of line numbers with SQL parameters
                           If provided, annotations are placed at these lines
                           If not, heuristic analysis is used
        
        Returns:
            List of placed annotations
        """
        # First pass: collect all lines to annotate
        lines_to_annotate = []
        
        if sql_param_lines:
            # Use specified lines
            for line_num in sql_param_lines:
                if line_num > 0 and line_num <= len(self.lines):
                    if self.is_valid_annotation_target(line_num):
                        line = self.lines[line_num - 1]
                        annotation = self.analyze_string_literal(line)
                        if annotation is None:
                            annotation = SqlQuotesAnnotationType.SQL_EVEN_QUOTES
                        lines_to_annotate.append((line_num, annotation))
        else:
            # Heuristic analysis - find SQL-related String declarations
            for i, line in enumerate(self.lines, 1):
                if self.is_sql_related(line) and self.is_valid_annotation_target(i):
                    annotation = self.analyze_string_literal(line)
                    if annotation:
                        lines_to_annotate.append((i, annotation))
        
        # Second pass: place annotations in reverse order to maintain line numbers
        for line_num, annotation in sorted(lines_to_annotate, reverse=True):
            self.place_annotation(line_num, annotation)
        
        return self.placements


def place_sql_quotes_annotations_from_predictions(
    predictions_file: str,
    project_root: str,
    output_dir: Optional[str] = None
) -> Dict[str, List[SqlQuotesPlacement]]:
    """
    Place SQL Quotes annotations based on model predictions
    
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
        
        placer = SqlQuotesAnnotationPlacer(full_path)
        
        # Place annotations based on predictions
        for pred in sorted(preds, key=lambda x: -x['line_number']):  # Reverse order
            annotation_type = pred.get('annotation_type', '@SqlEvenQuotes')
            if annotation_type == '@SqlEvenQuotes':
                annotation = SqlQuotesAnnotationType.SQL_EVEN_QUOTES
            elif annotation_type == '@SqlOddQuotes':
                annotation = SqlQuotesAnnotationType.SQL_ODD_QUOTES
            else:
                annotation = SqlQuotesAnnotationType.SQL_EVEN_QUOTES
            
            placer.place_annotation(
                pred['line_number'],
                annotation,
                pred.get('target_element', '')
            )
        
        # Add imports and save
        placer.add_imports()
        placer.save_file()
        
        results[file_path] = placer.placements
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Place SQL Quotes annotations')
    parser.add_argument('file_or_dir', help='Java file or directory to annotate')
    parser.add_argument('--predictions', help='JSON file with predictions')
    parser.add_argument('--output', help='Output directory (default: modify in place)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    
    args = parser.parse_args()
    
    path = Path(args.file_or_dir)
    
    if args.predictions:
        results = place_sql_quotes_annotations_from_predictions(
            args.predictions,
            str(path),
            args.output
        )
        print(f"Placed annotations in {len(results)} files")
        for file_path, placements in results.items():
            print(f"  {file_path}: {len(placements)} annotations")
    
    elif path.is_file():
        placer = SqlQuotesAnnotationPlacer(str(path))
        placements = placer.analyze_and_place()
        
        if not args.dry_run:
            placer.add_imports()
            placer.save_file()
        
        print(f"Placed {len(placements)} annotations in {path}")
        for p in placements:
            print(f"  Line {p.line_number}: {p.annotation.value}")
    
    elif path.is_dir():
        total = 0
        for java_file in path.rglob('*.java'):
            if '/test/' in str(java_file):
                continue
            
            placer = SqlQuotesAnnotationPlacer(str(java_file))
            placements = placer.analyze_and_place()
            
            if placements:
                if not args.dry_run:
                    placer.add_imports()
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
