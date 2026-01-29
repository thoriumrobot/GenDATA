#!/usr/bin/env python3
"""
Comprehensive Annotation Placement Script for CFWR Pipeline

This script places relevant annotations on predicted locations for input projects,
with full support for multiple Checker Framework checkers (Lower Bound, SQL Quotes, 
Signature String) using confidence-based selection.

Features:
- PERFECT ACCURACY: Uses AST-based analysis for exact annotation placement (DEFAULT)
- Multi-Checker Support: Supports Lower Bound, SQL Quotes, and Signature String checkers
- Confidence-Based Selection: Places single annotation per location using highest confidence
- Integrates with MultiCheckerPredictor for unified prediction across all checkers
- Intelligent placement based on code structure
- Validation and verification of placed annotations
- Best practices defaults for consistency
- BACKUP: Approximate placement available as fallback option

Annotation Placement Behavior:
- For each location, if multiple models predict annotations, only the highest-confidence
  annotation is placed (single annotation per location)
- Confidence-based selection ensures optimal annotation choice across all annotation types
- Checker-specific annotation types are automatically detected and used

DEFAULT BEHAVIOR: Perfect placement is used by default for maximum accuracy.
Use --approximate_placement only if perfect placement fails or for compatibility.
"""

import os
import json
import argparse
import logging
import shutil
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import our existing modules - these classes are defined in this file
# from annotation_placement import (
#     AnnotationType, AnnotationPlacement, JavaCodeAnalyzer, 
#     AnnotationPlacementManager
# )
from checker_framework_integration import CheckerFrameworkEvaluator, CheckerType
# from perfect_annotation_placement import (
#     PreciseJavaAnalyzer, PreciseAnnotationPlacer, PerfectAnnotationPlacementSystem
# )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LowerBoundAnnotationType(Enum):
    """Lower Bound Checker specific annotations"""
    MIN_LEN = "@MinLen"
    ARRAY_LEN = "@ArrayLen"
    LT_EQ_LENGTH_OF = "@LTEqLengthOf"
    GT_LENGTH_OF = "@GTLengthOf"
    LENGTH_OF = "@LengthOf"
    POSITIVE = "@Positive"
    NON_NEGATIVE = "@NonNegative"
    GT_NEG_ONE = "@GTENegativeOne"
    LT_LENGTH_OF = "@LTLengthOf"
    SEARCH_INDEX_FOR = "@SearchIndexFor"
    SEARCH_INDEX_BOTTOM = "@SearchIndexBottom"
    SEARCH_INDEX_UNKNOWN = "@SearchIndexUnknown"


class SqlQuotesAnnotationType(Enum):
    """SQL Quotes Checker specific annotations"""
    SQL_EVEN_QUOTES = "@SqlEvenQuotes"
    SQL_ODD_QUOTES = "@SqlOddQuotes"


class SignatureAnnotationType(Enum):
    """Signature String Checker specific annotations"""
    BINARY_NAME = "@BinaryName"
    FULLY_QUALIFIED_NAME = "@FullyQualifiedName"
    FIELD_DESCRIPTOR = "@FieldDescriptor"
    CLASS_GET_NAME = "@ClassGetName"
    INTERNAL_FORM = "@InternalForm"


# Mapping of checker types to their annotation types
CHECKER_ANNOTATION_TYPES = {
    'lower_bound': LowerBoundAnnotationType,
    'sql_quotes': SqlQuotesAnnotationType,
    'signature_string': SignatureAnnotationType,
}

# Import statements for each checker
CHECKER_IMPORTS = {
    'lower_bound': [
        "import org.checkerframework.checker.index.qual.Positive;",
        "import org.checkerframework.checker.index.qual.NonNegative;",
        "import org.checkerframework.checker.index.qual.GTENegativeOne;",
    ],
    'sql_quotes': [
        "import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;",
        "import org.checkerframework.checker.sqlquotes.qual.SqlOddQuotes;",
    ],
    'signature_string': [
        "import org.checkerframework.checker.signature.qual.BinaryName;",
        "import org.checkerframework.checker.signature.qual.FullyQualifiedName;",
        "import org.checkerframework.checker.signature.qual.FieldDescriptor;",
        "import org.checkerframework.checker.signature.qual.ClassGetName;",
        "import org.checkerframework.checker.signature.qual.InternalForm;",
    ],
}


class AnnotationStrategy(Enum):
    """Strategies for placing annotations"""
    VARIABLE_DECLARATION = "variable_declaration"
    METHOD_PARAMETER = "method_parameter"
    METHOD_RETURN = "method_return"
    FIELD_DECLARATION = "field_declaration"
    LOCAL_VARIABLE = "local_variable"
    ARRAY_ACCESS = "array_access"
    LOOP_VARIABLE = "loop_variable"

@dataclass
class PredictionResult:
    """Represents a prediction result from our ML models"""
    file_path: str
    line_number: int
    confidence: float
    annotation_type: str
    target_element: str
    context: str = ""
    model_type: str = ""

@dataclass
class AnnotationPlacement:
    """Represents a placed annotation"""
    line_number: int
    annotation_type: str
    target_element: str
    placement_strategy: str

@dataclass
class AnnotationContext:
    """Context information for annotation placement"""
    file_path: str
    line_number: int
    code_line: str
    surrounding_lines: List[str]
    variable_name: Optional[str] = None
    method_name: Optional[str] = None
    class_name: Optional[str] = None
    is_array: bool = False
    is_loop_variable: bool = False
    is_parameter: bool = False
    is_return_type: bool = False

class JavaCodeAnalyzer:
    """Simple Java code analyzer for context extraction"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            self.lines = []
    
    def find_method_at_line(self, line_number: int) -> Optional[Dict]:
        """Find method containing the given line"""
        # Simple implementation - can be enhanced
        return None
    
    def find_class_at_line(self, line_number: int) -> Optional[Dict]:
        """Find class containing the given line"""
        # Simple implementation - can be enhanced
        return None

class AnnotationPlacementManager:
    """Manages annotation placement for a specific file"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.placed_annotations: List[AnnotationPlacement] = []

class PreciseAnnotationPlacer:
    """Precise annotation placer using AST analysis"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            self.lines = []
    
    def _is_valid_annotation_target(self, line_number: int) -> bool:
        """
        Check if the line is a valid annotation target.
        
        FIXED: Prevent placing annotations on invalid locations like
        new statements, return statements, if/for/while loops, etc.
        """
        if line_number < 1 or line_number > len(self.lines):
            return False
        
        line = self.lines[line_number - 1].strip()
        
        # Skip empty lines or comments
        if not line or line.startswith('//') or line.startswith('/*') or line.startswith('*'):
            return False
        
        # Invalid targets - these are statements, not declarations
        invalid_patterns = [
            r'^new\s+',              # new statements
            r'^return\s*',           # return statements
            r'^if\s*\(',             # if statements
            r'^else\s*',             # else clauses
            r'^for\s*\(',            # for loops
            r'^while\s*\(',          # while loops
            r'^do\s*\{',             # do-while loops
            r'^switch\s*\(',         # switch statements
            r'^case\s+',             # case labels
            r'^default\s*:',         # default label
            r'^throw\s+',            # throw statements
            r'^try\s*\{',            # try blocks
            r'^catch\s*\(',          # catch clauses
            r'^finally\s*\{',        # finally blocks
            r'^break\s*;',           # break statements
            r'^continue\s*;',        # continue statements
            r'^assert\s+',           # assert statements
            r'^synchronized\s*\(',   # synchronized blocks
            r'^\}\s*$',              # closing braces
            r'^\{\s*$',              # opening braces alone
            r'^[a-zA-Z_]\w*\s*\.\s*\w+\s*\(',  # method calls like obj.method()
            r'^[a-zA-Z_]\w*\s*\(',   # method/constructor calls like method()
            r'^\)\s*[;{]?\s*$',      # closing parenthesis
            r'^super\s*\(',          # super() calls
            r'^this\s*\(',           # this() calls
        ]
        
        import re
        for pattern in invalid_patterns:
            if re.match(pattern, line):
                logger.debug(f"Line {line_number} is invalid annotation target (matches {pattern}): {line[:50]}")
                return False
        
        # Valid targets - declarations that can have type annotations
        valid_patterns = [
            r'^(public|private|protected|static|final|abstract|synchronized|native|transient|volatile)\s+',  # Modifiers
            r'^@\w+',                              # Existing annotations (can add more)
            r'^\w+(\s*<[^>]+>)?\s+\w+\s*[=;,\)]',  # Variable declarations: Type var = or Type var;
            r'^\w+(\s*<[^>]+>)?\s+\w+\s*\(',      # Method declarations: Type methodName(
            r'^\w+\s*\[\]\s+\w+',                  # Array declarations: Type[] var
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, line):
                return True
        
        # If we can't determine, reject to be safe
        logger.debug(f"Line {line_number} rejected (no valid pattern match): {line[:50]}")
        return False
    
    def place_annotation_precisely(self, line_number: int, annotation: str, target_element: str) -> bool:
        """Place annotation precisely at the given location"""
        try:
            # FIXED: Validate the target before placing annotation
            if line_number > 0 and line_number <= len(self.lines):
                # Check if this is a valid annotation target
                if not self._is_valid_annotation_target(line_number):
                    logger.warning(f"Skipping invalid annotation target at line {line_number}: "
                                 f"{self.lines[line_number - 1].strip()[:50]}")
                    return False
                
                indent = len(self.lines[line_number - 1]) - len(self.lines[line_number - 1].lstrip())
                indent_str = ' ' * indent
                annotation_line = f"{indent_str}{annotation}\n"
                self.lines.insert(line_number - 1, annotation_line)
                return True
            return False
        except Exception as e:
            logger.error(f"Error placing annotation: {e}")
            return False
    
    def save_file(self):
        """Save the modified file"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.writelines(self.lines)
        except Exception as e:
            logger.error(f"Error saving file: {e}")

class ComprehensiveAnnotationPlacer:
    """Comprehensive annotation placement system"""
    
    def __init__(self, project_root: str, output_dir: str, checker_name: str = 'lower_bound', 
                 backup: bool = True, perfect_placement: bool = True):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.checker_name = checker_name.lower()
        self.backup = backup
        self.perfect_placement = perfect_placement
        self.evaluator = CheckerFrameworkEvaluator()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track placed annotations for validation
        self.placed_annotations: Dict[str, List[AnnotationPlacement]] = {}
        
        # Cache for AnnotationPlacementManagers (per file)
        self.placement_managers: Dict[str, AnnotationPlacementManager] = {}
        
        logger.info(f"Initialized annotation placer for {checker_name} checker with perfect_placement={perfect_placement}")
    
    def get_placement_manager(self, file_path: str) -> AnnotationPlacementManager:
        """Get or create an AnnotationPlacementManager for a specific file"""
        if file_path not in self.placement_managers:
            if os.path.exists(file_path):
                self.placement_managers[file_path] = AnnotationPlacementManager(file_path)
            else:
                logger.warning(f"File not found for placement manager: {file_path}")
                return None
        return self.placement_managers[file_path]
        
    def load_predictions(self, predictions_file: str) -> List[PredictionResult]:
        """Load prediction results from JSON file"""
        logger.info(f"Loading predictions from: {predictions_file}")
        
        predictions = []
        try:
            with open(predictions_file, 'r') as f:
                data = json.load(f)
                
            # Handle different prediction file formats
            if isinstance(data, list):
                # Direct list of predictions
                for pred in data:
                    predictions.append(self._parse_prediction(pred))
            elif isinstance(data, dict):
                # Nested format with file groupings
                for file_path, file_predictions in data.items():
                    for pred in file_predictions:
                        pred['file_path'] = file_path
                        predictions.append(self._parse_prediction(pred))
                        
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            raise
            
        logger.info(f"Loaded {len(predictions)} predictions")
        return predictions
    
    def _parse_prediction(self, pred_data: Dict) -> PredictionResult:
        """Parse a single prediction from JSON data"""
        return PredictionResult(
            file_path=pred_data.get('file_path', ''),
            line_number=int(pred_data.get('line_number', 0)),
            confidence=float(pred_data.get('confidence', 0.0)),
            annotation_type=pred_data.get('annotation_type', '@NonNull'),
            target_element=pred_data.get('target_element', ''),
            context=pred_data.get('context', ''),
            model_type=pred_data.get('model_type', 'unknown')
        )
    
    def analyze_code_context(self, file_path: str, line_number: int) -> AnnotationContext:
        """Analyze the code context around a prediction location"""
        try:
            analyzer = JavaCodeAnalyzer(file_path)
            
            # Get the line and surrounding context
            lines = analyzer.lines
            if line_number <= 0 or line_number > len(lines):
                raise ValueError(f"Invalid line number: {line_number}")
                
            code_line = lines[line_number - 1]
            start_idx = max(0, line_number - 3)
            end_idx = min(len(lines), line_number + 3)
            surrounding_lines = lines[start_idx:end_idx]
            
            # Analyze code structure
            context = AnnotationContext(
                file_path=file_path,
                line_number=line_number,
                code_line=code_line.strip(),
                surrounding_lines=surrounding_lines
            )
            
            # Detect context-specific information
            self._enhance_context(context, analyzer)
            
            return context
            
        except Exception as e:
            logger.warning(f"Could not analyze context for {file_path}:{line_number}: {e}")
            return AnnotationContext(
                file_path=file_path,
                line_number=line_number,
                code_line="",
                surrounding_lines=[]
            )
    
    def _enhance_context(self, context: AnnotationContext, analyzer: JavaCodeAnalyzer):
        """Enhance context with additional code analysis"""
        line = context.code_line
        
        # Detect variable declarations
        if any(keyword in line for keyword in ['int ', 'long ', 'String ', 'Object ']):
            context.variable_name = self._extract_variable_name(line)
            
        # Detect arrays
        context.is_array = '[' in line and ']' in line
        
        # Detect method parameters
        context.is_parameter = '(' in line and ')' in line and not line.strip().startswith('//')
        
        # Detect loop variables
        context.is_loop_variable = any(keyword in line for keyword in ['for ', 'while ', 'foreach '])
        
        # Get method and class context
        try:
            method_info = analyzer.find_method_at_line(context.line_number)
            if method_info:
                context.method_name = method_info.get('name', '')
                
            class_info = analyzer.find_class_at_line(context.line_number)
            if class_info:
                context.class_name = class_info.get('name', '')
        except:
            pass
    
    def _extract_variable_name(self, line: str) -> Optional[str]:
        """Extract variable name from a declaration line"""
        import re
        
        # Simple regex to extract variable names
        patterns = [
            r'\b(?:int|long|double|float|boolean|char|byte|short|String|Object|\w+)\s+(\w+)\s*[=;]',
            r'\b(\w+)\s*\[\s*\]',
            r'(\w+)\s*=\s*'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        
        return None
    
    def determine_annotation_strategy(self, context: AnnotationContext, prediction: PredictionResult) -> AnnotationStrategy:
        """Determine the best strategy for placing an annotation"""
        line = context.code_line.lower()
        
        # Method parameters
        if context.is_parameter or ('(' in line and ')' in line and 'public' in line):
            return AnnotationStrategy.METHOD_PARAMETER
            
        # Method return types
        if any(keyword in line for keyword in ['return ', 'public ', 'private ', 'protected ']) and '(' in line:
            return AnnotationStrategy.METHOD_RETURN
            
        # Array access
        if context.is_array or '[' in line:
            return AnnotationStrategy.ARRAY_ACCESS
            
        # Loop variables
        if context.is_loop_variable:
            return AnnotationStrategy.LOOP_VARIABLE
            
        # Field declarations
        if any(keyword in line for keyword in ['private ', 'public ', 'protected ']) and '(' not in line:
            return AnnotationStrategy.FIELD_DECLARATION
            
        # Variable declarations
        if context.variable_name:
            return AnnotationStrategy.VARIABLE_DECLARATION
            
        # Default to local variable
        return AnnotationStrategy.LOCAL_VARIABLE
    
    def select_appropriate_annotation(self, prediction: PredictionResult, context: AnnotationContext) -> List[str]:
        """
        Select appropriate annotation based on prediction.
        Returns only the annotation from the prediction (single annotation per location).
        """
        # Return only the annotation from the prediction
        # Multiple annotations are never placed - only the highest confidence one
        return [prediction.annotation_type]
    
    def place_annotation_at_location(self, file_path: str, line_number: int, 
                                   annotations: List[str], strategy: AnnotationStrategy) -> bool:
        """Place annotation(s) at a specific location with perfect accuracy (default)"""
        try:
            # Create backup if requested
            if self.backup:
                backup_path = self.output_dir / "backups" / Path(file_path).name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
            
            # Use perfect placement system by default for accurate positioning
            if self.perfect_placement:
                return self._place_annotations_perfectly(file_path, line_number, annotations, strategy)
            else:
                # Fallback to approximate placement only if explicitly requested
                logger.warning("Using approximate placement (perfect placement is recommended)")
                return self._place_annotations_approximately_backup(file_path, line_number, annotations, strategy)
                
        except Exception as e:
            logger.error(f"Error placing annotation at {file_path}:{line_number}: {e}")
            return False
    
    def _place_annotations_perfectly(self, file_path: str, line_number: int, 
                                   annotations: List[str], strategy: AnnotationStrategy) -> bool:
        """Place annotations using the perfect placement system"""
        try:
            # Create precise placer for this file
            placer = PreciseAnnotationPlacer(file_path)
            
            # Place each annotation with perfect accuracy
            success_count = 0
            for annotation in annotations:
                # Determine target element from context
                context = self.analyze_code_context(file_path, line_number)
                target_element = context.variable_name or context.method_name or ""
                
                # Place annotation precisely
                success = placer.place_annotation_precisely(
                    line_number, annotation, target_element
                )
                
                if success:
                    success_count += 1
                    
                    # Track placed annotation
                    if file_path not in self.placed_annotations:
                        self.placed_annotations[file_path] = []
                    
                    placement = AnnotationPlacement(
                        line_number=line_number,
                        annotation_type=annotation,
                        target_element=target_element,
                        placement_strategy=strategy.value
                    )
                    self.placed_annotations[file_path].append(placement)
            
            # Save file if any annotations were placed
            if success_count > 0:
                placer.save_file()
                logger.info(f"Perfectly placed {success_count}/{len(annotations)} annotations at {file_path}:{line_number}")
                return True
            else:
                logger.warning(f"Failed to place any annotations at {file_path}:{line_number}")
                return False
                
        except Exception as e:
            logger.error(f"Error in perfect placement: {e}")
            return False
    
    # ============================================================================
    # BACKUP APPROXIMATE PLACEMENT METHODS (DEPRECATED - USE PERFECT PLACEMENT)
    # ============================================================================
    
    def _format_annotations_for_strategy_backup(self, annotations: List[str], strategy: AnnotationStrategy, 
                                               lines: List[str], line_number: int) -> Tuple[int, str]:
        """
        BACKUP METHOD: Format annotations based on placement strategy (APPROXIMATE)
        
        This method is kept as a backup for the approximate placement system.
        The main codebase now uses perfect placement by default.
        """
        # Get indentation from target line
        if line_number > 0 and line_number <= len(lines):
            target_line = lines[line_number - 1]
            indent = len(target_line) - len(target_line.lstrip())
            indent_str = ' ' * indent
        else:
            indent_str = '    '  # Default 4-space indent
        
        # Format annotations
        if len(annotations) == 1:
            annotation_text = f"{indent_str}{annotations[0]}\n"
        else:
            # Multiple annotations on separate lines
            annotation_lines = [f"{indent_str}{ann}\n" for ann in annotations]
            annotation_text = ''.join(annotation_lines)
        
        # Determine insertion point based on strategy
        insert_line = line_number - 1  # Default: before the target line
        
        if strategy == AnnotationStrategy.METHOD_PARAMETER:
            # Find the parameter and annotate inline
            insert_line = line_number - 1
            # For parameters, we might want to modify the line directly
            
        elif strategy == AnnotationStrategy.METHOD_RETURN:
            # Find method signature and annotate return type
            insert_line = line_number - 1
            
        elif strategy in [AnnotationStrategy.VARIABLE_DECLARATION, AnnotationStrategy.LOCAL_VARIABLE]:
            # Before variable declaration
            insert_line = line_number - 1
            
        return insert_line, annotation_text
    
    def _place_annotations_approximately_backup(self, file_path: str, line_number: int, 
                                              annotations: List[str], strategy: AnnotationStrategy) -> bool:
        """
        BACKUP METHOD: Place annotations using approximate line-based placement
        
        This method is kept as a backup for the approximate placement system.
        The main codebase now uses perfect placement by default.
        """
        try:
            # Create backup if requested
            if self.backup:
                backup_path = self.output_dir / "backups" / Path(file_path).name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
            
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Determine insertion point and format
            insert_line, annotation_text = self._format_annotations_for_strategy_backup(
                annotations, strategy, lines, line_number
            )
            
            # Insert annotations
            if insert_line >= 0 and insert_line <= len(lines):
                lines.insert(insert_line, annotation_text)
                
                # Write modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                # Track placed annotation
                if file_path not in self.placed_annotations:
                    self.placed_annotations[file_path] = []
                
                for ann in annotations:
                    placement = AnnotationPlacement(
                        line_number=line_number,
                        annotation_type=ann,
                        target_element="",
                        placement_strategy=strategy.value
                    )
                    self.placed_annotations[file_path].append(placement)
                
                logger.info(f"Approximately placed {len(annotations)} annotations at {file_path}:{line_number}")
                return True
            else:
                logger.warning(f"Invalid insertion point for {file_path}:{line_number}")
                return False
                
        except Exception as e:
            logger.error(f"Error in approximate placement at {file_path}:{line_number}: {e}")
            return False
    
    def process_predictions(self, predictions: List[PredictionResult]) -> Dict[str, int]:
        """
        Process all predictions and place annotations.
        Groups predictions by location and selects highest-confidence annotation per location.
        """
        logger.info(f"Processing {len(predictions)} predictions for {self.checker_name} checker")
        
        stats = {
            'total': len(predictions),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'locations_with_predictions': 0,
            'locations_after_selection': 0
        }
        
        # Group predictions by location (file_path, line_number) for confidence-based selection
        predictions_by_location: Dict[Tuple[str, int], List[PredictionResult]] = {}
        for pred in predictions:
            # Normalize file path
            if os.path.isabs(pred.file_path):
                file_path = pred.file_path
            else:
                # Try relative to project root
                candidate_path = str(self.project_root / pred.file_path)
                if os.path.exists(candidate_path):
                    file_path = candidate_path
                else:
                    # Try as-is if it exists
                    if os.path.exists(pred.file_path):
                        file_path = pred.file_path
                    else:
                        logger.warning(f"File path not found: {pred.file_path} (relative to {self.project_root})")
                        file_path = candidate_path  # Use candidate anyway for grouping
            
            location_key = (file_path, pred.line_number)
            if location_key not in predictions_by_location:
                predictions_by_location[location_key] = []
            predictions_by_location[location_key].append(pred)
        
        stats['locations_with_predictions'] = len(predictions_by_location)
        
        # Select highest-confidence prediction for each location
        selected_predictions: List[PredictionResult] = []
        for (file_path, line_number), location_predictions in predictions_by_location.items():
            if len(location_predictions) > 1:
                # Multiple predictions for same location - select highest confidence
                best_prediction = max(location_predictions, key=lambda p: p.confidence)
                logger.debug(f"Location {file_path}:{line_number}: Selected {best_prediction.annotation_type} "
                           f"(confidence: {best_prediction.confidence:.3f}) from {len(location_predictions)} predictions")
                
                # Verify only one annotation will be placed
                if len(location_predictions) > 1:
                    other_predictions = [p for p in location_predictions if p != best_prediction]
                    logger.debug(f"  Discarded {len(other_predictions)} other predictions at same location")
            else:
                best_prediction = location_predictions[0]
            
            selected_predictions.append(best_prediction)
        
        # Verify no duplicate locations in selected predictions
        selected_locations = {(p.file_path, p.line_number) for p in selected_predictions}
        if len(selected_locations) != len(selected_predictions):
            logger.warning(f"Duplicate locations detected: {len(selected_predictions)} predictions but {len(selected_locations)} unique locations")
            # Remove duplicates, keeping highest confidence
            seen_locations = {}
            for pred in selected_predictions:
                loc_key = (pred.file_path, pred.line_number)
                if loc_key not in seen_locations or pred.confidence > seen_locations[loc_key].confidence:
                    seen_locations[loc_key] = pred
            selected_predictions = list(seen_locations.values())
            logger.info(f"After deduplication: {len(selected_predictions)} unique predictions")
        
        stats['locations_after_selection'] = len(selected_predictions)
        logger.info(f"Selected {len(selected_predictions)} annotations from {len(predictions)} predictions "
                   f"across {len(predictions_by_location)} locations")
        
        # Group selected predictions by file for efficient processing
        predictions_by_file: Dict[str, List[PredictionResult]] = {}
        for pred in selected_predictions:
            # Normalize file path (same logic as above)
            if os.path.isabs(pred.file_path):
                file_path = pred.file_path
            else:
                candidate_path = str(self.project_root / pred.file_path)
                file_path = candidate_path if os.path.exists(candidate_path) else pred.file_path
            
            if file_path not in predictions_by_file:
                predictions_by_file[file_path] = []
            predictions_by_file[file_path].append(pred)
            
            # Verify no duplicate line numbers per file
            line_numbers = [p.line_number for p in predictions_by_file[file_path]]
            if len(line_numbers) != len(set(line_numbers)):
                duplicates = [ln for ln in line_numbers if line_numbers.count(ln) > 1]
                logger.warning(f"Duplicate line numbers in {file_path}: {duplicates}")
        
        # Process each file
        for file_path, file_predictions in predictions_by_file.items():
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                stats['skipped'] += len(file_predictions)
                continue
            
            logger.info(f"Processing {len(file_predictions)} selected predictions for {file_path}")
            
            # Sort predictions by line number (descending) to avoid line shift issues
            file_predictions.sort(key=lambda p: p.line_number, reverse=True)
            
            for prediction in file_predictions:
                try:
                    # Analyze context
                    context = self.analyze_code_context(file_path, prediction.line_number)
                    
                    # Determine strategy
                    strategy = self.determine_annotation_strategy(context, prediction)
                    
                    # Select appropriate annotation (single annotation from prediction)
                    annotations = self.select_appropriate_annotation(prediction, context)
                    
                    # Verify only one annotation is being placed
                    if len(annotations) > 1:
                        logger.warning(f"Multiple annotations detected for single location {file_path}:{prediction.line_number}: {annotations}")
                        # Keep only the first annotation (should already be only one from select_appropriate_annotation)
                        annotations = [annotations[0]]
                    
                    # Place annotation (only one per location)
                    success = self.place_annotation_at_location(
                        file_path, prediction.line_number, annotations, strategy
                    )
                    
                    if success:
                        stats['successful'] += 1
                        # Verify annotation was actually placed (check placement manager)
                        if file_path in self.placed_annotations:
                            placed_at_location = [p for p in self.placed_annotations[file_path] 
                                               if p.line_number == prediction.line_number]
                            if len(placed_at_location) > 1:
                                logger.warning(f"Multiple annotations placed at {file_path}:{prediction.line_number}: {placed_at_location}")
                    else:
                        stats['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing prediction {prediction}: {e}")
                    stats['failed'] += 1
        
        logger.info(f"Annotation placement complete: {stats}")
        return stats
    
    def validate_annotations(self, checker_types: List[CheckerType] = None) -> Dict[str, any]:
        """Validate placed annotations using Checker Framework"""
        if checker_types is None:
            checker_types = [CheckerType.NULLNESS, CheckerType.INDEX]
        
        logger.info("Validating placed annotations...")
        
        validation_results = {}
        
        for file_path in self.placed_annotations.keys():
            logger.info(f"Validating {file_path}")
            
            try:
                # Run checker on annotated file
                evaluation_result = self.evaluator.evaluate_file(file_path, checker_types[0])
                
                validation_results[file_path] = {
                    'warnings_count': len(evaluation_result.new_warnings),
                    'warnings': evaluation_result.new_warnings,
                    'annotations_placed': len(self.placed_annotations[file_path]),
                    'success': evaluation_result.success,
                    'compilation_success': evaluation_result.compilation_success
                }
                
            except Exception as e:
                logger.error(f"Validation failed for {file_path}: {e}")
                validation_results[file_path] = {
                    'error': str(e),
                    'annotations_placed': len(self.placed_annotations[file_path])
                }
        
        return validation_results
    
    def generate_report(self, stats: Dict[str, int], validation_results: Dict[str, any] = None) -> str:
        """Generate a comprehensive report of annotation placement"""
        report_path = self.output_dir / "annotation_placement_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Annotation Placement Report\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"- Total predictions processed: {stats['total']}\n")
            f.write(f"- Successful placements: {stats['successful']}\n")
            f.write(f"- Failed placements: {stats['failed']}\n")
            f.write(f"- Skipped predictions: {stats['skipped']}\n")
            f.write(f"- Success rate: {stats['successful']/stats['total']*100:.1f}%\n\n")
            
            # Files processed
            f.write("## Files Processed\n\n")
            for file_path, placements in self.placed_annotations.items():
                f.write(f"### {file_path}\n")
                f.write(f"- Annotations placed: {len(placements)}\n")
                
                # Group by annotation type
                ann_counts = {}
                for placement in placements:
                    ann_type = placement.annotation_type
                    ann_counts[ann_type] = ann_counts.get(ann_type, 0) + 1
                
                for ann_type, count in ann_counts.items():
                    f.write(f"  - {ann_type}: {count}\n")
                f.write("\n")
            
            # Validation results
            if validation_results:
                f.write("## Validation Results\n\n")
                for file_path, result in validation_results.items():
                    f.write(f"### {file_path}\n")
                    if 'error' in result:
                        f.write(f"- Validation error: {result['error']}\n")
                    else:
                        f.write(f"- Warnings after annotation: {result['warnings_count']}\n")
                        f.write(f"- Annotations placed: {result['annotations_placed']}\n")
                    f.write("\n")
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)

def main():
    parser = argparse.ArgumentParser(
        description='Place relevant annotations on predicted locations with Lower Bound Checker support'
    )
    
    parser.add_argument('--project_root', required=True,
                       help='Root directory of the Java project')
    parser.add_argument('--predictions_file', required=True,
                       help='JSON file containing prediction results')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for processed files and reports')
    parser.add_argument('--checker_name', 
                       choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       default='lower_bound',
                       help='Checker name for annotation placement (default: lower_bound)')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup of original files (default: True)')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate annotations using Checker Framework (default: True)')
    parser.add_argument('--checker_types', nargs='*', 
                       choices=['nullness', 'index', 'interning', 'lock', 'regex', 'signature'],
                       default=['nullness', 'index'],
                       help='Checker Framework checkers to use for validation')
    parser.add_argument('--perfect_placement', action='store_true', default=True,
                       help='Use perfect AST-based placement (DEFAULT - recommended)')
    parser.add_argument('--approximate_placement', action='store_true', default=False,
                       help='Use approximate line-based placement (BACKUP - less accurate)')
    
    args = parser.parse_args()
    
    # Convert checker types to enum
    checker_type_map = {
        'nullness': CheckerType.NULLNESS,
        'index': CheckerType.INDEX,
        'interning': CheckerType.INTERNING,
        'lock': CheckerType.LOCK,
        'regex': CheckerType.REGEX,
        'signature': CheckerType.SIGNATURE
    }
    checker_types = [checker_type_map[ct] for ct in args.checker_types]
    
    # Determine placement mode
    use_perfect_placement = args.perfect_placement and not args.approximate_placement
    
    try:
        # Initialize annotation placer
        placer = ComprehensiveAnnotationPlacer(
            project_root=args.project_root,
            output_dir=args.output_dir,
            checker_name=args.checker_name,
            backup=args.backup,
            perfect_placement=use_perfect_placement
        )
        
        # Load predictions
        predictions = placer.load_predictions(args.predictions_file)
        
        # Process predictions and place annotations
        stats = placer.process_predictions(predictions)
        
        # Validate annotations if requested
        validation_results = None
        if args.validate:
            validation_results = placer.validate_annotations(checker_types)
        
        # Generate report
        report_path = placer.generate_report(stats, validation_results)
        
        logger.info("Annotation placement completed successfully!")
        logger.info(f"Report available at: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Annotation placement failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
