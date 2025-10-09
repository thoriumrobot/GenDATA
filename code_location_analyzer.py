#!/usr/bin/env python3
"""
Code Location Analyzer

This module implements AST-based code location analysis to identify
transformation locations and their applicability for augmentation.
It analyzes Java code to identify method boundaries, statement positions,
and expression contexts to determine which transformations can be applied.
"""

import re
import ast
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

from recursive_augmentation_engine import TransformationType

logger = logging.getLogger(__name__)

class LocationType(Enum):
    """Types of code locations where transformations can be applied"""
    CLASS_LEVEL = "class_level"
    METHOD_LEVEL = "method_level"
    STATEMENT_LEVEL = "statement_level"
    EXPRESSION_LEVEL = "expression_level"
    BLOCK_LEVEL = "block_level"

class StatementType(Enum):
    """Types of statements in Java code"""
    ASSIGNMENT = "assignment"
    METHOD_CALL = "method_call"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    RETURN = "return"
    VARIABLE_DECLARATION = "variable_declaration"
    ARRAY_ACCESS = "array_access"
    FIELD_ACCESS = "field_access"
    CONSTRUCTOR_CALL = "constructor_call"
    EXPRESSION = "expression"

class ExpressionType(Enum):
    """Types of expressions in Java code"""
    ARITHMETIC = "arithmetic"
    LOGICAL = "logical"
    COMPARISON = "comparison"
    TERNARY = "ternary"
    METHOD_CALL = "method_call"
    ARRAY_ACCESS = "array_access"
    FIELD_ACCESS = "field_access"
    LITERAL = "literal"
    VARIABLE = "variable"

@dataclass
class CodeLocation:
    """Represents a specific location in Java code"""
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    location_type: LocationType
    context: Dict[str, any]
    code_snippet: str
    applicable_transformations: Set[TransformationType]

class CodeLocationAnalyzer:
    """Analyzes Java code to identify transformation locations"""
    
    def __init__(self):
        # Transformation applicability rules
        self.location_transformation_map = self._build_location_transformation_map()
        
        # Regex patterns for Java code analysis
        self.patterns = {
            'method': re.compile(r'^\s*(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE),
            'assignment': re.compile(r'^\s*(\w+)\s*=\s*([^;]+);', re.MULTILINE),
            'method_call': re.compile(r'(\w+)\.(\w+)\s*\('),
            'array_access': re.compile(r'(\w+)\[([^\]]+)\]'),
            'conditional': re.compile(r'if\s*\(([^)]+)\)\s*\{'),
            'loop': re.compile(r'(for|while)\s*\('),
            'return': re.compile(r'return\s+([^;]+);'),
            'variable_declaration': re.compile(r'^\s*(\w+)\s+(\w+)\s*=\s*([^;]+);', re.MULTILINE),
            'constructor_call': re.compile(r'new\s+(\w+)\s*\('),
            'ternary': re.compile(r'([^?]+\?[^:]+:[^;]+)'),
            'lambda': re.compile(r'\([^)]*\)\s*->\s*'),
            'stream': re.compile(r'\.stream\(\)'),
            'string_concat': re.compile(r'"[^"]*"\s*\+\s*'),
            'numeric_literal': re.compile(r'\b\d+\b'),
            'exception_handling': re.compile(r'try\s*\{'),
            'switch': re.compile(r'switch\s*\('),
        }
    
    def analyze_code(self, java_code: str) -> List[CodeLocation]:
        """Analyze Java code and return list of transformation locations"""
        locations = []
        lines = java_code.split('\n')
        
        # Find class-level locations
        locations.extend(self._find_class_level_locations(java_code, lines))
        
        # Find method-level locations
        locations.extend(self._find_method_level_locations(java_code, lines))
        
        # Find statement-level locations
        locations.extend(self._find_statement_level_locations(java_code, lines))
        
        # Find expression-level locations
        locations.extend(self._find_expression_level_locations(java_code, lines))
        
        # Find block-level locations
        locations.extend(self._find_block_level_locations(java_code, lines))
        
        return locations
    
    def get_transformation_applicability(self, location: CodeLocation) -> Set[TransformationType]:
        """Return transformations applicable at this location"""
        return location.applicable_transformations
    
    def _find_class_level_locations(self, code: str, lines: List[str]) -> List[CodeLocation]:
        """Find class-level transformation locations"""
        locations = []
        
        # Look for class declaration
        class_match = re.search(r'class\s+(\w+)', code)
        if class_match:
            start_line = code[:class_match.start()].count('\n') + 1
            end_line = len(lines)
            
            # Class-level transformations
            applicable_transforms = {
                TransformationType.RANDOM_METHOD_INSERTION,
                TransformationType.METHOD_EXTRACTION,
                TransformationType.BUILDER_PATTERN,
                TransformationType.FUNCTIONAL_CONVERSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=end_line,
                column_start=class_match.start(),
                column_end=class_match.end(),
                location_type=LocationType.CLASS_LEVEL,
                context={'class_name': class_match.group(1)},
                code_snippet=code[start_line-1:min(start_line+10, len(lines))],
                applicable_transformations=applicable_transforms
            ))
        
        return locations
    
    def _find_method_level_locations(self, code: str, lines: List[str]) -> List[CodeLocation]:
        """Find method-level transformation locations"""
        locations = []
        
        for match in self.patterns['method'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            method_name = match.group(4)
            
            # Find method end (simple heuristic)
            brace_count = 0
            end_pos = match.end()
            while end_pos < len(code):
                if code[end_pos] == '{':
                    brace_count += 1
                elif code[end_pos] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        break
                end_pos += 1
            
            end_line = code[:end_pos].count('\n') + 1
            
            # Method-level transformations
            applicable_transforms = {
                TransformationType.RANDOM_STATEMENT_INSERTION,
                TransformationType.LOOP_CONVERSION,
                TransformationType.GUARD_REVERSAL,
                TransformationType.METHOD_EXTRACTION,
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.EXCEPTION_HANDLING,
                TransformationType.LAMBDA_EXPRESSION,
                TransformationType.STREAM_API,
                TransformationType.SIMPLE_METHOD_CALL,
                TransformationType.SIMPLE_ASSIGNMENT,
                TransformationType.SIMPLE_CONDITIONAL,
                TransformationType.SIMPLE_RETURN_STATEMENT,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=end_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.METHOD_LEVEL,
                context={'method_name': method_name},
                code_snippet=code[match.start():min(match.end()+200, len(code))],
                applicable_transformations=applicable_transforms
            ))
        
        return locations
    
    def _find_statement_level_locations(self, code: str, lines: List[str]) -> List[CodeLocation]:
        """Find statement-level transformation locations"""
        locations = []
        
        # Assignment statements
        for match in self.patterns['assignment'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            var_name = match.group(1)
            expression = match.group(2)
            
            applicable_transforms = {
                TransformationType.MATHEMATICAL_EXPRESSION,
                TransformationType.VARIABLE_OPERATION,
                TransformationType.ARRAY_ACCESS_PATTERN,
                TransformationType.STRING_CONCATENATION,
                TransformationType.NUMERIC_LITERAL,
                TransformationType.SIMPLE_ASSIGNMENT,
                TransformationType.SIMPLE_NUMERIC_OPERATIONS,
                TransformationType.SIMPLE_STRING_OPERATION,
            }
            
            # Add expression-specific transformations
            if '?' in expression and ':' in expression:
                applicable_transforms.add(TransformationType.TERNARY_OPERATOR)
            if '+' in expression or '*' in expression or '-' in expression:
                applicable_transforms.add(TransformationType.MATHEMATICAL_EXPRESSION)
            if '"' in expression:
                applicable_transforms.add(TransformationType.STRING_CONCATENATION)
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.STATEMENT_LEVEL,
                context={
                    'statement_type': StatementType.ASSIGNMENT,
                    'variable_name': var_name,
                    'expression': expression
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Method call statements
        for match in self.patterns['method_call'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.SIMPLE_METHOD_CALL,
                TransformationType.METHOD_EXTRACTION,
                TransformationType.FUNCTIONAL_CONVERSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.STATEMENT_LEVEL,
                context={
                    'statement_type': StatementType.METHOD_CALL,
                    'object': match.group(1),
                    'method': match.group(2)
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Conditional statements
        for match in self.patterns['conditional'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            condition = match.group(1)
            
            applicable_transforms = {
                TransformationType.GUARD_REVERSAL,
                TransformationType.LOGICAL_EXPRESSION,
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.SIMPLE_CONDITIONAL,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.STATEMENT_LEVEL,
                context={
                    'statement_type': StatementType.CONDITIONAL,
                    'condition': condition
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Loop statements
        for match in self.patterns['loop'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            loop_type = match.group(1)
            
            applicable_transforms = {
                TransformationType.LOOP_CONVERSION,
                TransformationType.RANDOM_STATEMENT_INSERTION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.STATEMENT_LEVEL,
                context={
                    'statement_type': StatementType.LOOP,
                    'loop_type': loop_type
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Return statements
        for match in self.patterns['return'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            return_expr = match.group(1)
            
            applicable_transforms = {
                TransformationType.SIMPLE_RETURN_STATEMENT,
                TransformationType.VARIABLE_OPERATION,
                TransformationType.MATHEMATICAL_EXPRESSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.STATEMENT_LEVEL,
                context={
                    'statement_type': StatementType.RETURN,
                    'return_expression': return_expr
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        return locations
    
    def _find_expression_level_locations(self, code: str, lines: List[str]) -> List[CodeLocation]:
        """Find expression-level transformation locations"""
        locations = []
        
        # Array access expressions
        for match in self.patterns['array_access'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.ARRAY_ACCESS_PATTERN,
                TransformationType.SIMPLE_ARRAY_ACCESS,
                TransformationType.MATHEMATICAL_EXPRESSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.EXPRESSION_LEVEL,
                context={
                    'expression_type': ExpressionType.ARRAY_ACCESS,
                    'array_name': match.group(1),
                    'index': match.group(2)
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Ternary expressions
        for match in self.patterns['ternary'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.TERNARY_OPERATOR,
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.LOGICAL_EXPRESSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.EXPRESSION_LEVEL,
                context={
                    'expression_type': ExpressionType.TERNARY,
                },
                code_snippet=match.group(1),
                applicable_transformations=applicable_transforms
            ))
        
        # Lambda expressions
        for match in self.patterns['lambda'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.LAMBDA_EXPRESSION,
                TransformationType.FUNCTIONAL_CONVERSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.EXPRESSION_LEVEL,
                context={
                    'expression_type': ExpressionType.METHOD_CALL,
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Stream API expressions
        for match in self.patterns['stream'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.STREAM_API,
                TransformationType.FUNCTIONAL_CONVERSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.EXPRESSION_LEVEL,
                context={
                    'expression_type': ExpressionType.METHOD_CALL,
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # String concatenation
        for match in self.patterns['string_concat'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.STRING_CONCATENATION,
                TransformationType.SIMPLE_STRING_OPERATION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.EXPRESSION_LEVEL,
                context={
                    'expression_type': ExpressionType.ARITHMETIC,
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Numeric literals
        for match in self.patterns['numeric_literal'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.NUMERIC_LITERAL,
                TransformationType.MATHEMATICAL_EXPRESSION,
                TransformationType.SIMPLE_NUMERIC_OPERATIONS,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.EXPRESSION_LEVEL,
                context={
                    'expression_type': ExpressionType.LITERAL,
                    'literal_value': match.group(0)
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        return locations
    
    def _find_block_level_locations(self, code: str, lines: List[str]) -> List[CodeLocation]:
        """Find block-level transformation locations"""
        locations = []
        
        # Try-catch blocks
        for match in self.patterns['exception_handling'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.EXCEPTION_HANDLING,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.BLOCK_LEVEL,
                context={
                    'block_type': 'try_catch',
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        # Switch statements
        for match in self.patterns['switch'].finditer(code):
            start_line = code[:match.start()].count('\n') + 1
            
            applicable_transforms = {
                TransformationType.SWITCH_STATEMENT,
                TransformationType.CONDITIONAL_EXPRESSION,
            }
            
            locations.append(CodeLocation(
                line_start=start_line,
                line_end=start_line,
                column_start=match.start(),
                column_end=match.end(),
                location_type=LocationType.BLOCK_LEVEL,
                context={
                    'block_type': 'switch',
                },
                code_snippet=match.group(0),
                applicable_transformations=applicable_transforms
            ))
        
        return locations
    
    def _build_location_transformation_map(self) -> Dict[LocationType, Set[TransformationType]]:
        """Build mapping from location types to applicable transformations"""
        return {
            LocationType.CLASS_LEVEL: {
                TransformationType.RANDOM_METHOD_INSERTION,
                TransformationType.METHOD_EXTRACTION,
                TransformationType.BUILDER_PATTERN,
                TransformationType.FUNCTIONAL_CONVERSION,
            },
            LocationType.METHOD_LEVEL: {
                TransformationType.RANDOM_STATEMENT_INSERTION,
                TransformationType.LOOP_CONVERSION,
                TransformationType.GUARD_REVERSAL,
                TransformationType.METHOD_EXTRACTION,
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.EXCEPTION_HANDLING,
                TransformationType.LAMBDA_EXPRESSION,
                TransformationType.STREAM_API,
                TransformationType.SIMPLE_METHOD_CALL,
                TransformationType.SIMPLE_ASSIGNMENT,
                TransformationType.SIMPLE_CONDITIONAL,
                TransformationType.SIMPLE_RETURN_STATEMENT,
            },
            LocationType.STATEMENT_LEVEL: {
                TransformationType.MATHEMATICAL_EXPRESSION,
                TransformationType.VARIABLE_OPERATION,
                TransformationType.ARRAY_ACCESS_PATTERN,
                TransformationType.STRING_CONCATENATION,
                TransformationType.NUMERIC_LITERAL,
                TransformationType.TERNARY_OPERATOR,
                TransformationType.GUARD_REVERSAL,
                TransformationType.LOGICAL_EXPRESSION,
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.LOOP_CONVERSION,
                TransformationType.SIMPLE_ASSIGNMENT,
                TransformationType.SIMPLE_NUMERIC_OPERATIONS,
                TransformationType.SIMPLE_STRING_OPERATION,
                TransformationType.SIMPLE_METHOD_CALL,
                TransformationType.SIMPLE_CONDITIONAL,
                TransformationType.SIMPLE_RETURN_STATEMENT,
            },
            LocationType.EXPRESSION_LEVEL: {
                TransformationType.ARRAY_ACCESS_PATTERN,
                TransformationType.SIMPLE_ARRAY_ACCESS,
                TransformationType.MATHEMATICAL_EXPRESSION,
                TransformationType.TERNARY_OPERATOR,
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.LOGICAL_EXPRESSION,
                TransformationType.LAMBDA_EXPRESSION,
                TransformationType.FUNCTIONAL_CONVERSION,
                TransformationType.STREAM_API,
                TransformationType.STRING_CONCATENATION,
                TransformationType.SIMPLE_STRING_OPERATION,
                TransformationType.NUMERIC_LITERAL,
                TransformationType.SIMPLE_NUMERIC_OPERATIONS,
            },
            LocationType.BLOCK_LEVEL: {
                TransformationType.EXCEPTION_HANDLING,
                TransformationType.SWITCH_STATEMENT,
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.RANDOM_EXPRESSION_INSERTION,
            }
        }
