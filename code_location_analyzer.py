#!/usr/bin/env python3
"""
Code Location Analyzer

This module implements JDT-based code location analysis to identify
transformation locations and their applicability for augmentation.
It uses Eclipse JDT AST parsing instead of fragile regex patterns.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from jdt_service import JdtParserService, CodeLocation as JdtCodeLocation
from recursive_augmentation_engine import TransformationType

logger = logging.getLogger(__name__)

class LocationType(Enum):
    """Types of code locations where transformations can be applied"""
    CLASS_LEVEL = "CLASS_LEVEL"
    METHOD_LEVEL = "METHOD_LEVEL"
    STATEMENT_LEVEL = "STATEMENT_LEVEL"
    EXPRESSION_LEVEL = "EXPRESSION_LEVEL"
    BLOCK_LEVEL = "BLOCK_LEVEL"

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
    """Analyzes Java code to identify transformation locations using JDT AST parsing"""
    
    def __init__(self, jdt_jar_path: Optional[str] = None):
        """
        Initialize the analyzer with JDT parser service.
        
        Args:
            jdt_jar_path: Path to jdt-parser-all.jar. If None, will try to find it.
        """
        try:
            self.jdt_service = JdtParserService(jdt_jar_path)
            logger.info("Initialized CodeLocationAnalyzer with JDT parser service")
        except Exception as e:
            logger.error(f"Failed to initialize JDT parser service: {e}")
            raise RuntimeError(f"JDT parser service initialization failed: {e}")
        
        # Transformation applicability rules
        self.location_transformation_map = self._build_location_transformation_map()
    
    def analyze_code(self, java_code: str) -> List[CodeLocation]:
        """Analyze Java code and return list of transformation locations using JDT AST parsing"""
        try:
            # Use JDT service to parse code locations
            jdt_locations = self.jdt_service.parse_code_locations_from_string(java_code)
            
            # Convert JDT locations to our format
            locations = []
            for jdt_loc in jdt_locations:
                location = CodeLocation(
                    line_start=jdt_loc.line_start,
                    line_end=jdt_loc.line_end,
                    column_start=jdt_loc.column_start,
                    column_end=jdt_loc.column_end,
                    location_type=LocationType(jdt_loc.location_type),
                    context=jdt_loc.context,
                    code_snippet=jdt_loc.code_snippet,
                    applicable_transformations=self._convert_transformations(jdt_loc.applicable_transformations)
                )
                locations.append(location)
            
            logger.info(f"Analyzed {len(locations)} code locations using JDT AST parsing")
            return locations
            
        except Exception as e:
            logger.error(f"JDT code analysis failed: {e}")
            # Fallback to empty list rather than crashing
            return []
    
    def get_transformation_applicability(self, location: CodeLocation) -> Set[TransformationType]:
        """Return transformations applicable at this location"""
        return location.applicable_transformations
    
    def _convert_transformations(self, transformation_strings: List[str]) -> Set[TransformationType]:
        """Convert string transformation names to TransformationType enums"""
        transformations = set()
        for trans_str in transformation_strings:
            try:
                # Map string names to TransformationType enum values
                trans_type = self._string_to_transformation_type(trans_str)
                if trans_type:
                    transformations.add(trans_type)
            except (ValueError, KeyError):
                logger.warning(f"Unknown transformation type: {trans_str}")
                continue
        return transformations
    
    def _string_to_transformation_type(self, trans_str: str) -> Optional[TransformationType]:
        """Convert string transformation name to TransformationType enum"""
        # Map JDT transformation strings to our enum values
        transformation_map = {
            'GUARD_REVERSAL': TransformationType.GUARD_REVERSAL,
            'TERNARY_IF_ELSE': TransformationType.TERNARY_OPERATOR,
            'VARIABLE_INLINING': TransformationType.VARIABLE_OPERATION,
            'METHOD_EXTRACTION': TransformationType.METHOD_EXTRACTION,
            'SIMPLE_ASSIGNMENT': TransformationType.SIMPLE_ASSIGNMENT,
            'IDENTITY_MATH': TransformationType.MATHEMATICAL_EXPRESSION,
            'ARITHMETIC_PROPERTIES': TransformationType.MATHEMATICAL_EXPRESSION,
            'CONDITIONAL_RESTRUCTURING': TransformationType.CONDITIONAL_EXPRESSION,
            'BLOCK_RESTRUCTURING': TransformationType.SIMPLE_CONDITIONAL,
            'BUILDER_PATTERN': TransformationType.BUILDER_PATTERN,
            'FUNCTIONAL_CONVERSION': TransformationType.FUNCTIONAL_CONVERSION,
            'RANDOM_METHOD_INSERTION': TransformationType.RANDOM_METHOD_INSERTION,
        }
        return transformation_map.get(trans_str)
    
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
                TransformationType.METHOD_EXTRACTION,
                TransformationType.VARIABLE_OPERATION,
                TransformationType.GUARD_REVERSAL,
                TransformationType.TERNARY_OPERATOR,
            },
            LocationType.STATEMENT_LEVEL: {
                TransformationType.VARIABLE_OPERATION,
                TransformationType.SIMPLE_ASSIGNMENT,
                TransformationType.GUARD_REVERSAL,
                TransformationType.CONDITIONAL_EXPRESSION,
            },
            LocationType.EXPRESSION_LEVEL: {
                TransformationType.MATHEMATICAL_EXPRESSION,
                TransformationType.LOGICAL_EXPRESSION,
            },
            LocationType.BLOCK_LEVEL: {
                TransformationType.SIMPLE_CONDITIONAL,
            }
        }

# Convenience function for backward compatibility
def create_code_location_analyzer(jdt_jar_path: Optional[str] = None) -> CodeLocationAnalyzer:
    """Create and return a code location analyzer instance"""
    return CodeLocationAnalyzer(jdt_jar_path)