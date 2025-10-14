#!/usr/bin/env python3
"""
Unified Augmentation Registry

This module provides a unified registry that maps all transformation types
to their implementation functions, enabling location-aware transformation
application and centralized management of all augmentation techniques.
"""

import os
import random
from typing import List, Dict, Tuple, Optional, Set, Callable, Any
import logging

from recursive_augmentation_engine import (
    TransformationType, TransformationState, RecursiveAugmentationEngine
)
from enhanced_semantic_augment_slices import EnhancedSemanticTransformer
from simple_code_semantic_augment_slices import SimpleCodeSemanticTransformer
from augment_slices import (
    insert_random_methods, insert_random_statements, 
    generate_random_method, generate_random_statement
)
from code_location_analyzer import CodeLocation, CodeLocationAnalyzer
from transformation_caching import TransformationCache

logger = logging.getLogger(__name__)

class UnifiedAugmentationRegistry:
    """Registry that maps all transformation types to their implementations"""
    
    def __init__(self, seed: int = 42, enable_caching: bool = True):
        self.seed = seed
        random.seed(seed)
        
        # Initialize transformers
        self.enhanced_transformer = EnhancedSemanticTransformer(seed=seed)
        self.simple_transformer = SimpleCodeSemanticTransformer(seed=seed)
        self.recursive_engine = RecursiveAugmentationEngine(seed=seed)
        self.location_analyzer = CodeLocationAnalyzer()
        
        # Initialize caching system
        self.enable_caching = enable_caching
        self.cache = TransformationCache() if enable_caching else None
        
        # Build transformation mappings
        self.transformation_map = self._build_transformation_map()
        self.location_validators = self._build_location_validators()
        
        # Statistics
        self.transformation_stats = {
            'total_applications': 0,
            'successful_transformations': 0,
            'failed_transformations': 0,
            'location_specific_stats': {},
            'transformation_counts': {t.value: 0 for t in TransformationType}
        }
    
    def apply_transformation(self, code: str, transformation_type: TransformationType, 
                           location: Optional[CodeLocation] = None) -> str:
        """Apply transformation at specific location if provided"""
        import time
        start_time = time.time()
        
        try:
            self.transformation_stats['total_applications'] += 1
            self.transformation_stats['transformation_counts'][transformation_type.value] += 1
            
            # Check cache first if enabled
            if self.cache:
                cached_result = self.cache.get_transformation_result(code, transformation_type)
                if cached_result:
                    logger.debug(f"Cache hit for transformation {transformation_type.value}")
                    return cached_result.output_code
            
            # Get transformation function
            transform_func = self.transformation_map.get(transformation_type)
            if not transform_func:
                logger.warning(f"No implementation found for transformation: {transformation_type}")
                return code
            
            # Apply transformation
            if location:
                # Location-specific application
                result = self._apply_at_location(code, transformation_type, location)
            else:
                # Global application (fallback)
                result = transform_func(code)
            
            execution_time = time.time() - start_time
            
            # Validate result
            success = result and result != code
            if success:
                self.transformation_stats['successful_transformations'] += 1
                
                # Update location-specific stats
                if location:
                    location_key = f"{location.location_type.value}_{location.context.get('statement_type', 'unknown')}"
                    if location_key not in self.transformation_stats['location_specific_stats']:
                        self.transformation_stats['location_specific_stats'][location_key] = 0
                    self.transformation_stats['location_specific_stats'][location_key] += 1
                
                logger.debug(f"Successfully applied {transformation_type.value}")
            else:
                self.transformation_stats['failed_transformations'] += 1
                logger.debug(f"Transformation {transformation_type.value} had no effect")
            
            # Cache the result if caching is enabled
            if self.cache:
                location_context = {}
                if location:
                    location_context = {
                        'location_type': location.location_type.value,
                        'context': location.context
                    }
                
                self.cache.cache_transformation_result(
                    code=code,
                    transformation_type=transformation_type,
                    output_code=result,
                    success=success,
                    execution_time=execution_time,
                    location_context=location_context
                )
            
            return result
                
        except Exception as e:
            self.transformation_stats['failed_transformations'] += 1
            logger.error(f"Error applying transformation {transformation_type.value}: {e}")
            return code
    
    def get_valid_transformations(self, code: str, location: Optional[CodeLocation] = None) -> List[TransformationType]:
        """Return transformations valid at this location"""
        if location:
            return list(location.applicable_transformations)
        else:
            # Return all transformations if no specific location
            return list(TransformationType)
    
    def get_recommended_transformations(self, code: str, location: Optional[CodeLocation] = None) -> List[TransformationType]:
        """Get recommended transformations based on success patterns and location"""
        recommendations = []
        
        # Get cache-based recommendations if available
        if self.cache:
            cache_recommendations = self.cache.get_recommended_transformations(code, 
                location.context if location else None)
            recommendations.extend(cache_recommendations)
        
        # Get location-based valid transformations
        valid_transformations = self.get_valid_transformations(code, location)
        
        # Combine and deduplicate
        all_recommendations = recommendations + valid_transformations
        unique_recommendations = []
        seen = set()
        for t in all_recommendations:
            if t not in seen:
                unique_recommendations.append(t)
                seen.add(t)
        
        return unique_recommendations
    
    def analyze_code_locations(self, code: str) -> List[CodeLocation]:
        """Analyze code and return all transformation locations"""
        return self.location_analyzer.analyze_code(code)
    
    def apply_transformation_sequence(self, code: str, transformation_sequence: List[TransformationType],
                                    locations: Optional[List[CodeLocation]] = None) -> Tuple[str, List[bool]]:
        """Apply a sequence of transformations with optional location specification"""
        current_code = code
        success_flags = []
        
        for i, transformation_type in enumerate(transformation_sequence):
            location = locations[i] if locations and i < len(locations) else None
            original_code = current_code
            current_code = self.apply_transformation(current_code, transformation_type, location)
            success_flags.append(current_code != original_code)
        
        return current_code, success_flags
    
    def get_transformation_statistics(self) -> Dict[str, Any]:
        """Get transformation application statistics"""
        total = self.transformation_stats['total_applications']
        if total == 0:
            success_rate = 0.0
        else:
            success_rate = self.transformation_stats['successful_transformations'] / total
        
        stats = {
            'total_applications': total,
            'successful_transformations': self.transformation_stats['successful_transformations'],
            'failed_transformations': self.transformation_stats['failed_transformations'],
            'success_rate': success_rate,
            'transformation_counts': self.transformation_stats['transformation_counts'].copy(),
            'location_specific_stats': self.transformation_stats['location_specific_stats'].copy()
        }
        
        # Add cache statistics if caching is enabled
        if self.cache:
            stats['cache_statistics'] = self.cache.get_cache_statistics()
        
        return stats
    
    def _transform_with_jdt(self, transformation_name: str, mode: str) -> Callable[[str], str]:
        """Create a transformation function that uses JDT-based transformers"""
        def transform_func(code: str) -> str:
            try:
                import tempfile
                import os
                
                # Create temporary files for JDT transformer
                with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as input_file:
                    input_file.write(code)
                    input_file.flush()
                    
                    # Use appropriate transformer based on mode
                    if mode == 'enhanced':
                        result = self.enhanced_transformer.transform_file(input_file.name, 0)
                    else:  # simple
                        result = self.simple_transformer.transform_file(input_file.name, 0)
                    
                    # Clean up temporary files
                    os.unlink(input_file.name)
                    
                    return result
            except Exception as e:
                logger.warning(f"JDT transformation {transformation_name} failed: {e}")
                return code  # Return original code on error
        
        return transform_func
    
    def _build_transformation_map(self) -> Dict[TransformationType, Callable[[str], str]]:
        """Build mapping from transformation types to their implementation functions"""
        return {
            # Enhanced transformations (17 methods) - using JDT-based approach
            TransformationType.LOOP_CONVERSION: self._transform_with_jdt('loop_conversion', 'enhanced'),
            TransformationType.GUARD_REVERSAL: self._transform_with_jdt('guard_reversal', 'enhanced'),
            TransformationType.MATHEMATICAL_EXPRESSION: self._transform_with_jdt('mathematical_expression', 'enhanced'),
            TransformationType.LOGICAL_EXPRESSION: self._transform_with_jdt('logical_expression', 'enhanced'),
            TransformationType.TERNARY_OPERATOR: self._transform_with_jdt('ternary_operator', 'enhanced'),
            TransformationType.SWITCH_STATEMENT: self._transform_with_jdt('switch_statement', 'enhanced'),
            TransformationType.VARIABLE_OPERATION: self._transform_with_jdt('variable_operation', 'enhanced'),
            TransformationType.METHOD_EXTRACTION: self._transform_with_jdt('method_extraction', 'enhanced'),
            TransformationType.CONDITIONAL_EXPRESSION: self._transform_with_jdt('conditional_expression', 'enhanced'),
            TransformationType.ARRAY_ACCESS_PATTERN: self._transform_with_jdt('array_access_pattern', 'enhanced'),
            TransformationType.STRING_CONCATENATION: self._transform_with_jdt('string_concatenation', 'enhanced'),
            TransformationType.NUMERIC_LITERAL: self._transform_with_jdt('numeric_literal', 'enhanced'),
            TransformationType.EXCEPTION_HANDLING: self._transform_with_jdt('exception_handling', 'enhanced'),
            TransformationType.LAMBDA_EXPRESSION: self._transform_with_jdt('lambda_expression', 'enhanced'),
            TransformationType.STREAM_API: self._transform_with_jdt('stream_api', 'enhanced'),
            TransformationType.BUILDER_PATTERN: self._transform_with_jdt('builder_pattern', 'enhanced'),
            TransformationType.FUNCTIONAL_CONVERSION: self._transform_with_jdt('functional_conversion', 'enhanced'),
            
            # Simple transformations (10 methods) - using JDT-based approach
            TransformationType.SIMPLE_METHOD_CALL: self._transform_with_jdt('simple_method_call', 'simple'),
            TransformationType.SIMPLE_ASSIGNMENT: self._transform_with_jdt('simple_assignment', 'simple'),
            TransformationType.SIMPLE_CONDITIONAL: self._transform_with_jdt('simple_conditional', 'simple'),
            TransformationType.SIMPLE_ARRAY_ACCESS: self._transform_with_jdt('simple_array_access', 'simple'),
            TransformationType.SIMPLE_RETURN_STATEMENT: self._transform_with_jdt('simple_return_statement', 'simple'),
            TransformationType.SIMPLE_VARIABLE_DECLARATION: self._transform_with_jdt('simple_variable_declaration', 'simple'),
            TransformationType.SIMPLE_CONSTRUCTOR_CALL: self._transform_with_jdt('simple_constructor_call', 'simple'),
            TransformationType.SIMPLE_FIELD_ACCESS: self._transform_with_jdt('simple_field_access', 'simple'),
            TransformationType.SIMPLE_STRING_OPERATION: self._transform_with_jdt('simple_string_operation', 'simple'),
            TransformationType.SIMPLE_NUMERIC_OPERATION: self._transform_with_jdt('simple_numeric_operation', 'simple'),
            
            # Random augmentation transformations (3 methods)
            TransformationType.RANDOM_METHOD_INSERTION: self._apply_random_method_insertion,
            TransformationType.RANDOM_STATEMENT_INSERTION: self._apply_random_statement_insertion,
            TransformationType.RANDOM_EXPRESSION_INSERTION: self._apply_random_expression_insertion,
        }
    
    def _build_location_validators(self) -> Dict[TransformationType, Callable[[CodeLocation], bool]]:
        """Build mapping from transformation types to their location validation functions"""
        return {
            # Enhanced transformations
            TransformationType.LOOP_CONVERSION: lambda loc: loc.location_type.value in ['method_level', 'statement_level', 'block_level'],
            TransformationType.GUARD_REVERSAL: lambda loc: loc.context.get('statement_type') in ['conditional', 'if_statement'],
            TransformationType.MATHEMATICAL_EXPRESSION: lambda loc: loc.location_type.value in ['statement_level', 'expression_level'],
            TransformationType.LOGICAL_EXPRESSION: lambda loc: loc.context.get('expression_type') == 'logical',
            TransformationType.TERNARY_OPERATOR: lambda loc: loc.context.get('expression_type') == 'ternary',
            TransformationType.SWITCH_STATEMENT: lambda loc: loc.context.get('block_type') == 'switch',
            TransformationType.VARIABLE_OPERATION: lambda loc: loc.context.get('statement_type') == 'assignment',
            TransformationType.METHOD_EXTRACTION: lambda loc: loc.location_type.value in ['method_level', 'statement_level'],
            TransformationType.CONDITIONAL_EXPRESSION: lambda loc: loc.context.get('statement_type') in ['conditional', 'if_statement'],
            TransformationType.ARRAY_ACCESS_PATTERN: lambda loc: loc.context.get('expression_type') == 'array_access',
            TransformationType.STRING_CONCATENATION: lambda loc: loc.context.get('expression_type') in ['arithmetic', 'string'],
            TransformationType.NUMERIC_LITERAL: lambda loc: loc.context.get('expression_type') == 'literal',
            TransformationType.EXCEPTION_HANDLING: lambda loc: loc.context.get('block_type') == 'try_catch',
            TransformationType.LAMBDA_EXPRESSION: lambda loc: loc.context.get('expression_type') == 'method_call',
            TransformationType.STREAM_API: lambda loc: loc.context.get('expression_type') == 'method_call',
            TransformationType.BUILDER_PATTERN: lambda loc: loc.context.get('block_type') in ['constructor', 'method_level'],
            TransformationType.FUNCTIONAL_CONVERSION: lambda loc: loc.context.get('expression_type') == 'method_call',
            
            # Simple transformations
            TransformationType.SIMPLE_METHOD_CALL: lambda loc: loc.context.get('statement_type') == 'method_call',
            TransformationType.SIMPLE_ASSIGNMENT: lambda loc: loc.context.get('statement_type') == 'assignment',
            TransformationType.SIMPLE_CONDITIONAL: lambda loc: loc.context.get('statement_type') == 'conditional',
            TransformationType.SIMPLE_ARRAY_ACCESS: lambda loc: loc.context.get('expression_type') == 'array_access',
            TransformationType.SIMPLE_RETURN_STATEMENT: lambda loc: loc.context.get('statement_type') == 'return',
            TransformationType.SIMPLE_VARIABLE_DECLARATION: lambda loc: loc.context.get('statement_type') == 'variable_declaration',
            TransformationType.SIMPLE_CONSTRUCTOR_CALL: lambda loc: loc.context.get('statement_type') == 'constructor_call',
            TransformationType.SIMPLE_FIELD_ACCESS: lambda loc: loc.context.get('expression_type') == 'field_access',
            TransformationType.SIMPLE_STRING_OPERATION: lambda loc: loc.context.get('expression_type') in ['arithmetic', 'string'],
            TransformationType.SIMPLE_NUMERIC_OPERATION: lambda loc: loc.context.get('expression_type') in ['arithmetic', 'literal'],
            
            # Random augmentation transformations
            TransformationType.RANDOM_METHOD_INSERTION: lambda loc: loc.location_type.value in ['class_level', 'method_level'],
            TransformationType.RANDOM_STATEMENT_INSERTION: lambda loc: loc.location_type.value in ['method_level', 'block_level'],
            TransformationType.RANDOM_EXPRESSION_INSERTION: lambda loc: loc.location_type.value in ['statement_level', 'expression_level'],
        }
    
    def _apply_at_location(self, code: str, transformation_type: TransformationType, 
                          location: CodeLocation) -> str:
        """Apply transformation at specific location"""
        try:
            # Validate that transformation is applicable at this location
            validator = self.location_validators.get(transformation_type)
            if validator and not validator(location):
                logger.debug(f"Transformation {transformation_type.value} not applicable at location {location.location_type.value}")
                return code
            
            # Get transformation function
            transform_func = self.transformation_map.get(transformation_type)
            if not transform_func:
                return code
            
            # For location-specific transformations, we need to extract the relevant code section
            if location.line_start == location.line_end:
                # Single line transformation
                lines = code.split('\n')
                if location.line_start <= len(lines):
                    line_content = lines[location.line_start - 1]
                    transformed_line = transform_func(line_content)
                    if transformed_line != line_content:
                        lines[location.line_start - 1] = transformed_line
                        return '\n'.join(lines)
            else:
                # Multi-line transformation
                lines = code.split('\n')
                start_idx = max(0, location.line_start - 1)
                end_idx = min(len(lines), location.line_end)
                
                if start_idx < end_idx:
                    section_lines = lines[start_idx:end_idx]
                    section_code = '\n'.join(section_lines)
                    transformed_section = transform_func(section_code)
                    
                    if transformed_section != section_code:
                        # Replace the section
                        new_lines = lines[:start_idx] + transformed_section.split('\n') + lines[end_idx:]
                        return '\n'.join(new_lines)
            
            # Fallback to global transformation
            return transform_func(code)
            
        except Exception as e:
            logger.error(f"Error applying transformation {transformation_type.value} at location: {e}")
            return code
    
    # Random augmentation implementations
    def _apply_random_method_insertion(self, code: str) -> str:
        """Apply random method insertion transformation"""
        try:
            # Insert 1-2 random methods
            method_count = random.randint(1, 2)
            return insert_random_methods(code, method_count)
        except Exception as e:
            logger.error(f"Error in random method insertion: {e}")
            return code
    
    def _apply_random_statement_insertion(self, code: str) -> str:
        """Apply random statement insertion transformation"""
        try:
            # Insert 1-3 random statements
            stmt_count = random.randint(1, 3)
            return insert_random_statements(code, stmt_count)
        except Exception as e:
            logger.error(f"Error in random statement insertion: {e}")
            return code
    
    def _apply_random_expression_insertion(self, code: str) -> str:
        """Apply random expression insertion transformation"""
        try:
            # Insert random expressions in appropriate contexts
            lines = code.split('\n')
            modified_lines = []
            
            for line in lines:
                # Look for assignment statements
                if '=' in line and ';' in line:
                    # Insert random expression before assignment
                    if random.random() < 0.3:  # 30% chance
                        random_expr = generate_random_statement().strip()
                        modified_lines.append(random_expr)
                modified_lines.append(line)
            
            return '\n'.join(modified_lines)
        except Exception as e:
            logger.error(f"Error in random expression insertion: {e}")
            return code
