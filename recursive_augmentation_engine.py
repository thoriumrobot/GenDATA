#!/usr/bin/env python3
"""
Recursive Augmentation Engine

This module implements a recursive augmentation engine that can apply transformations
in chains with depth control and dependency tracking. It extends the existing
semantic augmentation systems to support recursive application of transformations.
"""

import os
import re
import random
import ast
import json
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

# Import existing augmentation transformers
from enhanced_semantic_augment_slices import EnhancedSemanticTransformer
from simple_code_semantic_augment_slices import SimpleCodeSemanticTransformer

logger = logging.getLogger(__name__)

class TransformationType(Enum):
    """Enumeration of transformation types"""
    # Enhanced transformations (17 methods)
    LOOP_CONVERSION = "loop_conversion"
    GUARD_REVERSAL = "guard_reversal"
    MATHEMATICAL_EXPRESSION = "mathematical_expression"
    LOGICAL_EXPRESSION = "logical_expression"
    TERNARY_OPERATOR = "ternary_operator"
    SWITCH_STATEMENT = "switch_statement"
    VARIABLE_OPERATION = "variable_operation"
    METHOD_EXTRACTION = "method_extraction"
    CONDITIONAL_EXPRESSION = "conditional_expression"
    ARRAY_ACCESS_PATTERN = "array_access_pattern"
    STRING_CONCATENATION = "string_concatenation"
    NUMERIC_LITERAL = "numeric_literal"
    EXCEPTION_HANDLING = "exception_handling"
    LAMBDA_EXPRESSION = "lambda_expression"
    STREAM_API = "stream_api"
    BUILDER_PATTERN = "builder_pattern"
    FUNCTIONAL_CONVERSION = "functional_conversion"
    
    # Simple transformations (10 methods)
    SIMPLE_METHOD_CALL = "simple_method_call"
    SIMPLE_ASSIGNMENT = "simple_assignment"
    SIMPLE_CONDITIONAL = "simple_conditional"
    SIMPLE_ARRAY_ACCESS = "simple_array_access"
    SIMPLE_RETURN_STATEMENT = "simple_return_statement"
    SIMPLE_VARIABLE_DECLARATION = "simple_variable_declaration"
    SIMPLE_CONSTRUCTOR_CALL = "simple_constructor_call"
    SIMPLE_FIELD_ACCESS = "simple_field_access"
    SIMPLE_STRING_OPERATION = "simple_string_operation"
    SIMPLE_NUMERIC_OPERATION = "simple_numeric_operation"
    
    # Random augmentation transformations (3 methods)
    RANDOM_METHOD_INSERTION = "random_method_insertion"
    RANDOM_STATEMENT_INSERTION = "random_statement_insertion"
    RANDOM_EXPRESSION_INSERTION = "random_expression_insertion"

@dataclass
class TransformationState:
    """Represents the state of code after transformations"""
    code: str
    transformation_history: List[TransformationType]
    depth: int
    complexity_score: float
    compilation_status: bool
    semantic_preservation: bool
    metadata: Dict[str, Any]

@dataclass
class TransformationDependency:
    """Represents dependencies between transformations"""
    source: TransformationType
    target: TransformationType
    weight: float  # 0.0 to 1.0, how likely this dependency is
    conditions: List[str]  # Conditions when this dependency applies

class RecursiveAugmentationEngine:
    """Engine for applying recursive augmentation transformations"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Initialize transformers
        self.enhanced_transformer = EnhancedSemanticTransformer(seed=seed)
        self.simple_transformer = SimpleCodeSemanticTransformer(seed=seed)
        
        # Transformation dependency graph
        self.dependency_graph = self._build_dependency_graph()
        
        # Transformation mappings
        self.enhanced_transformations = {
            TransformationType.LOOP_CONVERSION: self.enhanced_transformer._transform_loops,
            TransformationType.GUARD_REVERSAL: self.enhanced_transformer._transform_guards,
            TransformationType.MATHEMATICAL_EXPRESSION: self.enhanced_transformer._transform_mathematical_expressions,
            TransformationType.LOGICAL_EXPRESSION: self.enhanced_transformer._transform_logical_expressions,
            TransformationType.TERNARY_OPERATOR: self.enhanced_transformer._transform_ternary_operators,
            TransformationType.SWITCH_STATEMENT: self.enhanced_transformer._transform_switch_statements,
            TransformationType.VARIABLE_OPERATION: self.enhanced_transformer._transform_variable_operations,
            TransformationType.METHOD_EXTRACTION: self.enhanced_transformer._transform_method_extraction,
            TransformationType.CONDITIONAL_EXPRESSION: self.enhanced_transformer._transform_conditional_expressions,
            TransformationType.ARRAY_ACCESS_PATTERN: self.enhanced_transformer._transform_array_access_patterns,
            TransformationType.STRING_CONCATENATION: self.enhanced_transformer._transform_string_concatenation,
            TransformationType.NUMERIC_LITERAL: self.enhanced_transformer._transform_numeric_literals,
            TransformationType.EXCEPTION_HANDLING: self.enhanced_transformer._transform_exception_handling,
            TransformationType.LAMBDA_EXPRESSION: self.enhanced_transformer._transform_lambda_expressions,
            TransformationType.STREAM_API: self.enhanced_transformer._transform_stream_api,
            TransformationType.BUILDER_PATTERN: self.enhanced_transformer._transform_builder_patterns,
            TransformationType.FUNCTIONAL_CONVERSION: self.enhanced_transformer._transform_functional_conversions,
        }
        
        self.simple_transformations = {
            TransformationType.SIMPLE_METHOD_CALL: self.simple_transformer._transform_simple_method_calls,
            TransformationType.SIMPLE_ASSIGNMENT: self.simple_transformer._transform_simple_assignments,
            TransformationType.SIMPLE_CONDITIONAL: self.simple_transformer._transform_simple_conditionals,
            TransformationType.SIMPLE_ARRAY_ACCESS: self.simple_transformer._transform_simple_array_access,
            TransformationType.SIMPLE_RETURN_STATEMENT: self.simple_transformer._transform_simple_return_statements,
            TransformationType.SIMPLE_VARIABLE_DECLARATION: self.simple_transformer._transform_simple_variable_declarations,
            TransformationType.SIMPLE_CONSTRUCTOR_CALL: self.simple_transformer._transform_simple_constructor_calls,
            TransformationType.SIMPLE_FIELD_ACCESS: self.simple_transformer._transform_simple_field_access,
            TransformationType.SIMPLE_STRING_OPERATION: self.simple_transformer._transform_simple_string_operations,
            TransformationType.SIMPLE_NUMERIC_OPERATION: self.simple_transformer._transform_simple_numeric_operations,
        }
        
        # Random augmentation transformations (3 methods)
        self.random_transformations = {
            TransformationType.RANDOM_METHOD_INSERTION: self._apply_random_method_insertion,
            TransformationType.RANDOM_STATEMENT_INSERTION: self._apply_random_statement_insertion,
            TransformationType.RANDOM_EXPRESSION_INSERTION: self._apply_random_expression_insertion,
        }
        
        # All transformations
        self.all_transformations = {**self.enhanced_transformations, **self.simple_transformations, **self.random_transformations}
        
        # Statistics
        self.transformation_stats = {
            'total_applications': 0,
            'successful_transformations': 0,
            'failed_transformations': 0,
            'recursion_depths': {},
            'transformation_counts': {t.value: 0 for t in TransformationType}
        }

    def _build_dependency_graph(self) -> Dict[TransformationType, List[TransformationDependency]]:
        """Build transformation dependency graph"""
        dependencies = {}
        
        # Enhanced transformation dependencies
        enhanced_deps = [
            # Method extraction enables variable operations
            TransformationDependency(
                source=TransformationType.METHOD_EXTRACTION,
                target=TransformationType.VARIABLE_OPERATION,
                weight=0.8,
                conditions=["extracted_method_has_variables"]
            ),
            # Variable operations enable mathematical expressions
            TransformationDependency(
                source=TransformationType.VARIABLE_OPERATION,
                target=TransformationType.MATHEMATICAL_EXPRESSION,
                weight=0.7,
                conditions=["variables_in_expressions"]
            ),
            # Loop conversions enable guard reversals
            TransformationDependency(
                source=TransformationType.LOOP_CONVERSION,
                target=TransformationType.GUARD_REVERSAL,
                weight=0.6,
                conditions=["loop_has_conditions"]
            ),
            # Conditional expressions enable ternary operators
            TransformationDependency(
                source=TransformationType.CONDITIONAL_EXPRESSION,
                target=TransformationType.TERNARY_OPERATOR,
                weight=0.5,
                conditions=["simple_conditionals"]
            ),
            # Array access patterns enable numeric literals
            TransformationDependency(
                source=TransformationType.ARRAY_ACCESS_PATTERN,
                target=TransformationType.NUMERIC_LITERAL,
                weight=0.4,
                conditions=["array_indices_are_numeric"]
            ),
        ]
        
        # Simple transformation dependencies
        simple_deps = [
            # Simple assignments enable simple conditionals
            TransformationDependency(
                source=TransformationType.SIMPLE_ASSIGNMENT,
                target=TransformationType.SIMPLE_CONDITIONAL,
                weight=0.6,
                conditions=["assignments_in_conditions"]
            ),
            # Simple method calls enable simple field access
            TransformationDependency(
                source=TransformationType.SIMPLE_METHOD_CALL,
                target=TransformationType.SIMPLE_FIELD_ACCESS,
                weight=0.5,
                conditions=["method_returns_objects"]
            ),
            # Simple numeric operations enable simple array access
            TransformationDependency(
                source=TransformationType.SIMPLE_NUMERIC_OPERATION,
                target=TransformationType.SIMPLE_ARRAY_ACCESS,
                weight=0.4,
                conditions=["numeric_values_used_as_indices"]
            ),
        ]
        
        # Cross-system dependencies
        cross_deps = [
            # Simple transformations can enable enhanced ones
            TransformationDependency(
                source=TransformationType.SIMPLE_VARIABLE_DECLARATION,
                target=TransformationType.VARIABLE_OPERATION,
                weight=0.3,
                conditions=["complex_variables"]
            ),
            TransformationDependency(
                source=TransformationType.SIMPLE_CONDITIONAL,
                target=TransformationType.GUARD_REVERSAL,
                weight=0.3,
                conditions=["complex_conditionals"]
            ),
        ]
        
        all_deps = enhanced_deps + simple_deps + cross_deps
        
        # Build dependency graph
        for dep in all_deps:
            if dep.source not in dependencies:
                dependencies[dep.source] = []
            dependencies[dep.source].append(dep)
        
        return dependencies

    def apply_recursive_transformation(self, code: str, max_depth: int = 3, 
                                     transformation_sequence: Optional[List[TransformationType]] = None,
                                     complexity_threshold: float = 3.0) -> List[TransformationState]:
        """
        Apply recursive transformations to code
        
        Args:
            code: Input Java code
            max_depth: Maximum recursion depth
            transformation_sequence: Optional predefined sequence of transformations
            complexity_threshold: Threshold for choosing enhanced vs simple transformations
            
        Returns:
            List of transformation states at different depths
        """
        initial_state = TransformationState(
            code=code,
            transformation_history=[],
            depth=0,
            complexity_score=self._compute_complexity_score(code),
            compilation_status=True,
            semantic_preservation=True,
            metadata={'original_code': code}
        )
        
        states = [initial_state]
        
        if transformation_sequence:
            # Apply predefined sequence
            current_state = initial_state
            for i, transformation in enumerate(transformation_sequence):
                if current_state.depth >= max_depth:
                    break
                    
                new_state = self._apply_single_transformation(
                    current_state, transformation, i
                )
                if new_state:
                    states.append(new_state)
                    current_state = new_state
        else:
            # Apply random recursive transformations
            current_state = initial_state
            while current_state.depth < max_depth:
                # Get valid next transformations
                valid_transformations = self.get_valid_next_transformations(current_state)
                
                if not valid_transformations:
                    break
                
                # Choose transformation based on complexity and dependencies
                chosen_transformation = self._choose_transformation(
                    current_state, valid_transformations, complexity_threshold
                )
                
                new_state = self._apply_single_transformation(
                    current_state, chosen_transformation, current_state.depth
                )
                
                if new_state:
                    states.append(new_state)
                    current_state = new_state
                else:
                    break
        
        # Update statistics
        self.transformation_stats['total_applications'] += 1
        self.transformation_stats['successful_transformations'] += len(states) - 1
        
        return states

    def _apply_single_transformation(self, state: TransformationState, 
                                   transformation: TransformationType, 
                                   step: int) -> Optional[TransformationState]:
        """Apply a single transformation to a state"""
        try:
            # Apply transformation
            if transformation in self.enhanced_transformations:
                transformer_func = self.enhanced_transformations[transformation]
            elif transformation in self.simple_transformations:
                transformer_func = self.simple_transformations[transformation]
            elif transformation in self.random_transformations:
                transformer_func = self.random_transformations[transformation]
            else:
                logger.warning(f"Unknown transformation: {transformation}")
                return None
            
            # Apply transformation
            transformed_code = transformer_func(state.code)
            
            # Check if transformation was successful
            if transformed_code == state.code:
                logger.debug(f"Transformation {transformation} did not change code")
                return None
            
            # Create new state
            new_history = state.transformation_history + [transformation]
            new_complexity = self._compute_complexity_score(transformed_code)
            
            new_state = TransformationState(
                code=transformed_code,
                transformation_history=new_history,
                depth=state.depth + 1,
                complexity_score=new_complexity,
                compilation_status=self._check_compilation(transformed_code),
                semantic_preservation=self._verify_semantic_preservation(state.code, transformed_code),
                metadata={
                    **state.metadata,
                    'transformation_applied': transformation.value,
                    'step': step,
                    'previous_complexity': state.complexity_score
                }
            )
            
            # Update statistics
            self.transformation_stats['transformation_counts'][transformation.value] += 1
            if new_state.depth not in self.transformation_stats['recursion_depths']:
                self.transformation_stats['recursion_depths'][new_state.depth] = 0
            self.transformation_stats['recursion_depths'][new_state.depth] += 1
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error applying transformation {transformation}: {e}")
            self.transformation_stats['failed_transformations'] += 1
            return None

    def get_valid_next_transformations(self, state: TransformationState) -> List[TransformationType]:
        """Get valid next transformations based on current state and dependencies"""
        valid_transformations = []
        
        # Get dependent transformations
        if state.transformation_history:
            last_transformation = state.transformation_history[-1]
            if last_transformation in self.dependency_graph:
                for dep in self.dependency_graph[last_transformation]:
                    # Check if dependency conditions are met
                    if self._check_dependency_conditions(state, dep.conditions):
                        if random.random() < dep.weight:
                            valid_transformations.append(dep.target)
        
        # Add all transformations if no dependencies or if we want more variety
        if not valid_transformations or random.random() < 0.3:  # 30% chance to explore freely
            all_transforms = list(TransformationType)
            # Remove already applied transformations to avoid immediate repetition
            available_transforms = [t for t in all_transforms if t not in state.transformation_history[-3:]]
            valid_transformations.extend(available_transforms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_valid = []
        for t in valid_transformations:
            if t not in seen:
                seen.add(t)
                unique_valid.append(t)
        
        return unique_valid

    def _choose_transformation(self, state: TransformationState, 
                             valid_transformations: List[TransformationType],
                             complexity_threshold: float) -> TransformationType:
        """Choose transformation based on complexity and other factors"""
        # Prefer enhanced transformations for complex code
        if state.complexity_score >= complexity_threshold:
            enhanced_transforms = [t for t in valid_transformations 
                                 if t in self.enhanced_transformations]
            if enhanced_transforms:
                return random.choice(enhanced_transforms)
        
        # Prefer simple transformations for simple code
        else:
            simple_transforms = [t for t in valid_transformations 
                               if t in self.simple_transformations]
            if simple_transforms:
                return random.choice(simple_transforms)
        
        # Fall back to any valid transformation
        return random.choice(valid_transformations)

    def _compute_complexity_score(self, code: str) -> float:
        """Compute code complexity score"""
        score = 0.0
        
        # Count various complexity indicators
        score += code.count('for ') * 0.5
        score += code.count('while ') * 0.5
        score += code.count('if ') * 0.3
        score += code.count('switch ') * 0.8
        score += code.count('try ') * 0.6
        score += code.count('catch ') * 0.4
        score += code.count('->') * 0.7  # Lambda expressions
        score += code.count('.stream()') * 0.6  # Stream API
        score += code.count('new ') * 0.2
        score += code.count('throws ') * 0.5
        
        # Normalize by code length
        lines = len([line for line in code.split('\n') if line.strip()])
        if lines > 0:
            score = score / lines * 10  # Scale to 0-10 range
        
        return min(score, 10.0)

    def _check_dependency_conditions(self, state: TransformationState, 
                                   conditions: List[str]) -> bool:
        """Check if dependency conditions are met"""
        for condition in conditions:
            if condition == "extracted_method_has_variables":
                # Check if recent method extraction created variables
                if any(t == TransformationType.METHOD_EXTRACTION for t in state.transformation_history[-2:]):
                    return True
            elif condition == "variables_in_expressions":
                # Check if code has variable expressions
                return bool(re.search(r'\w+\s*[+\-*/]\s*\w+', state.code))
            elif condition == "loop_has_conditions":
                # Check if loops have conditions
                return bool(re.search(r'for\s*\([^)]*;[^)]*;[^)]*\)', state.code))
            # Add more conditions as needed
        
        return True  # Default to allowing transformation

    def _check_compilation(self, code: str) -> bool:
        """Basic compilation check"""
        # Simple heuristics for compilation success
        if not code.strip():
            return False
        
        # Check for balanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            return False
        
        # Check for balanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            return False
        
        return True

    def _verify_semantic_preservation(self, original: str, transformed: str) -> bool:
        """Verify that semantic meaning is preserved"""
        # This is a simplified check - in practice, you'd want more sophisticated analysis
        
        # Check that both have similar structure
        original_lines = len([l for l in original.split('\n') if l.strip()])
        transformed_lines = len([l for l in transformed.split('\n') if l.strip()])
        
        # Allow for some variation in line count due to transformations
        line_diff = abs(original_lines - transformed_lines)
        if line_diff > max(original_lines, transformed_lines) * 0.5:
            return False
        
        # Check that key identifiers are preserved
        original_ids = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', original))
        transformed_ids = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', transformed))
        
        # Most identifiers should be preserved (allowing for new ones from transformations)
        common_ids = original_ids.intersection(transformed_ids)
        if len(original_ids) > 0:
            preservation_ratio = len(common_ids) / len(original_ids)
            if preservation_ratio < 0.5:  # Less than 50% preservation
                return False
        
        return True

    def extract_method_recursive(self, code: str, depth: int = 2) -> List[TransformationState]:
        """Extract methods and recursively transform the extracted code"""
        states = []
        current_code = code
        current_depth = 0
        
        while current_depth < depth:
            # Find expressions that can be extracted
            extractable_expressions = self._find_extractable_expressions(current_code)
            
            if not extractable_expressions:
                break
            
            # Extract one expression
            extracted_code = self._extract_expression_to_method(current_code, extractable_expressions[0])
            
            if extracted_code != current_code:
                state = TransformationState(
                    code=extracted_code,
                    transformation_history=[TransformationType.METHOD_EXTRACTION] * (current_depth + 1),
                    depth=current_depth + 1,
                    complexity_score=self._compute_complexity_score(extracted_code),
                    compilation_status=self._check_compilation(extracted_code),
                    semantic_preservation=self._verify_semantic_preservation(code, extracted_code),
                    metadata={'extraction_depth': current_depth + 1}
                )
                states.append(state)
                current_code = extracted_code
                current_depth += 1
            else:
                break
        
        return states

    def extract_variables_recursive(self, code: str, depth: int = 2) -> List[TransformationState]:
        """Extract variables and recursively transform expressions"""
        states = []
        current_code = code
        current_depth = 0
        
        while current_depth < depth:
            # Find expressions that can be extracted to variables
            extractable_expressions = self._find_variable_extractable_expressions(current_code)
            
            if not extractable_expressions:
                break
            
            # Extract one expression to variable
            extracted_code = self._extract_expression_to_variable(current_code, extractable_expressions[0])
            
            if extracted_code != current_code:
                state = TransformationState(
                    code=extracted_code,
                    transformation_history=[TransformationType.VARIABLE_OPERATION] * (current_depth + 1),
                    depth=current_depth + 1,
                    complexity_score=self._compute_complexity_score(extracted_code),
                    compilation_status=self._check_compilation(extracted_code),
                    semantic_preservation=self._verify_semantic_preservation(code, extracted_code),
                    metadata={'variable_extraction_depth': current_depth + 1}
                )
                states.append(state)
                current_code = extracted_code
                current_depth += 1
            else:
                break
        
        return states

    def _find_extractable_expressions(self, code: str) -> List[str]:
        """Find expressions that can be extracted to methods"""
        # Look for complex expressions in assignments and returns
        patterns = [
            r'(\w+)\s*=\s*([^;]+);',  # Variable assignments
            r'return\s+([^;]+);',     # Return statements
        ]
        
        extractable = []
        for pattern in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                expression = match.group(-1).strip()  # Last group
                if len(expression) > 20 and self._is_complex_expression(expression):
                    extractable.append(expression)
        
        return extractable

    def _find_variable_extractable_expressions(self, code: str) -> List[str]:
        """Find expressions that can be extracted to variables"""
        # Look for repeated or complex expressions
        expressions = []
        patterns = [
            r'(\w+)\s*=\s*([^;]+);',
            r'return\s+([^;]+);',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                expression = match.group(-1).strip()
                if len(expression) > 15 and self._is_complex_expression(expression):
                    expressions.append(expression)
        
        # Find expressions that appear multiple times
        repeated_expressions = []
        for expr in expressions:
            count = code.count(expr)
            if count > 1:
                repeated_expressions.append(expr)
        
        return repeated_expressions

    def _is_complex_expression(self, expression: str) -> bool:
        """Check if expression is complex enough to warrant extraction"""
        # Count operators and method calls
        operators = expression.count('+') + expression.count('-') + expression.count('*') + expression.count('/')
        method_calls = expression.count('(')
        
        return operators > 2 or method_calls > 1 or len(expression) > 30

    def _extract_expression_to_method(self, code: str, expression: str) -> str:
        """Extract an expression to a method"""
        # Generate method name
        method_name = f"compute{random.randint(1, 100)}"
        
        # Replace expression with method call
        new_code = code.replace(expression, f"{method_name}()")
        
        # Add method definition
        method_def = f"""
    private int {method_name}() {{
        return {expression};
    }}"""
        
        # Insert method before the last closing brace
        last_brace = new_code.rfind('}')
        if last_brace != -1:
            new_code = new_code[:last_brace] + method_def + new_code[last_brace:]
        
        return new_code

    def _extract_expression_to_variable(self, code: str, expression: str) -> str:
        """Extract an expression to a variable"""
        # Generate variable name
        var_name = f"temp{random.randint(1, 100)}"
        
        # Find the first occurrence of the expression
        first_occurrence = code.find(expression)
        if first_occurrence == -1:
            return code
        
        # Replace with variable
        new_code = code.replace(expression, var_name, 1)
        
        # Add variable declaration before the first usage
        var_decl = f"int {var_name} = {expression};\n        "
        
        # Insert before the line containing the variable
        lines = new_code.split('\n')
        for i, line in enumerate(lines):
            if var_name in line and '=' not in line:
                lines[i] = var_decl + line
                break
        
        return '\n'.join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        return self.transformation_stats.copy()

    def reset_statistics(self):
        """Reset transformation statistics"""
        self.transformation_stats = {
            'total_applications': 0,
            'successful_transformations': 0,
            'failed_transformations': 0,
            'recursion_depths': {},
            'transformation_counts': {t.value: 0 for t in TransformationType}
        }
    
    def apply_transformation(self, code: str, transformation_type: TransformationType, 
                           deterministic: bool = False) -> Dict[str, Any]:
        """Apply a single transformation to code"""
        try:
            # Apply transformation
            if transformation_type in self.enhanced_transformations:
                transformer_func = self.enhanced_transformations[transformation_type]
            elif transformation_type in self.simple_transformations:
                transformer_func = self.simple_transformations[transformation_type]
            elif transformation_type in self.random_transformations:
                transformer_func = self.random_transformations[transformation_type]
            else:
                logger.warning(f"Unknown transformation: {transformation_type}")
                return {'success': False, 'error': f'Unknown transformation: {transformation_type}'}
            
            # Apply transformation
            transformed_code = transformer_func(code)
            
            # Check if transformation was successful
            success = transformed_code != code
            
            return {
                'success': success,
                'augmented_code': transformed_code,
                'transformation_type': transformation_type.value,
                'deterministic': deterministic
            }
            
        except Exception as e:
            logger.error(f"Error applying transformation {transformation_type}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _apply_random_method_insertion(self, code: str) -> str:
        """Apply random method insertion transformation"""
        try:
            from augment_slices import insert_random_methods
            # Insert 1-2 random methods
            method_count = random.randint(1, 2)
            return insert_random_methods(code, method_count)
        except Exception as e:
            logger.error(f"Error in random method insertion: {e}")
            return code
    
    def _apply_random_statement_insertion(self, code: str) -> str:
        """Apply random statement insertion transformation"""
        try:
            from augment_slices import insert_random_statements
            # Insert 1-3 random statements
            stmt_count = random.randint(1, 3)
            return insert_random_statements(code, stmt_count)
        except Exception as e:
            logger.error(f"Error in random statement insertion: {e}")
            return code
    
    def _apply_random_expression_insertion(self, code: str) -> str:
        """Apply random expression insertion transformation"""
        try:
            from augment_slices import generate_random_statement
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


def main():
    """Test the recursive augmentation engine"""
    # Test code
    test_code = """
public class TestClass {
    public int calculateSum(int[] arr) {
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum = sum + arr[i];
        }
        return sum;
    }
}
"""
    
    # Create engine
    engine = RecursiveAugmentationEngine(seed=42)
    
    # Test recursive transformation
    print("Testing recursive transformation...")
    states = engine.apply_recursive_transformation(test_code, max_depth=3)
    
    print(f"Generated {len(states)} transformation states:")
    for i, state in enumerate(states):
        print(f"\nState {i}:")
        print(f"  Depth: {state.depth}")
        print(f"  Transformations: {[t.value for t in state.transformation_history]}")
        print(f"  Complexity: {state.complexity_score:.2f}")
        print(f"  Compiles: {state.compilation_status}")
        print(f"  Semantic preserved: {state.semantic_preservation}")
    
    # Test method extraction
    print("\nTesting method extraction...")
    method_states = engine.extract_method_recursive(test_code, depth=2)
    print(f"Generated {len(method_states)} method extraction states")
    
    # Print statistics
    print("\nStatistics:")
    stats = engine.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
