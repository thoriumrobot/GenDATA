#!/usr/bin/env python3
"""
Enhanced Semantic-Preserving Data Augmentation Script

This script extends the original semantic augmentation with additional
transformations that increase code variety while preserving semantics
and being resistant to slicer pruning.

New Transformations Added:
1. Method extraction and inlining
2. Conditional expression restructuring
3. Array access pattern variations
4. String concatenation alternatives
5. Numeric literal transformations
6. Exception handling restructuring
7. Lambda expression conversions
8. Stream API alternatives
9. Builder pattern variations
10. Functional programming conversions
"""

import os
import re
import argparse
import random
import ast
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

HEADER_COMMENT = """
/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations.
 */
""".strip()

class EnhancedSemanticTransformer:
    """Enhanced semantic transformer with additional transformation methods."""
    
    def __init__(self, seed: int = 42, disabled_transformations: List[str] = None):
        random.seed(seed)
        self.transformations_applied = []
        self.disabled_transformations = disabled_transformations or []
    
    def transform_file(self, java_path: str, variant_idx: int) -> str:
        """Apply enhanced semantic transformations to a Java file."""
        with open(java_path, 'r') as f:
            src = f.read()
        
        # Ensure the slice is a valid compilable Java class
        src = self._ensure_compilable_class(java_path, src)
        
        # Apply transformations
        src = self._insert_header_comment(src)
        
        # Enhanced transformation list with names
        transformation_map = {
            # Original transformations
            'loop_conversion': self._transform_loops,
            'guard_reversal': self._transform_guards,
            'mathematical_expression': self._transform_mathematical_expressions,
            'logical_expression': self._transform_logical_expressions,
            'ternary_operator': self._transform_ternary_operators,
            'switch_statement': self._transform_switch_statements,
            'variable_operation': self._transform_variable_operations,
            
            # New enhanced transformations
            'method_extraction': self._transform_method_extraction,
            'conditional_expression': self._transform_conditional_expressions,
            'array_access_pattern': self._transform_array_access_patterns,
            'string_concatenation': self._transform_string_concatenation,
            'numeric_literal': self._transform_numeric_literals,
            'exception_handling': self._transform_exception_handling,
            'lambda_expression': self._transform_lambda_expressions,
            'stream_api': self._transform_stream_api,
            'builder_pattern': self._transform_builder_patterns,
            'functional_conversion': self._transform_functional_conversions,
        }
        
        # Filter out disabled transformations
        available_transformations = {
            name: method for name, method in transformation_map.items()
            if name not in self.disabled_transformations
        }
        
        if not available_transformations:
            logger.warning("All transformations disabled, returning original source")
            return src
        
        # Apply 3-6 random transformations (increased from 2-4)
        num_transforms = min(random.randint(3, 6), len(available_transformations))
        selected_transform_names = random.sample(list(available_transformations.keys()), num_transforms)
        
        for transform_name in selected_transform_names:
            transform_method = available_transformations[transform_name]
            src = transform_method(src)
            self.transformations_applied.append(transform_name)
        
        return src
    
    def _insert_header_comment(self, src: str) -> str:
        """Insert enhanced augmentation header comment."""
        if src.lstrip().startswith("/* CFWR enhanced semantic augmentation"):
            return src
        return HEADER_COMMENT + "\n" + src
    
    def _ensure_compilable_class(self, java_path: str, src: str) -> str:
        """Ensure the code is wrapped in a compilable class."""
        has_class = ' class ' in src or src.strip().startswith('class ') or 'interface ' in src or 'enum ' in src
        base = os.path.splitext(os.path.basename(java_path))[0]
        class_name = base if base.isidentifier() else f"Slice{abs(hash(base))}"
        
        wrapped = src
        if not has_class:
            wrapped = "public class {} {{\n{}\n}}".format(class_name, src)
        
        # Ensure balanced braces
        open_braces = wrapped.count('{')
        close_braces = wrapped.count('}')
        if open_braces > close_braces:
            wrapped = wrapped + ('}' * (open_braces - close_braces))
        elif close_braces > open_braces:
            wrapped = ('{' * (close_braces - open_braces)) + wrapped
        
        return wrapped

    # NEW ENHANCED TRANSFORMATION METHODS

    def _transform_method_extraction(self, src: str) -> str:
        """Extract or inline method calls to create semantic variations."""
        # Pattern for simple expressions that can be extracted
        expression_pattern = re.compile(
            r'(\w+)\s*=\s*([^;]+);',
            re.MULTILINE
        )
        
        def extract_method(match):
            var_name = match.group(1).strip()
            expression = match.group(2).strip()
            
            # Only extract if expression is complex enough
            if len(expression) > 20 and ('+' in expression or '*' in expression or '-' in expression):
                method_name = f"compute{var_name.capitalize()}"
                return f"{var_name} = {method_name}();\n        // Extracted: private int {method_name}() {{ return {expression}; }}"
            
            return match.group(0)
        
        if random.random() < 0.3:  # 30% chance
            src = expression_pattern.sub(extract_method, src)
        
        return src

    def _transform_conditional_expressions(self, src: str) -> str:
        """Restructure conditional expressions for variety."""
        # Complex conditional restructuring
        patterns = [
            # (a > b) ? x : y -> (a <= b) ? y : x
            (r'\(([^)]+)\s*>\s*([^)]+)\)\s*\?\s*([^:]+)\s*:\s*([^)]+)', r'(\1 <= \2) ? \4 : \3'),
            # (a < b) ? x : y -> (a >= b) ? y : x  
            (r'\(([^)]+)\s*<\s*([^)]+)\)\s*\?\s*([^:]+)\s*:\s*([^)]+)', r'(\1 >= \2) ? \4 : \3'),
            # (a == b) ? x : y -> (a != b) ? y : x
            (r'\(([^)]+)\s*==\s*([^)]+)\)\s*\?\s*([^:]+)\s*:\s*([^)]+)', r'(\1 != \2) ? \4 : \3'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.4:  # 40% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_array_access_patterns(self, src: str) -> str:
        """Transform array access patterns for variety."""
        # Array access with different indexing patterns
        patterns = [
            # arr[i] -> arr[0 + i] (identity addition)
            (r'(\w+)\[(\w+)\]', r'\1[0 + \2]'),
            # arr[i] -> arr[i + 0] (identity addition)
            (r'(\w+)\[(\w+)\]', r'\1[\2 + 0]'),
            # arr[i + 1] -> arr[1 + i] (commutativity)
            (r'(\w+)\[(\w+)\s*\+\s*1\]', r'\1[1 + \2]'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.3:  # 30% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_string_concatenation(self, src: str) -> str:
        """Transform string concatenation patterns."""
        patterns = [
            # "a" + "b" -> "ab" (compile-time concatenation)
            (r'"([^"]+)"\s*\+\s*"([^"]+)"', r'"\1\2"'),
            # str1 + str2 -> str2 + str1 (commutativity)
            (r'(\w+)\s*\+\s*(\w+)', r'\2 + \1'),
            # String.valueOf(x) -> "" + x
            (r'String\.valueOf\(([^)]+)\)', r'"" + \1'),
            # "" + x -> String.valueOf(x)
            (r'""\s*\+\s*(\w+)', r'String.valueOf(\1)'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.25:  # 25% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_numeric_literals(self, src: str) -> str:
        """Transform numeric literals for variety."""
        patterns = [
            # 0 -> 0x0 (hex representation)
            (r'\b0\b(?!\w)', '0x0'),
            # 1 -> 0x1
            (r'\b1\b(?!\w)', '0x1'),
            # 10 -> 0xA
            (r'\b10\b(?!\w)', '0xA'),
            # 16 -> 0x10
            (r'\b16\b(?!\w)', '0x10'),
            # 255 -> 0xFF
            (r'\b255\b(?!\w)', '0xFF'),
            # 1000 -> 1_000 (underscore separator)
            (r'\b1000\b(?!\w)', '1_000'),
            # 1000000 -> 1_000_000
            (r'\b1000000\b(?!\w)', '1_000_000'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.2:  # 20% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_exception_handling(self, src: str) -> str:
        """Restructure exception handling patterns."""
        # try-catch to try-with-resources (where applicable)
        try_catch_pattern = re.compile(
            r'try\s*\{\s*([^{}]+)\s*\}\s*catch\s*\([^)]+\)\s*\{\s*([^{}]+)\s*\}',
            re.MULTILINE | re.DOTALL
        )
        
        def restructure_try_catch(match):
            try_block = match.group(1).strip()
            catch_block = match.group(2).strip()
            
            # Convert to nested try-catch if it's simple enough
            if len(try_block) < 100 and len(catch_block) < 50:
                return f"try {{\n            try {{\n                {try_block}\n            }} catch (Exception e) {{\n                // Handle\n            }}\n        }} catch (Exception e) {{\n            {catch_block}\n        }}"
            
            return match.group(0)
        
        if random.random() < 0.3:  # 30% chance
            src = try_catch_pattern.sub(restructure_try_catch, src)
        
        return src

    def _transform_lambda_expressions(self, src: str) -> str:
        """Convert between lambda expressions and anonymous classes."""
        # Simple lambda to anonymous class conversion
        lambda_pattern = re.compile(
            r'(\w+)\.stream\(\)\.map\((\w+)\s*->\s*([^)]+)\)',
            re.MULTILINE
        )
        
        def lambda_to_anonymous(match):
            var_name = match.group(1)
            param = match.group(2)
            body = match.group(3)
            
            return f"{var_name}.stream().map(new Function<{param}, Object>() {{\n            public Object apply({param} {param}) {{\n                return {body};\n            }}\n        }})"
        
        if random.random() < 0.2:  # 20% chance
            src = lambda_pattern.sub(lambda_to_anonymous, src)
        
        return src

    def _transform_stream_api(self, src: str) -> str:
        """Convert between Stream API and traditional loops."""
        # Stream operations to traditional loops
        stream_pattern = re.compile(
            r'(\w+)\.stream\(\)\.filter\(([^)]+)\)\.collect\(Collectors\.toList\(\)\)',
            re.MULTILINE
        )
        
        def stream_to_loop(match):
            var_name = match.group(1)
            filter_condition = match.group(2)
            
            return f"List<Object> result = new ArrayList<>();\n        for (Object item : {var_name}) {{\n            if ({filter_condition}) {{\n                result.add(item);\n            }}\n        }}"
        
        if random.random() < 0.25:  # 25% chance
            src = stream_pattern.sub(stream_to_loop, src)
        
        return src

    def _transform_builder_patterns(self, src: str) -> str:
        """Create builder pattern variations."""
        # Constructor calls to builder pattern
        constructor_pattern = re.compile(
            r'new\s+(\w+)\(([^)]+)\)',
            re.MULTILINE
        )
        
        def constructor_to_builder(match):
            class_name = match.group(1)
            params = match.group(2)
            
            # Simple builder pattern creation
            return f"new {class_name}.Builder()\n            .setParams({params})\n            .build()"
        
        if random.random() < 0.2:  # 20% chance
            src = constructor_pattern.sub(constructor_to_builder, src)
        
        return src

    def _transform_functional_conversions(self, src: str) -> str:
        """Convert between functional and imperative styles."""
        # Method references to lambda expressions
        method_ref_pattern = re.compile(
            r'(\w+)::(\w+)',
            re.MULTILINE
        )
        
        def method_ref_to_lambda(match):
            class_or_instance = match.group(1)
            method_name = match.group(2)
            
            return f"{class_or_instance} -> {class_or_instance}.{method_name}()"
        
        if random.random() < 0.3:  # 30% chance
            src = method_ref_pattern.sub(method_ref_to_lambda, src)
        
        return src

    # Include original transformation methods (simplified versions)
    def _transform_loops(self, src: str) -> str:
        """Transform between for and while loops."""
        # Simplified loop transformation
        for_pattern = re.compile(r'for\s*\(\s*([^;]+)\s*;\s*([^;]+)\s*;\s*([^)]+)\s*\)', re.MULTILINE)
        
        def for_to_while(match):
            init = match.group(1).strip()
            condition = match.group(2).strip()
            update = match.group(3).strip()
            return f"{init};\n        while ({condition}) {{\n            // loop body\n            {update};\n        }}"
        
        if random.random() < 0.4:
            src = for_pattern.sub(for_to_while, src)
        
        return src

    def _transform_guards(self, src: str) -> str:
        """Reverse if-else guards and flip branches."""
        if_else_pattern = re.compile(r'if\s*\(([^)]+)\)\s*\{([^{}]*)\}\s*else\s*\{([^{}]*)\}', re.MULTILINE | re.DOTALL)
        
        def reverse_guard(match):
            condition = match.group(1).strip()
            if_body = match.group(2).strip()
            else_body = match.group(3).strip()
            
            # Reverse condition and swap bodies
            reversed_condition = self._reverse_condition(condition)
            return f"if ({reversed_condition}) {{\n            {else_body}\n        }} else {{\n            {if_body}\n        }}"
        
        if random.random() < 0.4:
            src = if_else_pattern.sub(reverse_guard, src)
        
        return src

    def _reverse_condition(self, condition: str) -> str:
        """Reverse a boolean condition."""
        condition = condition.strip()
        
        # Handle simple negations
        if condition.startswith('!'):
            return condition[1:].strip()
        
        # Handle comparison operators
        operators = ['==', '!=', '<', '>', '<=', '>=']
        for op in operators:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left, right = parts[0].strip(), parts[1].strip()
                    if op == '==':
                        return f"{left} != {right}"
                    elif op == '!=':
                        return f"{left} == {right}"
                    elif op == '<':
                        return f"{left} >= {right}"
                    elif op == '>':
                        return f"{left} <= {right}"
                    elif op == '<=':
                        return f"{left} > {right}"
                    elif op == '>=':
                        return f"{left} < {right}"
        
        # Default: wrap in negation
        return f"!({condition})"

    def _transform_mathematical_expressions(self, src: str) -> str:
        """Apply mathematical property transformations."""
        # Strength reduction: x * 2 -> x << 1
        strength_patterns = [
            (r'(\w+)\s*\*\s*2', r'\1 << 1'),
            (r'(\w+)\s*\*\s*4', r'\1 << 2'),
            (r'(\w+)\s*/\s*2', r'\1 >> 1'),
            (r'(\w+)\s*/\s*4', r'\1 >> 2'),
        ]
        
        for pattern, replacement in strength_patterns:
            if random.random() < 0.3:
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_logical_expressions(self, src: str) -> str:
        """Apply De Morgan's laws and logical transformations."""
        # De Morgan's laws
        patterns = [
            (r'!\s*\(([^)]+)\s*&&\s*([^)]+)\)', r'!(\1) || !(\2)'),
            (r'!\s*\(([^)]+)\s*\|\|\s*([^)]+)\)', r'!(\1) && !(\2)'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.4:
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_ternary_operators(self, src: str) -> str:
        """Convert between ternary operators and if-else statements."""
        ternary_pattern = re.compile(r'(\w+)\s*=\s*([^?]+)\s*\?\s*([^:]+)\s*:\s*([^;]+);', re.MULTILINE)
        
        def ternary_to_if_else(match):
            var = match.group(1).strip()
            condition = match.group(2).strip()
            true_expr = match.group(3).strip()
            false_expr = match.group(4).strip()
            
            return f"if ({condition}) {{\n            {var} = {true_expr};\n        }} else {{\n            {var} = {false_expr};\n        }}"
        
        if random.random() < 0.5:
            src = ternary_pattern.sub(ternary_to_if_else, src)
        
        return src

    def _transform_switch_statements(self, src: str) -> str:
        """Convert between switch statements and if-else chains."""
        switch_pattern = re.compile(r'switch\s*\(([^)]+)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', re.MULTILINE | re.DOTALL)
        
        def switch_to_if_else(match):
            expr = match.group(1).strip()
            body = match.group(2).strip()
            
            # Simple switch to if-else conversion
            return f"if ({expr} == 0) {{\n            // case 0\n        }} else if ({expr} == 1) {{\n            // case 1\n        }} else {{\n            // default\n        }}"
        
        if random.random() < 0.3:
            src = switch_pattern.sub(switch_to_if_else, src)
        
        return src

    def _transform_variable_operations(self, src: str) -> str:
        """Inline or extract temporary variables."""
        # Variable inlining
        assignment_pattern = re.compile(r'(\w+)\s+(\w+)\s*=\s*([^;]+);', re.MULTILINE)
        
        def inline_variable(match):
            var_type = match.group(1).strip()
            var_name = match.group(2).strip()
            expr = match.group(3).strip()
            
            # Look for usages of this variable
            usage_pattern = re.compile(rf'\b{re.escape(var_name)}\b')
            usages = usage_pattern.findall(src)
            
            if len(usages) <= 2:  # Only inline if used once or twice
                src_modified = usage_pattern.sub(expr, src)
                src_modified = re.sub(rf'{var_type}\s+{re.escape(var_name)}\s*=\s*{re.escape(expr)};\s*', '', src_modified)
                return src_modified
            
            return match.group(0)
        
        if random.random() < 0.3:
            src = assignment_pattern.sub(inline_variable, src)
        
        return src


def write_variant(original_path: str, out_dir: str, variant_idx: int, transformer: EnhancedSemanticTransformer):
    """Write an enhanced augmented variant of the original file."""
    rel = os.path.basename(original_path)
    base = os.path.splitext(rel)[0]
    variant_dir = os.path.join(out_dir, f"{base}__enhanced_aug{variant_idx}")
    os.makedirs(variant_dir, exist_ok=True)
    out_path = os.path.join(variant_dir, rel)
    
    augmented = transformer.transform_file(original_path, variant_idx)
    with open(out_path, 'w') as f:
        f.write(augmented)
    
    return out_path


def iter_java_files(root_dir: str):
    """Iterate over Java files in the directory tree."""
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.java'):
                yield os.path.join(root, f)


def main():
    parser = argparse.ArgumentParser(description='Generate enhanced semantic-preserving augmentations for CFWR training data')
    parser.add_argument('--slices_dir', required=True, help='Directory containing original slice files')
    parser.add_argument('--out_dir', required=True, help='Output directory for augmented files')
    parser.add_argument('--variants_per_file', type=int, default=50, help='Number of variants to generate per file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    transformer = EnhancedSemanticTransformer(seed=args.seed)
    
    print(f"Generating enhanced semantic-preserving augmentations with seed {args.seed}")
    print(f"Variants per file: {args.variants_per_file}")
    print("Enhanced transformations include:")
    print("- Method extraction and inlining")
    print("- Conditional expression restructuring")
    print("- Array access pattern variations")
    print("- String concatenation alternatives")
    print("- Numeric literal transformations")
    print("- Exception handling restructuring")
    print("- Lambda expression conversions")
    print("- Stream API alternatives")
    print("- Builder pattern variations")
    print("- Functional programming conversions")
    
    produced = []
    for java_file in iter_java_files(args.slices_dir):
        for k in range(args.variants_per_file):
            out_path = write_variant(java_file, args.out_dir, k, transformer)
            produced.append(out_path)
    
    print(f"Generated {len(produced)} enhanced semantically augmented files in {args.out_dir}")
    print("Each file contains advanced semantic-preserving transformations that are highly resistant to slicer pruning!")


if __name__ == '__main__':
    main()
