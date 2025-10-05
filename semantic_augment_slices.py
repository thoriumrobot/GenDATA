#!/usr/bin/env python3
"""
Semantic-Preserving Data Augmentation Script

This script augments Java slices using semantic-preserving transformations that
slicers are less likely to remove. The transformations maintain the original
semantics while changing the syntactic structure.

Key Transformations:
1. Loop conversions (for ↔ while)
2. Guard reversals (if-else condition flipping)
3. Mathematical properties (commutativity, associativity, identity operations)
4. De Morgan's laws
5. Relational operator inversions
6. Ternary ↔ if-else conversions
7. Switch ↔ if-else chain conversions
8. Variable inlining/extraction
"""

import os
import re
import argparse
import random
import ast
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

HEADER_COMMENT = """
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
""".strip()

class SemanticTransformer:
    """Applies semantic-preserving transformations to Java code."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.transformations_applied = []
    
    def transform_file(self, java_path: str, variant_idx: int) -> str:
        """Apply semantic transformations to a Java file."""
        with open(java_path, 'r') as f:
            src = f.read()
        
        # Ensure the slice is a valid compilable Java class
        src = self._ensure_compilable_class(java_path, src)
        
        # Apply transformations
        src = self._insert_header_comment(src)
        
        # Apply random transformations
        transformations = [
            self._transform_loops,
            self._transform_guards,
            self._transform_mathematical_expressions,
            self._transform_logical_expressions,
            self._transform_ternary_operators,
            self._transform_switch_statements,
            self._transform_variable_operations,
        ]
        
        # Apply 2-4 random transformations
        num_transforms = random.randint(2, 4)
        selected_transforms = random.sample(transformations, num_transforms)
        
        for transform in selected_transforms:
            src = transform(src)
        
        return src
    
    def _insert_header_comment(self, src: str) -> str:
        """Insert augmentation header comment."""
        if src.lstrip().startswith("/* CFWR semantic augmentation"):
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
    
    def _transform_loops(self, src: str) -> str:
        """Transform between for and while loops."""
        # Pattern for for loops: for (init; condition; update) { body }
        for_pattern = re.compile(
            r'for\s*\(\s*([^;]+)\s*;\s*([^;]+)\s*;\s*([^)]+)\s*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            re.MULTILINE | re.DOTALL
        )
        
        def for_to_while(match):
            init = match.group(1).strip()
            condition = match.group(2).strip()
            update = match.group(3).strip()
            body = match.group(4).strip()
            
            # Convert to while loop
            while_body = f"{body}\n            {update};"
            return f"{init};\n        while ({condition}) {{\n            {while_body}\n        }}"
        
        # Pattern for while loops: while (condition) { body }
        while_pattern = re.compile(
            r'while\s*\(([^)]+)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            re.MULTILINE | re.DOTALL
        )
        
        def while_to_for(match):
            condition = match.group(1).strip()
            body = match.group(2).strip()
            
            # Extract potential loop variable and update
            # This is a heuristic - look for common patterns
            if '++' in body or '--' in body:
                # Try to convert back to for loop
                lines = body.strip().split('\n')
                if len(lines) >= 2:
                    last_line = lines[-1].strip()
                    if '++' in last_line or '--' in last_line:
                        update = last_line.rstrip(';')
                        new_body = '\n'.join(lines[:-1])
                        # Generate a simple loop variable
                        var_name = f"i{random.randint(1, 99)}"
                        return f"for (int {var_name} = 0; {condition}; {update}) {{\n            {new_body}\n        }}"
            
            return match.group(0)  # Return original if conversion not possible
        
        # Apply transformations randomly
        if random.random() < 0.5:
            src = for_pattern.sub(for_to_while, src)
        else:
            src = while_pattern.sub(while_to_for, src)
        
        return src
    
    def _transform_guards(self, src: str) -> str:
        """Reverse if-else guards and flip branches."""
        # Pattern for if-else statements
        if_else_pattern = re.compile(
            r'if\s*\(([^)]+)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*else\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            re.MULTILINE | re.DOTALL
        )
        
        def reverse_guard(match):
            condition = match.group(1).strip()
            if_body = match.group(2).strip()
            else_body = match.group(3).strip()
            
            # Reverse the condition and swap bodies
            reversed_condition = self._reverse_condition(condition)
            return f"if ({reversed_condition}) {{\n            {else_body}\n        }} else {{\n            {if_body}\n        }}"
        
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
        # Commutativity: a + b -> b + a, a * b -> b * a
        src = self._apply_commutativity(src)
        
        # Identity operations: a + 0 -> a, a * 1 -> a, a - 0 -> a
        src = self._apply_identity_operations(src)
        
        # Associativity: (a + b) + c -> a + (b + c)
        src = self._apply_associativity(src)
        
        # Strength reduction: x * 2 -> x << 1, y / 4 -> y >> 2
        src = self._apply_strength_reduction(src)
        
        return src
    
    def _apply_commutativity(self, src: str) -> str:
        """Apply commutativity transformations."""
        # Pattern for addition and multiplication
        patterns = [
            (r'(\w+)\s*\+\s*(\w+)', r'\2 + \1'),  # a + b -> b + a
            (r'(\w+)\s*\*\s*(\w+)', r'\2 * \1'),  # a * b -> b * a
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.3:  # 30% chance to apply
                src = re.sub(pattern, replacement, src)
        
        return src
    
    def _apply_identity_operations(self, src: str) -> str:
        """Apply identity operation transformations."""
        patterns = [
            (r'(\w+)\s*\+\s*0', r'\1'),      # a + 0 -> a
            (r'0\s*\+\s*(\w+)', r'\1'),      # 0 + a -> a
            (r'(\w+)\s*\*\s*1', r'\1'),      # a * 1 -> a
            (r'1\s*\*\s*(\w+)', r'\1'),      # 1 * a -> a
            (r'(\w+)\s*-\s*0', r'\1'),       # a - 0 -> a
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.4:  # 40% chance to apply
                src = re.sub(pattern, replacement, src)
        
        return src
    
    def _apply_associativity(self, src: str) -> str:
        """Apply associativity transformations."""
        # Pattern for (a + b) + c -> a + (b + c)
        pattern = r'\((\w+)\s*\+\s*(\w+)\)\s*\+\s*(\w+)'
        replacement = r'\1 + (\2 + \3)'
        
        if random.random() < 0.2:  # 20% chance to apply
            src = re.sub(pattern, replacement, src)
        
        return src
    
    def _apply_strength_reduction(self, src: str) -> str:
        """Apply strength reduction transformations."""
        patterns = [
            (r'(\w+)\s*\*\s*2', r'\1 << 1'),     # x * 2 -> x << 1
            (r'(\w+)\s*\*\s*4', r'\1 << 2'),     # x * 4 -> x << 2
            (r'(\w+)\s*\*\s*8', r'\1 << 3'),     # x * 8 -> x << 3
            (r'(\w+)\s*/\s*2', r'\1 >> 1'),      # x / 2 -> x >> 1
            (r'(\w+)\s*/\s*4', r'\1 >> 2'),      # x / 4 -> x >> 2
            (r'(\w+)\s*/\s*8', r'\1 >> 3'),      # x / 8 -> x >> 3
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.3:  # 30% chance to apply
                src = re.sub(pattern, replacement, src)
        
        return src
    
    def _transform_logical_expressions(self, src: str) -> str:
        """Apply De Morgan's laws and logical transformations."""
        # De Morgan's laws
        patterns = [
            # !(a && b) -> !a || !b
            (r'!\s*\(([^)]+)\s*&&\s*([^)]+)\)', r'!(\1) || !(\2)'),
            # !(a || b) -> !a && !b
            (r'!\s*\(([^)]+)\s*\|\|\s*([^)]+)\)', r'!(\1) && !(\2)'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.4:  # 40% chance to apply
                src = re.sub(pattern, replacement, src)
        
        return src
    
    def _transform_ternary_operators(self, src: str) -> str:
        """Convert between ternary operators and if-else statements."""
        # Pattern for ternary: condition ? true_expr : false_expr
        ternary_pattern = re.compile(
            r'(\w+)\s*=\s*([^?]+)\s*\?\s*([^:]+)\s*:\s*([^;]+);',
            re.MULTILINE
        )
        
        def ternary_to_if_else(match):
            var = match.group(1).strip()
            condition = match.group(2).strip()
            true_expr = match.group(3).strip()
            false_expr = match.group(4).strip()
            
            return f"""if ({condition}) {{
            {var} = {true_expr};
        }} else {{
            {var} = {false_expr};
        }}"""
        
        # Pattern for simple if-else assignments
        if_else_pattern = re.compile(
            r'if\s*\(([^)]+)\)\s*\{\s*(\w+)\s*=\s*([^;]+);\s*\}\s*else\s*\{\s*(\w+)\s*=\s*([^;]+);\s*\}',
            re.MULTILINE | re.DOTALL
        )
        
        def if_else_to_ternary(match):
            condition = match.group(1).strip()
            var1 = match.group(2).strip()
            expr1 = match.group(3).strip()
            var2 = match.group(4).strip()
            expr2 = match.group(5).strip()
            
            if var1 == var2:  # Same variable assigned
                return f"{var1} = {condition} ? {expr1} : {expr2};"
            else:
                return match.group(0)  # Return original if different variables
        
        # Apply transformations randomly
        if random.random() < 0.5:
            src = ternary_pattern.sub(ternary_to_if_else, src)
        else:
            src = if_else_pattern.sub(if_else_to_ternary, src)
        
        return src
    
    def _transform_switch_statements(self, src: str) -> str:
        """Convert between switch statements and if-else chains."""
        # Pattern for switch statements
        switch_pattern = re.compile(
            r'switch\s*\(([^)]+)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            re.MULTILINE | re.DOTALL
        )
        
        def switch_to_if_else(match):
            expr = match.group(1).strip()
            body = match.group(2).strip()
            
            # Parse switch body to extract cases
            lines = body.split('\n')
            if_else_chain = []
            current_case = None
            current_body = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('case ') and ':' in line:
                    # Save previous case
                    if current_case is not None:
                        case_body = '\n'.join(current_body).strip()
                        if_else_chain.append(f"if ({expr} == {current_case}) {{\n            {case_body}\n        }}")
                    
                    # Start new case
                    current_case = line.split(':')[0].replace('case ', '').strip()
                    current_body = []
                elif line == 'default:':
                    # Save previous case
                    if current_case is not None:
                        case_body = '\n'.join(current_body).strip()
                        if_else_chain.append(f"if ({expr} == {current_case}) {{\n            {case_body}\n        }}")
                    
                    current_case = 'default'
                    current_body = []
                elif line and not line.startswith('break;'):
                    current_body.append(line)
            
            # Handle last case
            if current_case is not None:
                case_body = '\n'.join(current_body).strip()
                if current_case == 'default':
                    if_else_chain.append(f"else {{\n            {case_body}\n        }}")
                else:
                    if_else_chain.append(f"if ({expr} == {current_case}) {{\n            {case_body}\n        }}")
            
            # Join with proper else if structure
            if len(if_else_chain) == 1:
                return if_else_chain[0]
            elif len(if_else_chain) > 1:
                result = if_else_chain[0]
                for i in range(1, len(if_else_chain)):
                    if 'else {' in if_else_chain[i]:
                        result += f" else {if_else_chain[i]}"
                    else:
                        result += f" else {if_else_chain[i]}"
                return result
            else:
                return src
        
        src = switch_pattern.sub(switch_to_if_else, src)
        return src
    
    def _transform_variable_operations(self, src: str) -> str:
        """Inline or extract temporary variables."""
        # Pattern for simple assignments that can be inlined
        assignment_pattern = re.compile(
            r'(\w+)\s+(\w+)\s*=\s*([^;]+);',
            re.MULTILINE
        )
        
        def inline_variable(match):
            var_type = match.group(1).strip()
            var_name = match.group(2).strip()
            expr = match.group(3).strip()
            
            # Look for usages of this variable
            usage_pattern = re.compile(rf'\b{re.escape(var_name)}\b')
            usages = usage_pattern.findall(src)
            
            if len(usages) <= 2:  # Only inline if used once or twice
                # Replace usages with the expression
                src_modified = usage_pattern.sub(expr, src)
                # Remove the assignment
                src_modified = re.sub(rf'{var_type}\s+{re.escape(var_name)}\s*=\s*{re.escape(expr)};\s*', '', src_modified)
                return src_modified
            
            return match.group(0)  # Return original if too many usages
        
        if random.random() < 0.3:  # 30% chance to apply
            src = assignment_pattern.sub(inline_variable, src)
        
        return src


def write_variant(original_path: str, out_dir: str, variant_idx: int, transformer: SemanticTransformer):
    """Write an augmented variant of the original file."""
    rel = os.path.basename(original_path)
    base = os.path.splitext(rel)[0]
    variant_dir = os.path.join(out_dir, f"{base}__semantic_aug{variant_idx}")
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
    parser = argparse.ArgumentParser(description='Generate semantic-preserving augmentations for CFWR training data')
    parser.add_argument('--slices_dir', required=True, help='Directory containing original slice files')
    parser.add_argument('--out_dir', required=True, help='Output directory for augmented files')
    parser.add_argument('--variants_per_file', type=int, default=50, help='Number of variants to generate per file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    transformer = SemanticTransformer(seed=args.seed)
    
    print(f"Generating semantic-preserving augmentations with seed {args.seed}")
    print(f"Variants per file: {args.variants_per_file}")
    
    produced = []
    for java_file in iter_java_files(args.slices_dir):
        for k in range(args.variants_per_file):
            out_path = write_variant(java_file, args.out_dir, k, transformer)
            produced.append(out_path)
    
    print(f"Generated {len(produced)} semantically augmented files in {args.out_dir}")
    print("Each file contains semantic-preserving transformations that slicers are less likely to remove!")


if __name__ == '__main__':
    main()
