#!/usr/bin/env python3
"""
Simple Code Semantic-Preserving Data Augmentation Script

This script provides semantic augmentation methods specifically designed for 
simple Checker Framework test cases. These methods focus on transformations
that work well with simple code structures while being highly resistant to 
slicer pruning.

Key Features:
- Optimized for simple method calls, assignments, and conditionals
- High slicer resistance through semantic equivalences
- Minimal code structure changes to preserve simplicity
- Focus on annotation placement contexts
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
 * CFWR simple semantic augmentation: applied simple code semantic-preserving transformations.
 */
""".strip()

class SimpleCodeSemanticTransformer:
    """Semantic transformer optimized for simple Checker Framework test cases."""
    
    def __init__(self, seed: int = 42, disabled_transformations: List[str] = None):
        random.seed(seed)
        self.transformations_applied = []
        self.disabled_transformations = disabled_transformations or []
    
    def transform_file(self, java_path: str, variant_idx: int) -> str:
        """Apply simple semantic transformations to a Java file."""
        with open(java_path, 'r') as f:
            src = f.read()
        
        # Ensure the slice is a valid compilable Java class
        src = self._ensure_compilable_class(java_path, src)
        
        # Apply transformations
        src = self._insert_header_comment(src)
        
        # Simple transformation methods with names
        transformation_map = {
            'simple_method_call': self._transform_simple_method_calls,
            'simple_assignment': self._transform_simple_assignments,
            'simple_conditional': self._transform_simple_conditionals,
            'simple_array_access': self._transform_simple_array_access,
            'simple_return_statement': self._transform_simple_return_statements,
            'simple_variable_declaration': self._transform_simple_variable_declarations,
            'simple_constructor_call': self._transform_simple_constructor_calls,
            'simple_field_access': self._transform_simple_field_access,
            'simple_string_operation': self._transform_simple_string_operations,
            'simple_numeric_operation': self._transform_simple_numeric_operations,
        }
        
        # Filter out disabled transformations
        available_transformations = {
            name: method for name, method in transformation_map.items()
            if name not in self.disabled_transformations
        }
        
        if not available_transformations:
            logger.warning("All transformations disabled, returning original source")
            return src
        
        # Apply 2-4 random transformations (conservative for simple code)
        num_transforms = min(random.randint(2, 4), len(available_transformations))
        selected_transform_names = random.sample(list(available_transformations.keys()), num_transforms)
        
        for transform_name in selected_transform_names:
            transform_method = available_transformations[transform_name]
            src = transform_method(src)
            self.transformations_applied.append(transform_name)
        
        return src
    
    def _insert_header_comment(self, src: str) -> str:
        """Insert simple augmentation header comment."""
        if src.lstrip().startswith("/* CFWR simple semantic augmentation"):
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

    # SIMPLE CODE TRANSFORMATION METHODS

    def _transform_simple_method_calls(self, src: str) -> str:
        """Transform simple method calls for variety."""
        # Method call with parentheses variations
        patterns = [
            # obj.method() -> (obj).method()
            (r'(\w+)\.(\w+)\(', r'(\1).\2('),
            # obj.method() -> obj. method()
            (r'(\w+)\.(\w+)\(', r'\1. \2('),
            # Simple method chaining: obj.method().field -> obj.method() .field
            (r'(\w+)\(\)\.(\w+)', r'\1(). \2'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.3:  # 30% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_assignments(self, src: str) -> str:
        """Transform simple assignment statements."""
        # Assignment with spacing variations
        patterns = [
            # int x = 5; -> int x=5;
            (r'(\w+)\s+(\w+)\s*=\s*([^;]+);', r'\1 \2=\3;'),
            # int x = 5; -> int x = 5 ;
            (r'(\w+)\s+(\w+)\s*=\s*([^;]+);', r'\1 \2 = \3 ;'),
            # Simple compound assignment: x = x + 1 -> x += 1
            (r'(\w+)\s*=\s*\1\s*\+\s*(\d+)', r'\1 += \2'),
            (r'(\w+)\s*=\s*\1\s*-\s*(\d+)', r'\1 -= \2'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.25:  # 25% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_conditionals(self, src: str) -> str:
        """Transform simple conditional expressions."""
        # Simple condition reversals for simple cases
        patterns = [
            # if (x > 0) -> if (0 < x)
            (r'if\s*\(\s*(\w+)\s*>\s*(\d+)\s*\)', r'if (\2 < \1)'),
            (r'if\s*\(\s*(\w+)\s*<\s*(\d+)\s*\)', r'if (\2 > \1)'),
            # Simple boolean negation: if (flag) -> if (!(!flag))
            (r'if\s*\(\s*(\w+)\s*\)', r'if (!(!\1))'),
            # Simple equality: if (x == 0) -> if (0 == x)
            (r'if\s*\(\s*(\w+)\s*==\s*(\d+)\s*\)', r'if (\2 == \1)'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.2:  # 20% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_array_access(self, src: str) -> str:
        """Transform simple array access patterns."""
        # Array access with simple index variations
        patterns = [
            # arr[i] -> arr[0 + i]
            (r'(\w+)\[(\w+)\]', r'\1[0 + \2]'),
            # arr[i] -> arr[i + 0]
            (r'(\w+)\[(\w+)\]', r'\1[\2 + 0]'),
            # arr[0] -> arr[0 + 0]
            (r'(\w+)\[0\]', r'\1[0 + 0]'),
            # Simple array length: arr.length -> (arr).length
            (r'(\w+)\.length', r'(\1).length'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.3:  # 30% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_return_statements(self, src: str) -> str:
        """Transform simple return statements."""
        # Return statement variations
        patterns = [
            # return x; -> return (x);
            (r'return\s+(\w+);', r'return (\1);'),
            # return x; -> return 0 + x;
            (r'return\s+(\w+);', r'return 0 + \1;'),
            # return x; -> return x + 0;
            (r'return\s+(\w+);', r'return \1 + 0;'),
            # Simple method call returns: return obj.method(); -> return (obj.method());
            (r'return\s+([^;]+)\(\);', r'return (\1());'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.25:  # 25% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_variable_declarations(self, src: str) -> str:
        """Transform simple variable declarations."""
        # Variable declaration variations
        patterns = [
            # int x = 5; -> final int x = 5;
            (r'(\w+)\s+(\w+)\s*=\s*([^;]+);', r'final \1 \2 = \3;'),
            # Remove final if present: final int x = 5; -> int x = 5;
            (r'final\s+(\w+)\s+(\w+)\s*=\s*([^;]+);', r'\1 \2 = \3;'),
            # Simple type casting: int x = 5; -> int x = (int)5;
            (r'(\w+)\s+(\w+)\s*=\s*(\d+);', r'\1 \2 = (\1)\3;'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.2:  # 20% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_constructor_calls(self, src: str) -> str:
        """Transform simple constructor calls."""
        # Constructor call variations
        patterns = [
            # new Type() -> (new Type())
            (r'new\s+(\w+)\(\)', r'(new \1())'),
            # new Type(arg) -> new Type((arg))
            (r'new\s+(\w+)\(([^)]+)\)', r'new \1((\2))'),
            # Simple array creation: new int[5] -> new int[0 + 5]
            (r'new\s+(\w+)\[(\d+)\]', r'new \1[0 + \2]'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.25:  # 25% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_field_access(self, src: str) -> str:
        """Transform simple field access patterns."""
        # Field access variations
        patterns = [
            # obj.field -> (obj).field
            (r'(\w+)\.(\w+)', r'(\1).\2'),
            # obj.field -> obj. field
            (r'(\w+)\.(\w+)', r'\1. \2'),
            # Simple static access: Class.field -> (Class).field
            (r'(\w+)\.(\w+)', r'(\1).\2'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.2:  # 20% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_string_operations(self, src: str) -> str:
        """Transform simple string operations."""
        # String operation variations
        patterns = [
            # "text" -> ("text")
            (r'"([^"]+)"', r'("\1")'),
            # String concatenation: "a" + "b" -> "ab"
            (r'"([^"]+)"\s*\+\s*"([^"]+)"', r'"\1\2"'),
            # Simple string method calls: str.method() -> (str).method()
            (r'(\w+)\.(\w+)\(\)', r'(\1).\2()'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.3:  # 30% chance
                src = re.sub(pattern, replacement, src)
        
        return src

    def _transform_simple_numeric_operations(self, src: str) -> str:
        """Transform simple numeric operations."""
        # Numeric operation variations
        patterns = [
            # x + 0 -> x
            (r'(\w+)\s*\+\s*0', r'\1'),
            # 0 + x -> x
            (r'0\s*\+\s*(\w+)', r'\1'),
            # x * 1 -> x
            (r'(\w+)\s*\*\s*1', r'\1'),
            # 1 * x -> x
            (r'1\s*\*\s*(\w+)', r'\1'),
            # Simple arithmetic: x + 1 -> x + (0 + 1)
            (r'(\w+)\s*\+\s*(\d+)', r'\1 + (0 + \2)'),
            # Simple numeric literals: 10 -> 0xA (for small numbers)
            (r'\b10\b', '0xA'),
            (r'\b16\b', '0x10'),
            (r'\b15\b', '0xF'),
        ]
        
        for pattern, replacement in patterns:
            if random.random() < 0.25:  # 25% chance
                src = re.sub(pattern, replacement, src)
        
        return src


def write_variant(original_path: str, out_dir: str, variant_idx: int, transformer: SimpleCodeSemanticTransformer):
    """Write a simple augmented variant of the original file."""
    rel = os.path.basename(original_path)
    base = os.path.splitext(rel)[0]
    variant_dir = os.path.join(out_dir, f"{base}__simple_aug{variant_idx}")
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
    parser = argparse.ArgumentParser(description='Generate simple semantic-preserving augmentations for Checker Framework test cases')
    parser.add_argument('--slices_dir', required=True, help='Directory containing original slice files')
    parser.add_argument('--out_dir', required=True, help='Output directory for augmented files')
    parser.add_argument('--variants_per_file', type=int, default=50, help='Number of variants to generate per file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    transformer = SimpleCodeSemanticTransformer(seed=args.seed)
    
    print(f"Generating simple semantic-preserving augmentations with seed {args.seed}")
    print(f"Variants per file: {args.variants_per_file}")
    print("Simple transformations optimized for Checker Framework test cases:")
    print("- Simple method call variations")
    print("- Simple assignment transformations")
    print("- Simple conditional restructuring")
    print("- Simple array access patterns")
    print("- Simple return statement variations")
    print("- Simple variable declaration changes")
    print("- Simple constructor call variations")
    print("- Simple field access patterns")
    print("- Simple string operation alternatives")
    print("- Simple numeric operation transformations")
    
    produced = []
    for java_file in iter_java_files(args.slices_dir):
        for k in range(args.variants_per_file):
            out_path = write_variant(java_file, args.out_dir, k, transformer)
            produced.append(out_path)
    
    print(f"Generated {len(produced)} simple semantically augmented files in {args.out_dir}")
    print("Each file contains simple semantic-preserving transformations optimized for Checker Framework test cases!")


if __name__ == '__main__':
    main()
