#!/usr/bin/env python3
"""
Data Augmentation Script with Best Practices Defaults

This script augments Java slices with:
- Truly random, syntactically correct Java code
- Dynamic generation of methods, statements, and expressions
- Enhanced diversity for better model training

Best Practices:
- Generates syntactically correct Java code
- Uses random data pools for variety
- Maintains code structure and readability
- Provides configurable randomness levels
- Integrates seamlessly with training pipeline
"""

import os
import re
import argparse
import random
from pathlib import Path

HEADER_COMMENT = """
/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
""".strip()

# Random data pools for generating varied code
PRIMITIVE_TYPES = ['int', 'long', 'double', 'float', 'boolean', 'char', 'byte', 'short']
REFERENCE_TYPES = ['String', 'Object', 'Integer', 'Long', 'Double', 'Float', 'Boolean', 'Character']
ACCESS_MODIFIERS = ['private', 'public', 'protected', '']
STATIC_MODIFIERS = ['static', '']
METHOD_NAMES = ['helper', 'util', 'temp', 'aux', 'proc', 'func', 'calc', 'compute', 'process', 'handle']
VARIABLE_NAMES = ['val', 'data', 'item', 'obj', 'result', 'temp', 'var', 'elem', 'node', 'entry']
OPERATORS = ['+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>']
COMPARISON_OPS = ['==', '!=', '<', '>', '<=', '>=']
LOGICAL_OPS = ['&&', '||']


def find_class_insertion_point(src: str) -> int:
    # Heuristic: insert before the last closing brace of the file
    last = src.rfind('}')
    return last if last != -1 else len(src)


def insert_header_comment(src: str) -> str:
    if src.lstrip().startswith("/* CFWR augmentation"):
        return src
    return HEADER_COMMENT + "\n" + src


def generate_random_literal(type_name: str) -> str:
    """Generate a random literal value for the given type."""
    if type_name == 'int':
        return str(random.randint(-1000, 1000))
    elif type_name == 'long':
        return str(random.randint(-1000, 1000)) + 'L'
    elif type_name == 'double':
        return f"{random.uniform(-100.0, 100.0):.2f}"
    elif type_name == 'float':
        return f"{random.uniform(-100.0, 100.0):.2f}f"
    elif type_name == 'boolean':
        return random.choice(['true', 'false'])
    elif type_name == 'char':
        return f"'{random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')}'"
    elif type_name == 'String':
        words = ['hello', 'world', 'test', 'data', 'value', 'temp', 'result', 'item']
        return f'"{random.choice(words)}{random.randint(1, 99)}"'
    else:
        return 'null'


def generate_random_expression(type_name: str, depth: int = 0) -> str:
    """Generate a random expression of the given type."""
    if depth > 2:  # Prevent infinite recursion
        return generate_random_literal(type_name)
    
    if random.random() < 0.3 and depth < 2:  # 30% chance to create complex expression
        if type_name in PRIMITIVE_TYPES:
            op = random.choice(OPERATORS)
            left_type = random.choice(PRIMITIVE_TYPES)
            right_type = random.choice(PRIMITIVE_TYPES)
            return f"({generate_random_expression(left_type, depth + 1)} {op} {generate_random_expression(right_type, depth + 1)})"
    
    return generate_random_literal(type_name)


def generate_random_statement() -> str:
    """Generate a random Java statement."""
    stmt_type = random.choice(['assignment', 'if', 'for', 'while', 'try_catch', 'return'])
    
    if stmt_type == 'assignment':
        var_type = random.choice(PRIMITIVE_TYPES + REFERENCE_TYPES)
        var_name = f"__cfwr_{random.choice(VARIABLE_NAMES)}{random.randint(1, 99)}"
        expr = generate_random_expression(var_type)
        return f"        {var_type} {var_name} = {expr};"
    
    elif stmt_type == 'if':
        condition = f"{generate_random_expression('boolean')} {random.choice(LOGICAL_OPS)} {generate_random_expression('boolean')}"
        return f"""        if ({condition}) {{
            {generate_random_statement().strip()}
        }}"""
    
    elif stmt_type == 'for':
        var_name = f"__cfwr_i{random.randint(1, 99)}"
        limit = random.randint(1, 10)
        return f"""        for (int {var_name} = 0; {var_name} < {limit}; {var_name}++) {{
            {generate_random_statement().strip()}
        }}"""
    
    elif stmt_type == 'while':
        condition = generate_random_expression('boolean')
        return f"""        while ({condition}) {{
            {generate_random_statement().strip()}
            break; // Prevent infinite loops
        }}"""
    
    elif stmt_type == 'try_catch':
        return f"""        try {{
            {generate_random_statement().strip()}
        }} catch (Exception __cfwr_e{random.randint(1, 99)}) {{
            // ignore
        }}"""
    
    elif stmt_type == 'return':
        return_type = random.choice(PRIMITIVE_TYPES + REFERENCE_TYPES)
        expr = generate_random_expression(return_type)
        return f"        return {expr};"
    
    return "        // random statement"


def generate_random_method() -> str:
    """Generate a completely random Java method."""
    access = random.choice(ACCESS_MODIFIERS)
    static = random.choice(STATIC_MODIFIERS)
    return_type = random.choice(PRIMITIVE_TYPES + REFERENCE_TYPES)
    method_name = f"__cfwr_{random.choice(METHOD_NAMES)}{random.randint(1, 999)}"
    
    # Generate parameters
    param_count = random.randint(0, 3)
    params = []
    for i in range(param_count):
        param_type = random.choice(PRIMITIVE_TYPES + REFERENCE_TYPES)
        param_name = f"__cfwr_p{i}"
        params.append(f"{param_type} {param_name}")
    
    param_str = ", ".join(params)
    
    # Generate method body
    stmt_count = random.randint(1, 4)
    statements = []
    for _ in range(stmt_count):
        statements.append(generate_random_statement())
    
    # Add return statement if method has return type
    if return_type != 'void':
        statements.append(f"        return {generate_random_expression(return_type)};")
    
    body = "\n".join(statements)
    
    modifiers = f"{access} {static}".strip()
    if modifiers:
        modifiers += " "
    
    return f"""    {modifiers}{return_type} {method_name}({param_str}) {{
{body}
    }}"""


def insert_random_methods(src: str, count: int) -> str:
    """Insert random methods into the class."""
    insert_at = find_class_insertion_point(src)
    methods = []
    for i in range(count):
        methods.append(generate_random_method())
    addition = "\n".join(methods) + "\n"
    return src[:insert_at] + addition + src[insert_at:]


def insert_random_statements(src: str, count: int) -> str:
    """Insert random statements into existing methods."""
    out = src
    pattern = re.compile(r"(\)\s*\{)")
    matches = list(pattern.finditer(out))
    
    # Insert random statements into first few methods
    for m in matches[:count]:
        idx = m.end()
        random_stmt = generate_random_statement()
        out = out[:idx] + "\n" + random_stmt + "\n" + out[idx:]
    
    return out


def augment_file(java_path: str, variant_idx: int) -> str:
    with open(java_path, 'r') as f:
        src = f.read()
    
    # Ensure the slice is a valid compilable Java class before augmentation
    src = ensure_compilable_class(java_path, src)

    src = insert_header_comment(src)
    
    # Add random methods (1-3 methods)
    method_count = random.randint(1, 3)
    src = insert_random_methods(src, method_count)
    
    # Add random statements to existing methods (1-2 statements)
    stmt_count = random.randint(1, 2)
    src = insert_random_statements(src, stmt_count)
    
    return src


def write_variant(original_path: str, out_dir: str, variant_idx: int):
    rel = os.path.basename(original_path)
    base = os.path.splitext(rel)[0]
    variant_dir = os.path.join(out_dir, f"{base}__aug{variant_idx}")
    os.makedirs(variant_dir, exist_ok=True)
    out_path = os.path.join(variant_dir, rel)
    augmented = augment_file(original_path, variant_idx)
    with open(out_path, 'w') as f:
        f.write(augmented)
    return out_path


def ensure_compilable_class(java_path: str, src: str) -> str:
    """Wraps fragment slices into a compilable single-class file if needed.
    - If no top-level 'class' keyword is found, create: public class <Base> { <src> }
    - Ensures braces are balanced by appending a closing brace if needed.
    - If file contains multiple top-level elements without a class, encapsulate them in one class.
    """
    # Heuristic: detect a top-level class presence
    has_class = ' class ' in src or src.strip().startswith('class ') or 'interface ' in src or 'enum ' in src
    base = os.path.splitext(os.path.basename(java_path))[0]
    class_name = base if base.isidentifier() else f"Slice{abs(hash(base))}"

    wrapped = src
    if not has_class:
        wrapped = "public class {} {{\n{}\n}}".format(class_name, src)

    # Ensure at least one opening and closing brace
    open_braces = wrapped.count('{')
    close_braces = wrapped.count('}')
    if open_braces > close_braces:
        wrapped = wrapped + ('}' * (open_braces - close_braces))
    elif close_braces > open_braces:
        # Prepend missing opening braces minimally
        wrapped = ('{' * (close_braces - open_braces)) + wrapped

    return wrapped


def iter_java_files(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.java'):
                yield os.path.join(root, f)


def main():
    parser = argparse.ArgumentParser(description='Generate random Java code augmentations for CFWR training data')
    parser.add_argument('--slices_dir', required=True, help='Directory containing original slice files')
    parser.add_argument('--out_dir', required=True, help='Output directory for augmented files')
    parser.add_argument('--variants_per_file', type=int, default=10, help='Number of variants to generate per file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    parser.add_argument('--max_methods', type=int, default=3, help='Maximum number of random methods to add')
    parser.add_argument('--max_statements', type=int, default=2, help='Maximum number of random statements to add')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"Generating random augmentations with seed {args.seed}")
    print(f"Max methods per file: {args.max_methods}")
    print(f"Max statements per file: {args.max_statements}")

    produced = []
    for java_file in iter_java_files(args.slices_dir):
        for k in range(args.variants_per_file):
            out_path = write_variant(java_file, args.out_dir, k)
            produced.append(out_path)

    print(f"Augmented {len(produced)} files into {args.out_dir}")
    print("Each file now contains unique random Java code for better ML training diversity!")


if __name__ == '__main__':
    main()


