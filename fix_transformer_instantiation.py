#!/usr/bin/env python3
"""
Script to fix SemanticTransformer instantiation in test files.
"""
import os
import glob
import re

def fix_transformer_instantiation(file_path):
    """Fix SemanticTransformer instantiation in a test file."""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the line that creates a new SemanticTransformer instance
    content = re.sub(r'\s*transformer = new SemanticTransformer\(42L\); // Seeded for determinism\s*\n', '\n', content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Fix all test files."""
    test_files = glob.glob('src/test/java/cfwr/jdt/transformations/**/*TransformationTest.java', recursive=True)
    
    print(f"Found {len(test_files)} test files to fix")
    
    for file_path in test_files:
        fix_transformer_instantiation(file_path)
    
    print("All test files fixed!")

if __name__ == '__main__':
    main()
