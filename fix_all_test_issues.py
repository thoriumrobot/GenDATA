#!/usr/bin/env python3
"""
Script to fix all issues in test files.
"""
import os
import glob
import re

def fix_test_file(file_path):
    """Fix all issues in a test file."""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix repeated 'public' modifiers
    content = re.sub(r'public public void', 'public void', content)
    
    # Fix empty string literal issue
    content = re.sub(r'assertTrue\(transformed\.contains\(""";\)', 
                     r'assertTrue(transformed.contains("\\"\\";")', content)
    
    # Fix any other string literal issues with triple quotes
    content = re.sub(r'contains\("""\)', 'contains("\\"\\"")', content)
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Fix all test files."""
    test_files = glob.glob('src/test/java/cfwr/jdt/transformations/**/*TransformationTest.java', recursive=True)
    
    print(f"Found {len(test_files)} test files to fix")
    
    for file_path in test_files:
        fix_test_file(file_path)
    
    print("All test files fixed!")

if __name__ == '__main__':
    main()
