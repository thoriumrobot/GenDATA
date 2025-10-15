#!/usr/bin/env python3
"""
Script to fix duplicate TRANSFORMATION lines in test files.
"""
import os
import glob
import re

def fix_duplicate_lines(file_path):
    """Fix duplicate TRANSFORMATION lines in a test file."""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove duplicate TRANSFORMATION lines
    # Pattern: find incomplete TRANSFORMATION line followed by complete one
    pattern = r'private static final String TRANSFORMATION = "\s*\n\s*private static final String TRANSFORMATION = "([^"]+)";'
    replacement = r'private static final String TRANSFORMATION = "\1";'
    
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Fix all test files with duplicate lines."""
    test_files = glob.glob('src/test/java/cfwr/jdt/transformations/**/*TransformationTest.java', recursive=True)
    
    print(f"Found {len(test_files)} test files to fix")
    
    for file_path in test_files:
        fix_duplicate_lines(file_path)
    
    print("All test files fixed!")

if __name__ == '__main__':
    main()
