#!/usr/bin/env python3
"""
Script to fix test quality issues in all transformation test files.
"""
import os
import re
import glob

def fix_test_file(file_path):
    """Fix test quality issues in a single test file."""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add BeforeEach import if not present
    if 'import org.junit.jupiter.api.BeforeEach;' not in content:
        content = content.replace(
            'import org.junit.jupiter.api.Test;',
            'import org.junit.jupiter.api.Test;\nimport org.junit.jupiter.api.BeforeEach;'
        )
    
    # Add setUp method if not present
    if '@Override\n    @BeforeEach' not in content:
        # Find the class declaration and add setUp method after it
        class_match = re.search(r'class (\w+) extends TransformationTestBase \{', content)
        if class_match:
            class_name = class_match.group(1)
            setup_method = f'''    
    @Override
    @BeforeEach
    public void setUp() {{
        super.setUp();
        transformer = new SemanticTransformer(42L); // Seeded for determinism
    }}
    
    private static final String TRANSFORMATION = "'''
            
            # Find where to insert the setUp method
            insert_pos = class_match.end()
            content = content[:insert_pos] + setup_method + content[insert_pos:]
    
    # Make all test methods public
    content = re.sub(r'(\s+)void test(\w+)', r'\1public void test\2', content)
    
    # Fix assertion messages to be more descriptive
    content = re.sub(
        r'assertTransformationApplied\(([^,]+), ([^,]+), "([^"]*)"\)',
        r'assertTransformationApplied(\1, \2, "\3 should transform code")',
        content
    )
    
    content = re.sub(
        r'assertCompiles\(([^,]+), "([^"]*)"\)',
        r'assertCompiles(\1, "\2 should produce compilable code")',
        content
    )
    
    content = re.sub(
        r'assertSemanticallyEquivalent\(([^,]+), ([^,]+), "([^"]*)"\)',
        r'assertSemanticallyEquivalent(\1, \2, "\3 should preserve semantics")',
        content
    )
    
    # Fix assertTrue messages
    content = re.sub(
        r'assertTrue\(([^,]+), "([^"]*)"\)',
        r'assertTrue(\1, "Should have \2")',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Fix all transformation test files."""
    test_files = glob.glob('src/test/java/cfwr/jdt/transformations/**/*TransformationTest.java', recursive=True)
    
    print(f"Found {len(test_files)} test files to fix")
    
    for file_path in test_files:
        fix_test_file(file_path)
    
    print("All test files fixed!")

if __name__ == '__main__':
    main()
