#!/usr/bin/env python3
"""
Script to fix the new test files that still have SemanticTransformer instantiation.
"""
import os
import glob

def fix_new_test_file(file_path):
    """Fix SemanticTransformer instantiation in new test files."""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the line that creates a new SemanticTransformer instance
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if 'transformer = new SemanticTransformer(42L);' in line:
            # Skip this line
            continue
        elif 'transformer = new SemanticTransformer(42L); // Seeded for determinism' in line:
            # Skip this line
            continue
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.write('\n'.join(fixed_lines))

def main():
    """Fix all new test files."""
    # List of new test files created
    new_test_files = [
        'src/test/java/cfwr/jdt/transformations/enhanced/BitwiseOperationTransformationTest.java',
        'src/test/java/cfwr/jdt/transformations/enhanced/ComparisonOperationTransformationTest.java',
        'src/test/java/cfwr/jdt/transformations/enhanced/TypeConversionTransformationTest.java',
        'src/test/java/cfwr/jdt/transformations/enhanced/NullCheckPatternTransformationTest.java',
        'src/test/java/cfwr/jdt/transformations/enhanced/ConstantFoldingTransformationTest.java',
        'src/test/java/cfwr/jdt/transformations/random/DeadCodeInsertionTransformationTest.java',
        'src/test/java/cfwr/jdt/transformations/enhanced/MethodChainTransformationTest.java',
        'src/test/java/cfwr/jdt/transformations/enhanced/VariableRenamingTransformationTest.java'
    ]
    
    print(f"Found {len(new_test_files)} new test files to fix")
    
    for file_path in new_test_files:
        if os.path.exists(file_path):
            fix_new_test_file(file_path)
        else:
            print(f"File not found: {file_path}")
    
    print("All new test files fixed!")

if __name__ == '__main__':
    main()
