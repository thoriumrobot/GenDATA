#!/usr/bin/env python3
"""
Test script for semantic augmentation system
"""

import os
import tempfile
import shutil
from semantic_augment_slices import SemanticTransformer

def create_test_java_file():
    """Create a test Java file with various constructs to transform."""
    return """
public class TestClass {
    public int calculateSum(int[] array) {
        int sum = 0;
        
        // For loop that can be converted to while
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        
        // If-else that can have guard reversed
        if (sum > 0) {
            System.out.println("Positive sum");
        } else {
            System.out.println("Non-positive sum");
        }
        
        // Mathematical expressions
        int result = sum * 2 + 0;
        result = result * 1;
        
        // Ternary operator
        String message = result > 10 ? "Large result" : "Small result";
        
        // Switch statement
        switch (result % 3) {
            case 0:
                return result;
            case 1:
                return result + 1;
            default:
                return result + 2;
        }
    }
    
    public boolean checkConditions(boolean a, boolean b) {
        // Logical expressions for De Morgan's laws
        if (!(a && b)) {
            return true;
        }
        return false;
    }
}
"""

def test_semantic_transformations():
    """Test the semantic transformation system."""
    print("Testing Semantic Augmentation System")
    print("=" * 50)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        test_file = os.path.join(temp_dir, "TestClass.java")
        with open(test_file, 'w') as f:
            f.write(create_test_java_file())
        
        print(f"Original file created: {test_file}")
        print("\nOriginal content:")
        print("-" * 30)
        with open(test_file, 'r') as f:
            print(f.read())
        
        # Create transformer
        transformer = SemanticTransformer(seed=42)
        
        # Generate 3 variants
        print("\nGenerating 3 semantic variants...")
        print("=" * 50)
        
        for i in range(3):
            print(f"\nVariant {i + 1}:")
            print("-" * 30)
            
            transformed = transformer.transform_file(test_file, i)
            print(transformed)
            
            # Save variant
            variant_file = os.path.join(temp_dir, f"TestClass_variant_{i + 1}.java")
            with open(variant_file, 'w') as f:
                f.write(transformed)
            
            print(f"Saved to: {variant_file}")
        
        print("\n" + "=" * 50)
        print("Semantic augmentation test completed!")
        print("Each variant applies different semantic-preserving transformations:")
        print("- Loop conversions (for ↔ while)")
        print("- Guard reversals (if-else condition flipping)")
        print("- Mathematical properties (commutativity, identity operations)")
        print("- De Morgan's laws")
        print("- Ternary ↔ if-else conversions")
        print("- Switch ↔ if-else chain conversions")
        print("- Variable inlining/extraction")

if __name__ == "__main__":
    test_semantic_transformations()


