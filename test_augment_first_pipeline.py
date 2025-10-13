#!/usr/bin/env python3
"""
Test script for the augment-first pipeline approach
"""

import os
import tempfile
import shutil
from pathlib import Path

def create_test_java_project():
    """Create a test Java project with multiple files for testing."""
    
    test_project = """
public class ArrayProcessor {
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
        
        return sum;
    }
    
    public int findMax(int[] array) {
        if (array.length == 0) {
            return -1;
        }
        
        int max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        
        return max;
    }
}
"""

    test_utils = """
public class MathUtils {
    public static int multiply(int a, int b) {
        // Mathematical expressions for transformation
        int result = a * b + 0;
        result = result * 1;
        return result;
    }
    
    public static boolean isEven(int n) {
        // Ternary operator
        return n % 2 == 0 ? true : false;
    }
}
"""

    return {
        "ArrayProcessor.java": test_project,
        "MathUtils.java": test_utils
    }

def create_test_warnings_file():
    """Create a test warnings file."""
    return """
ArrayProcessor.java:8:20: compiler.err.proc.messager: [array.length.negative] Variable used in array creation could be negative.
found   : int
required: an integer >= 0 (@NonNegative or @Positive)
ArrayProcessor.java:15:15: compiler.err.proc.messager: [assignment] incompatible types in assignment.
found   : int
required: @NonNegative int
MathUtils.java:5:25: compiler.err.proc.messager: [assignment] incompatible types in assignment.
found   : int
required: @Positive int
"""

def test_augment_first_approach():
    """Test the augment-first pipeline approach."""
    print("Testing Augment-First Pipeline Approach")
    print("=" * 60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test project structure
        project_root = temp_path / "test_project"
        project_root.mkdir()
        
        # Create test Java files
        java_files = create_test_java_project()
        for filename, content in java_files.items():
            file_path = project_root / filename
            file_path.write_text(content)
            print(f"Created: {file_path}")
        
        # Create test warnings file
        warnings_file = temp_path / "test_warnings.out"
        warnings_file.write_text(create_test_warnings_file())
        print(f"Created: {warnings_file}")
        
        print(f"\nProject structure:")
        print(f"  Project root: {project_root}")
        print(f"  Warnings file: {warnings_file}")
        
        # Test the semantic augmentation on the original code
        print(f"\nStep 1: Testing semantic augmentation on original code")
        print("-" * 50)
        
        from semantic_augment_slices import SemanticTransformer, iter_java_files
        
        # Create augmentation output directory
        augmented_dir = temp_path / "augmented_code"
        augmented_dir.mkdir()
        
        transformer = SemanticTransformer(seed=42)
        augmented_count = 0
        
        # Augment each Java file with 3 variants
        for java_file in iter_java_files(str(project_root)):
            rel_path = os.path.relpath(java_file, str(project_root))
            base_name = os.path.splitext(rel_path)[0]
            
            print(f"Augmenting: {rel_path}")
            
            for variant_idx in range(3):  # 3 variants for testing
                variant_dir = augmented_dir / f"{base_name}__variant_{variant_idx}"
                variant_dir.mkdir()
                output_path = variant_dir / os.path.basename(rel_path)
                
                # Apply semantic transformations
                augmented_content = transformer.transform_file(java_file, variant_idx)
                output_path.write_text(augmented_content)
                augmented_count += 1
                
                print(f"  Variant {variant_idx + 1}: {output_path}")
        
        print(f"\nGenerated {augmented_count} augmented variants")
        
        # Show example of augmented content
        print(f"\nExample augmented content (ArrayProcessor variant 1):")
        print("-" * 50)
        example_file = augmented_dir / "ArrayProcessor__variant_0" / "ArrayProcessor.java"
        if example_file.exists():
            print(example_file.read_text())
        
        print(f"\n" + "=" * 60)
        print("Augment-First Approach Benefits:")
        print("1. Original code is augmented first with semantic transformations")
        print("2. Each augmented variant maintains the same semantics")
        print("3. Slicers will work on semantically equivalent but syntactically different code")
        print("4. This may produce more diverse slicing patterns")
        print("5. Models will see how the same semantic intent can be expressed differently")
        
        print(f"\nNext steps would be:")
        print("1. Run slicing on each augmented variant")
        print("2. Generate CFGs from all slices")
        print("3. Train models on the diverse slice data")
        
        # Demonstrate the pipeline structure
        print(f"\nPipeline Structure:")
        print(f"  Original Code → Semantic Augmentation → Multiple Variants")
        print(f"  Each Variant → Slicing → Slices")
        print(f"  All Slices → CFG Generation → CFGs")
        print(f"  CFGs → Model Training → Trained Models")

if __name__ == "__main__":
    test_augment_first_approach()



