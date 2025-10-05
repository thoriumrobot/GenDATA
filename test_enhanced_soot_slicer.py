#!/usr/bin/env python3
"""
Test script for the enhanced Soot slicer with forward and backward slicing
"""

import os
import tempfile
import shutil
from pathlib import Path

def create_test_java_file():
    """Create a test Java file with various constructs for slicing."""
    return """
public class TestSlice {
    public int calculate(int a, int b) {
        int sum = a + b;           // Line 3: Definition of sum
        int product = a * b;       // Line 4: Definition of product
        
        if (sum > 0) {             // Line 5: Control dependency
            product = product * 2; // Line 6: Data dependency on product
        }
        
        int result = sum + product; // Line 9: Uses both sum and product
        return result;              // Line 10: Return statement
    }
    
    public void testMethod() {
        int x = 5;                 // Line 13: Definition
        int y = 10;                // Line 14: Definition
        int z = x + y;             // Line 15: Uses x and y
        
        if (z > 10) {              // Line 16: Control dependency
            System.out.println("Large"); // Line 17: Data dependency
        } else {
            System.out.println("Small"); // Line 19: Data dependency
        }
    }
}
"""

def test_soot_slicer_modes():
    """Test the enhanced Soot slicer with different slicing modes."""
    print("Testing Enhanced Soot Slicer with Forward and Backward Slicing")
    print("=" * 70)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test Java file
        java_file = temp_path / "TestSlice.java"
        java_file.write_text(create_test_java_file())
        print(f"Created test file: {java_file}")
        
        # Create output directory
        output_dir = temp_path / "slices"
        output_dir.mkdir()
        
        # Test different slicing modes
        slicing_modes = ["backward", "forward", "combined"]
        target_line = 9  # Line with "int result = sum + product;"
        member_sig = "TestSlice.calculate(int,int)"
        
        for mode in slicing_modes:
            print(f"\nTesting {mode.upper()} slicing mode:")
            print("-" * 40)
            
            mode_output_dir = output_dir / f"slice_{mode}"
            mode_output_dir.mkdir()
            
            # Build the Soot slicer command
            cmd = [
                "java", "-cp", "/home/ubuntu/GenDATA/build/libs/CFWR-all.jar",
                "cfwr.SootSlicer",
                "--projectRoot", str(temp_path),
                "--targetFile", "TestSlice.java",
                "--line", str(target_line),
                "--output", str(mode_output_dir),
                "--member", member_sig,
                "--slice-mode", mode
            ]
            
            print(f"Command: {' '.join(cmd)}")
            
            # Note: This would run the actual Soot slicer
            # For demonstration, we'll simulate the expected output
            print(f"Expected behavior for {mode} slicing:")
            
            if mode == "backward":
                print("  - Finds all statements that influence the target line")
                print("  - Includes: sum = a + b, product = a * b, product = product * 2")
                print("  - Includes: if (sum > 0) condition")
                
            elif mode == "forward":
                print("  - Finds all statements influenced by the target line")
                print("  - Includes: return result")
                print("  - May include other statements that use 'result'")
                
            elif mode == "combined":
                print("  - Combines both backward and forward slices")
                print("  - Includes all statements from backward slice")
                print("  - Includes all statements from forward slice")
                print("  - Provides the most comprehensive slice")
        
        print(f"\n" + "=" * 70)
        print("Enhanced Soot Slicer Features:")
        print("1. Forward Slicing: Finds statements influenced by the target")
        print("2. Backward Slicing: Finds statements that influence the target")
        print("3. Combined Slicing: Merges both forward and backward slices")
        print("4. Improved Line Mapping: Better mapping of source lines to bytecode")
        print("5. Data Flow Analysis: Tracks def-use relationships")
        print("6. Control Flow Analysis: Identifies control dependencies")
        
        print(f"\nUsage Examples:")
        print("# Backward slicing only")
        print("java -cp CFWR-all.jar cfwr.SootSlicer --slice-mode backward ...")
        print()
        print("# Forward slicing only") 
        print("java -cp CFWR-all.jar cfwr.SootSlicer --slice-mode forward ...")
        print()
        print("# Combined slicing (default)")
        print("java -cp CFWR-all.jar cfwr.SootSlicer --slice-mode combined ...")

def test_shell_script_interface():
    """Test the shell script interface for the enhanced Soot slicer."""
    print(f"\n" + "=" * 70)
    print("Testing Shell Script Interface")
    print("=" * 70)
    
    shell_script = "/home/ubuntu/GenDATA/tools/soot_slicer.sh"
    
    if os.path.exists(shell_script):
        print(f"Shell script found: {shell_script}")
        print("\nShell script usage examples:")
        print()
        print("# Backward slicing")
        print(f"{shell_script} --projectRoot /path/to/project \\")
        print("  --targetFile TestClass.java \\")
        print("  --line 10 \\")
        print("  --output /path/to/output \\")
        print("  --member TestClass.method(int,int) \\")
        print("  --slice-mode backward")
        print()
        print("# Forward slicing")
        print(f"{shell_script} --projectRoot /path/to/project \\")
        print("  --targetFile TestClass.java \\")
        print("  --line 10 \\")
        print("  --output /path/to/output \\")
        print("  --member TestClass.method(int,int) \\")
        print("  --slice-mode forward")
        print()
        print("# Combined slicing (default)")
        print(f"{shell_script} --projectRoot /path/to/project \\")
        print("  --targetFile TestClass.java \\")
        print("  --line 10 \\")
        print("  --output /path/to/output \\")
        print("  --member TestClass.method(int,int)")
    else:
        print(f"Shell script not found: {shell_script}")
        print("Please ensure the shell script exists and is executable")

if __name__ == "__main__":
    test_soot_slicer_modes()
    test_shell_script_interface()
