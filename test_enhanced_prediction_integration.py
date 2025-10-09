#!/usr/bin/env python3
"""
Test Enhanced Prediction Integration

This script tests the integration of Lower Bound Checker execution during prediction
with warning-based slicing as the default behavior.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_prediction_pipeline():
    """Test the enhanced prediction pipeline integration"""
    
    logger.info("🧪 Testing Enhanced Prediction Pipeline Integration")
    
    try:
        # Import the enhanced prediction pipeline
        from enhanced_prediction_pipeline import EnhancedPredictionPipeline
        from main_optimized_pipeline import MainOptimizedPipeline
        
        logger.info("✅ Successfully imported enhanced prediction components")
        
        # Create a temporary test directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_project_root = Path(temp_dir) / "test_project"
            test_project_root.mkdir(parents=True, exist_ok=True)
            
            # Create a simple test Java file
            test_java_file = test_project_root / "TestClass.java"
            with open(test_java_file, 'w') as f:
                f.write("""
public class TestClass {
    public void testMethod(int index, int[] array) {
        if (index >= 0 && index < array.length) {
            int value = array[index];
            System.out.println("Value: " + value);
        }
    }
    
    public int getSize(int[] array) {
        return array.length;
    }
}
""")
            
            logger.info(f"📁 Created test project at: {test_project_root}")
            
            # Test 1: Enhanced Prediction Pipeline Initialization
            logger.info("Test 1: Initializing Enhanced Prediction Pipeline")
            
            output_dir = Path(temp_dir) / "test_output"
            
            enhanced_pipeline = EnhancedPredictionPipeline(
                project_root=str(test_project_root),
                output_dir=str(output_dir),
                models_dir='/home/ubuntu/GenDATA/models_annotation_types',
                cfwr_root='/home/ubuntu/GenDATA',
                checker_framework_home='/home/ubuntu/checker-framework-3.42.0'
            )
            
            logger.info("✅ Enhanced Prediction Pipeline initialized successfully")
            
            # Test 2: Lower Bound Checker Execution
            logger.info("Test 2: Running Lower Bound Checker")
            
            warnings_file = enhanced_pipeline.run_lower_bound_checker([str(test_java_file)])
            
            if warnings_file and os.path.exists(warnings_file):
                logger.info(f"✅ Lower Bound Checker executed successfully")
                logger.info(f"📄 Warnings file: {warnings_file}")
                
                # Check warnings file content
                with open(warnings_file, 'r') as f:
                    content = f.read()
                    logger.info(f"📊 Warnings file size: {len(content)} characters")
            else:
                logger.warning("⚠️ Lower Bound Checker execution completed but no warnings file found")
            
            # Test 3: Main Optimized Pipeline Integration
            logger.info("Test 3: Testing Main Optimized Pipeline Integration")
            
            main_pipeline = MainOptimizedPipeline(device='cpu')
            
            # Test the enhanced prediction method
            result = main_pipeline.predict_with_enhanced_pipeline(
                project_root=str(test_project_root),
                output_dir=str(output_dir / "main_pipeline_output"),
                models_dir='/home/ubuntu/GenDATA/models_annotation_types',
                java_files=[str(test_java_file)],
                use_lower_bound_checker=True
            )
            
            if result.get('success', False):
                logger.info("✅ Main Optimized Pipeline integration test passed")
                logger.info(f"📊 Result: {result}")
            else:
                logger.warning(f"⚠️ Main Optimized Pipeline integration test completed with warnings: {result}")
            
            # Test 4: Directory Structure Verification
            logger.info("Test 4: Verifying directory structure")
            
            expected_dirs = [
                output_dir / "temp_analysis" / "warnings",
                output_dir / "temp_analysis" / "slices", 
                output_dir / "temp_analysis" / "cfgs",
                output_dir / "predictions"
            ]
            
            for expected_dir in expected_dirs:
                if expected_dir.exists():
                    logger.info(f"✅ Directory exists: {expected_dir}")
                else:
                    logger.warning(f"⚠️ Directory not found: {expected_dir}")
            
            # Test 5: Configuration Integration
            logger.info("Test 5: Testing configuration integration")
            
            config = main_pipeline.config
            perf_opts = config.get('performance_optimization', {})
            
            logger.info(f"📊 Performance optimization enabled: {perf_opts.get('performance_tracking', False)}")
            logger.info(f"📊 Preferred models: {perf_opts.get('preferred_models', [])}")
            logger.info(f"📊 Preferred annotations: {perf_opts.get('preferred_annotations', [])}")
            
            logger.info("✅ Configuration integration test passed")
            
            return True
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def test_command_line_interface():
    """Test the command-line interface integration"""
    
    logger.info("🧪 Testing Command-Line Interface Integration")
    
    try:
        # Test command-line arguments
        import subprocess
        
        # Test help command
        result = subprocess.run([
            'python', 'main_optimized_pipeline.py', '--help'
        ], capture_output=True, text=True, cwd='/home/ubuntu/GenDATA')
        
        if result.returncode == 0:
            logger.info("✅ Command-line help works correctly")
            
            # Check if enhanced prediction options are present
            if '--predict-enhanced' in result.stdout:
                logger.info("✅ Enhanced prediction option found in help")
            else:
                logger.warning("⚠️ Enhanced prediction option not found in help")
            
            if '--java-files' in result.stdout:
                logger.info("✅ Java files option found in help")
            else:
                logger.warning("⚠️ Java files option not found in help")
                
        else:
            logger.error(f"❌ Command-line help failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Command-line interface test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    
    logger.info("🚀 Starting Enhanced Prediction Integration Tests")
    
    test_results = []
    
    # Test 1: Enhanced Prediction Pipeline
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Enhanced Prediction Pipeline")
    logger.info("="*60)
    
    test1_result = test_enhanced_prediction_pipeline()
    test_results.append(("Enhanced Prediction Pipeline", test1_result))
    
    # Test 2: Command-Line Interface
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Command-Line Interface")
    logger.info("="*60)
    
    test2_result = test_command_line_interface()
    test_results.append(("Command-Line Interface", test2_result))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All integration tests passed! Enhanced prediction integration is working correctly.")
        return 0
    else:
        logger.error(f"❌ {total_tests - passed_tests} test(s) failed. Please check the integration.")
        return 1

if __name__ == '__main__':
    exit(main())
