#!/usr/bin/env python3
"""
Comprehensive test for ablation study pipeline
Tests all three ablation study types with small dataset
"""

import os
import sys
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_study_pipeline import AblationStudyPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_environment():
    """Create a minimal test environment with sample warnings file"""
    test_dir = tempfile.mkdtemp(prefix='ablation_test_')
    logger.info(f"Created test directory: {test_dir}")
    
    # Create sample warnings file
    warnings_file = os.path.join(test_dir, 'test_warnings.out')
    with open(warnings_file, 'w') as f:
        f.write("""TestFile.java:10:15: compiler.err.proc.messager: [index] Possible out-of-bounds access
TestFile.java:25:8: compiler.warn.proc.messager: [type.anno.before.modifier] write type annotation @IndexFor immediately before type
TestFile.java:30:20: compiler.err.proc.messager: [assignment] incompatible types in assignment
""")
    
    # Create minimal project structure
    project_dir = os.path.join(test_dir, 'test_project')
    os.makedirs(project_dir, exist_ok=True)
    
    # Create sample Java file
    java_file = os.path.join(project_dir, 'TestFile.java')
    with open(java_file, 'w') as f:
        f.write("""public class TestFile {
    private int[] array = new int[10];
    
    public int getValue(int index) {
        return array[index];  // Potential out-of-bounds access
    }
    
    public void setValue(int index, int value) {
        array[index] = value;
    }
}""")
    
    return test_dir, warnings_file, project_dir

def test_no_augmentation():
    """Test ablation study without any augmentation"""
    logger.info("=== Testing No Augmentation Ablation ===")
    
    test_dir, warnings_file, project_dir = create_test_environment()
    
    try:
        # Initialize ablation pipeline
        pipeline = AblationStudyPipeline(
            project_root=project_dir,
            warnings_file=warnings_file,
            cfwr_root=test_dir,
            output_dir=os.path.join(test_dir, 'test_output'),
            device='cpu'  # Use CPU for testing
        )
        
        # Run no augmentation study with minimal episodes
        results = pipeline.run_no_augmentation_study(episodes=2)
        
        # Verify results
        assert results, "No augmentation study should return results"
        assert 'metrics' in results, "Results should contain metrics"
        
        metrics = results['metrics']
        assert 'slices_generated' in metrics, "Should track slices generated"
        assert 'models_trained' in metrics, "Should track models trained"
        assert 'reduction_percentage' in metrics, "Should track warning reduction"
        
        logger.info("✓ No augmentation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ No augmentation test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

def test_single_transformation_ablation():
    """Test removing one transformation (e.g., loop_conversion)"""
    logger.info("=== Testing Single Transformation Ablation ===")
    
    test_dir, warnings_file, project_dir = create_test_environment()
    
    try:
        # Initialize ablation pipeline
        pipeline = AblationStudyPipeline(
            project_root=project_dir,
            warnings_file=warnings_file,
            cfwr_root=test_dir,
            output_dir=os.path.join(test_dir, 'test_output'),
            device='cpu'  # Use CPU for testing
        )
        
        # Run transformation ablation study with minimal episodes
        results = pipeline.run_transformation_ablation_study('loop_conversion', episodes=2)
        
        # Verify results
        assert results, "Transformation ablation study should return results"
        assert 'metrics' in results, "Results should contain metrics"
        
        metrics = results['metrics']
        assert 'slices_generated' in metrics, "Should track slices generated"
        assert 'models_trained' in metrics, "Should track models trained"
        assert 'reduction_percentage' in metrics, "Should track warning reduction"
        
        logger.info("✓ Single transformation ablation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Single transformation ablation test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

def test_no_random_walk():
    """Test without random walk optimization"""
    logger.info("=== Testing No Random Walk Ablation ===")
    
    test_dir, warnings_file, project_dir = create_test_environment()
    
    try:
        # Initialize ablation pipeline
        pipeline = AblationStudyPipeline(
            project_root=project_dir,
            warnings_file=warnings_file,
            cfwr_root=test_dir,
            output_dir=os.path.join(test_dir, 'test_output'),
            device='cpu'  # Use CPU for testing
        )
        
        # Run no random walk study with minimal episodes
        results = pipeline.run_no_random_walk_study(episodes=2)
        
        # Verify results
        assert results, "No random walk study should return results"
        assert 'metrics' in results, "Results should contain metrics"
        
        metrics = results['metrics']
        assert 'slices_generated' in metrics, "Should track slices generated"
        assert 'models_trained' in metrics, "Should track models trained"
        assert 'reduction_percentage' in metrics, "Should track warning reduction"
        
        logger.info("✓ No random walk test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ No random walk test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

def test_full_pipeline():
    """Run all three studies in sequence and verify isolation"""
    logger.info("=== Testing Full Pipeline ===")
    
    test_dir, warnings_file, project_dir = create_test_environment()
    
    try:
        # Initialize ablation pipeline
        pipeline = AblationStudyPipeline(
            project_root=project_dir,
            warnings_file=warnings_file,
            cfwr_root=test_dir,
            output_dir=os.path.join(test_dir, 'test_output'),
            device='cpu'  # Use CPU for testing
        )
        
        # Run all studies with minimal episodes
        results = pipeline.run_all_ablations(episodes=2)
        
        # Verify all studies completed
        expected_cases = ['baseline', 'no_augmentation', 'no_random_walk']
        for case in expected_cases:
            assert case in results, f"Results should contain {case}"
            assert 'metrics' in results[case], f"{case} should contain metrics"
        
        # Verify directory isolation
        output_dir = Path(pipeline.output_dir)
        for case in expected_cases:
            case_dir = output_dir / case
            assert case_dir.exists(), f"Directory for {case} should exist"
            assert (case_dir / 'slices').exists(), f"Slices directory for {case} should exist"
            assert (case_dir / 'cfg_output').exists(), f"CFG output directory for {case} should exist"
            assert (case_dir / 'models').exists(), f"Models directory for {case} should exist"
        
        # Verify comprehensive results file
        results_file = output_dir / 'ablation_results_summary.json'
        assert results_file.exists(), "Comprehensive results file should exist"
        
        with open(results_file, 'r') as f:
            comprehensive_results = json.load(f)
        
        assert 'ablation_study_summary' in comprehensive_results, "Should contain study summary"
        assert 'individual_results' in comprehensive_results, "Should contain individual results"
        assert 'performance_comparison' in comprehensive_results, "Should contain performance comparison"
        
        logger.info("✓ Full pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Full pipeline test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

def test_warning_reduction_measurement():
    """Test warning reduction measurement functionality"""
    logger.info("=== Testing Warning Reduction Measurement ===")
    
    test_dir, warnings_file, project_dir = create_test_environment()
    
    try:
        # Initialize ablation pipeline
        pipeline = AblationStudyPipeline(
            project_root=project_dir,
            warnings_file=warnings_file,
            cfwr_root=test_dir,
            output_dir=os.path.join(test_dir, 'test_output'),
            device='cpu'  # Use CPU for testing
        )
        
        # Test baseline warning counting
        baseline_warnings = pipeline._count_baseline_warnings()
        assert baseline_warnings > 0, "Should count baseline warnings"
        logger.info(f"Counted {baseline_warnings} baseline warnings")
        
        # Test warning reduction measurement (with empty model directory)
        test_case_dir = Path(pipeline.output_dir) / 'test_case'
        test_case_dir.mkdir(parents=True, exist_ok=True)
        (test_case_dir / 'models').mkdir(exist_ok=True)
        
        reduction_metrics = pipeline._measure_warning_reduction(test_case_dir)
        assert 'baseline_warnings' in reduction_metrics, "Should contain baseline warnings"
        assert 'remaining_warnings' in reduction_metrics, "Should contain remaining warnings"
        assert 'reduction_percentage' in reduction_metrics, "Should contain reduction percentage"
        
        logger.info(f"Warning reduction metrics: {reduction_metrics}")
        logger.info("✓ Warning reduction measurement test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Warning reduction measurement test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)

def main():
    """Run all comprehensive tests"""
    logger.info("Starting comprehensive ablation pipeline tests")
    
    test_results = {}
    
    # Run individual tests
    test_results['no_augmentation'] = test_no_augmentation()
    test_results['single_transformation'] = test_single_transformation_ablation()
    test_results['no_random_walk'] = test_no_random_walk()
    test_results['warning_reduction'] = test_warning_reduction_measurement()
    test_results['full_pipeline'] = test_full_pipeline()
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    logger.info(f"\n=== Test Results Summary ===")
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Ablation pipeline is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1

if __name__ == '__main__':
    exit(main())
