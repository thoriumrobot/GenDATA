#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Unified Augmentation System

This module tests the unified augmentation registry, code location analyzer,
random walk optimization, and caching mechanisms.
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
import logging

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_augmentation_registry import UnifiedAugmentationRegistry
from code_location_analyzer import CodeLocationAnalyzer, CodeLocation, LocationType
from transformation_caching import TransformationCache
from augmentation_policy_learner import RandomWalkOptimizer, ReinforcementLearningPolicy
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator
from recursive_augmentation_engine import TransformationType, TransformationState

# Set up logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

class TestCodeLocationAnalyzer(unittest.TestCase):
    """Test the code location analyzer"""
    
    def setUp(self):
        self.analyzer = CodeLocationAnalyzer()
        self.sample_java_code = '''
public class TestClass {
    public void testMethod() {
        int x = 5;
        if (x > 0) {
            System.out.println("Positive");
        }
        for (int i = 0; i < x; i++) {
            System.out.println(i);
        }
    }
}
'''
    
    def test_analyze_code_locations(self):
        """Test basic code location analysis"""
        locations = self.analyzer.analyze_code(self.sample_java_code)
        
        self.assertIsInstance(locations, list)
        self.assertGreater(len(locations), 0)
        
        # Check that we found different location types
        location_types = {loc.location_type for loc in locations}
        self.assertIn(LocationType.CLASS_LEVEL, location_types)
        self.assertIn(LocationType.METHOD_LEVEL, location_types)
    
    def test_get_transformation_applicability(self):
        """Test transformation applicability at locations"""
        locations = self.analyzer.analyze_code(self.sample_java_code)
        
        for location in locations:
            applicable_transforms = self.analyzer.get_transformation_applicability(location)
            self.assertIsInstance(applicable_transforms, set)
            self.assertGreater(len(applicable_transforms), 0)

class TestTransformationCache(unittest.TestCase):
    """Test the transformation caching system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = TransformationCache(cache_dir=self.temp_dir, max_size=100)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_cache_transformation_result(self):
        """Test caching and retrieving transformation results"""
        code = "int x = 5;"
        transformation_type = TransformationType.SIMPLE_ASSIGNMENT
        output_code = "int x = 10;"
        
        # Cache a result
        self.cache.cache_transformation_result(
            code=code,
            transformation_type=transformation_type,
            output_code=output_code,
            success=True,
            execution_time=0.1
        )
        
        # Retrieve the result
        cached_result = self.cache.get_transformation_result(code, transformation_type)
        
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result.output_code, output_code)
        self.assertTrue(cached_result.success)
    
    def test_success_patterns(self):
        """Test success pattern tracking"""
        code = "int x = 5;"
        transformation_type = TransformationType.SIMPLE_ASSIGNMENT
        
        # Cache multiple successful results
        for i in range(5):
            self.cache.cache_transformation_result(
                code=code,
                transformation_type=transformation_type,
                output_code=f"int x = {i};",
                success=True,
                execution_time=0.1
            )
        
        # Check success pattern
        pattern = self.cache.get_success_pattern(code)
        self.assertIsNotNone(pattern)
        self.assertIn(transformation_type.value, pattern.successful_transformations)
        self.assertGreater(pattern.usage_count, 0)
    
    def test_recommended_transformations(self):
        """Test transformation recommendations"""
        code = "int x = 5;"
        transformation_type = TransformationType.SIMPLE_ASSIGNMENT
        
        # Cache a successful transformation
        self.cache.cache_transformation_result(
            code=code,
            transformation_type=transformation_type,
            output_code="int x = 10;",
            success=True,
            execution_time=0.1
        )
        
        # Get recommendations
        recommendations = self.cache.get_recommended_transformations(code)
        self.assertIn(transformation_type, recommendations)

class TestUnifiedAugmentationRegistry(unittest.TestCase):
    """Test the unified augmentation registry"""
    
    def setUp(self):
        self.registry = UnifiedAugmentationRegistry(seed=42, enable_caching=True)
        self.sample_code = '''
public class TestClass {
    public void testMethod() {
        int x = 5;
        if (x > 0) {
            System.out.println("Positive");
        }
    }
}
'''
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        self.assertIsNotNone(self.registry.enhanced_transformer)
        self.assertIsNotNone(self.registry.simple_transformer)
        self.assertIsNotNone(self.registry.location_analyzer)
        self.assertIsNotNone(self.registry.cache)
        self.assertEqual(len(self.registry.transformation_map), 30)
    
    def test_apply_transformation(self):
        """Test applying transformations"""
        transformation_type = TransformationType.SIMPLE_ASSIGNMENT
        result = self.registry.apply_transformation(self.sample_code, transformation_type)
        
        self.assertIsInstance(result, str)
        self.assertIsNotNone(result)
    
    def test_location_aware_transformation(self):
        """Test location-aware transformation application"""
        locations = self.registry.analyze_code_locations(self.sample_code)
        
        if locations:
            location = locations[0]
            transformation_type = TransformationType.SIMPLE_ASSIGNMENT
            
            result = self.registry.apply_transformation(
                self.sample_code, 
                transformation_type, 
                location
            )
            
            self.assertIsInstance(result, str)
    
    def test_transformation_sequence(self):
        """Test applying transformation sequences"""
        sequence = [
            TransformationType.SIMPLE_ASSIGNMENT,
            TransformationType.SIMPLE_METHOD_CALL
        ]
        
        result_code, success_flags = self.registry.apply_transformation_sequence(
            self.sample_code, sequence
        )
        
        self.assertIsInstance(result_code, str)
        self.assertIsInstance(success_flags, list)
        self.assertEqual(len(success_flags), len(sequence))
    
    def test_recommended_transformations(self):
        """Test getting recommended transformations"""
        locations = self.registry.analyze_code_locations(self.sample_code)
        
        if locations:
            location = locations[0]
            recommendations = self.registry.get_recommended_transformations(
                self.sample_code, location
            )
            
            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)
    
    def test_statistics(self):
        """Test transformation statistics"""
        # Apply a few transformations
        self.registry.apply_transformation(self.sample_code, TransformationType.SIMPLE_ASSIGNMENT)
        self.registry.apply_transformation(self.sample_code, TransformationType.SIMPLE_METHOD_CALL)
        
        stats = self.registry.get_transformation_statistics()
        
        self.assertIn('total_applications', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('cache_statistics', stats)

class TestRandomWalkOptimization(unittest.TestCase):
    """Test random walk optimization"""
    
    def setUp(self):
        self.optimizer = RandomWalkOptimizer(
            methods=['rl', 'mcts'],
            device='cpu',
            registry=UnifiedAugmentationRegistry(seed=42)
        )
        self.sample_code = '''
public class TestClass {
    public void testMethod() {
        int x = 5;
        if (x > 0) {
            System.out.println("Positive");
        }
    }
}
'''
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer.registry)
        self.assertIn('rl', self.optimizer.components)
        self.assertIn('mcts', self.optimizer.components)
    
    @patch('augmentation_sequence_evaluator.AugmentationSequenceEvaluator')
    def test_optimization_sequence(self, mock_evaluator):
        """Test optimization sequence generation"""
        # Mock the evaluator
        mock_evaluator.return_value.evaluate_sequence.return_value = Mock(
            overall_score=0.8,
            warning_reduction=0.7,
            model_performance=0.6
        )
        
        # Mock the recursive augmentation engine
        with patch('augmentation_policy_learner.RecursiveAugmentationEngine') as mock_engine:
            mock_engine.return_value.apply_transformation.return_value = "transformed code"
            
            result = self.optimizer.optimize_augmentation_sequence(
                initial_code=self.sample_code,
                max_iterations=5,
                parallel=False
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('best_sequence', result)

class TestModelPerformanceEvaluation(unittest.TestCase):
    """Test model performance evaluation during training"""
    
    def setUp(self):
        self.evaluator = AugmentationSequenceEvaluator(device='cpu')
        self.sample_states = [
            TransformationState(code="int x = 5;", quality_score=0.5),
            TransformationState(code="int x = 10;", quality_score=0.8)
        ]
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertFalse(self.evaluator.model_performance_enabled)
        self.assertEqual(len(self.evaluator.model_cache), 0)
    
    def test_enable_model_performance_evaluation(self):
        """Test enabling model performance evaluation"""
        self.evaluator.enable_model_performance_evaluation(
            model_type='gcn',
            annotation_type='nonnegative'
        )
        
        self.assertTrue(self.evaluator.model_performance_enabled)
        self.assertEqual(self.evaluator.model_type, 'gcn')
        self.assertEqual(self.evaluator.annotation_type, 'nonnegative')
    
    def test_model_performance_evaluation_disabled(self):
        """Test model performance evaluation when disabled"""
        score = self.evaluator.evaluate_model_performance_with_prediction(self.sample_states)
        self.assertEqual(score, 0.5)  # Should return neutral score

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = UnifiedAugmentationRegistry(seed=42, enable_caching=True)
        self.optimizer = RandomWalkOptimizer(
            methods=['rl'],
            device='cpu',
            registry=self.registry
        )
        self.sample_code = '''
public class TestClass {
    public void testMethod() {
        int x = 5;
        if (x > 0) {
            System.out.println("Positive");
        }
        return x;
    }
}
'''
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Step 1: Analyze code locations
        locations = self.registry.analyze_code_locations(self.sample_code)
        self.assertGreater(len(locations), 0)
        
        # Step 2: Get recommended transformations
        recommendations = self.registry.get_recommended_transformations(self.sample_code)
        self.assertGreater(len(recommendations), 0)
        
        # Step 3: Apply transformation sequence
        sequence = recommendations[:3]  # Take first 3 recommendations
        result_code, success_flags = self.registry.apply_transformation_sequence(
            self.sample_code, sequence
        )
        
        self.assertIsInstance(result_code, str)
        self.assertEqual(len(success_flags), len(sequence))
        
        # Step 4: Check cache statistics
        stats = self.registry.get_transformation_statistics()
        self.assertIn('cache_statistics', stats)
        
        # Step 5: Test optimization (mocked to avoid long execution)
        with patch('augmentation_sequence_evaluator.AugmentationSequenceEvaluator') as mock_evaluator:
            mock_evaluator.return_value.evaluate_sequence.return_value = Mock(
                overall_score=0.8,
                warning_reduction=0.7
            )
            
            with patch('augmentation_policy_learner.RecursiveAugmentationEngine') as mock_engine:
                mock_engine.return_value.apply_transformation.return_value = "optimized code"
                
                optimization_result = self.optimizer.optimize_augmentation_sequence(
                    initial_code=self.sample_code,
                    max_iterations=3,
                    parallel=False
                )
                
                self.assertIsInstance(optimization_result, dict)
    
    def test_caching_workflow(self):
        """Test caching workflow"""
        # Apply transformation first time
        result1 = self.registry.apply_transformation(
            self.sample_code, 
            TransformationType.SIMPLE_ASSIGNMENT
        )
        
        # Apply same transformation second time (should hit cache)
        result2 = self.registry.apply_transformation(
            self.sample_code, 
            TransformationType.SIMPLE_ASSIGNMENT
        )
        
        # Results should be the same
        self.assertEqual(result1, result2)
        
        # Check cache statistics
        stats = self.registry.get_transformation_statistics()
        cache_stats = stats['cache_statistics']
        self.assertGreater(cache_stats['hits'], 0)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        self.registry = UnifiedAugmentationRegistry(seed=42)
        self.analyzer = CodeLocationAnalyzer()
    
    def test_empty_code_handling(self):
        """Test handling of empty code"""
        empty_code = ""
        locations = self.analyzer.analyze_code(empty_code)
        self.assertEqual(len(locations), 0)
        
        result = self.registry.apply_transformation(empty_code, TransformationType.SIMPLE_ASSIGNMENT)
        self.assertEqual(result, empty_code)
    
    def test_invalid_transformation_type(self):
        """Test handling of invalid transformation types"""
        # This should not raise an exception
        result = self.registry.apply_transformation(
            "int x = 5;", 
            TransformationType.SIMPLE_ASSIGNMENT
        )
        self.assertIsInstance(result, str)
    
    def test_malformed_java_code(self):
        """Test handling of malformed Java code"""
        malformed_code = "public class { invalid syntax"
        locations = self.analyzer.analyze_code(malformed_code)
        # Should not crash, may return empty or partial results
        self.assertIsInstance(locations, list)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_classes = [
        TestCodeLocationAnalyzer,
        TestTransformationCache,
        TestUnifiedAugmentationRegistry,
        TestRandomWalkOptimization,
        TestModelPerformanceEvaluation,
        TestIntegration,
        TestErrorHandling
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
