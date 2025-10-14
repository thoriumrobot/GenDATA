#!/usr/bin/env python3
"""
Integration tests for JDT pipeline components
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from code_location_analyzer import CodeLocationAnalyzer, CodeLocation, LocationType
from enhanced_semantic_augment_slices import EnhancedSemanticTransformer
from simple_code_semantic_augment_slices import SimpleCodeSemanticTransformer
from recursive_augmentation_engine import RecursiveAugmentationEngine, TransformationType


class TestJdtPipelineIntegration(unittest.TestCase):
    """Integration tests for JDT-based pipeline components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_java_code = '''
public class TestClass {
    private int field;
    
    public void testMethod(int x) {
        if (x > 0) {
            System.out.println("Positive: " + x);
            field = x;
        } else {
            System.out.println("Non-positive: " + x);
        }
        
        for (int i = 0; i < x; i++) {
            System.out.println("Iteration: " + i);
        }
    }
    
    public int getField() {
        return field;
    }
}
'''
    
    @patch('code_location_analyzer.JdtParserService')
    def test_code_location_analyzer_integration(self, mock_jdt_service):
        """Test CodeLocationAnalyzer integration with JDT"""
        # Mock JDT service response
        mock_locations = [
            MockCodeLocation(
                line_start=1, line_end=15, column_start=0, column_end=0,
                location_type="CLASS_LEVEL", context={"class_name": "TestClass"},
                code_snippet="", applicable_transformations=["RANDOM_METHOD_INSERTION"]
            ),
            MockCodeLocation(
                line_start=4, line_end=12, column_start=0, column_end=0,
                location_type="METHOD_LEVEL", context={"method_name": "testMethod"},
                code_snippet="", applicable_transformations=["GUARD_REVERSAL"]
            )
        ]
        
        mock_service_instance = MagicMock()
        mock_service_instance.parse_code_locations_from_string.return_value = mock_locations
        mock_jdt_service.return_value = mock_service_instance
        
        # Test the analyzer
        analyzer = CodeLocationAnalyzer()
        locations = analyzer.analyze_code(self.test_java_code)
        
        self.assertIsInstance(locations, list)
        self.assertEqual(len(locations), 2)
        self.assertEqual(locations[0].location_type, LocationType.CLASS_LEVEL)
        self.assertEqual(locations[1].location_type, LocationType.METHOD_LEVEL)
    
    @patch('enhanced_semantic_augment_slices.JdtSemanticTransformer')
    def test_enhanced_semantic_transformer_integration(self, mock_jdt_transformer):
        """Test EnhancedSemanticTransformer integration with JDT"""
        # Mock JDT transformer response
        transformed_code = self.test_java_code.replace("if (x > 0)", "if (!(x <= 0))")
        
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.transform_code.return_value = transformed_code
        mock_transformer_instance.get_available_transformations.return_value = [
            'loop_conversion', 'guard_reversal', 'mathematical_expression'
        ]
        mock_jdt_transformer.return_value = mock_transformer_instance
        
        # Test the transformer
        transformer = EnhancedSemanticTransformer(seed=42)
        result = transformer.transform_file("test.java", 0)
        
        self.assertIsInstance(result, str)
        self.assertIn("if (!(x <= 0))", result)  # Should contain transformed code
    
    @patch('simple_code_semantic_augment_slices.JdtSemanticTransformer')
    def test_simple_semantic_transformer_integration(self, mock_jdt_transformer):
        """Test SimpleCodeSemanticTransformer integration with JDT"""
        # Mock JDT transformer response
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.transform_code.return_value = self.test_java_code
        mock_transformer_instance.get_available_transformations.return_value = [
            'simple_assignment', 'simple_method_call', 'simple_conditional'
        ]
        mock_jdt_transformer.return_value = mock_transformer_instance
        
        # Test the transformer
        transformer = SimpleCodeSemanticTransformer(seed=42)
        result = transformer.transform_file("test.java", 0)
        
        self.assertIsInstance(result, str)
        self.assertFalse(result.strip().isEmpty())
    
    @patch('semantic_augment_slices.JdtSemanticTransformer')
    def test_semantic_transformer_integration(self, mock_jdt_transformer):
        """Test SemanticTransformer integration with JDT"""
        # Mock JDT transformer response
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.transform_code.return_value = self.test_java_code
        mock_transformer_instance.get_available_transformations.return_value = {
            'enhanced': ['loop_conversion', 'guard_reversal'],
            'simple': ['simple_assignment', 'simple_method_call']
        }
        mock_jdt_transformer.return_value = mock_transformer_instance
        
        # Test the transformer
        transformer = SemanticTransformer(seed=42)
        result = transformer.transform_file("test.java", 0, 'enhanced')
        
        self.assertIsInstance(result, str)
        self.assertFalse(result.strip().isEmpty())
    
    @patch('recursive_augmentation_engine.JdtParserService')
    def test_recursive_augmentation_engine_integration(self, mock_jdt_service):
        """Test RecursiveAugmentationEngine integration with JDT"""
        # Mock JDT service responses
        mock_service_instance = MagicMock()
        mock_service_instance.extract_identifiers.return_value = {
            'variables': ['x', 'i', 'field'],
            'methods': ['testMethod', 'println'],
            'types': ['TestClass', 'String']
        }
        mock_service_instance.parse_code_locations_from_string.return_value = []
        mock_jdt_service.return_value = mock_service_instance
        
        # Mock the transformer methods
        with patch.object(RecursiveAugmentationEngine, '_apply_jdt_transformation', return_value=self.test_java_code):
            # Test the engine
            engine = RecursiveAugmentationEngine(seed=42)
            
            # Test that JDT service is available
            self.assertIsNotNone(engine.jdt_service)
            
            # Test identifier extraction
            identifiers = engine._extract_identifiers_jdt(self.test_java_code)
            self.assertIsInstance(identifiers, set)
            self.assertIn('x', identifiers)
            self.assertIn('testMethod', identifiers)
    
    @patch('code_location_analyzer.JdtParserService')
    @patch('enhanced_semantic_augment_slices.JdtSemanticTransformer')
    def test_end_to_end_pipeline_integration(self, mock_jdt_transformer, mock_jdt_service):
        """Test end-to-end pipeline integration"""
        # Mock JDT service for code location analysis
        mock_locations = [
            MockCodeLocation(
                line_start=4, line_end=12, column_start=0, column_end=0,
                location_type="METHOD_LEVEL", context={"method_name": "testMethod"},
                code_snippet="", applicable_transformations=["GUARD_REVERSAL"]
            )
        ]
        
        mock_service_instance = MagicMock()
        mock_service_instance.parse_code_locations_from_string.return_value = mock_locations
        mock_jdt_service.return_value = mock_service_instance
        
        # Mock JDT transformer for semantic augmentation
        transformed_code = self.test_java_code.replace("if (x > 0)", "if (!(x <= 0))")
        
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.transform_code.return_value = transformed_code
        mock_transformer_instance.get_available_transformations.return_value = [
            'loop_conversion', 'guard_reversal'
        ]
        mock_jdt_transformer.return_value = mock_transformer_instance
        
        # Test the complete pipeline
        analyzer = CodeLocationAnalyzer()
        locations = analyzer.analyze_code(self.test_java_code)
        
        transformer = EnhancedSemanticTransformer(seed=42)
        result = transformer.transform_file("test.java", 0)
        
        # Verify pipeline components work together
        self.assertEqual(len(locations), 1)
        self.assertIn("if (!(x <= 0))", result)
    
    def test_transformation_type_compatibility(self):
        """Test that transformation types are compatible across components"""
        # Test that all transformation types are properly defined
        expected_enhanced_types = [
            'loop_conversion', 'guard_reversal', 'mathematical_expression',
            'logical_expression', 'ternary_operator', 'switch_statement',
            'variable_operation', 'method_extraction', 'conditional_expression',
            'array_access_pattern', 'string_concatenation', 'numeric_literal',
            'exception_handling', 'lambda_expression', 'stream_api',
            'builder_pattern', 'functional_conversion'
        ]
        
        expected_simple_types = [
            'simple_method_call', 'simple_assignment', 'simple_conditional',
            'simple_array_access', 'simple_return_statement', 'simple_variable_declaration',
            'simple_constructor_call', 'simple_field_access', 'simple_string_operation',
            'simple_numeric_operation'
        ]
        
        # Verify transformation types exist in enum
        for trans_type in expected_enhanced_types + expected_simple_types:
            # Convert to enum name format
            enum_name = trans_type.upper()
            self.assertTrue(hasattr(TransformationType, enum_name), 
                          f"TransformationType.{enum_name} should exist")


class MockCodeLocation:
    """Mock CodeLocation for testing"""
    def __init__(self, line_start, line_end, column_start, column_end, 
                 location_type, context, code_snippet, applicable_transformations):
        self.line_start = line_start
        self.line_end = line_end
        self.column_start = column_start
        self.column_end = column_end
        self.location_type = location_type
        self.context = context
        self.code_snippet = code_snippet
        self.applicable_transformations = applicable_transformations


if __name__ == '__main__':
    unittest.main()
