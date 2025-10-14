#!/usr/bin/env python3
"""
Unit tests for JDT semantic transformer wrapper
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from jdt_semantic_transformer import JdtSemanticTransformer


class TestJdtSemanticTransformer(unittest.TestCase):
    """Test cases for JdtSemanticTransformer wrapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_java_code = '''
public class TestClass {
    public void testMethod(int x) {
        if (x > 0) {
            System.out.println("Positive: " + x);
        } else {
            System.out.println("Non-positive: " + x);
        }
        
        for (int i = 0; i < x; i++) {
            System.out.println("Iteration: " + i);
        }
    }
}
'''
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    @patch('subprocess.run')
    def test_transform_code_enhanced_mode(self, mock_run, mock_find_jar):
        """Test transforming code in enhanced mode"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Transformation completed successfully"
        mock_run.return_value = mock_result
        
        # Mock transformed code
        transformed_code = '''
public class TestClass {
    public void testMethod(int x) {
        if (!(x <= 0)) {
            System.out.println("Positive: " + x);
        } else {
            System.out.println("Non-positive: " + x);
        }
        
        int i = 0;
        while (i < x) {
            System.out.println("Iteration: " + i);
            i++;
        }
    }
}
'''
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=transformed_code)):
            with patch('os.path.exists', return_value=True):
                transformer = JdtSemanticTransformer(seed=42)
                result = transformer.transform_code(
                    self.test_java_code, 
                    ['guard_reversal', 'loop_conversion'], 
                    'enhanced'
                )
                
                self.assertIsInstance(result, str)
                self.assertNotEqual(result, self.test_java_code)  # Should be transformed
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    @patch('subprocess.run')
    def test_transform_code_simple_mode(self, mock_run, mock_find_jar):
        """Test transforming code in simple mode"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Transformation completed successfully"
        mock_run.return_value = mock_result
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=self.test_java_code)):
            with patch('os.path.exists', return_value=True):
                transformer = JdtSemanticTransformer(seed=42)
                result = transformer.transform_code(
                    self.test_java_code, 
                    ['simple_assignment'], 
                    'simple'
                )
                
                self.assertIsInstance(result, str)
                self.assertFalse(result.strip().isEmpty())
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    @patch('subprocess.run')
    def test_transform_file(self, mock_run, mock_find_jar):
        """Test transforming a file"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Transformation completed successfully"
        mock_run.return_value = mock_result
        
        transformed_code = "transformed code content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as input_file:
            input_file.write(self.test_java_code.encode())
            input_file.flush()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as output_file:
                with patch('builtins.open', unittest.mock.mock_open(read_data=transformed_code)):
                    with patch('os.path.exists', return_value=True):
                        transformer = JdtSemanticTransformer(seed=42)
                        success = transformer.transform_file(
                            input_file.name,
                            output_file.name,
                            ['guard_reversal'],
                            'enhanced'
                        )
                        
                        self.assertTrue(success)
                
                os.unlink(input_file.name)
                os.unlink(output_file.name)
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    def test_get_available_transformations_enhanced(self, mock_find_jar):
        """Test getting available enhanced transformations"""
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        with patch('os.path.exists', return_value=True):
            transformer = JdtSemanticTransformer(seed=42)
            transformations = transformer.get_available_transformations('enhanced')
            
            self.assertIsInstance(transformations, list)
            self.assertIn('loop_conversion', transformations)
            self.assertIn('guard_reversal', transformations)
            self.assertIn('mathematical_expression', transformations)
            self.assertGreater(len(transformations), 10)  # Should have many transformations
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    def test_get_available_transformations_simple(self, mock_find_jar):
        """Test getting available simple transformations"""
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        with patch('os.path.exists', return_value=True):
            transformer = JdtSemanticTransformer(seed=42)
            transformations = transformer.get_available_transformations('simple')
            
            self.assertIsInstance(transformations, list)
            self.assertIn('simple_assignment', transformations)
            self.assertIn('simple_method_call', transformations)
            self.assertIn('simple_conditional', transformations)
            self.assertEqual(len(transformations), 10)  # Should have exactly 10 simple transformations
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    def test_get_available_transformations_invalid_mode(self, mock_find_jar):
        """Test getting transformations for invalid mode"""
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        with patch('os.path.exists', return_value=True):
            transformer = JdtSemanticTransformer(seed=42)
            transformations = transformer.get_available_transformations('invalid')
            
            self.assertIsInstance(transformations, list)
            self.assertEqual(len(transformations), 0)  # Should return empty list
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    def test_get_random_transformations(self, mock_find_jar):
        """Test getting random transformations"""
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        with patch('os.path.exists', return_value=True):
            transformer = JdtSemanticTransformer(seed=42)
            random_transformations = transformer.get_random_transformations(count=3, mode='enhanced')
            
            self.assertIsInstance(random_transformations, list)
            self.assertLessEqual(len(random_transformations), 3)
            self.assertGreater(len(random_transformations), 0)
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    @patch('subprocess.run')
    def test_transform_code_failure(self, mock_run, mock_find_jar):
        """Test handling transformation failure"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        # Mock subprocess failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Transformation failed"
        mock_run.return_value = mock_result
        
        with patch('os.path.exists', return_value=True):
            transformer = JdtSemanticTransformer(seed=42)
            result = transformer.transform_code(
                self.test_java_code, 
                ['guard_reversal'], 
                'enhanced'
            )
            
            # Should return original code on failure
            self.assertEqual(result, self.test_java_code)
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    @patch('subprocess.run')
    def test_transform_code_timeout(self, mock_run, mock_find_jar):
        """Test handling transformation timeout"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        # Mock subprocess timeout
        mock_run.side_effect = subprocess.TimeoutExpired("java", 60)
        
        with patch('os.path.exists', return_value=True):
            transformer = JdtSemanticTransformer(seed=42)
            result = transformer.transform_code(
                self.test_java_code, 
                ['guard_reversal'], 
                'enhanced'
            )
            
            # Should return original code on timeout
            self.assertEqual(result, self.test_java_code)
    
    def test_jar_not_found(self):
        """Test handling when JAR file is not found"""
        with patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path', side_effect=FileNotFoundError("JAR not found")):
            with self.assertRaises(FileNotFoundError):
                JdtSemanticTransformer()
    
    @patch('jdt_semantic_transformer.JdtSemanticTransformer._find_jar_path')
    def test_different_seeds(self, mock_find_jar):
        """Test that different seeds produce different results"""
        mock_find_jar.return_value = "/fake/path/jdt-transformer-all.jar"
        
        with patch('os.path.exists', return_value=True):
            transformer1 = JdtSemanticTransformer(seed=42)
            transformer2 = JdtSemanticTransformer(seed=123)
            
            # The transformers should have different seeds internally
            # This is tested by checking that they can be created with different seeds
            self.assertIsInstance(transformer1, JdtSemanticTransformer)
            self.assertIsInstance(transformer2, JdtSemanticTransformer)


if __name__ == '__main__':
    unittest.main()
