#!/usr/bin/env python3
"""
Unit tests for Signature String feature extraction

Tests the string feature extractor and source code extractor to ensure
they correctly extract features for Signature String Checker.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signature_string_feature_extractor import (
    SignatureStringFeatureExtractor,
    StringPatternAnalyzer,
    FormatDetector,
    StructuralAnalyzer,
    ContextAnalyzer
)
from source_code_feature_extractor import (
    SourceCodeFeatureExtractor,
    SourceCodeReader,
    FallbackExtractor
)
from signature_string_checker import SignatureStringChecker

class TestStringPatternAnalyzer(unittest.TestCase):
    """Test StringPatternAnalyzer"""
    
    def setUp(self):
        self.analyzer = StringPatternAnalyzer()
    
    def test_dotted_string(self):
        """Test pattern analysis for dotted string (FullyQualifiedName)"""
        patterns = self.analyzer.analyze_patterns("java.lang.String")
        self.assertEqual(patterns['dot_count'], 2.0)
        self.assertEqual(patterns['slash_count'], 0.0)
        self.assertEqual(patterns['capital_letter_count'], 1.0)  # S
    
    def test_slashed_string(self):
        """Test pattern analysis for slashed string (BinaryName)"""
        patterns = self.analyzer.analyze_patterns("java/lang/String")
        self.assertEqual(patterns['dot_count'], 0.0)
        self.assertEqual(patterns['slash_count'], 2.0)
        self.assertEqual(patterns['capital_letter_count'], 1.0)  # S
    
    def test_field_descriptor(self):
        """Test pattern analysis for field descriptor"""
        patterns = self.analyzer.analyze_patterns("Ljava/lang/String;")
        self.assertEqual(patterns['semicolon_count'], 1.0)
        self.assertEqual(patterns['slash_count'], 2.0)
        self.assertEqual(patterns['string_length'], 18.0)
    
    def test_empty_string(self):
        """Test pattern analysis for empty string"""
        patterns = self.analyzer.analyze_patterns("")
        self.assertEqual(patterns['string_length'], 0.0)
        self.assertEqual(patterns['dot_count'], 0.0)

class TestFormatDetector(unittest.TestCase):
    """Test FormatDetector"""
    
    def setUp(self):
        self.detector = FormatDetector()
    
    def test_fully_qualified_name(self):
        """Test format detection for FullyQualifiedName"""
        format_features = self.detector.detect_format("java.lang.String")
        self.assertEqual(format_features['has_dots'], 1.0)
        self.assertEqual(format_features['has_slashes'], 0.0)
        self.assertGreater(format_features['fully_qualified_confidence'], 0.5)
    
    def test_binary_name(self):
        """Test format detection for BinaryName"""
        format_features = self.detector.detect_format("java/lang/String")
        self.assertEqual(format_features['has_dots'], 0.0)
        self.assertEqual(format_features['has_slashes'], 1.0)
        self.assertGreater(format_features['binary_confidence'], 0.5)
    
    def test_field_descriptor(self):
        """Test format detection for FieldDescriptor"""
        format_features = self.detector.detect_format("Ljava/lang/String;")
        self.assertEqual(format_features['is_field_descriptor_format'], 1.0)
        self.assertGreater(format_features['field_descriptor_confidence'], 0.5)

class TestStructuralAnalyzer(unittest.TestCase):
    """Test StructuralAnalyzer"""
    
    def setUp(self):
        self.analyzer = StructuralAnalyzer()
    
    def test_package_depth(self):
        """Test package depth calculation"""
        structure = self.analyzer.analyze_structure("java.lang.String")
        self.assertEqual(structure['package_depth'], 2.0)  # java.lang
        self.assertGreater(structure['class_name_length'], 0.0)
    
    def test_array_type(self):
        """Test array type detection"""
        structure = self.analyzer.analyze_structure("Ljava/lang/String;[")
        self.assertEqual(structure['has_array_brackets'], 1.0)
    
    def test_primitive_type(self):
        """Test primitive type detection"""
        structure = self.analyzer.analyze_structure("I")
        self.assertEqual(structure['has_primitive_type'], 1.0)

class TestSourceCodeExtractor(unittest.TestCase):
    """Test SourceCodeFeatureExtractor"""
    
    def setUp(self):
        self.extractor = SourceCodeFeatureExtractor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_extract_string_from_line(self):
        """Test string extraction from a line"""
        line = 'String className = "java.lang.String";'
        result = self.extractor.extract_string_at_line("dummy", 1)
        # This will return None since file doesn't exist, but we can test the fallback
        self.assertIsNotNone(self.extractor.fallback_extractor._extract_string_from_line(line))
    
    def test_create_java_file(self):
        """Test extracting string from actual Java file"""
        java_file = os.path.join(self.temp_dir, "Test.java")
        with open(java_file, 'w') as f:
            f.write('public class Test {\n')
            f.write('    String name = "java.lang.String";\n')
            f.write('}\n')
        
        result = self.extractor.extract_string_at_line(java_file, 2)
        self.assertIsNotNone(result)
        self.assertEqual(result, "java.lang.String")

class TestSignatureStringFeatureExtractor(unittest.TestCase):
    """Test SignatureStringFeatureExtractor integration"""
    
    def setUp(self):
        self.extractor = SignatureStringFeatureExtractor()
    
    def test_extract_features_fully_qualified(self):
        """Test feature extraction for FullyQualifiedName"""
        cfg_data = {
            'nodes': [],
            'control_edges': [],
            'dataflow_edges': [],
            'java_file': ''
        }
        node = {
            'id': 1,
            'label': 'String className = "java.lang.String"',
            'node_type': 'variable',
            'line': 10
        }
        
        features = self.extractor.extract_features(
            string_value="java.lang.String",
            label=node['label'],
            node_type=node['node_type'],
            cfg_data=cfg_data,
            node=node
        )
        
        self.assertEqual(len(features), 30)
        self.assertGreater(features[0], 0.0)  # has_dots
        self.assertEqual(features[1], 0.0)  # has_slashes
    
    def test_extract_features_binary(self):
        """Test feature extraction for BinaryName"""
        cfg_data = {
            'nodes': [],
            'control_edges': [],
            'dataflow_edges': [],
            'java_file': ''
        }
        node = {
            'id': 1,
            'label': 'String className = "java/lang/String"',
            'node_type': 'variable',
            'line': 10
        }
        
        features = self.extractor.extract_features(
            string_value="java/lang/String",
            label=node['label'],
            node_type=node['node_type'],
            cfg_data=cfg_data,
            node=node
        )
        
        self.assertEqual(len(features), 30)
        self.assertEqual(features[0], 0.0)  # has_dots
        self.assertGreater(features[1], 0.0)  # has_slashes
    
    def test_extract_features_field_descriptor(self):
        """Test feature extraction for FieldDescriptor"""
        cfg_data = {
            'nodes': [],
            'control_edges': [],
            'dataflow_edges': [],
            'java_file': ''
        }
        node = {
            'id': 1,
            'label': 'String desc = "Ljava/lang/String;"',
            'node_type': 'variable',
            'line': 10
        }
        
        features = self.extractor.extract_features(
            string_value="Ljava/lang/String;",
            label=node['label'],
            node_type=node['node_type'],
            cfg_data=cfg_data,
            node=node
        )
        
        self.assertEqual(len(features), 30)
        self.assertGreater(features[2], 0.0)  # is_field_descriptor_format

class TestSignatureStringChecker(unittest.TestCase):
    """Test SignatureStringChecker integration"""
    
    def setUp(self):
        self.checker = SignatureStringChecker()
    
    def test_extract_features(self):
        """Test feature extraction from SignatureStringChecker"""
        cfg_data = {
            'nodes': [{'id': 1, 'label': 'test', 'node_type': 'stmt', 'line': 1}],
            'control_edges': [],
            'dataflow_edges': [],
            'java_file': ''
        }
        node = {
            'id': 1,
            'label': 'String className = "java.lang.String"',
            'node_type': 'variable',
            'line': 10
        }
        
        features = self.checker.extract_features(cfg_data, node)
        
        # Should return 30 features
        self.assertEqual(len(features), 30)
        self.assertTrue(all(isinstance(f, float) for f in features))

if __name__ == '__main__':
    unittest.main()

