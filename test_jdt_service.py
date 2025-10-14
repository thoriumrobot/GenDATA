#!/usr/bin/env python3
"""
Unit tests for JDT service wrapper
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from jdt_service import JdtParserService, CodeLocation, WarningInfo


class TestJdtParserService(unittest.TestCase):
    """Test cases for JdtParserService wrapper"""
    
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
    }
}
'''
        self.test_warnings = '''TestClass.java:3:5: compiler.warn.proc.messager: [nullness] potential null pointer dereference
TestClass.java:5:9: compiler.err.proc.messager: [nullness] null assignment to non-null field'''
    
    @patch('jdt_service.JdtParserService._find_jar_path')
    @patch('subprocess.run')
    def test_parse_code_locations_from_string(self, mock_run, mock_find_jar):
        """Test parsing code locations from Java code string"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-parser-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Parsed 3 code locations"
        mock_run.return_value = mock_result
        
        # Mock JSON output
        expected_json = [
            {
                "lineStart": 1,
                "lineEnd": 7,
                "columnStart": 0,
                "columnEnd": 0,
                "locationType": "CLASS_LEVEL",
                "context": {"class_name": "TestClass"},
                "codeSnippet": "",
                "applicableTransformations": ["RANDOM_METHOD_INSERTION"]
            }
        ]
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(expected_json))):
            with patch('os.path.exists', return_value=True):
                service = JdtParserService()
                locations = service.parse_code_locations_from_string(self.test_java_code)
                
                self.assertIsInstance(locations, list)
                self.assertEqual(len(locations), 1)
                self.assertEqual(locations[0].location_type, "CLASS_LEVEL")
    
    @patch('jdt_service.JdtParserService._find_jar_path')
    @patch('subprocess.run')
    def test_parse_warnings(self, mock_run, mock_find_jar):
        """Test parsing warnings file"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-parser-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Parsed 2 warnings"
        mock_run.return_value = mock_result
        
        # Mock JSON output
        expected_json = [
            {
                "lineNumber": 1,
                "filePath": "TestClass.java",
                "line": 3,
                "column": 5,
                "severity": "warning",
                "checker": "nullness",
                "message": "potential null pointer dereference"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.test_warnings)
            f.flush()
            
            with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(expected_json))):
                with patch('os.path.exists', return_value=True):
                    service = JdtParserService()
                    warnings = service.parse_warnings(f.name)
                    
                    self.assertIsInstance(warnings, list)
                    self.assertEqual(len(warnings), 1)
                    self.assertEqual(warnings[0].file_path, "TestClass.java")
                    self.assertEqual(warnings[0].severity, "warning")
            
            os.unlink(f.name)
    
    @patch('jdt_service.JdtParserService._find_jar_path')
    @patch('subprocess.run')
    def test_extract_identifiers(self, mock_run, mock_find_jar):
        """Test extracting identifiers from Java code"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-parser-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Extracted identifiers"
        mock_run.return_value = mock_result
        
        # Mock JSON output
        expected_json = {
            "variables": ["x"],
            "methods": ["testMethod", "println"],
            "types": ["TestClass", "String"],
            "fields": []
        }
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(expected_json))):
            with patch('os.path.exists', return_value=True):
                service = JdtParserService()
                identifiers = service.extract_identifiers(self.test_java_code)
                
                self.assertIsInstance(identifiers, dict)
                self.assertIn("variables", identifiers)
                self.assertIn("methods", identifiers)
                self.assertIn("types", identifiers)
                self.assertEqual(identifiers["variables"], ["x"])
    
    @patch('jdt_service.JdtParserService._find_jar_path')
    @patch('subprocess.run')
    def test_validate_syntax(self, mock_run, mock_find_jar):
        """Test Java syntax validation"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-parser-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Syntax validation result"
        mock_run.return_value = mock_result
        
        # Mock JSON output for valid syntax
        expected_json = {"valid": True, "message": "Syntax is valid"}
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(expected_json))):
            with patch('os.path.exists', return_value=True):
                service = JdtParserService()
                is_valid = service.validate_syntax(self.test_java_code)
                
                self.assertTrue(is_valid)
    
    @patch('jdt_service.JdtParserService._find_jar_path')
    @patch('subprocess.run')
    def test_validate_syntax_invalid(self, mock_run, mock_find_jar):
        """Test Java syntax validation with invalid code"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-parser-all.jar"
        
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Syntax validation result"
        mock_run.return_value = mock_result
        
        # Mock JSON output for invalid syntax
        expected_json = {"valid": False, "message": "Syntax errors found"}
        
        invalid_java_code = '''
public class InvalidClass {
    public void invalidMethod() {
        System.out.println("Invalid"  // Missing closing parenthesis
    }
}
'''
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(expected_json))):
            with patch('os.path.exists', return_value=True):
                service = JdtParserService()
                is_valid = service.validate_syntax(invalid_java_code)
                
                self.assertFalse(is_valid)
    
    @patch('jdt_service.JdtParserService._find_jar_path')
    @patch('subprocess.run')
    def test_service_failure(self, mock_run, mock_find_jar):
        """Test service failure handling"""
        # Mock JAR path
        mock_find_jar.return_value = "/fake/path/jdt-parser-all.jar"
        
        # Mock subprocess failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "JDT parser failed"
        mock_run.return_value = mock_result
        
        with patch('os.path.exists', return_value=True):
            service = JdtParserService()
            
            # Should raise RuntimeError on service failure
            with self.assertRaises(RuntimeError):
                service.parse_code_locations_from_string(self.test_java_code)
    
    def test_jar_not_found(self):
        """Test handling when JAR file is not found"""
        with patch('jdt_service.JdtParserService._find_jar_path', side_effect=FileNotFoundError("JAR not found")):
            with self.assertRaises(FileNotFoundError):
                JdtParserService()


class TestCodeLocation(unittest.TestCase):
    """Test cases for CodeLocation dataclass"""
    
    def test_code_location_creation(self):
        """Test creating a CodeLocation instance"""
        location = CodeLocation(
            line_start=1,
            line_end=5,
            column_start=0,
            column_end=0,
            location_type="METHOD_LEVEL",
            context={"method_name": "testMethod"},
            code_snippet="public void testMethod() { }",
            applicable_transformations=["GUARD_REVERSAL", "METHOD_EXTRACTION"]
        )
        
        self.assertEqual(location.line_start, 1)
        self.assertEqual(location.line_end, 5)
        self.assertEqual(location.location_type, "METHOD_LEVEL")
        self.assertEqual(location.context["method_name"], "testMethod")
        self.assertEqual(len(location.applicable_transformations), 2)


class TestWarningInfo(unittest.TestCase):
    """Test cases for WarningInfo dataclass"""
    
    def test_warning_info_creation(self):
        """Test creating a WarningInfo instance"""
        warning = WarningInfo(
            line_number=1,
            file_path="TestClass.java",
            line=3,
            column=5,
            severity="warning",
            checker="nullness",
            message="potential null pointer dereference"
        )
        
        self.assertEqual(warning.file_path, "TestClass.java")
        self.assertEqual(warning.line, 3)
        self.assertEqual(warning.severity, "warning")
        self.assertEqual(warning.checker, "nullness")


if __name__ == '__main__':
    unittest.main()
