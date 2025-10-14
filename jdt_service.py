#!/usr/bin/env python3
"""
JDT Service Wrapper

Python wrapper for Eclipse JDT-based parsing services.
Replaces regex-based parsing with robust AST parsing.
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LocationType(Enum):
    """Types of code locations where transformations can be applied"""
    CLASS_LEVEL = "CLASS_LEVEL"
    METHOD_LEVEL = "METHOD_LEVEL"
    STATEMENT_LEVEL = "STATEMENT_LEVEL"
    EXPRESSION_LEVEL = "EXPRESSION_LEVEL"
    BLOCK_LEVEL = "BLOCK_LEVEL"

@dataclass
class CodeLocation:
    """Represents a specific location in Java code"""
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    location_type: str
    context: Dict[str, Any]
    code_snippet: str
    applicable_transformations: List[str]

@dataclass
class WarningInfo:
    """Represents a warning from Checker Framework"""
    line_number: int
    file_path: str
    line: int
    column: int
    severity: str
    checker: str
    message: str

class JdtParserService:
    """Python wrapper for JDT parser service"""
    
    def __init__(self, jar_path: Optional[str] = None):
        """
        Initialize JDT parser service.
        
        Args:
            jar_path: Path to jdt-parser-all.jar. If None, will try to find it.
        """
        if jar_path is None:
            jar_path = self._find_jar_path()
        
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"JDT parser JAR not found: {jar_path}")
        
        self.jar_path = jar_path
        self.java_cmd = self._get_java_command()
        
        logger.info(f"Initialized JDT parser service with JAR: {jar_path}")
    
    def _find_jar_path(self) -> str:
        """Find the JDT parser JAR in the build directory"""
        possible_paths = [
            "build/libs/jdt-parser-all.jar",
            "/home/ubuntu/GenDATA/build/libs/jdt-parser-all.jar",
            "build/libs/jdt-parser.jar",
            "/home/ubuntu/GenDATA/build/libs/jdt-parser.jar"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find JDT parser JAR. Please build it first with: ./gradlew jdtParserJar")
    
    def _get_java_command(self) -> str:
        """Get Java command with proper classpath"""
        return "java"
    
    def parse_code_locations(self, java_file: str) -> List[CodeLocation]:
        """
        Parse Java file and extract code locations.
        
        Args:
            java_file: Path to Java source file
            
        Returns:
            List of CodeLocation objects
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            cmd = [
                self.java_cmd, "-jar", self.jar_path,
                "--operation", "parse-code-locations",
                "--input", java_file,
                "--output", output_path
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"JDT parser failed: {result.stderr}")
            
            # Read and parse JSON output
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            locations = []
            for item in data:
                location = CodeLocation(
                    line_start=item['lineStart'],
                    line_end=item['lineEnd'],
                    column_start=item['columnStart'],
                    column_end=item['columnEnd'],
                    location_type=item['locationType'],
                    context=item['context'],
                    code_snippet=item['codeSnippet'],
                    applicable_transformations=item['applicableTransformations']
                )
                locations.append(location)
            
            logger.info(f"Parsed {len(locations)} code locations from {java_file}")
            return locations
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def parse_code_locations_from_string(self, java_code: str) -> List[CodeLocation]:
        """
        Parse Java code string and extract code locations.
        
        Args:
            java_code: Java source code as string
            
        Returns:
            List of CodeLocation objects
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as input_file:
            input_file.write(java_code)
            input_path = input_file.name
        
        try:
            return self.parse_code_locations(input_path)
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def parse_warnings(self, warnings_file: str) -> List[WarningInfo]:
        """
        Parse Checker Framework warnings file.
        
        Args:
            warnings_file: Path to warnings file
            
        Returns:
            List of WarningInfo objects
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            cmd = [
                self.java_cmd, "-jar", self.jar_path,
                "--operation", "parse-warnings",
                "--input", warnings_file,
                "--output", output_path
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"JDT warning parser failed: {result.stderr}")
            
            # Read and parse JSON output
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            warnings = []
            for item in data:
                warning = WarningInfo(
                    line_number=item['lineNumber'],
                    file_path=item['filePath'],
                    line=item['line'],
                    column=item['column'],
                    severity=item['severity'],
                    checker=item['checker'],
                    message=item['message']
                )
                warnings.append(warning)
            
            logger.info(f"Parsed {len(warnings)} warnings from {warnings_file}")
            return warnings
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def extract_identifiers(self, java_code: str) -> Dict[str, List[str]]:
        """
        Extract identifiers from Java code.
        
        Args:
            java_code: Java source code as string
            
        Returns:
            Dictionary with categorized identifiers
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as input_file:
            input_file.write(java_code)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            cmd = [
                self.java_cmd, "-jar", self.jar_path,
                "--operation", "parse-identifiers",
                "--input", input_path,
                "--output", output_path
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"JDT identifier extractor failed: {result.stderr}")
            
            # Read and parse JSON output
            with open(output_path, 'r') as f:
                identifiers = json.load(f)
            
            logger.info(f"Extracted identifiers: {list(identifiers.keys())}")
            return identifiers
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def validate_syntax(self, java_code: str) -> bool:
        """
        Validate Java syntax.
        
        Args:
            java_code: Java source code as string
            
        Returns:
            True if syntax is valid, False otherwise
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as input_file:
            input_file.write(java_code)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            cmd = [
                self.java_cmd, "-jar", self.jar_path,
                "--operation", "validate-syntax",
                "--input", input_path,
                "--output", output_path
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"JDT syntax validation failed: {result.stderr}")
                return False
            
            # Read and parse JSON output
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            is_valid = data.get('valid', False)
            logger.debug(f"Syntax validation result: {is_valid}")
            return is_valid
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

# Convenience function for backward compatibility
def create_jdt_parser_service(jar_path: Optional[str] = None) -> JdtParserService:
    """Create and return a JDT parser service instance"""
    return JdtParserService(jar_path)
