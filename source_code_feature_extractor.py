#!/usr/bin/env python3
"""
Source Code Feature Extractor

This module provides utilities to extract string values from Java source code
for Signature String Checker feature extraction.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class SourceCodeReader:
    """Reads Java files and extracts string literals at specific line numbers"""
    
    def __init__(self):
        self.cache = {}  # Cache for file contents
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read Java file content, with caching"""
        if file_path in self.cache:
            return self.cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                self.cache[file_path] = content
                return content
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return None
    
    def get_line(self, file_path: str, line_number: int) -> Optional[str]:
        """Get specific line from file"""
        content = self.read_file(file_path)
        if content is None:
            return None
        
        lines = content.splitlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
        return None
    
    def get_context(self, file_path: str, line_number: int, context_lines: int = 5) -> List[str]:
        """Get surrounding context lines"""
        content = self.read_file(file_path)
        if content is None:
            return []
        
        lines = content.splitlines()
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        return lines[start:end]

class ASTStringExtractor:
    """Uses Eclipse JDT to parse and extract string values from AST"""
    
    def __init__(self):
        self.jdt_available = self._check_jdt_available()
    
    def _check_jdt_available(self) -> bool:
        """Check if JDT services are available"""
        try:
            # Check if JDT parser JAR exists
            jdt_jar = Path('/home/ubuntu/GenDATA/build/libs/jdtParser.jar')
            return jdt_jar.exists()
        except Exception:
            return False
    
    def extract_string_at_line(self, file_path: str, line_number: int) -> Optional[str]:
        """
        Extract string literal/expression at specific line using JDT AST parsing.
        
        Returns the actual string value if found, None otherwise.
        """
        if not self.jdt_available:
            return None
        
        try:
            # Use JDT service to parse and extract string
            # This would require calling Java service via subprocess
            # For now, return None and fall back to regex-based extraction
            return None
        except Exception as e:
            logger.debug(f"JDT extraction failed for {file_path}:{line_number}: {e}")
            return None

class FallbackExtractor:
    """Regex-based string extraction if AST parsing unavailable"""
    
    def __init__(self, source_reader: SourceCodeReader):
        self.reader = source_reader
    
    def extract_string_at_line(self, file_path: str, line_number: int) -> Optional[str]:
        """
        Extract string literal at line using regex patterns.
        
        Looks for string literals in common patterns:
        - Class.forName("...")
        - Class.getName()
        - String variable assignments
        - Method parameters
        """
        line = self.reader.get_line(file_path, line_number)
        if not line:
            return None
        
        # Pattern 1: String literal in quotes: "java.lang.String"
        string_literal_pattern = r'["\']([^"\']+)["\']'
        matches = re.findall(string_literal_pattern, line)
        if matches:
            # Return the first string literal found
            return matches[0]
        
        # Pattern 2: Class.forName("...")
        forname_pattern = r'Class\.forName\(["\']([^"\']+)["\']\)'
        match = re.search(forname_pattern, line, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 3: Variable assignment: String name = "..."
        assignment_pattern = r'String\s+\w+\s*=\s*["\']([^"\']+)["\']'
        match = re.search(assignment_pattern, line, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 4: Method parameter: method("...")
        param_pattern = r'\(["\']([^"\']+)["\']\)'
        match = re.search(param_pattern, line)
        if match:
            return match.group(1)
        
        return None
    
    def extract_string_context(self, file_path: str, line_number: int, context_lines: int = 5) -> Dict[str, Any]:
        """Extract string value and surrounding context"""
        context = self.reader.get_context(file_path, line_number, context_lines)
        string_value = self.extract_string_at_line(file_path, line_number)
        
        return {
            'string_value': string_value,
            'line': self.reader.get_line(file_path, line_number),
            'context': context,
            'line_number': line_number
        }
    
    def parse_string_value(self, source_code: str, line_number: int) -> Optional[str]:
        """Parse and normalize string value from source code"""
        lines = source_code.splitlines()
        if 1 <= line_number <= len(lines):
            line = lines[line_number - 1]
            return self._extract_string_from_line(line)
        return None
    
    def _extract_string_from_line(self, line: str) -> Optional[str]:
        """Extract string value from a single line"""
        # Try various patterns
        patterns = [
            r'["\']([^"\']+)["\']',  # Simple string literal
            r'Class\.forName\(["\']([^"\']+)["\']\)',  # Class.forName
            r'String\s+\w+\s*=\s*["\']([^"\']+)["\']',  # String assignment
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

class SourceCodeFeatureExtractor:
    """Main interface for extracting string values from Java source code"""
    
    def __init__(self):
        self.reader = SourceCodeReader()
        self.ast_extractor = ASTStringExtractor()
        self.fallback_extractor = FallbackExtractor(self.reader)
    
    def extract_string_at_line(self, file_path: str, line_number: int) -> Optional[str]:
        """
        Extract string literal/expression at specific line.
        
        Tries AST extraction first, falls back to regex-based extraction.
        """
        # Try AST extraction first
        if self.ast_extractor.jdt_available:
            result = self.ast_extractor.extract_string_at_line(file_path, line_number)
            if result:
                return result
        
        # Fall back to regex-based extraction
        return self.fallback_extractor.extract_string_at_line(file_path, line_number)
    
    def extract_string_context(self, file_path: str, line_number: int, context_lines: int = 5) -> Dict[str, Any]:
        """Extract string value and surrounding context"""
        return self.fallback_extractor.extract_string_context(file_path, line_number, context_lines)
    
    def parse_string_value(self, source_code: str, line_number: int) -> Optional[str]:
        """Parse and normalize string value from source code"""
        return self.fallback_extractor.parse_string_value(source_code, line_number)

