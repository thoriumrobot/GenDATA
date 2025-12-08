#!/usr/bin/env python3
"""
Signature String Checker Implementation

This module implements the CheckerInterface for the Signature String Checker,
which tracks string format types for Java type signatures.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import logging
from checker_interface import CheckerInterface

logger = logging.getLogger(__name__)

class SignatureStringChecker(CheckerInterface):
    """Implementation of CheckerInterface for Signature String Checker"""
    
    def get_checker_name(self) -> str:
        return "SignatureString"
    
    def get_checker_processor(self) -> str:
        return "org.checkerframework.checker.signature.qual.SignatureChecker"
    
    def get_annotation_types(self) -> List[str]:
        return ['@FullyQualifiedName', '@BinaryName', '@FieldDescriptor']
    
    def parse_warnings(self, warnings_file: str) -> List[Dict[str, Any]]:
        """
        Parse Signature String Checker warnings from output file.
        
        Warning format example:
        ClassLoader.java:45: error: [signature.type.incompatible] Expected FullyQualifiedName but got BinaryName
        """
        warnings = []
        
        if not Path(warnings_file).exists():
            logger.warning(f"Warnings file not found: {warnings_file}")
            return warnings
        
        try:
            with open(warnings_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse warning line
                    match = re.match(r'(.+?):(\d+):\s*(error|warning):\s*\[(.+?)\]\s*(.+)', line)
                    if match:
                        file_path, line_num, level, checker_msg, message = match.groups()
                        warnings.append({
                            'file': file_path,
                            'line': int(line_num),
                            'column': 0,
                            'level': level,
                            'checker_message': checker_msg,
                            'message': message,
                            'annotation_type': self._infer_annotation_type(checker_msg, message)
                        })
        except Exception as e:
            logger.error(f"Error parsing warnings file {warnings_file}: {e}")
        
        return warnings
    
    def _infer_annotation_type(self, checker_msg: str, message: str) -> Optional[str]:
        """Infer likely annotation type from warning message"""
        msg_lower = message.lower()
        
        if 'fullyqualifiedname' in msg_lower or 'fully qualified' in msg_lower:
            return '@FullyQualifiedName'
        elif 'binaryname' in msg_lower or 'binary name' in msg_lower:
            return '@BinaryName'
        elif 'fielddescriptor' in msg_lower or 'field descriptor' in msg_lower:
            return '@FieldDescriptor'
        
        return None
    
    def extract_features(self, cfg_data: Dict[str, Any], node: Dict[str, Any]) -> List[float]:
        """
        Extract Signature String Checker-specific features.
        
        Features include:
        - String format patterns (dotted vs slashed)
        - Type name patterns
        - Method signature patterns
        - Class.forName usage
        """
        features = []
        label = node.get('label', '').lower()
        node_type = node.get('node_type', '').lower()
        
        # Feature 1: Dotted format (FullyQualifiedName)
        has_dots = '.' in label and not label.startswith('.')
        features.append(1.0 if has_dots else 0.0)
        
        # Feature 2: Slashed format (BinaryName)
        has_slashes = '/' in label
        features.append(1.0 if has_slashes else 0.0)
        
        # Feature 3: Field descriptor format (L...;)
        is_field_descriptor = label.startswith('l') and label.endswith(';') and '/' in label
        features.append(1.0 if is_field_descriptor else 0.0)
        
        # Feature 4: Class.forName usage
        is_forname = 'class.forname' in label or 'forname(' in label
        features.append(1.0 if is_forname else 0.0)
        
        # Feature 5: Class.getName usage
        is_getname = 'class.getname' in label or '.getname(' in label
        features.append(1.0 if is_getname else 0.0)
        
        # Feature 6: Method descriptor pattern
        is_method_descriptor = '(' in label and ')' in label and any(c in label for c in ['i', 'l', 'd', 'f', 'z'])
        features.append(1.0 if is_method_descriptor else 0.0)
        
        # Feature 7: Package name pattern
        has_package = '.' in label and len(label.split('.')) > 1
        features.append(1.0 if has_package else 0.0)
        
        # Feature 8: Array type pattern
        is_array_type = '[' in label
        features.append(1.0 if is_array_type else 0.0)
        
        return features
    
    def validate_annotation(self, annotation_type: str, location: Dict[str, Any]) -> bool:
        """Validate annotation placement"""
        if annotation_type not in self.get_annotation_types():
            return False
        
        target_type = location.get('target_type', '')
        valid_targets = ['parameter', 'local_variable', 'return']
        
        return target_type in valid_targets
    
    def get_training_data_source(self) -> str:
        # Signature String Checker test suite location
        return '/home/ubuntu/checker-framework/checker/tests/signature/'
    
    def get_warning_patterns(self) -> List[str]:
        return [
            'signature.type.incompatible',
            'signature.type',
            'fullyqualifiedname',
            'binaryname',
            'fielddescriptor',
            'signature'
        ]

