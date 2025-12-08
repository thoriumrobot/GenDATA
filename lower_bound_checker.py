#!/usr/bin/env python3
"""
Lower Bound Checker Implementation

This module implements the CheckerInterface for the Lower Bound Checker,
which is currently the primary checker supported by GenDATA.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import logging
from checker_interface import CheckerInterface

logger = logging.getLogger(__name__)

class LowerBoundChecker(CheckerInterface):
    """Implementation of CheckerInterface for Lower Bound Checker"""
    
    def get_checker_name(self) -> str:
        return "LowerBound"
    
    def get_checker_processor(self) -> str:
        return "org.checkerframework.checker.index.IndexChecker"
    
    def get_annotation_types(self) -> List[str]:
        return ['@Positive', '@NonNegative', '@GTENegativeOne']
    
    def parse_warnings(self, warnings_file: str) -> List[Dict[str, Any]]:
        """
        Parse Lower Bound Checker warnings from output file.
        
        Warning format example:
        StringMethods.java:46: error: [array.access.unsafe.high] array access might be out of bounds
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
                    
                    # Parse warning line: file:line: error: [checker.message] message
                    match = re.match(r'(.+?):(\d+):\s*(error|warning):\s*\[(.+?)\]\s*(.+)', line)
                    if match:
                        file_path, line_num, level, checker_msg, message = match.groups()
                        warnings.append({
                            'file': file_path,
                            'line': int(line_num),
                            'column': 0,  # Column not always available
                            'level': level,
                            'checker_message': checker_msg,
                            'message': message,
                            'annotation_type': self._infer_annotation_type(checker_msg, message)
                        })
        except Exception as e:
            logger.error(f"Error parsing warnings file {warnings_file}: {e}")
        
        return warnings
    
    def _infer_annotation_type(self, checker_msg: str, message: str) -> Optional[str]:
        """
        Infer likely annotation type from warning message.
        
        This is a heuristic - actual annotation type should be determined
        by the model during prediction.
        """
        msg_lower = message.lower()
        
        if 'negative' in msg_lower or 'lower bound' in msg_lower:
            if 'positive' in msg_lower or 'greater than zero' in msg_lower:
                return '@Positive'
            elif 'greater than or equal to -1' in msg_lower or 'gtenegativeone' in msg_lower:
                return '@GTENegativeOne'
            else:
                return '@NonNegative'
        
        return None
    
    def extract_features(self, cfg_data: Dict[str, Any], node: Dict[str, Any]) -> List[float]:
        """
        Extract Lower Bound Checker-specific features.
        
        Features include:
        - Array index usage patterns
        - Loop variable patterns
        - Numeric comparison patterns
        - Subtraction result patterns
        """
        features = []
        label = node.get('label', '').lower()
        node_type = node.get('node_type', '').lower()
        
        # Feature 1: Array index usage
        is_array_index = ('[' in label or ']' in label) and any(v in label for v in ['index', 'i', 'j', 'k', 'idx'])
        features.append(1.0 if is_array_index else 0.0)
        
        # Feature 2: Loop variable
        is_loop_var = any(pattern in label for pattern in ['for', 'while', 'iterator']) and \
                     any(v in label for v in ['i', 'j', 'k', 'idx', 'index', 'counter'])
        features.append(1.0 if is_loop_var else 0.0)
        
        # Feature 3: Subtraction result
        is_subtraction = any(pattern in label for pattern in [' - ', '- ', 'length -', 'size -', 'count -'])
        features.append(1.0 if is_subtraction else 0.0)
        
        # Feature 4: Comparison with length/size
        compared_with_length = any(pattern in label for pattern in [
            '< length', '< size', '<= length', '<= size',
            'length >', 'size >', 'length >=', 'size >='
        ])
        features.append(1.0 if compared_with_length else 0.0)
        
        # Feature 5: Initialization to zero
        initialized_to_zero = any(pattern in label for pattern in ['= 0', '=0', ':= 0', ':=0'])
        features.append(1.0 if initialized_to_zero else 0.0)
        
        # Feature 6: Nonnegative check
        nonnegative_check = any(pattern in label for pattern in ['>= 0', '>=0', '>= -1', '>=-1'])
        features.append(1.0 if nonnegative_check else 0.0)
        
        return features
    
    def validate_annotation(self, annotation_type: str, location: Dict[str, Any]) -> bool:
        """Validate annotation placement"""
        if annotation_type not in self.get_annotation_types():
            return False
        
        # Basic validation - can be extended with more sophisticated checks
        target_type = location.get('target_type', '')
        valid_targets = ['parameter', 'field', 'local_variable', 'return']
        
        return target_type in valid_targets
    
    def get_training_data_source(self) -> str:
        return '/home/ubuntu/checker-framework/checker/tests/index/'
    
    def get_warning_patterns(self) -> List[str]:
        return [
            'array.access.unsafe',
            'array.access.unsafe.high',
            'array.access.unsafe.low',
            'lowerbound',
            'positive',
            'nonnegative'
        ]

