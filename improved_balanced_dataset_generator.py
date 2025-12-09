#!/usr/bin/env python3
"""
Improved Balanced Dataset Generator for Annotation Type Models

This module creates a balanced training dataset using REAL code examples where:
- Positive examples: Code nodes that actually need the specific annotation type
- Negative examples: Code nodes that don't need the specific annotation type (real code, not artificial)

This ensures models learn from meaningful code patterns rather than artificial feature modifications.
"""

import os
import json
import random
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import checker-specific modules
try:
    from checker_config import CheckerType
    from value_pattern_detector import ValuePatternDetector
    CHECKER_MODULES_AVAILABLE = True
except ImportError:
    CHECKER_MODULES_AVAILABLE = False
    CheckerType = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RealBalancedExample:
    """Represents a balanced training example using real code"""
    node_id: int
    file_path: str
    method_name: str
    line_number: int
    node_type: str
    node_label: str
    features: List[float]
    annotation_type: str  # The target annotation type
    is_positive: bool     # True if this annotation should be present, False if absent
    confidence: float     # Synthetic confidence score
    code_context: str     # The actual code context for this node

class ImprovedBalancedDatasetGenerator:
    """Generates balanced training datasets using real code examples"""
    
    def __init__(self, target_balance: float = 0.5, random_seed: int = 42, checker_type: Optional[CheckerType] = None, checker_name: Optional[str] = None):
        """
        Initialize the improved balanced dataset generator
        
        Args:
            target_balance: Target ratio of positive examples (0.5 = 50% positive, 50% negative)
            random_seed: Random seed for reproducible results
            checker_type: Optional checker type for checker-specific value pattern extraction
            checker_name: Optional checker name ('lower_bound', 'sql_quotes', 'signature_string') to determine annotation types
        """
        self.target_balance = target_balance
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Checker type for value pattern extraction
        self.checker_type = checker_type
        self.checker_name = checker_name or 'lower_bound'  # Default to lower_bound
        if CHECKER_MODULES_AVAILABLE and checker_type is not None:
            self.pattern_detector = ValuePatternDetector()
        else:
            self.pattern_detector = None
        
        # Annotation types to balance - determined by checker name
        if self.checker_name == 'sql_quotes':
            self.annotation_types = ['@SqlEvenQuotes', '@SqlOddQuotes']
        elif self.checker_name == 'signature_string':
            self.annotation_types = ['@FullyQualifiedName', '@BinaryName', '@FieldDescriptor']
        else:  # Default: lower_bound
            self.annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        
        # Statistics tracking
        self.generation_stats = {
            'total_examples': 0,
            'positive_examples': 0,
            'negative_examples': 0,
            'annotation_type_counts': {ann_type: {'positive': 0, 'negative': 0} 
                                     for ann_type in self.annotation_types}
        }
    
    def load_cfg_files(self, cfg_directory: str) -> List[Dict[str, Any]]:
        """Load all CFG files from a directory"""
        cfg_files = []
        
        if not os.path.exists(cfg_directory):
            logger.warning(f"CFG directory does not exist: {cfg_directory}")
            return cfg_files
        
        for root, dirs, files in os.walk(cfg_directory):
            for file in files:
                if file.endswith('.json'):
                    cfg_path = os.path.join(root, file)
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg_data = json.load(f)
                        
                        # Extract method name from path
                        method_name = os.path.splitext(file)[0]
                        
                        cfg_files.append({
                            'file': cfg_path,
                            'method': method_name,
                            'data': cfg_data
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load CFG file {cfg_path}: {e}")
        
        logger.info(f"Loaded {len(cfg_files)} CFG files from {cfg_directory}")
        return cfg_files
    
    def extract_node_features(self, node: Dict, cfg_data: Dict, include_checker_patterns: bool = True) -> List[float]:
        """
        Extract features from a CFG node (enhanced version with semantic patterns)
        
        Args:
            node: CFG node dictionary
            cfg_data: Full CFG data dictionary
            include_checker_patterns: Whether to include checker-specific value patterns
            
        Returns:
            List of feature values
        """
        # Use checker-specific feature extraction if available
        if self.checker_name == 'sql_quotes':
            return self._extract_sql_quotes_features(node, cfg_data)
        elif self.checker_name == 'signature_string':
            return self._extract_signature_string_features(node, cfg_data)
        
        # Default: Lower Bound Checker features
        label = node.get('label', '').lower()
        node_type = node.get('type', '').lower()
        line = node.get('line', 0)
        
        # Check for comparison patterns in label and surrounding context
        has_strict_positive = any(pattern in label for pattern in ['> 0', '>0', 'greater than 0', 'strictly positive'])
        has_nonnegative = any(pattern in label for pattern in ['>= 0', '>=0', 'greater than or equal to 0', 'nonnegative', 'non-negative'])
        has_gtenegativeone = any(pattern in label for pattern in ['>= -1', '>=-1', 'greater than or equal to -1', '>= - 1'])
        has_array_length_minus_one = any(pattern in label for pattern in ['length - 1', 'length-1', 'size - 1', 'size-1'])
        
        # Check surrounding nodes for comparison context
        nodes = cfg_data.get('nodes', [])
        current_idx = next((i for i, n in enumerate(nodes) if n.get('id') == node.get('id')), -1)
        surrounding_labels = []
        if current_idx >= 0:
            # Check previous and next nodes
            for offset in [-2, -1, 1, 2]:
                idx = current_idx + offset
                if 0 <= idx < len(nodes):
                    surrounding_labels.append(nodes[idx].get('label', '').lower())
        
        surrounding_text = ' '.join(surrounding_labels)
        has_strict_positive_context = any(pattern in surrounding_text for pattern in ['> 0', '>0'])
        has_nonnegative_context = any(pattern in surrounding_text for pattern in ['>= 0', '>=0'])
        
        # "Could be zero" detection patterns
        label_lower = label.lower()
        
        # Pattern 1: Array index usage (indices can be 0)
        is_used_as_array_index = (
            ('[' in label or ']' in label or 'array[' in label_lower or 'list[' in label_lower) and
            any(var in label_lower for var in ['index', 'i', 'j', 'k', 'idx', 'pos'])
        )
        
        # Pattern 2: Loop iteration variable (often start at 0)
        is_loop_variable = (
            any(pattern in label_lower for pattern in ['for', 'while', 'iterator', 'iter', 'loop']) and
            any(var in label_lower for var in ['i', 'j', 'k', 'idx', 'index', 'counter'])
        )
        
        # Pattern 3: Subtraction result that could be 0 (length - 1, size - offset)
        is_subtraction_result = any(pattern in label_lower for pattern in [
            ' - ', '- ', 'length -', 'size -', 'count -', '.length -', '.size -'
        ])
        
        # Pattern 4: Parameter used in array access context (check surrounding nodes)
        is_param_in_array_context = False
        if 'parameter' in node_type and current_idx >= 0:
            for offset in [-3, -2, -1, 1, 2, 3]:
                idx = current_idx + offset
                if 0 <= idx < len(nodes):
                    nearby_label = nodes[idx].get('label', '').lower()
                    if '[' in nearby_label and ']' in nearby_label:
                        is_param_in_array_context = True
                        break
        
        # Pattern 5: Comparison with length/size (suggests can start at 0)
        compared_with_length = any(pattern in label_lower for pattern in [
            '< length', '< size', '<= length', '<= size',
            'length >', 'size >', 'length >=', 'size >=',
            '.length', '.size()'
        ])
        
        # Pattern 6: Initialization to 0
        initialized_to_zero = any(pattern in label_lower for pattern in [
            '= 0', '=0', ':= 0', ':=0', 'equals 0', 'equals zero', 'zero'
        ])
        
        # Pattern 7: Used in >= 0 check (explicit nonnegative check)
        used_in_nonnegative_check = (
            has_nonnegative or has_nonnegative_context or
            any(pattern in label_lower for pattern in ['>= 0', '>=0', '>= -1', '>=-1'])
        )
        
        # Pattern 8: Offset/position variable (often can be 0)
        is_offset_or_position = any(pattern in label_lower for pattern in [
            'offset', 'position', 'pos', 'start', 'begin', 'beginning'
        ])
        
        # Aggregated "could be zero" score (emphasized feature)
        could_be_zero_indicators = [
            is_used_as_array_index, is_loop_variable, is_subtraction_result,
            is_param_in_array_context, compared_with_length, initialized_to_zero,
            used_in_nonnegative_check, is_offset_or_position
        ]
        could_be_zero_score = sum(could_be_zero_indicators) / max(len(could_be_zero_indicators), 1)
        
        # Enhanced features for annotation type prediction
        features = [
            float(len(label)),  # label_length
            float(line if line is not None else 0),  # line_number
            float('method' in node_type),  # is_method
            float('field' in node_type),  # is_field
            float('parameter' in node_type),  # is_parameter
            float('variable' in node_type),  # is_variable
            float('positive' in label),  # contains_positive
            float('negative' in label),  # contains_negative
            float('int' in label),  # contains_int
            float('array' in label),  # contains_array
            float('length' in label),  # contains_length
            float('index' in label),  # contains_index
            float('size' in label),  # contains_size
            float('count' in label),  # contains_count
            float('bound' in label),  # contains_bound
            float('string' in label),  # contains_string
            float('collection' in label),  # contains_collection
            float('loop' in label),  # contains_loop
            float('condition' in label),  # contains_condition
            float('return' in label),  # contains_return
            float('call' in label),  # contains_call
            # New semantic pattern features
            float(has_strict_positive),  # has_strict_positive_comparison (> 0)
            float(has_nonnegative),  # has_nonnegative_comparison (>= 0)
            float(has_gtenegativeone),  # has_gtenegativeone_comparison (>= -1)
            float(has_array_length_minus_one),  # is_array_length_minus_one
            float(has_strict_positive_context),  # strict_positive_in_context
            float(has_nonnegative_context),  # nonnegative_in_context
            # "Could be zero" features (emphasized - placed early and scaled)
            float(is_used_as_array_index) * 2.0,  # Scaled for emphasis
            float(is_loop_variable) * 2.0,
            float(is_subtraction_result) * 1.5,
            float(is_param_in_array_context) * 2.0,
            float(compared_with_length) * 1.5,
            float(initialized_to_zero) * 2.0,
            float(used_in_nonnegative_check) * 2.0,
            float(is_offset_or_position) * 1.5,
            float(could_be_zero_score) * 3.0,  # Aggregated score, highly emphasized
        ]
        
        # Add checker-specific value patterns (raw features, will be emphasized during training)
        if include_checker_patterns and self.pattern_detector is not None and self.checker_type is not None:
            checker_patterns = self.pattern_detector.get_pattern_features(node, cfg_data, self.checker_type)
            features.extend(checker_patterns)
        
        return features
    
    def _extract_sql_quotes_features(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract SQL Quotes Checker-specific features"""
        label = node.get('label', '')
        node_type = node.get('type', '').lower()
        label_lower = label.lower()
        
        # Quote-related features
        single_quote_count = label.count("'")
        double_quote_count = label.count('"')
        total_quotes = single_quote_count + double_quote_count
        is_even_quotes = (total_quotes % 2 == 0) if total_quotes > 0 else True
        
        # SQL method patterns
        has_sql_method = any(pattern in label_lower for pattern in [
            'executequery', 'executeprepared', 'executeupdate', 'preparedstatement',
            'statement.execute', 'connection.prepare'
        ])
        
        # String concatenation
        has_concatenation = '+' in label and ('string' in node_type or 'str' in label_lower)
        
        # Prepared statement
        has_prepared = 'preparedstatement' in label_lower or 'preparestatement' in label_lower
        
        # Sanitization
        has_sanitization = any(pattern in label_lower for pattern in [
            'quote(', 'escape(', 'sanitize(', 'escapeSql'
        ])
        
        features = [
            float(len(label)),  # label_length
            float(node.get('line', 0) if node.get('line') is not None else 0),  # line_number
            float('method' in node_type),  # is_method
            float('field' in node_type),  # is_field
            float('parameter' in node_type),  # is_parameter
            float('variable' in node_type),  # is_variable
            float(total_quotes > 0),  # has_quotes
            float(is_even_quotes),  # is_even_quotes
            float(total_quotes),  # quote_count
            float(has_concatenation),  # has_concatenation
            float(has_sql_method),  # has_sql_method
            float(has_sanitization),  # has_sanitization
            float(has_prepared),  # has_prepared_statement
        ]
        
        return features
    
    def _extract_signature_string_features(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract Signature String Checker-specific features"""
        try:
            from signature_string_feature_extractor import SignatureStringFeatureExtractor
            from source_code_feature_extractor import SourceCodeFeatureExtractor
            
            string_feature_extractor = SignatureStringFeatureExtractor()
            source_extractor = SourceCodeFeatureExtractor()
            
            label = node.get('label', '')
            node_type = node.get('type', '')
            line_number = node.get('line', 0)
            if line_number is None:
                line_number = 0
            
            # Try to extract actual string value from source code
            string_value = None
            java_file = cfg_data.get('java_file', '')
            if java_file and line_number > 0:
                try:
                    string_value = source_extractor.extract_string_at_line(java_file, line_number)
                except Exception:
                    pass
            
            # Extract comprehensive features
            features = string_feature_extractor.extract_features(
                string_value=string_value,
                label=label,
                node_type=node_type,
                cfg_data=cfg_data,
                node=node
            )
            
            return features
        except ImportError:
            # Fallback to basic features
            label = node.get('label', '')
            node_type = node.get('type', '').lower()
            
            has_dots = '.' in label and not label.startswith('.')
            has_slashes = '/' in label
            is_field_descriptor = label.startswith('L') and label.endswith(';') and '/' in label
            
            features = [
                float(len(label)),  # label_length
                float(node.get('line', 0) if node.get('line') is not None else 0),  # line_number
                float('method' in node_type),  # is_method
                float('field' in node_type),  # is_field
                float('parameter' in node_type),  # is_parameter
                float('variable' in node_type),  # is_variable
                float(has_dots),  # has_dots
                float(has_slashes),  # has_slashes
                float(is_field_descriptor),  # is_field_descriptor_format
                float(label.count('.')),  # dot_count
                float(label.count('/')),  # slash_count
                float(label.count(';')),  # semicolon_count
            ]
            
            return features
    
    def get_code_context(self, node: Dict, cfg_data: Dict) -> str:
        """Extract the actual code context for a node"""
        label = node.get('label', '')
        node_type = node.get('type', '')
        line = node.get('line', 0)
        
        # Create a meaningful code context string
        context_parts = []
        
        if label:
            context_parts.append(f"Label: {label}")
        if node_type:
            context_parts.append(f"Type: {node_type}")
        if line and line > 0:
            context_parts.append(f"Line: {line}")
        
        # Add context from surrounding nodes if available
        nodes = cfg_data.get('nodes', [])
        if nodes:
            current_idx = next((i for i, n in enumerate(nodes) if n.get('id') == node.get('id')), -1)
            if current_idx >= 0:
                # Add context from previous and next nodes
                if current_idx > 0:
                    prev_node = nodes[current_idx - 1]
                    if prev_node.get('label'):
                        context_parts.append(f"Prev: {prev_node['label'][:30]}...")
                if current_idx < len(nodes) - 1:
                    next_node = nodes[current_idx + 1]
                    if next_node.get('label'):
                        context_parts.append(f"Next: {next_node['label'][:30]}...")
        
        return " | ".join(context_parts)
    
    def determine_annotation_type(self, node: Dict, cfg_data: Dict) -> str:
        """Determine the most appropriate annotation type for a node using enhanced rules"""
        label = node.get('label', '').lower()
        node_type = node.get('type', '').lower()
        
        # Enhanced rule-based annotation type determination
        # Rule 1: Array and index-related annotations
        if any(keyword in label for keyword in ['array', 'index', 'subscript']):
            if 'length' in label or 'size' in label:
                return '@NonNegative'
            elif 'bound' in label or 'limit' in label:
                return '@GTENegativeOne'
            else:
                return '@Positive'
        
        # Rule 2: Loop and iteration variables
        if any(keyword in label for keyword in ['loop', 'iter', 'i', 'j', 'k']):
            if 'array' in label or 'list' in label:
                return '@NonNegative'
            else:
                return '@GTENegativeOne'
        
        # Rule 3: Size and length related (FIXED: Better match Index Checker semantics)
        if any(keyword in label for keyword in ['length', 'size', 'count', 'capacity']):
            # Check for explicit comparison patterns
            if any(pattern in label for pattern in ['> 0', '>0', 'greater than 0']):
                return '@Positive'
            elif any(pattern in label for pattern in ['>= 0', '>=0', 'greater than or equal to 0']):
                return '@NonNegative'
            # Default: parameters often need @NonNegative (can be 0), not @Positive
            elif 'parameter' in node_type:
                return '@NonNegative'  # Changed from @Positive to @NonNegative
            else:
                return '@NonNegative'
        
        # Rule 4: Numeric types and parameters
        if 'parameter' in node_type:
            if any(keyword in label for keyword in ['int', 'long', 'double', 'float']):
                return '@NonNegative'
            else:
                return '@Positive'
        
        # Rule 5: String and collection types
        if any(keyword in label for keyword in ['string', 'list', 'map', 'set']):
            return '@Positive'
        
        # Rule 6: Method calls and complex patterns
        if 'method' in node_type or 'call' in label:
            return '@Positive'
        
        # Rule 7: Variable declarations
        if 'variable' in node_type:
            if any(keyword in label for keyword in ['temp', 'result', 'value']):
                return '@GTENegativeOne'
            else:
                return '@NonNegative'
        
        # Default based on context
        if 'positive' in label:
            return '@Positive'
        elif 'negative' in label:
            return '@GTENegativeOne'
        else:
            return '@NonNegative'
    
    def classify_node_for_annotation_type(self, node: Dict, cfg_data: Dict, target_annotation: str) -> Tuple[bool, float]:
        """
        Classify whether a node should have the target annotation type
        
        Returns:
            (is_positive, confidence): Whether the node needs the annotation and confidence score
        """
        # Use checker-specific classification methods
        if self.checker_name == 'sql_quotes':
            return self.classify_node_for_sql_quotes_annotation(node, cfg_data, target_annotation)
        elif self.checker_name == 'signature_string':
            return self.classify_node_for_signature_string_annotation(node, cfg_data, target_annotation)
        else:
            # Default: Lower Bound Checker classification
            predicted_annotation = self.determine_annotation_type(node, cfg_data)
            
            # Check if the node actually needs this annotation type
            is_positive = (predicted_annotation == target_annotation)
            
            # Calculate confidence based on how well the node matches the target annotation
            label = node.get('label', '').lower()
            node_type = node.get('type', '').lower()
            
            confidence = 0.5  # Base confidence
            
            if target_annotation == '@Positive':
                # Features that suggest @Positive annotation
                if any(keyword in label for keyword in ['size', 'length', 'count', 'capacity']):
                    confidence += 0.3
                if 'parameter' in node_type:
                    confidence += 0.2
                if any(keyword in label for keyword in ['int', 'long', 'double']):
                    confidence += 0.1
            
            elif target_annotation == '@NonNegative':
                # Features that suggest @NonNegative annotation
                if any(keyword in label for keyword in ['index', 'offset', 'position']):
                    confidence += 0.3
                if any(keyword in label for keyword in ['array', 'list']):
                    confidence += 0.2
                if 'parameter' in node_type:
                    confidence += 0.2
            
            elif target_annotation == '@GTENegativeOne':
                # Features that suggest @GTENegativeOne annotation
                if any(keyword in label for keyword in ['bound', 'limit', 'capacity']):
                    confidence += 0.3
                if any(keyword in label for keyword in ['variable', 'temp', 'result']):
                    confidence += 0.2
                if any(keyword in label for keyword in ['count', 'size']):
                    confidence += 0.1
            
            # Ensure confidence is in [0.1, 1.0] range
            confidence = max(0.1, min(1.0, confidence))
            
            return is_positive, confidence
    
    def classify_node_for_sql_quotes_annotation(self, node: Dict, cfg_data: Dict, target_annotation: str) -> Tuple[bool, float]:
        """
        Classify whether a node should have SQL Quotes annotation type
        
        Args:
            node: CFG node dictionary
            cfg_data: Full CFG data dictionary
            target_annotation: '@SqlEvenQuotes' or '@SqlOddQuotes'
            
        Returns:
            (is_positive, confidence): Whether the node needs the annotation and confidence score
        """
        label = node.get('label', '')
        node_type = node.get('type', '').lower()
        label_lower = label.lower()
        
        # Count single quotes in string literals
        single_quote_count = label.count("'")
        double_quote_count = label.count('"')
        total_quotes = single_quote_count + double_quote_count
        
        # Determine quote parity
        is_even_quotes = (total_quotes % 2 == 0) if total_quotes > 0 else True
        
        # Check for SQL-related patterns
        has_sql_method = any(pattern in label_lower for pattern in [
            'executequery', 'executeprepared', 'executeupdate', 'preparedstatement',
            'statement.execute', 'connection.prepare', 'sql', 'query'
        ])
        
        has_string_concatenation = '+' in label and ('string' in node_type or 'str' in label_lower)
        has_prepared_statement = 'preparedstatement' in label_lower or 'preparestatement' in label_lower
        
        # Classify based on target annotation
        if target_annotation == '@SqlEvenQuotes':
            # Positive if even quotes or prepared statement (safe)
            is_positive = is_even_quotes or has_prepared_statement
            confidence = 0.5
            
            if is_even_quotes and total_quotes > 0:
                confidence += 0.3
            if has_prepared_statement:
                confidence += 0.2
            if has_sql_method and is_even_quotes:
                confidence += 0.1
                
        elif target_annotation == '@SqlOddQuotes':
            # Positive if odd quotes (unsafe)
            is_positive = not is_even_quotes and total_quotes > 0
            confidence = 0.5
            
            if not is_even_quotes and total_quotes > 0:
                confidence += 0.3
            if has_string_concatenation and not is_even_quotes:
                confidence += 0.2
            if has_sql_method and not is_even_quotes:
                confidence += 0.1
        else:
            # Unknown annotation type
            is_positive = False
            confidence = 0.1
        
        # Ensure confidence is in [0.1, 1.0] range
        confidence = max(0.1, min(1.0, confidence))
        
        return is_positive, confidence
    
    def classify_node_for_signature_string_annotation(self, node: Dict, cfg_data: Dict, target_annotation: str) -> Tuple[bool, float]:
        """
        Classify whether a node should have Signature String annotation type
        
        Args:
            node: CFG node dictionary
            cfg_data: Full CFG data dictionary
            target_annotation: '@FullyQualifiedName', '@BinaryName', or '@FieldDescriptor'
            
        Returns:
            (is_positive, confidence): Whether the node needs the annotation and confidence score
        """
        label = node.get('label', '')
        node_type = node.get('type', '').lower()
        
        # Try to extract string value from source code if available
        string_value = None
        try:
            from source_code_feature_extractor import SourceCodeFeatureExtractor
            java_file = cfg_data.get('java_file', '')
            line_number = node.get('line', 0)
            if java_file and line_number and line_number > 0:
                source_extractor = SourceCodeFeatureExtractor()
                string_value = source_extractor.extract_string_at_line(java_file, line_number)
        except Exception:
            pass
        
        # If no string value from source, try to extract from label
        if not string_value:
            # Look for string literals in label
            import re
            string_match = re.search(r'["\']([^"\']*)["\']', label)
            if string_match:
                string_value = string_match.group(1)
        
        # Use format detector if available
        try:
            from signature_string_feature_extractor import FormatDetector
            format_detector = FormatDetector()
            format_scores = format_detector.detect_format(string_value or label)
        except Exception:
            format_scores = {
                'fully_qualified_confidence': 0.0,
                'binary_confidence': 0.0,
                'field_descriptor_confidence': 0.0
            }
        
        # Check for reflection API patterns
        has_reflection = any(pattern in label.lower() for pattern in [
            'class.forname', 'getclass', 'getname', 'getsignature',
            'classloader', 'reflection', 'method.invoke'
        ])
        
        # Classify based on target annotation
        if target_annotation == '@FullyQualifiedName':
            # Positive if dotted format (package.Class)
            has_dots = '.' in (string_value or label) and not (string_value or label).startswith('.')
            is_positive = has_dots or format_scores.get('fully_qualified_confidence', 0.0) > 0.5
            confidence = format_scores.get('fully_qualified_confidence', 0.5)
            
            if has_dots:
                confidence += 0.2
            if 'getname' in label.lower() or 'class.getname' in label.lower():
                confidence += 0.1
                
        elif target_annotation == '@BinaryName':
            # Positive if slashed format (package/Class)
            has_slashes = '/' in (string_value or label)
            is_positive = has_slashes or format_scores.get('binary_confidence', 0.0) > 0.5
            confidence = format_scores.get('binary_confidence', 0.5)
            
            if has_slashes:
                confidence += 0.2
            if 'class.forname' in label.lower() or has_reflection:
                confidence += 0.1
                
        elif target_annotation == '@FieldDescriptor':
            # Positive if JVM format (Lpackage/Class;)
            string_val = string_value or label
            is_field_descriptor = string_val.startswith('L') and string_val.endswith(';') and '/' in string_val
            is_positive = is_field_descriptor or format_scores.get('field_descriptor_confidence', 0.0) > 0.5
            confidence = format_scores.get('field_descriptor_confidence', 0.5)
            
            if is_field_descriptor:
                confidence += 0.3
            if has_reflection:
                confidence += 0.1
        else:
            # Unknown annotation type
            is_positive = False
            confidence = 0.1
        
        # Ensure confidence is in [0.1, 1.0] range
        confidence = max(0.1, min(1.0, confidence))
        
        return is_positive, confidence
    
    def generate_balanced_examples(self, cfg_files: List[Dict[str, Any]], 
                                 examples_per_annotation: int = 1000) -> Dict[str, List[RealBalancedExample]]:
        """
        Generate balanced examples using REAL code examples
        
        Args:
            cfg_files: List of CFG file data
            examples_per_annotation: Target number of examples per annotation type
            
        Returns:
            Dictionary mapping annotation types to lists of balanced examples
        """
        balanced_datasets = {ann_type: [] for ann_type in self.annotation_types}
        
        logger.info(f"Generating balanced datasets with {examples_per_annotation} examples per annotation type")
        logger.info(f"Target balance: {self.target_balance*100:.1f} percent positive, {(1-self.target_balance)*100:.1f} percent negative")
        
        for ann_type in self.annotation_types:
            logger.info(f"\nGenerating examples for {ann_type}...")
            
            # Collect all nodes and classify them for this annotation type
            positive_nodes = []
            negative_nodes = []
            
            for cfg_file in cfg_files:
                cfg_data = cfg_file['data']
                for node in cfg_data.get('nodes', []):
                    is_positive, confidence = self.classify_node_for_annotation_type(node, cfg_data, ann_type)
                    
                    node_info = {
                        'node': node,
                        'cfg_data': cfg_data,
                        'file_path': cfg_file['file'],
                        'method_name': cfg_file['method'],
                        'confidence': confidence
                    }
                    
                    if is_positive:
                        positive_nodes.append(node_info)
                    else:
                        negative_nodes.append(node_info)
            
            logger.info(f"Found {len(positive_nodes)} positive and {len(negative_nodes)} negative nodes for {ann_type}")
            
            if len(positive_nodes) == 0 or len(negative_nodes) == 0:
                logger.warning(f"Insufficient nodes for {ann_type}: {len(positive_nodes)} positive, {len(negative_nodes)} negative")
                continue
            
            # Generate positive examples (real nodes that need this annotation)
            num_positive = int(examples_per_annotation * self.target_balance)
            positive_examples = self._generate_real_positive_examples(
                positive_nodes, ann_type, num_positive
            )
            
            # Generate negative examples (real nodes that don't need this annotation)
            num_negative = examples_per_annotation - num_positive
            negative_examples = self._generate_real_negative_examples(
                negative_nodes, ann_type, num_negative
            )
            
            # Combine and shuffle
            all_examples = positive_examples + negative_examples
            random.shuffle(all_examples)
            
            balanced_datasets[ann_type] = all_examples
            
            # Update statistics
            self.generation_stats['annotation_type_counts'][ann_type]['positive'] = len(positive_examples)
            self.generation_stats['annotation_type_counts'][ann_type]['negative'] = len(negative_examples)
            
            logger.info(f"Generated {len(positive_examples)} positive and {len(negative_examples)} negative examples for {ann_type}")
        
        # Update overall statistics
        self.generation_stats['total_examples'] = sum(len(examples) for examples in balanced_datasets.values())
        self.generation_stats['positive_examples'] = sum(
            stats['positive'] for stats in self.generation_stats['annotation_type_counts'].values()
        )
        self.generation_stats['negative_examples'] = sum(
            stats['negative'] for stats in self.generation_stats['annotation_type_counts'].values()
        )
        
        return balanced_datasets
    
    def _generate_real_positive_examples(self, positive_nodes: List[Dict], 
                                       annotation_type: str, num_examples: int) -> List[RealBalancedExample]:
        """Generate positive examples using real nodes that need the annotation"""
        examples = []
        
        # Sample nodes with replacement if needed, prioritizing higher confidence
        positive_nodes_sorted = sorted(positive_nodes, key=lambda x: x['confidence'], reverse=True)
        
        for i in range(num_examples):
            node_info = positive_nodes_sorted[i % len(positive_nodes_sorted)]
            node = node_info['node']
            cfg_data = node_info['cfg_data']
            
            features = self.extract_node_features(node, cfg_data)
            code_context = self.get_code_context(node, cfg_data)
            
            example = RealBalancedExample(
                node_id=node.get('id', i),
                file_path=node_info['file_path'],
                method_name=node_info['method_name'],
                line_number=node.get('line', 0),
                node_type=node.get('type', ''),
                node_label=node.get('label', ''),
                features=features,
                annotation_type=annotation_type,
                is_positive=True,
                confidence=node_info['confidence'],
                code_context=code_context
            )
            
            examples.append(example)
        
        return examples
    
    def _generate_real_negative_examples(self, negative_nodes: List[Dict], 
                                       annotation_type: str, num_examples: int) -> List[RealBalancedExample]:
        """Generate negative examples using real nodes that don't need the annotation"""
        examples = []
        
        # Sample nodes with replacement if needed, prioritizing lower confidence (more certain negatives)
        negative_nodes_sorted = sorted(negative_nodes, key=lambda x: x['confidence'])
        
        for i in range(num_examples):
            node_info = negative_nodes_sorted[i % len(negative_nodes_sorted)]
            node = node_info['node']
            cfg_data = node_info['cfg_data']
            
            features = self.extract_node_features(node, cfg_data)
            code_context = self.get_code_context(node, cfg_data)
            
            example = RealBalancedExample(
                node_id=node.get('id', i),
                file_path=node_info['file_path'],
                method_name=node_info['method_name'],
                line_number=node.get('line', 0),
                node_type=node.get('type', ''),
                node_label=node.get('label', ''),
                features=features,
                annotation_type=annotation_type,
                is_positive=False,
                confidence=node_info['confidence'],
                code_context=code_context
            )
            
            examples.append(example)
        
        return examples
    
    def save_balanced_dataset(self, balanced_datasets: Dict[str, List[RealBalancedExample]], 
                            output_dir: str):
        """Save the balanced datasets to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for ann_type, examples in balanced_datasets.items():
            if not examples:
                continue
            
            # Save as JSON
            output_file = os.path.join(output_dir, f"{ann_type.replace('@', '').lower()}_real_balanced_dataset.json")
            
            dataset_data = {
                'annotation_type': ann_type,
                'total_examples': len(examples),
                'positive_examples': sum(1 for ex in examples if ex.is_positive),
                'negative_examples': sum(1 for ex in examples if not ex.is_positive),
                'balance_ratio': sum(1 for ex in examples if ex.is_positive) / len(examples),
                'examples': [
                    {
                        'node_id': ex.node_id,
                        'file_path': ex.file_path,
                        'method_name': ex.method_name,
                        'line_number': ex.line_number,
                        'node_type': ex.node_type,
                        'node_label': ex.node_label,
                        'features': ex.features,
                        'is_positive': ex.is_positive,
                        'confidence': ex.confidence,
                        'code_context': ex.code_context
                    }
                    for ex in examples
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(dataset_data, f, indent=2)
            
            logger.info(f"Saved {len(examples)} real examples for {ann_type} to {output_file}")
        
        # Save overall statistics
        stats_file = os.path.join(output_dir, "real_generation_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(self.generation_stats, f, indent=2)
        
        logger.info(f"Saved generation statistics to {stats_file}")
    
    def print_statistics(self):
        """Print generation statistics"""
        print("\n" + "="*60)
        print("REAL BALANCED DATASET GENERATION STATISTICS")
        print("="*60)
        
        print(f"Total examples generated: {self.generation_stats['total_examples']}")
        print(f"Overall balance: {self.generation_stats['positive_examples']} positive, {self.generation_stats['negative_examples']} negative")
        
        if self.generation_stats['total_examples'] > 0:
            overall_balance = self.generation_stats['positive_examples'] / self.generation_stats['total_examples']
            print(f"Overall balance ratio: {overall_balance:.3f} (target: {self.target_balance:.3f})")
        
        print("\nPer-annotation-type statistics:")
        for ann_type, stats in self.generation_stats['annotation_type_counts'].items():
            total = stats['positive'] + stats['negative']
            if total > 0:
                balance = stats['positive'] / total
                print(f"  {ann_type}: {stats['positive']} positive, {stats['negative']} negative (balance: {balance:.3f})")
        
        print("="*60)


def main():
    """Main function to generate balanced datasets using real code examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate balanced training datasets using real code examples')
    parser.add_argument('--cfg_dir', required=True, help='Directory containing CFG files')
    parser.add_argument('--output_dir', required=True, help='Output directory for balanced datasets')
    parser.add_argument('--examples_per_annotation', type=int, default=1000, 
                       help='Number of examples to generate per annotation type')
    parser.add_argument('--target_balance', type=float, default=0.5,
                       help='Target balance ratio for positive examples (0.5 = 50 percent positive)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible results')
    parser.add_argument('--checker_name', type=str, default='lower_bound',
                       choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       help='Checker name to determine annotation types')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ImprovedBalancedDatasetGenerator(
        target_balance=args.target_balance,
        random_seed=args.random_seed,
        checker_name=args.checker_name
    )
    
    # Load CFG files
    cfg_files = generator.load_cfg_files(args.cfg_dir)
    
    if not cfg_files:
        logger.error("No CFG files found. Exiting.")
        return 1
    
    # Generate balanced datasets
    balanced_datasets = generator.generate_balanced_examples(
        cfg_files, 
        examples_per_annotation=args.examples_per_annotation
    )
    
    # Save datasets
    generator.save_balanced_dataset(balanced_datasets, args.output_dir)
    
    # Print statistics
    generator.print_statistics()
    
    return 0


if __name__ == '__main__':
    exit(main())
