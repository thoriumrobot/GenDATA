#!/usr/bin/env python3
"""
Signature String Feature Extractor

This module provides comprehensive string feature extraction for Signature String Checker.
It analyzes string values to extract features that help distinguish between
@FullyQualifiedName, @BinaryName, and @FieldDescriptor annotation types.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class StringPatternAnalyzer:
    """Analyzes string literals and expressions for format patterns"""
    
    def analyze_patterns(self, string_value: str) -> Dict[str, float]:
        """Analyze character-level patterns in string"""
        if not string_value:
            return self._empty_patterns()
        
        patterns = {
            'dot_count': float(string_value.count('.')),
            'slash_count': float(string_value.count('/')),
            'semicolon_count': float(string_value.count(';')),
            'capital_letter_count': float(sum(1 for c in string_value if c.isupper())),
            'lowercase_letter_count': float(sum(1 for c in string_value if c.islower())),
            'digit_count': float(sum(1 for c in string_value if c.isdigit())),
            'bracket_count': float(string_value.count('[') + string_value.count(']')),
            'paren_count': float(string_value.count('(') + string_value.count(')')),
            'string_length': float(len(string_value)),
        }
        
        return patterns
    
    def _empty_patterns(self) -> Dict[str, float]:
        """Return empty pattern dictionary"""
        return {
            'dot_count': 0.0,
            'slash_count': 0.0,
            'semicolon_count': 0.0,
            'capital_letter_count': 0.0,
            'lowercase_letter_count': 0.0,
            'digit_count': 0.0,
            'bracket_count': 0.0,
            'paren_count': 0.0,
            'string_length': 0.0,
        }

class FormatDetector:
    """Detects which format a string matches (FullyQualifiedName, BinaryName, FieldDescriptor)"""
    
    def detect_format(self, string_value: str) -> Dict[str, float]:
        """Detect format and return confidence scores"""
        if not string_value:
            return self._empty_format()
        
        # Format indicators
        has_dots = '.' in string_value and not string_value.startswith('.')
        has_slashes = '/' in string_value
        is_field_descriptor = string_value.startswith('L') and string_value.endswith(';') and '/' in string_value
        
        # Calculate format confidence scores
        fully_qualified_score = self._score_fully_qualified(string_value)
        binary_score = self._score_binary(string_value)
        field_descriptor_score = self._score_field_descriptor(string_value)
        
        # Normalize scores
        total_score = fully_qualified_score + binary_score + field_descriptor_score
        if total_score > 0:
            fully_qualified_score /= total_score
            binary_score /= total_score
            field_descriptor_score /= total_score
        
        # Format ambiguity (multiple formats possible)
        scores = [fully_qualified_score, binary_score, field_descriptor_score]
        max_score = max(scores)
        ambiguity = 1.0 - max_score if max_score > 0 else 1.0
        
        # Format transition indicators
        has_dot_to_slash = '.' in string_value and '/' in string_value
        has_slash_to_dot = '/' in string_value and '.' in string_value
        
        return {
            'has_dots': 1.0 if has_dots else 0.0,
            'has_slashes': 1.0 if has_slashes else 0.0,
            'is_field_descriptor_format': 1.0 if is_field_descriptor else 0.0,
            'fully_qualified_confidence': fully_qualified_score,
            'binary_confidence': binary_score,
            'field_descriptor_confidence': field_descriptor_score,
            'format_ambiguity': ambiguity,
            'has_dot_to_slash_transition': 1.0 if has_dot_to_slash else 0.0,
        }
    
    def _score_fully_qualified(self, string_value: str) -> float:
        """Score how well string matches FullyQualifiedName format (dotted)"""
        score = 0.0
        
        # Must have dots
        if '.' in string_value:
            score += 2.0
        
        # Should not have slashes (unless it's a mixed format)
        if '/' not in string_value:
            score += 1.0
        
        # Should not start with L or end with ;
        if not string_value.startswith('L') and not string_value.endswith(';'):
            score += 1.0
        
        # Should have package-like structure (lowercase segments)
        segments = string_value.split('.')
        if len(segments) > 1:
            score += 1.0
            # Check if segments look like package names (lowercase) and class names (Capitalized)
            for i, seg in enumerate(segments[:-1]):  # Package segments
                if seg and seg[0].islower():
                    score += 0.5
            if segments[-1] and segments[-1][0].isupper():  # Class name
                score += 0.5
        
        return score
    
    def _score_binary(self, string_value: str) -> float:
        """Score how well string matches BinaryName format (slashed)"""
        score = 0.0
        
        # Must have slashes
        if '/' in string_value:
            score += 2.0
        
        # Should not have dots (unless it's a mixed format)
        if '.' not in string_value:
            score += 1.0
        
        # Should not start with L or end with ; (unless it's a field descriptor)
        if not string_value.startswith('L') and not string_value.endswith(';'):
            score += 1.0
        
        # Should have package-like structure (slashed segments)
        segments = string_value.split('/')
        if len(segments) > 1:
            score += 1.0
        
        return score
    
    def _score_field_descriptor(self, string_value: str) -> float:
        """Score how well string matches FieldDescriptor format (L...;)"""
        score = 0.0
        
        # Must start with L and end with ;
        if string_value.startswith('L') and string_value.endswith(';'):
            score += 3.0
        
        # Should have slashes (internal format)
        if '/' in string_value:
            score += 1.0
        
        # Should have dots removed (converted to slashes)
        if '.' not in string_value:
            score += 1.0
        
        # Check for primitive type indicators
        primitive_types = ['I', 'J', 'D', 'F', 'Z', 'B', 'C', 'S']
        if string_value in primitive_types:
            score += 2.0
        
        return score
    
    def _empty_format(self) -> Dict[str, float]:
        """Return empty format detection dictionary"""
        return {
            'has_dots': 0.0,
            'has_slashes': 0.0,
            'is_field_descriptor_format': 0.0,
            'fully_qualified_confidence': 0.0,
            'binary_confidence': 0.0,
            'field_descriptor_confidence': 0.0,
            'format_ambiguity': 1.0,
            'has_dot_to_slash_transition': 0.0,
        }

class StructuralAnalyzer:
    """Extracts structural features (package depth, class name patterns, etc.)"""
    
    def analyze_structure(self, string_value: str) -> Dict[str, float]:
        """Analyze structural features of string"""
        if not string_value:
            return self._empty_structure()
        
        # Remove field descriptor markers for analysis
        clean_value = string_value
        if clean_value.startswith('L') and clean_value.endswith(';'):
            clean_value = clean_value[1:-1]
        
        # Package depth (number of segments)
        dot_segments = clean_value.split('.')
        slash_segments = clean_value.split('/')
        package_depth_dots = len(dot_segments) - 1 if '.' in clean_value else 0
        package_depth_slashes = len(slash_segments) - 1 if '/' in clean_value else 0
        package_depth = max(package_depth_dots, package_depth_slashes)
        
        # Class name (last segment)
        if '.' in clean_value:
            class_name = dot_segments[-1] if dot_segments else ''
        elif '/' in clean_value:
            class_name = slash_segments[-1] if slash_segments else ''
        else:
            class_name = clean_value
        
        class_name_length = len(class_name)
        
        # Array indicators
        has_array_brackets = '[' in string_value
        
        # Method descriptor indicators
        has_method_descriptor = '(' in string_value and ')' in string_value
        
        # Primitive type indicators
        primitive_indicators = ['I', 'J', 'D', 'F', 'Z', 'B', 'C', 'S']
        has_primitive_type = any(pt in string_value for pt in primitive_indicators)
        
        # Object type indicators (L...;)
        has_object_type = string_value.startswith('L') and string_value.endswith(';')
        
        # Segment count
        if '.' in clean_value:
            segment_count = len(dot_segments)
        elif '/' in clean_value:
            segment_count = len(slash_segments)
        else:
            segment_count = 1
        
        return {
            'package_depth': float(package_depth),
            'class_name_length': float(class_name_length),
            'has_array_brackets': 1.0 if has_array_brackets else 0.0,
            'has_method_descriptor': 1.0 if has_method_descriptor else 0.0,
            'has_primitive_type': 1.0 if has_primitive_type else 0.0,
            'has_object_type': 1.0 if has_object_type else 0.0,
            'segment_count': float(segment_count),
        }
    
    def _empty_structure(self) -> Dict[str, float]:
        """Return empty structure dictionary"""
        return {
            'package_depth': 0.0,
            'class_name_length': 0.0,
            'has_array_brackets': 0.0,
            'has_method_descriptor': 0.0,
            'has_primitive_type': 0.0,
            'has_object_type': 0.0,
            'segment_count': 0.0,
        }

class ContextAnalyzer:
    """Analyzes how strings are used (Class.forName, Class.getName, method parameters, etc.)"""
    
    def analyze_context(self, label: str, node_type: str, cfg_data: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, float]:
        """Analyze context features from CFG node and surrounding code"""
        label_lower = label.lower()
        
        # Context features from label analysis
        used_in_forname = 'class.forname' in label_lower or 'forname(' in label_lower
        used_in_getname = 'class.getname' in label_lower or '.getname(' in label_lower
        used_as_parameter = 'parameter' in node_type.lower()
        used_as_return = 'return' in label_lower or 'return' in node_type.lower()
        used_as_field = 'field' in node_type.lower()
        
        # Reflection API calls
        reflection_patterns = [
            'class.forname', 'class.getclass', 'class.getdeclaredmethod',
            'method.invoke', 'constructor.newinstance'
        ]
        used_in_reflection = any(pattern in label_lower for pattern in reflection_patterns)
        
        # Type conversion context
        type_conversion_patterns = [
            'class.cast', 'class.asSubclass', 'class.getComponentType'
        ]
        used_in_type_conversion = any(pattern in label_lower for pattern in type_conversion_patterns)
        
        return {
            'used_in_forname': 1.0 if used_in_forname else 0.0,
            'used_in_getname': 1.0 if used_in_getname else 0.0,
            'used_as_parameter': 1.0 if used_as_parameter else 0.0,
            'used_as_return': 1.0 if used_as_return else 0.0,
            'used_in_reflection': 1.0 if used_in_reflection else 0.0,
            'used_in_type_conversion': 1.0 if used_in_type_conversion else 0.0,
        }

class SignatureStringFeatureExtractor:
    """Main interface for extracting comprehensive string features"""
    
    def __init__(self):
        self.pattern_analyzer = StringPatternAnalyzer()
        self.format_detector = FormatDetector()
        self.structural_analyzer = StructuralAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def extract_features(self, string_value: Optional[str], label: str, node_type: str, 
                        cfg_data: Dict[str, Any], node: Dict[str, Any]) -> List[float]:
        """
        Extract comprehensive string features (20-30 features).
        
        Args:
            string_value: Actual string value from source code (if available)
            label: CFG node label
            node_type: CFG node type
            cfg_data: Complete CFG data
            node: Individual CFG node
            
        Returns:
            List of feature values (floats)
        """
        features = []
        
        # Use string_value if available, otherwise try to extract from label
        if not string_value:
            # Try to extract string from label
            string_value = self._extract_string_from_label(label)
        
        # Format Detection Features (6 features)
        format_features = self.format_detector.detect_format(string_value or '')
        features.extend([
            format_features['has_dots'],
            format_features['has_slashes'],
            format_features['is_field_descriptor_format'],
            format_features['fully_qualified_confidence'],
            format_features['format_ambiguity'],
            format_features['has_dot_to_slash_transition'],
        ])
        
        # Structural Features (8 features)
        structural_features = self.structural_analyzer.analyze_structure(string_value or '')
        features.extend([
            structural_features['package_depth'],
            structural_features['class_name_length'],
            structural_features['has_array_brackets'],
            structural_features['has_method_descriptor'],
            structural_features['has_primitive_type'],
            structural_features['has_object_type'],
            structural_features['segment_count'],
            float(len(string_value)) if string_value else 0.0,  # String length
        ])
        
        # Pattern Features (6 features)
        pattern_features = self.pattern_analyzer.analyze_patterns(string_value or '')
        features.extend([
            pattern_features['dot_count'],
            pattern_features['slash_count'],
            pattern_features['semicolon_count'],
            pattern_features['capital_letter_count'],
            pattern_features['lowercase_letter_count'],
            pattern_features['bracket_count'] + pattern_features['paren_count'],  # Special char count
        ])
        
        # Context Features (6 features)
        context_features = self.context_analyzer.analyze_context(label, node_type, cfg_data, node)
        features.extend([
            context_features['used_in_forname'],
            context_features['used_in_getname'],
            context_features['used_as_parameter'],
            context_features['used_as_return'],
            context_features['used_in_reflection'],
            context_features['used_in_type_conversion'],
        ])
        
        # CFG Context Features (4 features)
        node_id = node.get('id', 0)
        control_edges = cfg_data.get('control_edges', [])
        dataflow_edges = cfg_data.get('dataflow_edges', [])
        
        in_degree = float(sum(1 for edge in control_edges if edge.get('target') == node_id))
        out_degree = float(sum(1 for edge in control_edges if edge.get('source') == node_id))
        dataflow_in = float(sum(1 for edge in dataflow_edges if edge.get('target') == node_id))
        dataflow_out = float(sum(1 for edge in dataflow_edges if edge.get('source') == node_id))
        
        # Node type encoding
        node_type_encoded = self._encode_node_type(node_type)
        
        features.extend([
            node_type_encoded,
            in_degree,
            out_degree,
            dataflow_in + dataflow_out,  # Total dataflow connections
        ])
        
        # Ensure we have exactly 30 features
        while len(features) < 30:
            features.append(0.0)
        if len(features) > 30:
            features = features[:30]
        
        return features
    
    def _extract_string_from_label(self, label: str) -> Optional[str]:
        """Try to extract string value from CFG label"""
        # Look for string literals in label
        string_pattern = r'["\']([^"\']+)["\']'
        match = re.search(string_pattern, label)
        if match:
            return match.group(1)
        
        # Look for Class.forName patterns
        forname_pattern = r'Class\.forName\(["\']([^"\']+)["\']\)'
        match = re.search(forname_pattern, label, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return None
    
    def _encode_node_type(self, node_type: str) -> float:
        """Encode node type as numeric feature"""
        node_type_lower = node_type.lower()
        if 'parameter' in node_type_lower:
            return 1.0
        elif 'variable' in node_type_lower:
            return 2.0
        elif 'return' in node_type_lower:
            return 3.0
        elif 'field' in node_type_lower:
            return 4.0
        else:
            return 0.0

