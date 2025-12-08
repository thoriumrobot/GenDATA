#!/usr/bin/env python3
"""
Value Pattern Detector

Detects checker-relevant value patterns in CFG nodes for automatic emphasis learning.
Extracts raw value features that will be passed through learnable attention mechanisms.
"""

import re
from typing import Dict, List, Any
from checker_config import CheckerType, get_checker_config

class ValuePatternDetector:
    """Detect checker-relevant value patterns in code"""
    
    def __init__(self):
        self.pattern_cache = {}
    
    def detect_patterns(self, node: Dict, cfg_data: Dict, checker_type: CheckerType) -> Dict[str, float]:
        """
        Detect value patterns for a specific checker type
        
        Args:
            node: CFG node dictionary
            cfg_data: Full CFG data dictionary
            checker_type: Type of checker to detect patterns for
            
        Returns:
            Dictionary of pattern name -> feature value (0.0 or 1.0)
        """
        if checker_type == CheckerType.INDEX:
            return self.detect_lower_bound_patterns(node, cfg_data)
        elif checker_type == CheckerType.NULLNESS:
            return self.detect_null_patterns(node, cfg_data)
        elif checker_type == CheckerType.SIGNATURE:
            return self.detect_string_patterns(node, cfg_data)
        elif checker_type == CheckerType.INTERNING:
            return self.detect_interning_patterns(node, cfg_data)
        elif checker_type == CheckerType.LOCK:
            return self.detect_lock_patterns(node, cfg_data)
        elif checker_type == CheckerType.REGEX:
            return self.detect_regex_patterns(node, cfg_data)
        else:
            return {}
    
    def detect_lower_bound_patterns(self, node: Dict, cfg_data: Dict) -> Dict[str, float]:
        """Detect 0, -1, positive, nonnegative patterns for Lower Bound Checker"""
        label = node.get('label', '')
        label_lower = label.lower()
        node_type = str(node.get('node_type', '')).lower()
        
        # Pattern 1: Zero detection
        has_zero = float(
            '0' in label or 
            'zero' in label_lower or 
            '= 0' in label or 
            '=0' in label or
            'equals 0' in label_lower
        )
        
        # Pattern 2: Negative one detection
        has_negative_one = float(
            '-1' in label or 
            'negative one' in label_lower or
            '>= -1' in label or
            '>=-1' in label or
            'greater than or equal to -1' in label_lower
        )
        
        # Pattern 3: Positive number patterns
        has_positive = float(
            '> 0' in label or 
            '>0' in label or
            'positive' in label_lower or
            'strictly positive' in label_lower or
            'greater than 0' in label_lower
        )
        
        # Pattern 4: Nonnegative patterns
        has_nonnegative = float(
            '>= 0' in label or 
            '>=0' in label or
            'nonnegative' in label_lower or
            'non-negative' in label_lower or
            'greater than or equal to 0' in label_lower
        )
        
        # Pattern 5: Array index usage
        is_array_index = float(
            ('[' in label or ']' in label) and
            any(var in label_lower for var in ['index', 'i', 'j', 'k', 'idx', 'pos'])
        )
        
        # Pattern 6: Loop variable
        is_loop_variable = float(
            any(pattern in label_lower for pattern in ['for', 'while', 'iterator', 'iter', 'loop']) and
            any(var in label_lower for var in ['i', 'j', 'k', 'idx', 'index', 'counter'])
        )
        
        # Pattern 7: Subtraction result
        is_subtraction_result = float(
            any(pattern in label_lower for pattern in [
                ' - ', '- ', 'length -', 'size -', 'count -', '.length -', '.size -'
            ])
        )
        
        # Pattern 8: Comparison with length/size
        compared_with_length = float(
            any(pattern in label_lower for pattern in [
                '< length', '< size', '<= length', '<= size',
                'length >', 'size >', 'length >=', 'size >=',
                '.length', '.size()'
            ])
        )
        
        # Pattern 9: Initialization to zero
        initialized_to_zero = float(
            any(pattern in label_lower for pattern in [
                '= 0', '=0', ':= 0', ':=0', 'equals 0', 'equals zero'
            ])
        )
        
        # Pattern 10: Offset/position variable
        is_offset_position = float(
            any(pattern in label_lower for pattern in [
                'offset', 'position', 'pos', 'start', 'begin', 'beginning'
            ])
        )
        
        return {
            'zero': has_zero,
            'negative_one': has_negative_one,
            'positive': has_positive,
            'nonnegative': has_nonnegative,
            'array_index': is_array_index,
            'loop_variable': is_loop_variable,
            'subtraction_result': is_subtraction_result,
            'compared_with_length': compared_with_length,
            'initialized_to_zero': initialized_to_zero,
            'offset_position': is_offset_position
        }
    
    def detect_null_patterns(self, node: Dict, cfg_data: Dict) -> Dict[str, float]:
        """Detect null-related patterns for Null Checker"""
        label = node.get('label', '')
        label_lower = label.lower()
        node_type = str(node.get('node_type', '')).lower()
        
        # Pattern 1: Null literal
        has_null_literal = float(
            'null' in label_lower and
            not any(word in label_lower for word in ['nullable', 'nonnull', 'notnull'])
        )
        
        # Pattern 2: Null check
        has_null_check = float(
            '== null' in label or
            '!= null' in label or
            'equals null' in label_lower or
            'not equals null' in label_lower
        )
        
        # Pattern 3: Nullable type annotation
        has_nullable_type = float(
            '@Nullable' in label or
            '@NonNull' in label or
            'nullable' in label_lower or
            'nonnull' in label_lower
        )
        
        # Pattern 4: Null assignment
        has_null_assignment = float(
            '= null' in label or
            ':= null' in label or
            'equals null' in label_lower
        )
        
        # Pattern 5: Null return
        has_null_return = float(
            'return null' in label_lower or
            'returning null' in label_lower
        )
        
        # Pattern 6: Null parameter
        has_null_parameter = float(
            'parameter' in node_type and
            ('null' in label_lower or 'nullable' in label_lower)
        )
        
        # Pattern 7: Null comparison
        has_null_comparison = float(
            any(pattern in label for pattern in [
                '== null', '!= null', 'equals(null)', '!equals(null)'
            ])
        )
        
        # Pattern 8: Null dereference (potential)
        has_null_dereference = float(
            '.' in label and
            ('null' in label_lower or 'nullable' in label_lower)
        )
        
        return {
            'null_literal': has_null_literal,
            'null_check': has_null_check,
            'nullable_type': has_nullable_type,
            'null_assignment': has_null_assignment,
            'null_return': has_null_return,
            'null_parameter': has_null_parameter,
            'null_comparison': has_null_comparison,
            'null_dereference': has_null_dereference
        }
    
    def detect_string_patterns(self, node: Dict, cfg_data: Dict) -> Dict[str, float]:
        """Detect string-related patterns for Signature String Checker"""
        label = node.get('label', '')
        label_lower = label.lower()
        node_type = str(node.get('node_type', '')).lower()
        
        # Pattern 1: String literal
        has_string_literal = float(
            '"' in label or
            "'" in label or
            'string literal' in label_lower
        )
        
        # Pattern 2: String operation
        has_string_operation = float(
            any(pattern in label for pattern in [
                '.length()', '.substring', '.charAt', '.indexOf',
                '.equals(', '.compareTo', '.startsWith', '.endsWith',
                '.concat(', '.replace(', '.split('
            ])
        )
        
        # Pattern 3: Signature pattern
        has_signature_pattern = float(
            'signature' in label_lower or
            'class' in label_lower or
            'method' in label_lower or
            'getname' in label_lower or
            'getclass' in label_lower
        )
        
        # Pattern 4: Class name
        has_class_name = float(
            'class' in node_type or
            'classname' in label_lower or
            'getclass' in label_lower or
            '.class' in label
        )
        
        # Pattern 5: Method name
        has_method_name = float(
            'method' in node_type or
            'methodname' in label_lower or
            'getmethod' in label_lower
        )
        
        # Pattern 6: Fully qualified name
        has_fully_qualified_name = float(
            '.' in label and
            ('package' in label_lower or 'qualified' in label_lower or
             'fqn' in label_lower or label.count('.') >= 2)
        )
        
        # Pattern 7: String concatenation
        has_string_concatenation = float(
            '+' in label and
            ('string' in label_lower or '"' in label or "'" in label)
        )
        
        # Pattern 8: String comparison
        has_string_comparison = float(
            any(pattern in label for pattern in [
                '.equals(', '.compareTo(', '.equalsIgnoreCase(',
                '== "' in label, '!= "' in label
            ])
        )
        
        return {
            'string_literal': has_string_literal,
            'string_operation': has_string_operation,
            'signature_pattern': has_signature_pattern,
            'class_name': has_class_name,
            'method_name': has_method_name,
            'fully_qualified_name': has_fully_qualified_name,
            'string_concatenation': has_string_concatenation,
            'string_comparison': has_string_comparison
        }
    
    def detect_interning_patterns(self, node: Dict, cfg_data: Dict) -> Dict[str, float]:
        """Detect interning-related patterns for Interning Checker"""
        label = node.get('label', '')
        label_lower = label.lower()
        
        # Pattern 1: Interned string
        has_interned_string = float(
            '.intern()' in label or
            'interned' in label_lower
        )
        
        # Pattern 2: String constant
        has_string_constant = float(
            'constant' in label_lower and
            ('string' in label_lower or '"' in label)
        )
        
        # Pattern 3: String comparison
        has_string_comparison = float(
            any(pattern in label for pattern in [
                '.equals(', '== "' in label, '!= "' in label,
                '.compareTo(', '.equalsIgnoreCase('
            ])
        )
        
        # Pattern 4: Intern method call
        has_intern_method_call = float(
            '.intern()' in label or
            'intern(' in label_lower
        )
        
        # Pattern 5: String literal
        has_string_literal = float(
            '"' in label or "'" in label
        )
        
        # Pattern 6: Constant string
        has_constant_string = float(
            'final' in label_lower and
            ('string' in label_lower or '"' in label)
        )
        
        return {
            'interned_string': has_interned_string,
            'string_constant': has_string_constant,
            'string_comparison': has_string_comparison,
            'intern_method_call': has_intern_method_call,
            'string_literal': has_string_literal,
            'constant_string': has_constant_string
        }
    
    def detect_lock_patterns(self, node: Dict, cfg_data: Dict) -> Dict[str, float]:
        """Detect lock-related patterns for Lock Checker"""
        label = node.get('label', '')
        label_lower = label.lower()
        node_type = str(node.get('node_type', '')).lower()
        
        # Pattern 1: Lock operation
        has_lock_operation = float(
            any(pattern in label_lower for pattern in [
                'lock', 'unlock', 'acquire', 'release',
                'lock()', 'unlock()', 'trylock'
            ])
        )
        
        # Pattern 2: Synchronized block
        has_synchronized_block = float(
            'synchronized' in label_lower or
            'synchronize' in label_lower
        )
        
        # Pattern 3: Lock variable
        has_lock_variable = float(
            'lock' in label_lower and
            ('variable' in node_type or 'field' in node_type)
        )
        
        # Pattern 4: Lock acquire
        has_lock_acquire = float(
            any(pattern in label_lower for pattern in [
                'lock()', 'acquire', 'trylock', 'lock.lock'
            ])
        )
        
        # Pattern 5: Lock release
        has_lock_release = float(
            any(pattern in label_lower for pattern in [
                'unlock()', 'release', 'lock.unlock'
            ])
        )
        
        # Pattern 6: Synchronization pattern
        has_synchronization_pattern = float(
            any(pattern in label_lower for pattern in [
                'synchronized', 'mutex', 'semaphore', 'monitor',
                'guarded', 'holding', 'releases'
            ])
        )
        
        return {
            'lock_operation': has_lock_operation,
            'synchronized_block': has_synchronized_block,
            'lock_variable': has_lock_variable,
            'lock_acquire': has_lock_acquire,
            'lock_release': has_lock_release,
            'synchronization_pattern': has_synchronization_pattern
        }
    
    def detect_regex_patterns(self, node: Dict, cfg_data: Dict) -> Dict[str, float]:
        """Detect regex-related patterns for Regex Checker"""
        label = node.get('label', '')
        label_lower = label.lower()
        
        # Pattern 1: Regex pattern
        has_regex_pattern = float(
            'regex' in label_lower or
            'pattern' in label_lower or
            re.search(r'/[^/]+/', label) is not None  # Regex literal like /pattern/
        )
        
        # Pattern 2: Pattern matching
        has_pattern_matching = float(
            any(pattern in label_lower for pattern in [
                '.matches(', '.find(', '.replaceall(',
                'pattern.compile', 'matcher'
            ])
        )
        
        # Pattern 3: String pattern
        has_string_pattern = float(
            'pattern' in label_lower and
            ('string' in label_lower or '"' in label)
        )
        
        # Pattern 4: Regex literal
        has_regex_literal = float(
            re.search(r'/[^/]+/', label) is not None or
            (label.startswith('"') and '.*' in label) or
            (label.startswith("'") and '.*' in label)
        )
        
        # Pattern 5: Pattern compile
        has_pattern_compile = float(
            'pattern.compile' in label_lower or
            'compile(' in label_lower
        )
        
        # Pattern 6: Matcher operation
        has_matcher_operation = float(
            any(pattern in label_lower for pattern in [
                'matcher', '.matches(', '.find(', '.group(',
                '.replaceall(', '.replacefirst('
            ])
        )
        
        return {
            'regex_pattern': has_regex_pattern,
            'pattern_matching': has_pattern_matching,
            'string_pattern': has_string_pattern,
            'regex_literal': has_regex_literal,
            'pattern_compile': has_pattern_compile,
            'matcher_operation': has_matcher_operation
        }
    
    def get_pattern_features(self, node: Dict, cfg_data: Dict, checker_type: CheckerType) -> List[float]:
        """
        Get pattern features as a list of floats for a specific checker
        
        Args:
            node: CFG node dictionary
            cfg_data: Full CFG data dictionary
            checker_type: Type of checker
            
        Returns:
            List of feature values in order defined by checker config
        """
        patterns = self.detect_patterns(node, cfg_data, checker_type)
        config = get_checker_config(checker_type)
        pattern_names = config.get('value_patterns', [])
        
        # Return features in the order specified by config
        return [patterns.get(name, 0.0) for name in pattern_names]

