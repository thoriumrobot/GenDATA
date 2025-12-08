#!/usr/bin/env python3
"""
Checker Configuration Module

Defines checker-specific configurations including target values, annotation types,
and value patterns for automatic emphasis learning.
"""

from enum import Enum
from typing import Dict, List, Any

class CheckerType(Enum):
    """Types of Checker Framework checkers"""
    NULLNESS = "org.checkerframework.checker.nullness.NullnessChecker"
    INDEX = "org.checkerframework.checker.index.IndexChecker"
    INTERNING = "org.checkerframework.checker.interning.InterningChecker"
    LOCK = "org.checkerframework.checker.lock.LockChecker"
    REGEX = "org.checkerframework.checker.regex.RegexChecker"
    SIGNATURE = "org.checkerframework.checker.signature.SignatureChecker"

# Checker-specific configurations
CHECKER_CONFIGS: Dict[CheckerType, Dict[str, Any]] = {
    CheckerType.INDEX: {
        'name': 'Lower Bound Checker',
        'target_values': ['0', '-1'],
        'annotation_types': ['@Positive', '@NonNegative', '@GTENegativeOne'],
        'value_patterns': [
            'zero', 'negative_one', 'positive', 'nonnegative',
            'array_index', 'loop_variable', 'subtraction_result',
            'compared_with_length', 'initialized_to_zero', 'offset_position'
        ],
        'description': 'Emphasizes values 0 and -1, and patterns indicating nonnegative semantics'
    },
    CheckerType.NULLNESS: {
        'name': 'Null Checker',
        'target_values': ['null'],
        'annotation_types': ['@Nullable', '@NonNull'],
        'value_patterns': [
            'null_literal', 'null_check', 'nullable_type',
            'null_assignment', 'null_return', 'null_parameter',
            'null_comparison', 'null_dereference'
        ],
        'description': 'Emphasizes null literals and null-related patterns'
    },
    CheckerType.SIGNATURE: {
        'name': 'Signature String Checker',
        'target_values': ['string'],
        'annotation_types': ['@ClassGetName', '@MethodGetName', '@FullyQualifiedName'],
        'value_patterns': [
            'string_literal', 'string_operation', 'signature_pattern',
            'class_name', 'method_name', 'fully_qualified_name',
            'string_concatenation', 'string_comparison'
        ],
        'description': 'Emphasizes string literals and signature-related patterns'
    },
    CheckerType.INTERNING: {
        'name': 'Interning Checker',
        'target_values': ['interned_string'],
        'annotation_types': ['@Interned', '@InternedDistinct'],
        'value_patterns': [
            'interned_string', 'string_constant', 'string_comparison',
            'intern_method_call', 'string_literal', 'constant_string'
        ],
        'description': 'Emphasizes interned strings and string constants'
    },
    CheckerType.LOCK: {
        'name': 'Lock Checker',
        'target_values': ['lock'],
        'annotation_types': ['@GuardedBy', '@Holding', '@ReleasesNoLocks'],
        'value_patterns': [
            'lock_operation', 'synchronized_block', 'lock_variable',
            'lock_acquire', 'lock_release', 'synchronization_pattern'
        ],
        'description': 'Emphasizes lock operations and synchronization patterns'
    },
    CheckerType.REGEX: {
        'name': 'Regex Checker',
        'target_values': ['regex_pattern'],
        'annotation_types': ['@Regex', '@RegexBottom'],
        'value_patterns': [
            'regex_pattern', 'pattern_matching', 'string_pattern',
            'regex_literal', 'pattern_compile', 'matcher_operation'
        ],
        'description': 'Emphasizes regex patterns and pattern matching operations'
    }
}

def get_checker_config(checker_type: CheckerType) -> Dict[str, Any]:
    """Get configuration for a specific checker"""
    return CHECKER_CONFIGS.get(checker_type, {})

def get_all_checker_types() -> List[CheckerType]:
    """Get list of all supported checker types"""
    return list(CheckerType)

def get_checker_by_name(name: str) -> CheckerType:
    """Get checker type by name (case-insensitive)"""
    name_lower = name.lower()
    for checker_type, config in CHECKER_CONFIGS.items():
        if config['name'].lower() == name_lower:
            return checker_type
    # Try enum name
    try:
        return CheckerType[name.upper()]
    except KeyError:
        raise ValueError(f"Unknown checker name: {name}")

