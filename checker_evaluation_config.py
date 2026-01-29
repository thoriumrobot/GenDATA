#!/usr/bin/env python3
"""
Checker Evaluation Configuration

This module provides checker-specific configuration for evaluation.
"""

from typing import Dict, List
from pathlib import Path

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
MODELS_DIR = GEN_DATA_ROOT / 'models_annotation_types'
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'

# Checker configurations
CHECKER_CONFIGS = {
    'lower_bound': {
        'name': 'Lower Bound Checker',
        'processor': 'org.checkerframework.checker.index.IndexChecker',
        'test_suite': '/home/ubuntu/checker-framework/checker/tests/index',
        'annotation_types': ['@Positive', '@NonNegative', '@GTENegativeOne'],
        'base_models': ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n'],
        'expected_models': 21,  # 7 base models × 3 annotation types
        'evaluation_projects': ['pom-tuner', 'commons-lang', 'commons-io'],  # 3 GitHub projects with >=5 warnings
        'model_naming_pattern': '{annotation}_{model}',  # e.g., positive_gcn
    },
    'sql_quotes': {
        'name': 'SQL Quotes Checker',
        'processor': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
        'test_suite': '/home/ubuntu/checker-framework/checker/tests/sqlquotes',
        'annotation_types': ['@SqlEvenQuotes', '@SqlOddQuotes'],
        'base_models': ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n'],
        'expected_models': 14,  # 7 base models × 2 annotation types
        # Real projects for SQL Quotes evaluation - 3 GitHub projects
        'evaluation_projects': ['commons-dbcp', 'mybatis-3', 'commons-dbutils'],
        'model_naming_pattern': '{annotation}_{model}',  # e.g., sqlevenquotes_gcn
    },
    'signature_string': {
        'name': 'Signature String Checker',
        'processor': 'org.checkerframework.checker.signature.SignatureChecker',
        'test_suite': '/home/ubuntu/checker-framework/checker/tests/signature',
        'annotation_types': ['@FullyQualifiedName', '@BinaryName', '@FieldDescriptor'],
        'base_models': ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n'],
        'expected_models': 21,  # 7 base models × 3 annotation types
        # Real projects for Signature String evaluation - 3 GitHub projects with >=5 warnings
        'evaluation_projects': ['javassist', 'reflections', 'guice'],
        'model_naming_pattern': '{annotation}_{model}',  # e.g., fullyqualifiedname_gcn
    }
}

def get_checker_config(checker_name: str) -> Dict:
    """Get configuration for a specific checker."""
    return CHECKER_CONFIGS.get(checker_name.lower(), {})

def get_all_checker_names() -> List[str]:
    """Get list of all supported checker names."""
    return list(CHECKER_CONFIGS.keys())

def get_checker_annotation_types(checker_name: str) -> List[str]:
    """Get annotation types for a checker."""
    config = get_checker_config(checker_name)
    return config.get('annotation_types', [])

def get_checker_base_models(checker_name: str) -> List[str]:
    """Get base model types for a checker."""
    config = get_checker_config(checker_name)
    return config.get('base_models', [])

def build_model_name(checker_name: str, annotation_type: str, base_model: str) -> str:
    """Build model name from components."""
    config = get_checker_config(checker_name)
    pattern = config.get('model_naming_pattern', '{annotation}_{model}')
    
    # Normalize annotation type (remove @, lowercase)
    ann_normalized = annotation_type.replace('@', '').lower()
    
    return pattern.format(annotation=ann_normalized, model=base_model)

def get_evaluation_projects(checker_name: str) -> List[str]:
    """Get list of evaluation projects for a checker."""
    config = get_checker_config(checker_name)
    return config.get('evaluation_projects', [])


# Import statements for each checker's annotations
CHECKER_ANNOTATION_IMPORTS = {
    'lower_bound': [
        "import org.checkerframework.checker.index.qual.Positive;",
        "import org.checkerframework.checker.index.qual.NonNegative;",
        "import org.checkerframework.checker.index.qual.GTENegativeOne;",
    ],
    'sql_quotes': [
        "import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;",
        "import org.checkerframework.checker.sqlquotes.qual.SqlOddQuotes;",
    ],
    'signature_string': [
        "import org.checkerframework.checker.signature.qual.BinaryName;",
        "import org.checkerframework.checker.signature.qual.FullyQualifiedName;",
        "import org.checkerframework.checker.signature.qual.FieldDescriptor;",
        "import org.checkerframework.checker.signature.qual.ClassGetName;",
        "import org.checkerframework.checker.signature.qual.InternalForm;",
    ],
}


# Annotation to normalized name mapping
ANNOTATION_NORMALIZED_NAMES = {
    # Lower Bound
    '@Positive': 'positive',
    '@NonNegative': 'nonnegative',
    '@GTENegativeOne': 'gtenegativeone',
    # SQL Quotes
    '@SqlEvenQuotes': 'sqlevenquotes',
    '@SqlOddQuotes': 'sqloddquotes',
    # Signature String
    '@BinaryName': 'binaryname',
    '@FullyQualifiedName': 'fullyqualifiedname',
    '@FieldDescriptor': 'fielddescriptor',
    '@ClassGetName': 'classgetname',
    '@InternalForm': 'internalform',
}


def get_checker_imports(checker_name: str) -> List[str]:
    """Get import statements for a checker's annotations."""
    return CHECKER_ANNOTATION_IMPORTS.get(checker_name.lower(), [])


def normalize_annotation_name(annotation: str) -> str:
    """Normalize an annotation name for model naming."""
    return ANNOTATION_NORMALIZED_NAMES.get(annotation, annotation.replace('@', '').lower())


def get_training_projects(checker_name: str) -> List[str]:
    """Get list of training projects for a checker (with injected annotations)."""
    training_map = {
        'sql_quotes': ['commons-dbutils', 'training_sql_quotes', 'cf_sqlquotes_tests'],
        'signature_string': ['kryo', 'guice', 'training_signature', 'cf_signature_tests'],
        'lower_bound': ['agrona', 'hipparchus', 'eclipse-collections'],
    }
    return training_map.get(checker_name.lower(), [])

