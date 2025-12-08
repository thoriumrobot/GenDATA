#!/usr/bin/env python3
"""
Checker Registry - Registry for Checker Implementations

This module maintains a registry of all available checker implementations
and provides factory methods to create checker instances.
"""

from typing import Dict, Type, Optional, List
import logging
from checker_interface import CheckerInterface

logger = logging.getLogger(__name__)

# Registry of checker implementations
_CHECKER_REGISTRY: Dict[str, Type[CheckerInterface]] = {}

def register_checker(name: str, checker_class: Type[CheckerInterface]):
    """
    Register a checker implementation.
    
    Args:
        name: Unique name for the checker (e.g., 'lower_bound', 'sql_quotes')
        checker_class: Class implementing CheckerInterface
    """
    if not issubclass(checker_class, CheckerInterface):
        raise ValueError(f"{checker_class} must implement CheckerInterface")
    
    _CHECKER_REGISTRY[name.lower()] = checker_class
    logger.info(f"Registered checker: {name} -> {checker_class.__name__}")

def get_checker(name: str) -> Optional[CheckerInterface]:
    """
    Get a checker instance by name.
    
    Args:
        name: Checker name (case-insensitive)
        
    Returns:
        CheckerInterface instance or None if not found
    """
    checker_class = _CHECKER_REGISTRY.get(name.lower())
    if checker_class:
        return checker_class()
    return None

def list_checkers() -> List[str]:
    """
    List all registered checker names.
    
    Returns:
        List of checker names
    """
    return list(_CHECKER_REGISTRY.keys())

def is_checker_registered(name: str) -> bool:
    """
    Check if a checker is registered.
    
    Args:
        name: Checker name (case-insensitive)
        
    Returns:
        True if checker is registered, False otherwise
    """
    return name.lower() in _CHECKER_REGISTRY

# Import and register built-in checkers
def _register_builtin_checkers():
    """Register built-in checker implementations"""
    try:
        from lower_bound_checker import LowerBoundChecker
        register_checker('lower_bound', LowerBoundChecker)
        register_checker('lowerbound', LowerBoundChecker)
        register_checker('index', LowerBoundChecker)  # Alias for Index Checker
    except ImportError:
        logger.warning("LowerBoundChecker not available")
    
    try:
        from sql_quotes_checker import SqlQuotesChecker
        register_checker('sql_quotes', SqlQuotesChecker)
        register_checker('sqlquotes', SqlQuotesChecker)
    except ImportError:
        logger.debug("SqlQuotesChecker not available (expected if not implemented)")
    
    try:
        from signature_string_checker import SignatureStringChecker
        register_checker('signature_string', SignatureStringChecker)
        register_checker('signaturestring', SignatureStringChecker)
    except ImportError:
        logger.debug("SignatureStringChecker not available (expected if not implemented)")

# Auto-register on import
_register_builtin_checkers()

