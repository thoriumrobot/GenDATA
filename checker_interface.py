#!/usr/bin/env python3
"""
Checker Interface - Abstract Base Class for Checker Implementations

This module defines the abstract interface that all checker implementations must follow.
This allows the pipeline to support multiple checkers (Lower Bound, SQL Quotes, Signature String, etc.)
in a unified way.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CheckerInterface(ABC):
    """
    Abstract base class for checker implementations.
    
    Each checker (Lower Bound, SQL Quotes, Signature String, etc.) must implement
    this interface to integrate with the GenDATA pipeline.
    """
    
    @abstractmethod
    def get_checker_name(self) -> str:
        """Return the name of the checker (e.g., 'LowerBound', 'SqlQuotes')"""
        pass
    
    @abstractmethod
    def get_checker_processor(self) -> str:
        """Return the Checker Framework processor class name"""
        pass
    
    @abstractmethod
    def get_annotation_types(self) -> List[str]:
        """
        Return list of annotation types this checker supports.
        
        Returns:
            List of annotation type names (e.g., ['@Positive', '@NonNegative', '@GTENegativeOne'])
        """
        pass
    
    @abstractmethod
    def parse_warnings(self, warnings_file: str) -> List[Dict[str, Any]]:
        """
        Parse warnings from Checker Framework output file.
        
        Args:
            warnings_file: Path to warnings output file
            
        Returns:
            List of warning dictionaries with keys: file, line, column, message, annotation_type
        """
        pass
    
    @abstractmethod
    def extract_features(self, cfg_data: Dict[str, Any], node: Dict[str, Any]) -> List[float]:
        """
        Extract checker-specific features from CFG node.
        
        Args:
            cfg_data: Complete CFG data structure
            node: Individual node from CFG
            
        Returns:
            List of feature values (floats)
        """
        pass
    
    @abstractmethod
    def validate_annotation(self, annotation_type: str, location: Dict[str, Any]) -> bool:
        """
        Validate if an annotation can be placed at a given location.
        
        Args:
            annotation_type: The annotation type to place (e.g., '@Positive')
            location: Location dictionary with keys: file, line, column, target_type
            
        Returns:
            True if annotation is valid for this location, False otherwise
        """
        pass
    
    @abstractmethod
    def get_training_data_source(self) -> str:
        """
        Return path to training data source (Checker Framework test suite directory).
        
        Returns:
            Path to test suite directory for this checker
        """
        pass
    
    def get_model_name_prefix(self) -> str:
        """
        Return prefix for model file names.
        
        Default implementation uses checker name in lowercase.
        Can be overridden for custom naming.
        """
        return self.get_checker_name().lower().replace(' ', '_')
    
    def get_warning_patterns(self) -> List[str]:
        """
        Return list of warning message patterns to match.
        
        Default implementation returns empty list.
        Can be overridden for checker-specific pattern matching.
        """
        return []

