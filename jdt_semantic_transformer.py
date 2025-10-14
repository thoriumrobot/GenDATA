#!/usr/bin/env python3
"""
JDT Semantic Transformer Wrapper

Python wrapper for Eclipse JDT-based semantic transformations.
Replaces regex-based transformations with robust AST-based transformations.
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import List, Optional, Dict, Any
import random

logger = logging.getLogger(__name__)

class JdtSemanticTransformer:
    """Python wrapper for JDT semantic transformer service"""
    
    def __init__(self, jar_path: Optional[str] = None, seed: int = 42):
        """
        Initialize JDT semantic transformer.
        
        Args:
            jar_path: Path to jdt-transformer-all.jar. If None, will try to find it.
            seed: Random seed for reproducible transformations.
        """
        if jar_path is None:
            jar_path = self._find_jar_path()
        
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"JDT transformer JAR not found: {jar_path}")
        
        self.jar_path = jar_path
        self.java_cmd = self._get_java_command()
        self.seed = seed
        
        logger.info(f"Initialized JDT semantic transformer with JAR: {jar_path}, seed: {seed}")
    
    def _find_jar_path(self) -> str:
        """Find the JDT transformer JAR in the build directory"""
        possible_paths = [
            "build/libs/jdt-transformer-all.jar",
            "/home/ubuntu/GenDATA/build/libs/jdt-transformer-all.jar",
            "build/libs/jdt-transformer.jar",
            "/home/ubuntu/GenDATA/build/libs/jdt-transformer.jar"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find JDT transformer JAR. Please build it first with: ./gradlew jdtTransformerJar")
    
    def _get_java_command(self) -> str:
        """Get Java command with proper classpath"""
        return "java"
    
    def transform_file(self, input_file: str, output_file: str, 
                      transformations: List[str], mode: str = 'enhanced') -> bool:
        """
        Transform a Java file using JDT-based transformations.
        
        Args:
            input_file: Path to input Java file
            output_file: Path to output Java file
            transformations: List of transformation types to apply
            mode: Transformation mode ('enhanced' or 'simple')
            
        Returns:
            True if transformation was successful, False otherwise
        """
        try:
            # Read input file
            with open(input_file, 'r') as f:
                java_code = f.read()
            
            # Transform the code
            transformed_code = self.transform_code(java_code, transformations, mode)
            
            # Write output file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(transformed_code)
            
            logger.info(f"Successfully transformed {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transform file {input_file}: {e}")
            return False
    
    def transform_code(self, java_code: str, transformations: List[str],
                      mode: str = 'enhanced', force_transformation: bool = True) -> str:
        """
        Transform Java code string using JDT-based transformations.
        
        Args:
            java_code: Java source code as string
            transformations: List of transformation types to apply
            mode: Transformation mode ('enhanced' or 'simple')
            force_transformation: If True, retry with different transformations if none apply
            
        Returns:
            Transformed Java code
        """
        original_code = java_code
        
        # Try the requested transformations first
        transformed_code = self._try_transformations(java_code, transformations, mode)
        
        # If no changes were made and force_transformation is True, try other transformations
        if force_transformation and transformed_code == original_code and transformations:
            available_transformations = self.get_available_transformations(mode)
            other_transformations = [t for t in available_transformations if t not in transformations]
            
            if other_transformations:
                # Try with a subset of other transformations
                retry_transformations = other_transformations[:min(3, len(other_transformations))]
                logger.info(f"No changes with {transformations}, retrying with {retry_transformations}")
                transformed_code = self._try_transformations(java_code, retry_transformations, mode)
        
        return transformed_code

    def transform_code_with_flag(self, java_code: str, transformations: List[str],
                                 mode: str = 'enhanced', force_transformation: bool = True) -> (str, bool):
        """
        Transform code and also report whether a textual change occurred.

        Returns:
            (transformed_code, mutated)
        """
        transformed = self.transform_code(java_code, transformations, mode, force_transformation)
        return transformed, (transformed != java_code)
    
    def _try_transformations(self, java_code: str, transformations: List[str], mode: str) -> str:
        """Try applying transformations and return transformed code."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as input_file:
                input_file.write(java_code)
                input_path = input_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as output_file:
                output_path = output_file.name
            
            try:
                # Use a different seed for this transformation
                current_seed = self.seed + random.randint(0, 10000)
                
                cmd = [
                    self.java_cmd, "-jar", self.jar_path,
                    "--input", input_path,
                    "--output", output_path,
                    "--transformations", ",".join(transformations),
                    "--mode", mode,
                    "--seed", str(current_seed)
                ]
                
                logger.debug(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    logger.warning(f"JDT transformation failed: {result.stderr}")
                    return java_code  # Return original code on failure
                
                # Read transformed code
                with open(output_path, 'r') as f:
                    transformed_code = f.read()
                
                logger.info(f"Successfully applied {len(transformations)} transformations in {mode} mode")
                return transformed_code
                
            finally:
                # Clean up temp files
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
                    
        except Exception as e:
            logger.error(f"JDT code transformation failed: {e}")
            return java_code  # Return original code on failure
    
    def get_available_transformations(self, mode: str = 'enhanced') -> List[str]:
        """
        Get list of available transformations for the specified mode.
        
        Args:
            mode: Transformation mode ('enhanced' or 'simple')
            
        Returns:
            List of available transformation names
        """
        if mode == 'enhanced':
            return [
                'loop_conversion',
                'guard_reversal', 
                'mathematical_expression',
                'logical_expression',
                'ternary_operator',
                'switch_statement',
                'variable_operation',
                'brace_normalization',
                'string_concatenation',
                'numeric_literal'
            ]
        elif mode == 'simple':
            return [
                'simple_method_call',
                'simple_assignment',
                'simple_conditional',
                'simple_array_access',
                'simple_return_statement',
                'simple_variable_declaration',
                'simple_constructor_call',
                'simple_field_access',
                'simple_string_operation',
                'simple_numeric_operation'
            ]
        else:
            return []
    
    def get_random_transformations(self, count: int = 3, mode: str = 'enhanced') -> List[str]:
        """
        Get a random selection of transformations.
        
        Args:
            count: Number of transformations to select
            mode: Transformation mode ('enhanced' or 'simple')
            
        Returns:
            List of randomly selected transformation names
        """
        available = self.get_available_transformations(mode)
        if len(available) <= count:
            return available
        
        return random.sample(available, count)

# Convenience function for backward compatibility
def create_jdt_semantic_transformer(jar_path: Optional[str] = None, seed: int = 42) -> JdtSemanticTransformer:
    """Create and return a JDT semantic transformer instance"""
    return JdtSemanticTransformer(jar_path, seed)
