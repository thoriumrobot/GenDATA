#!/usr/bin/env python3
"""
Simple Semantic-Preserving Data Augmentation Script

This script uses JDT-based simple semantic transformations instead of regex-based ones.
It provides simple transformations that increase code variety while preserving semantics
and being resistant to slicer pruning.

JDT-Based Simple Transformations:
1. Simple method call variations
2. Simple assignment transformations
3. Simple conditional restructuring
4. Simple array access patterns
5. Simple return statement variations
6. Simple variable declaration changes
7. Simple constructor call variations
8. Simple field access patterns
9. Simple string operation alternatives
10. Simple numeric operation transformations
"""

import os
import argparse
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from jdt_semantic_transformer import JdtSemanticTransformer

# Set up logging
logger = logging.getLogger(__name__)

HEADER_COMMENT = """
/*
 * CFWR simple semantic augmentation: applied simple semantic-preserving transformations using JDT AST parsing.
 */
""".strip()

class SimpleCodeSemanticTransformer:
    """Simple semantic transformer using JDT-based transformations."""
    
    def __init__(self, seed: int = 42, disabled_transformations: List[str] = None, jdt_jar_path: Optional[str] = None):
        random.seed(seed)
        self.seed = seed
        self.transformations_applied = []
        self.disabled_transformations = disabled_transformations or []
        
        # Initialize JDT transformer
        try:
            self.jdt_transformer = JdtSemanticTransformer(jdt_jar_path, seed)
            logger.info("Initialized SimpleCodeSemanticTransformer with JDT-based transformations")
        except Exception as e:
            logger.error(f"Failed to initialize JDT transformer: {e}")
            raise RuntimeError(f"JDT transformer initialization failed: {e}")
    
    def transform_file(self, java_path: str, variant_idx: int) -> str:
        """Apply simple semantic transformations to a Java file using JDT."""
        try:
            with open(java_path, 'r') as f:
                original_code = f.read()
            
            # Get available simple transformations
            available_transformations = self.jdt_transformer.get_available_transformations('simple')
            
            # Filter out disabled transformations
            enabled_transformations = [t for t in available_transformations 
                                     if t not in self.disabled_transformations]
            
            if not enabled_transformations:
                logger.warning("No transformations available after filtering disabled ones")
                return original_code
            
            # Select random transformations to apply (fewer for simple mode)
            num_transformations = min(random.randint(1, 3), len(enabled_transformations))
            selected_transformations = random.sample(enabled_transformations, num_transformations)
            
            logger.info(f"Applying simple transformations: {selected_transformations}")
            
            # Apply transformations using JDT
            transformed_code = self.jdt_transformer.transform_code(
                original_code, 
                selected_transformations, 
                'simple'
            )
            
            # Record applied transformations
            self.transformations_applied.extend(selected_transformations)
            
            # Add header comment
            if transformed_code != original_code:
                transformed_code = self._add_header_comment(transformed_code, selected_transformations)
            
            return transformed_code
            
        except Exception as e:
            logger.error(f"Failed to transform file {java_path}: {e}")
            return original_code
    
    def _add_header_comment(self, code: str, transformations: List[str]) -> str:
        """Add header comment with applied transformations."""
        transformation_list = ", ".join(transformations)
        header = f"{HEADER_COMMENT}\n// Applied simple transformations: {transformation_list}\n\n"
        return header + code
    
    def transform_directory(self, input_dir: str, output_dir: str, num_variants: int = 3) -> Dict[str, Any]:
        """Transform all Java files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'files_processed': 0,
            'variants_created': 0,
            'transformations_applied': [],
            'errors': []
        }
        
        java_files = list(input_path.glob('**/*.java'))
        logger.info(f"Found {len(java_files)} Java files to process")
        
        for java_file in java_files:
            try:
                relative_path = java_file.relative_to(input_path)
                
                # Create variants
                for variant_idx in range(num_variants):
                    transformed_code = self.transform_file(str(java_file), variant_idx)
                    
                    # Create output path
                    variant_name = f"{java_file.stem}_simple_variant_{variant_idx}{java_file.suffix}"
                    output_file = output_path / relative_path.parent / variant_name
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write transformed file
                    with open(output_file, 'w') as f:
                        f.write(transformed_code)
                    
                    results['variants_created'] += 1
                    logger.info(f"Created simple variant: {output_file}")
                
                results['files_processed'] += 1
                results['transformations_applied'].extend(self.transformations_applied)
                
            except Exception as e:
                error_msg = f"Error processing {java_file}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics about applied transformations."""
        transformation_counts = {}
        for transformation in self.transformations_applied:
            transformation_counts[transformation] = transformation_counts.get(transformation, 0) + 1
        
        return {
            'total_transformations': len(self.transformations_applied),
            'unique_transformations': len(set(self.transformations_applied)),
            'transformation_counts': transformation_counts,
            'available_transformations': self.jdt_transformer.get_available_transformations('simple'),
            'disabled_transformations': self.disabled_transformations
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Simple semantic augmentation using JDT')
    parser.add_argument('input', help='Input Java file or directory')
    parser.add_argument('output', help='Output file or directory')
    parser.add_argument('--variants', type=int, default=3, help='Number of variants to create')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--disabled', nargs='*', default=[], help='Disabled transformations')
    parser.add_argument('--jdt-jar', help='Path to JDT transformer JAR')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize transformer
        transformer = SimpleCodeSemanticTransformer(
            seed=args.seed,
            disabled_transformations=args.disabled,
            jdt_jar_path=args.jdt_jar
        )
        
        # Process input
        if os.path.isfile(args.input):
            # Single file
            transformed_code = transformer.transform_file(args.input, 0)
            
            with open(args.output, 'w') as f:
                f.write(transformed_code)
            
            print(f"Transformed file: {args.input} -> {args.output}")
            
        elif os.path.isdir(args.input):
            # Directory
            results = transformer.transform_directory(args.input, args.output, args.variants)
            
            print(f"Processing complete:")
            print(f"  Files processed: {results['files_processed']}")
            print(f"  Variants created: {results['variants_created']}")
            print(f"  Errors: {len(results['errors'])}")
            
            if results['errors']:
                print("Errors encountered:")
                for error in results['errors']:
                    print(f"  - {error}")
        
        else:
            raise FileNotFoundError(f"Input path not found: {args.input}")
        
        # Print transformation statistics
        stats = transformer.get_transformation_stats()
        print(f"\nSimple Transformation Statistics:")
        print(f"  Total transformations applied: {stats['total_transformations']}")
        print(f"  Unique transformation types: {stats['unique_transformations']}")
        print(f"  Available transformations: {len(stats['available_transformations'])}")
        
        if stats['transformation_counts']:
            print("  Most used transformations:")
            for trans, count in sorted(stats['transformation_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {trans}: {count}")
        
    except Exception as e:
        logger.error(f"Simple transformation failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())