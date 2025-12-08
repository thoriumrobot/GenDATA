#!/usr/bin/env python3
"""
Shared Dataset Generation Utility for Ablation Studies

This module provides a unified interface for generating balanced datasets
from CFG directories for different ablation study conditions.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AblationDatasetGenerator:
    """Utility class for generating balanced datasets for ablation studies"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the dataset generator
        
        Args:
            random_seed: Random seed for reproducible dataset generation
        """
        self.random_seed = random_seed
    
    def generate_dataset(self, cfg_dir: str, output_dir: str, 
                        examples_per_annotation: int = 2000,
                        target_balance: float = 0.5,
                        timeout: int = 3600) -> Tuple[bool, Optional[str]]:
        """
        Generate a balanced dataset from CFG files
        
        Args:
            cfg_dir: Directory containing CFG JSON files
            output_dir: Output directory for generated datasets
            examples_per_annotation: Number of examples per annotation type
            target_balance: Target balance ratio (0.5 = 50% positive, 50% negative)
            timeout: Timeout in seconds for dataset generation
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        cfg_path = Path(cfg_dir)
        output_path = Path(output_dir)
        
        # Verify CFG directory exists
        if not cfg_path.exists():
            error_msg = f"CFG directory does not exist: {cfg_dir}"
            logger.error(error_msg)
            return False, error_msg
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if improved_balanced_dataset_generator.py exists
        generator_script = Path('improved_balanced_dataset_generator.py')
        if not generator_script.exists():
            error_msg = f"Dataset generator script not found: {generator_script}"
            logger.error(error_msg)
            return False, error_msg
        
        # Build command
        cmd = [
            sys.executable,
            str(generator_script),
            '--cfg_dir', str(cfg_path),
            '--output_dir', str(output_path),
            '--examples_per_annotation', str(examples_per_annotation),
            '--target_balance', str(target_balance),
            '--random_seed', str(self.random_seed)
        ]
        
        logger.info(f"Generating dataset from CFG directory: {cfg_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Examples per annotation: {examples_per_annotation}")
        logger.info(f"Target balance: {target_balance}")
        logger.info(f"Random seed: {self.random_seed}")
        
        try:
            # Run dataset generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                error_msg = f"Dataset generation failed with return code {result.returncode}"
                logger.error(error_msg)
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False, error_msg
            
            # Verify datasets were created
            annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
            created_count = 0
            
            for ann_type in annotation_types:
                dataset_file = output_path / f"{ann_type}_real_balanced_dataset.json"
                if dataset_file.exists():
                    created_count += 1
                    logger.info(f"Created dataset: {dataset_file}")
                else:
                    logger.warning(f"Expected dataset file not found: {dataset_file}")
            
            if created_count == 0:
                error_msg = "No dataset files were created"
                logger.error(error_msg)
                return False, error_msg
            
            logger.info(f"Successfully generated {created_count} dataset files")
            return True, None
            
        except subprocess.TimeoutExpired:
            error_msg = f"Dataset generation timed out after {timeout} seconds"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error during dataset generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def verify_dataset_exists(self, dataset_dir: str) -> bool:
        """
        Verify that required dataset files exist in the directory
        
        Args:
            dataset_dir: Directory containing dataset files
            
        Returns:
            True if all required datasets exist, False otherwise
        """
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            logger.warning(f"Dataset directory does not exist: {dataset_dir}")
            return False
        
        annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
        missing = []
        
        for ann_type in annotation_types:
            dataset_file = dataset_path / f"{ann_type}_real_balanced_dataset.json"
            if not dataset_file.exists():
                missing.append(ann_type)
        
        if missing:
            logger.warning(f"Missing dataset files for: {', '.join(missing)}")
            return False
        
        logger.info(f"All required dataset files exist in: {dataset_dir}")
        return True


def main():
    """CLI interface for dataset generation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate balanced datasets for ablation studies'
    )
    parser.add_argument(
        '--cfg_dir',
        required=True,
        help='Directory containing CFG JSON files'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for generated datasets'
    )
    parser.add_argument(
        '--examples_per_annotation',
        type=int,
        default=2000,
        help='Number of examples per annotation type (default: 2000)'
    )
    parser.add_argument(
        '--target_balance',
        type=float,
        default=0.5,
        help='Target balance ratio (default: 0.5)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout in seconds (default: 3600)'
    )
    
    args = parser.parse_args()
    
    generator = AblationDatasetGenerator(random_seed=args.random_seed)
    success, error = generator.generate_dataset(
        cfg_dir=args.cfg_dir,
        output_dir=args.output_dir,
        examples_per_annotation=args.examples_per_annotation,
        target_balance=args.target_balance,
        timeout=args.timeout
    )
    
    if success:
        logger.info("Dataset generation completed successfully")
        return 0
    else:
        logger.error(f"Dataset generation failed: {error}")
        return 1


if __name__ == '__main__':
    exit(main())

