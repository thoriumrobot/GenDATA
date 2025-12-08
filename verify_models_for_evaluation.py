#!/usr/bin/env python3
"""
Verify models available for evaluation.

This script checks which models are available and matches them with expected naming conventions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Expected base models
EXPECTED_BASE_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n']

# Annotation types
ANNOTATION_TYPES = ['positive', 'nonnegative', 'gtenegativeone']

# Model name mappings (handle both enhanced_causal and dgcrf)
MODEL_NAME_MAPPING = {
    'dgcrf': 'enhanced_causal',
    'enhanced_causal': 'enhanced_causal',
    'enhancedcausal': 'enhanced_causal',
    'gcn': 'gcn',
    'hgt': 'hgt',
    'gbt': 'gbt',
    'causal': 'causal',
    'gcsn': 'gcsn',
    'dg2n': 'dg2n'
}


def find_model_files(models_dir: Path) -> Dict[str, List[Path]]:
    """Find all model files in models directory."""
    models_by_name: Dict[str, List[Path]] = {}
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return models_by_name
    
    # Look for .pth files (PyTorch models)
    for model_file in models_dir.rglob('*.pth'):
        # Extract model name from filename
        filename = model_file.stem.lower()
        
        # Try to identify model type
        model_name = None
        annotation_type = None
        
        # Check for base model names
        for model_key, normalized_model in MODEL_NAME_MAPPING.items():
            if model_key in filename:
                model_name = normalized_model
                break
        
        # Check for annotation types
        for ann_type in ANNOTATION_TYPES:
            if ann_type in filename:
                annotation_type = ann_type
                break
        
        # Create model key
        if model_name:
            if annotation_type:
                model_key = f"{model_name}_{annotation_type}"
            else:
                model_key = model_name
            
            if model_key not in models_by_name:
                models_by_name[model_key] = []
            models_by_name[model_key].append(model_file)
    
    return models_by_name


def verify_models(models_dir: Path) -> Dict:
    """Verify which models are available."""
    logger.info(f"Verifying models in {models_dir}")
    
    models_by_name = find_model_files(models_dir)
    
    # Check for expected models
    expected_models = set()
    for base_model in EXPECTED_BASE_MODELS:
        for ann_type in ANNOTATION_TYPES:
            expected_models.add(f"{base_model}_{ann_type}")
    
    available_models = set(models_by_name.keys())
    missing_models = expected_models - available_models
    
    # Also check for base models without annotation suffix
    base_models_available = set()
    for model_key in available_models:
        for base_model in EXPECTED_BASE_MODELS:
            if model_key.startswith(base_model):
                base_models_available.add(base_model)
                break
    
    result = {
        'models_dir': str(models_dir),
        'available_models': sorted(available_models),
        'expected_models': sorted(expected_models),
        'missing_models': sorted(missing_models),
        'base_models_available': sorted(base_models_available),
        'models_by_name': {k: [str(p) for p in v] for k, v in models_by_name.items()},
        'total_models_found': len(available_models),
        'total_models_expected': len(expected_models)
    }
    
    logger.info(f"Found {len(available_models)} models")
    logger.info(f"Expected {len(expected_models)} models")
    logger.info(f"Missing {len(missing_models)} models")
    
    if missing_models:
        logger.warning(f"Missing models: {sorted(missing_models)}")
    
    return result


def get_available_models_for_evaluation(models_dir: Path) -> List[str]:
    """Get list of available models for evaluation."""
    models_by_name = find_model_files(models_dir)
    
    # Return model names that match expected pattern
    available = []
    for model_key in models_by_name.keys():
        # Check if it matches expected pattern
        for base_model in EXPECTED_BASE_MODELS:
            if model_key.startswith(base_model):
                available.append(model_key)
                break
    
    return sorted(available)


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to standard format."""
    model_name_lower = model_name.lower()
    
    # Check for mappings
    for key, normalized in MODEL_NAME_MAPPING.items():
        if key in model_name_lower:
            base = normalized
            # Check for annotation type
            for ann_type in ANNOTATION_TYPES:
                if ann_type in model_name_lower:
                    return f"{base}_{ann_type}"
            return base
    
    return model_name


def main():
    """Main function."""
    import sys
    
    models_dir = Path('/home/ubuntu/GenDATA/models_annotation_types')
    
    if len(sys.argv) > 1:
        models_dir = Path(sys.argv[1])
    
    result = verify_models(models_dir)
    
    print(f"Models Directory: {result['models_dir']}")
    print(f"Total Models Found: {result['total_models_found']}")
    print(f"Total Models Expected: {result['total_models_expected']}")
    print(f"\nAvailable Models ({len(result['available_models'])}):")
    for model in result['available_models']:
        print(f"  - {model}")
    
    if result['missing_models']:
        print(f"\nMissing Models ({len(result['missing_models'])}):")
        for model in result['missing_models']:
            print(f"  - {model}")
    
    print(f"\nBase Models Available: {', '.join(result['base_models_available'])}")
    
    # Return available models for evaluation
    available = get_available_models_for_evaluation(models_dir)
    print(f"\nModels for Evaluation: {len(available)}")
    for model in available:
        print(f"  - {model}")


if __name__ == '__main__':
    main()

