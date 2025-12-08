#!/usr/bin/env python3
"""
Retrain graph-based models (GCN, HGT, GCSN) with enhanced "could be zero" features.
This script retrains all graph-based models for all annotation types using the updated pipeline.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Graph-based model types
GRAPH_MODELS = ['gcn', 'hgt', 'gcsn']

# Annotation types
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

# Training scripts
TRAINING_SCRIPTS = {
    '@Positive': 'annotation_type_rl_positive.py',
    '@NonNegative': 'annotation_type_rl_nonnegative.py',
    '@GTENegativeOne': 'annotation_type_rl_gtenegativeone.py'
}

def retrain_graph_model(annotation_type: str, base_model: str, episodes: int = 100, 
                       cfg_dir: str = None, device: str = 'cpu') -> bool:
    """
    Retrain a single graph-based model for a specific annotation type.
    
    Args:
        annotation_type: One of '@Positive', '@NonNegative', '@GTENegativeOne'
        base_model: One of 'gcn', 'hgt', 'gcsn'
        episodes: Number of training episodes
        cfg_dir: Directory containing CFG JSON files (defaults to env var or 'cfg_output_specimin')
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        True if training succeeded, False otherwise
    """
    if annotation_type not in TRAINING_SCRIPTS:
        logger.error(f"Unknown annotation type: {annotation_type}")
        return False
    
    if base_model not in GRAPH_MODELS:
        logger.error(f"Unknown graph model: {base_model}")
        return False
    
    script = TRAINING_SCRIPTS[annotation_type]
    
    # Determine CFG directory
    if cfg_dir is None:
        cfg_dir = os.environ.get('CFG_OUTPUT_DIR', 
                                 os.environ.get('PREDICTION_CFG_DIR', 
                                               'cfg_output_specimin'))
    
    if not os.path.isdir(cfg_dir):
        logger.warning(f"CFG directory not found: {cfg_dir}, using default")
        cfg_dir = 'cfg_output_specimin'
    
    logger.info(f"Retraining {annotation_type} with {base_model} model")
    logger.info(f"  CFG directory: {cfg_dir}")
    logger.info(f"  Episodes: {episodes}")
    logger.info(f"  Device: {device}")
    
    # Build command
    cmd = [
        sys.executable, script,
        '--mode', 'train',
        '--base_model', base_model,
        '--episodes', str(episodes),
        '--device', device
    ]
    
    # Set environment variable for CFG directory
    env = os.environ.copy()
    env['CFG_OUTPUT_DIR'] = cfg_dir
    env['PREDICTION_CFG_DIR'] = cfg_dir
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully retrained {annotation_type} with {base_model}")
            return True
        else:
            logger.error(f"❌ Failed to retrain {annotation_type} with {base_model}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Training timeout for {annotation_type} with {base_model}")
        return False
    except Exception as e:
        logger.error(f"❌ Exception during training: {e}")
        return False


def retrain_all_graph_models(episodes: int = 100, cfg_dir: str = None, device: str = 'cpu') -> dict:
    """
    Retrain all graph-based models for all annotation types.
    
    Args:
        episodes: Number of training episodes per model
        cfg_dir: Directory containing CFG JSON files
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Dictionary with training results
    """
    results = {}
    total = len(GRAPH_MODELS) * len(ANNOTATION_TYPES)
    completed = 0
    
    logger.info("=" * 80)
    logger.info("Retraining Graph-Based Models with Enhanced 'Could Be Zero' Features")
    logger.info("=" * 80)
    logger.info(f"Models: {', '.join(GRAPH_MODELS)}")
    logger.info(f"Annotation types: {', '.join(ANNOTATION_TYPES)}")
    logger.info(f"Total models to train: {total}")
    logger.info("=" * 80)
    
    for base_model in GRAPH_MODELS:
        for annotation_type in ANNOTATION_TYPES:
            model_key = f"{annotation_type}_{base_model}"
            logger.info(f"\n[{completed + 1}/{total}] Training {model_key}...")
            
            success = retrain_graph_model(
                annotation_type=annotation_type,
                base_model=base_model,
                episodes=episodes,
                cfg_dir=cfg_dir,
                device=device
            )
            
            results[model_key] = success
            completed += 1
            
            if success:
                logger.info(f"✅ Progress: {completed}/{total} models trained")
            else:
                logger.warning(f"⚠️  Progress: {completed}/{total} models trained (1 failed)")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    
    successful = sum(1 for v in results.values() if v)
    failed = total - successful
    
    logger.info(f"✅ Successful: {successful}/{total}")
    logger.info(f"❌ Failed: {failed}/{total}")
    
    if failed > 0:
        logger.info("\nFailed models:")
        for model_key, success in results.items():
            if not success:
                logger.info(f"  - {model_key}")
    
    logger.info("=" * 80)
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Retrain graph-based models (GCN, HGT, GCSN) with enhanced features'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of training episodes per model (default: 100)'
    )
    parser.add_argument(
        '--cfg_dir',
        type=str,
        default=None,
        help='Directory containing CFG JSON files (default: from env or cfg_output_specimin)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training (default: cpu)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=GRAPH_MODELS,
        help='Train only a specific model type (default: all)'
    )
    parser.add_argument(
        '--annotation_type',
        type=str,
        default=None,
        choices=ANNOTATION_TYPES,
        help='Train only a specific annotation type (default: all)'
    )
    
    args = parser.parse_args()
    
    # If specific model or annotation type requested, train only those
    if args.model or args.annotation_type:
        models_to_train = [args.model] if args.model else GRAPH_MODELS
        annotations_to_train = [args.annotation_type] if args.annotation_type else ANNOTATION_TYPES
        
        results = {}
        total = len(models_to_train) * len(annotations_to_train)
        completed = 0
        
        logger.info(f"Training {total} specific model(s)...")
        
        for base_model in models_to_train:
            for annotation_type in annotations_to_train:
                model_key = f"{annotation_type}_{base_model}"
                logger.info(f"\n[{completed + 1}/{total}] Training {model_key}...")
                
                success = retrain_graph_model(
                    annotation_type=annotation_type,
                    base_model=base_model,
                    episodes=args.episodes,
                    cfg_dir=args.cfg_dir,
                    device=args.device
                )
                
                results[model_key] = success
                completed += 1
        
        successful = sum(1 for v in results.values() if v)
        logger.info(f"\n✅ Successfully trained: {successful}/{total}")
        
    else:
        # Train all models
        results = retrain_all_graph_models(
            episodes=args.episodes,
            cfg_dir=args.cfg_dir,
            device=args.device
        )
        
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        
        if successful == total:
            logger.info("🎉 All graph models retrained successfully!")
            return 0
        else:
            logger.warning(f"⚠️  {successful}/{total} models retrained successfully")
            return 1


if __name__ == '__main__':
    exit(main())

