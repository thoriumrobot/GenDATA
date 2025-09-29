#!/usr/bin/env python3
"""
Fixed version of predict_all_models_on_case_studies.py
This version generates CFGs for case study files first, then runs predictions.
"""

import os
import sys
import json
import subprocess
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CASE_STUDIES_ROOT = os.path.join(os.getcwd(), 'case_studies')
CASE_STUDY_CFG_DIR = os.path.join(os.getcwd(), 'case_study_cfg_output')
PRED_OUT_DIR = os.path.join(os.getcwd(), 'predictions_annotation_types')
MODELS_DIR = os.path.join(os.getcwd(), 'models_annotation_types')

def run(cmd: List[str]):
    """Run a command and exit if it fails"""
    logger.info("$ " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"Command failed with return code: {res.returncode}")
        sys.exit(res.returncode)

def ensure_case_study_cfgs():
    """Generate CFGs for case study files"""
    logger.info("Generating CFGs for case study files...")
    
    # Check if CFGs already exist
    if os.path.exists(CASE_STUDY_CFG_DIR) and len(os.listdir(CASE_STUDY_CFG_DIR)) > 0:
        logger.info("Case study CFGs already exist, skipping generation")
        return
    
    # Generate CFGs for case study files
    cmd = [sys.executable, 'generate_case_study_cfgs.py']
    run(cmd)

def list_java_files(root: str) -> List[str]:
    """List all Java files in the given directory tree"""
    files: List[str] = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.java'):
                files.append(os.path.join(r, f))
    return files

def predict_for_file(predictor, java_file: str, base_model_type: str) -> List[Dict]:
    """Predict annotations for a Java file using case study CFGs"""
    from model_based_predictor import ModelBasedPredictor
    
    # Find CFG data for this Java file in case study CFG directory
    java_basename = os.path.splitext(os.path.basename(java_file))[0]
    cfg_file = os.path.join(CASE_STUDY_CFG_DIR, java_basename, 'cfg.json')
    
    if not os.path.exists(cfg_file):
        logger.warning(f"No CFG file found for {java_file}; skipping")
        return []
    
    try:
        # Use the existing prediction method but with case study CFGs
        preds = predictor.predict_annotations_for_file_with_cfg(java_file, CASE_STUDY_CFG_DIR, threshold=0.3)
        
        # Tag with model_type if not present
        for p in preds:
            p.setdefault('model_type', base_model_type)
        
        return preds
    except Exception as e:
        logger.error(f"Prediction failed for {java_file}: {e}")
        return []

def main():
    """Main function to run predictions on case studies"""
    os.makedirs(PRED_OUT_DIR, exist_ok=True)
    
    # 1) Generate case study CFGs if they don't exist
    ensure_case_study_cfgs()
    
    # 2) Prepare predictor
    from model_based_predictor import ModelBasedPredictor
    predictor = ModelBasedPredictor(models_dir=MODELS_DIR, auto_train=True)
    
    base_models = ['enhanced_causal', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
    java_files = list_java_files(CASE_STUDIES_ROOT)
    logger.info(f"Found {len(java_files)} Java files under case_studies/")
    
    total_predictions = 0
    
    for base in base_models:
        logger.info(f"== Base model: {base} ==")
        
        if not predictor.load_or_train_models(base_model_type=base, episodes=10, 
                                            project_root='/home/ubuntu/checker-framework/checker/tests/index'):
            logger.warning(f"Skipping {base}: load/train failed")
            continue
        
        per_file_results: Dict[str, List[Dict]] = {}
        model_predictions = 0
        
        for jf in java_files:
            try:
                preds = predict_for_file(predictor, jf, base)
                if preds:
                    per_file_results.setdefault(jf, []).extend(preds)
                    model_predictions += len(preds)
            except Exception as e:
                logger.warning(f"Prediction failed for {jf} ({base}): {e}")
        
        # Save grouped predictions for this base model
        out_path = os.path.join(PRED_OUT_DIR, f"case_studies_{base}.predictions.json")
        with open(out_path, 'w') as f:
            json.dump(per_file_results, f, indent=2)
        
        logger.info(f"Saved {model_predictions} predictions for {base} to: {out_path}")
        total_predictions += model_predictions
    
    logger.info(f"Total predictions generated: {total_predictions}")
    logger.info("✅ Case study predictions completed successfully!")

if __name__ == '__main__':
    main()
