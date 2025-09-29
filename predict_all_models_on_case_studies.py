#!/usr/bin/env python3
"""
Predict all 7 base models × 3 annotation types over projects in case_studies/.

Steps:
1) Ensure prediction CFGs exist for case_studies/ (invoke simple pipeline predict once)
2) For each base model type, load/train annotation-type models
3) For each Java file in case_studies/, load its CFG and run predictions per base model
4) Save per-file, per-base-model predictions under predictions_annotation_types/
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
PRED_CFG_DIR = os.path.join(os.getcwd(), 'test_case_study_cfg_output')  # Use test CFG directory
PRED_OUT_DIR = os.path.join(os.getcwd(), 'predictions_annotation_types')
MODELS_DIR = os.path.join(os.getcwd(), 'models_annotation_types')


def run(cmd: List[str]):
    print("$ "+" ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def ensure_prediction_cfgs():
    """Generate CFGs for case study files specifically."""
    logger.info("Generating CFGs for case study files...")
    
    # Find all Java files in case studies
    java_files = []
    for root, dirs, files in os.walk(CASE_STUDIES_ROOT):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(java_files)} Java files in case studies")
    
    # Generate CFGs for each case study file (limit to first 5 for testing)
    processed_count = 0
    for java_file in java_files[:5]:
        try:
            logger.info(f"Generating CFG for {os.path.basename(java_file)}...")
            cmd = [sys.executable, 'simple_annotation_type_pipeline.py', '--mode', 'predict', '--target_file', java_file]
            run(cmd)
            processed_count += 1
        except Exception as e:
            logger.warning(f"Failed to generate CFG for {java_file}: {e}")
    
    logger.info(f"Successfully generated CFGs for {processed_count} case study files")


def list_java_files(root: str) -> List[str]:
    files: List[str] = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.java'):
                files.append(os.path.join(r, f))
    return files


def predict_for_file(predictor, java_file: str, base_model_type: str) -> List[Dict]:
    from model_based_predictor import ModelBasedPredictor
    # prediction_cfg_output/<basename>/cfg.json is expected
    cfg_dir = PRED_CFG_DIR
    preds = predictor.predict_annotations_for_file_with_cfg(java_file, cfg_dir, threshold=0.3)
    # Tag with model_type if not present
    for p in preds:
        p.setdefault('model_type', base_model_type)
    return preds


def main():
    os.makedirs(PRED_OUT_DIR, exist_ok=True)

    # 1) Build prediction CFGs
    ensure_prediction_cfgs()

    # 2) Prepare predictor
    from model_based_predictor import ModelBasedPredictor
    predictor = ModelBasedPredictor(models_dir=MODELS_DIR, auto_train=True)

    base_models = ['enhanced_causal', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
    java_files = list_java_files(CASE_STUDIES_ROOT)
    print(f"Found {len(java_files)} Java files under case_studies/")

    for base in base_models:
        print(f"== Base model: {base} ==")
        if not predictor.load_or_train_models(base_model_type=base, episodes=10, project_root='/home/ubuntu/checker-framework/checker/tests/index'):
            print(f"[WARN] Skipping {base}: load/train failed")
            continue

        per_file_results: Dict[str, List[Dict]] = {}
        for jf in java_files:
            try:
                preds = predict_for_file(predictor, jf, base)
                if preds:
                    per_file_results.setdefault(jf, []).extend(preds)
            except Exception as e:
                print(f"[WARN] Prediction failed for {jf} ({base}): {e}")

        # Save grouped predictions for this base model
        out_path = os.path.join(PRED_OUT_DIR, f"case_studies_{base}.predictions.json")
        with open(out_path, 'w') as f:
            json.dump(per_file_results, f, indent=2)
        print(f"Saved predictions: {out_path}")


if __name__ == '__main__':
    main()


