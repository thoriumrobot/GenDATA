#!/usr/bin/env python3
"""
Verify all 21 model-annotation combinations are trainable/loaded and that
element-wise prediction pipeline runs to completion.

Steps:
1) Trigger element-wise prediction slicing/CFG generation via simple pipeline
2) For each base model type (7), auto-train/load the 3 annotation models
3) Save a summary JSON with per-base-model status
"""

import os
import sys
import json
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_MODELS = ['enhanced_causal', 'enhanced_graph_causal', 'graph_causal', 'graphite', 'causal', 'hgt', 'gcn']

def ensure_elementwise_prediction_setup(cfwr_root: str, project_root: str, sample_java: str = None):
    """Run the simple pipeline in predict mode to build element-wise slices/CFGs."""
    cmd = [sys.executable, os.path.join(cfwr_root, 'simple_annotation_type_pipeline.py'), '--mode', 'predict', '--project_root', project_root, '--cfwr_root', cfwr_root]
    if sample_java:
        cmd += ['--target_file', sample_java]
    logger.info("Running element-wise setup: %s", ' '.join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError("Element-wise setup failed")

def verify_all_models(cfwr_root: str):
    models_dir = os.path.join(cfwr_root, 'models_annotation_types')
    os.makedirs(models_dir, exist_ok=True)
    from enhanced_graph_predictor import EnhancedGraphPredictor as ModelBasedPredictor
    predictor = ModelBasedPredictor(models_dir=models_dir, auto_train=True)
    summary = {}
    for base in BASE_MODELS:
        ok = predictor.load_or_train_models(base_model_type=base, epochs=5)
        summary[base] = bool(ok)
        logger.info("%s: %s", base, 'OK' if ok else 'FAILED')
    out = os.path.join(models_dir, 'verify_21_elementwise_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", out)
    return summary

def main():
    cfwr_root = os.getcwd()
    project_root = os.environ.get('PROJECT_ROOT', '/home/ubuntu/checker-framework/checker/tests/index')
    sample_java = os.environ.get('SAMPLE_JAVA')  # optional
    ensure_elementwise_prediction_setup(cfwr_root, project_root, sample_java)
    summary = verify_all_models(cfwr_root)
    # Exit nonzero if any failed
    if not all(summary.values()):
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())


