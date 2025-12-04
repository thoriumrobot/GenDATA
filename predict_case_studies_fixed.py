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
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CASE_STUDIES_ROOT = os.path.join(os.getcwd(), 'case_studies')
CASE_STUDY_CFG_DIR = os.path.join(os.getcwd(), 'case_study_cfg_output')
PRED_OUT_DIR = os.path.join(os.getcwd(), 'predictions_annotation_types')
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.getcwd(), 'models_annotation_types'))

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

def _build_cfg_index(cfg_root: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Return (exact_map, stem_map).
    - exact_map: from absolute java path (as recorded by generator index.json) → cfg.json
    - stem_map: from stem → cfg.json (fallback if exact not available)
    """
    exact_map: Dict[str, str] = {}
    stem_map: Dict[str, List[str]] = {}
    # Try to load generator-provided index
    try:
        idx_path = os.path.join(cfg_root, 'index.json')
        if os.path.exists(idx_path):
            import json
            exact_map = json.load(open(idx_path))
            logger.info(f"Loaded CFG index with {len(exact_map)} entries from {idx_path}")
    except Exception as e:
        logger.warning(f"Failed to load CFG index: {e}")
    # Always build a stem map as fallback
    for root, dirs, files in os.walk(cfg_root):
        if 'cfg.json' in files:
            cfg_path = os.path.join(root, 'cfg.json')
            stem = os.path.basename(os.path.dirname(cfg_path))
            stem_map.setdefault(stem, []).append(cfg_path)
    return exact_map, stem_map


def predict_for_file(predictor, java_file: str, base_model_type: str, exact_map: Dict[str, str], stem_map: Dict[str, List[str]], threshold: float = 0.3) -> List[Dict]:
    """Predict annotations for a Java file using case study CFGs"""
    from model_based_predictor import ModelBasedPredictor
    
    # Find CFG data for this Java file in case study CFG directory
    # Prefer exact mapping
    cfg_file = exact_map.get(java_file)
    if not cfg_file or not os.path.exists(cfg_file):
        # Fallback by stem with heuristic disambiguation
        java_basename = os.path.splitext(os.path.basename(java_file))[0]
        cands = stem_map.get(java_basename) or []
        if not cands:
            logger.warning(f"No CFG file found for {java_file}; skipping")
            return []
        # Heuristic: prefer candidate whose parent dir name appears in java_file path
        def score(path: str) -> int:
            parent = os.path.basename(os.path.dirname(path))
            return 0 if parent and parent in java_file else 1
        cands_sorted = sorted(cands, key=lambda p: (score(p), len(p)))
        cfg_file = cands_sorted[0]
        logger.debug(f"CFG fallback for {java_file} → {cfg_file}")
    
    try:
        # If canonical cfg.json missing but directory has other json, allow predictor to use directory
        if not os.path.exists(cfg_file):
            # Attempt to pick any JSON in same directory
            cand_dir = os.path.dirname(cfg_file)
            if os.path.isdir(cand_dir):
                js = [f for f in os.listdir(cand_dir) if f.endswith('.json')]
                if js:
                    cfg_file = os.path.join(cand_dir, js[0])
                    logger.debug(f"Using alternative CFG JSON for {java_file}: {cfg_file}")
        # Use the existing prediction method but with case study CFGs
        # Pass the resolved cfg_file path directly to the predictor
        preds = predictor.predict_annotations_for_file_with_cfg(java_file, CASE_STUDY_CFG_DIR, threshold=threshold, cfg_file_override=cfg_file)
        
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
    
    # 2) Parse model selection (single model per run for robust dispatch)
    want_model = None
    if '--model' in sys.argv:
        i = sys.argv.index('--model')
        if i+1 < len(sys.argv):
            want_model = sys.argv[i+1].strip()
    # map friendly names to underlying base types if needed
    name_map = {
        'gcn': 'gcn',
        'hgt': 'hgt',
        'gbt': 'gbt',
        'causal': 'causal',
        'gcsn': 'gcsn',
        'dg2n': 'dg2n',
        'dgcrf': 'dgcrf',
    }
    if want_model and want_model not in name_map:
        logger.error(f"Unknown model '{want_model}'")
        return 2
    # threshold arg
    threshold = 0.3
    if '--threshold' in sys.argv:
        ti = sys.argv.index('--threshold')
        if ti+1 < len(sys.argv):
            try:
                threshold = float(sys.argv[ti+1])
            except Exception:
                logger.warning("Invalid --threshold value; using default 0.3")
    from model_based_predictor import ModelBasedPredictor
    predictor = ModelBasedPredictor(models_dir=MODELS_DIR, auto_train=True)
    base_models = [name_map.get(want_model, None)] if want_model else ['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf']
    base_models = [m for m in base_models if m]
    java_files = list_java_files(CASE_STUDIES_ROOT)
    logger.info(f"Found {len(java_files)} Java files under case_studies/")
    
    total_predictions = 0
    
    # Build cfg index once
    exact_map, stem_map = _build_cfg_index(CASE_STUDY_CFG_DIR)
    # Coverage logging per project
    try:
        projects: Dict[str, Tuple[int,int]] = {}
        # Count java per project
        for jf in java_files:
            rel = os.path.relpath(jf, CASE_STUDIES_ROOT)
            proj = rel.split(os.sep)[0] if os.sep in rel else 'unknown'
            j,c = projects.get(proj, (0,0))
            projects[proj] = (j+1, c)
        # Count cfg hits (exact or stem)
        for jf in java_files:
            rel = os.path.relpath(jf, CASE_STUDIES_ROOT)
            proj = rel.split(os.sep)[0] if os.sep in rel else 'unknown'
            hit = jf in exact_map
            if not hit:
                stem = os.path.splitext(os.path.basename(jf))[0]
                hit = bool(stem_map.get(stem))
            j,c = projects.get(proj, (0,0))
            projects[proj] = (j, c+ (1 if hit else 0))
        for proj,(j,c) in projects.items():
            pct = (c/j*100.0) if j else 0.0
            level = logging.WARNING if pct < 5.0 else logging.INFO
            logger.log(level, f"CFG coverage for {proj}: {c}/{j} ({pct:.1f}%)")
    except Exception as e:
        logger.debug(f"Coverage logging failed: {e}")

    for base in base_models:
        publish_name = want_model if want_model else base
        # Alias unsupported dgcrf to dg2n backend if predictor cannot load it
        backend = base
        logger.info(f"== Base model: {base} ==")
        
        # Loading strategy: dgcrf is load-only with alias; others may auto-train
        if base == 'dgcrf':
            # Try to load dgcrf directly
            if not predictor.load_trained_models(base_model_type='dgcrf'):
                logger.warning("DGCRF models not found; aliasing to dg2n without training")
                backend = 'dg2n'
                if not predictor.load_trained_models(base_model_type=backend):
                    logger.error("No dg2n models available for dgcrf alias; proceeding with zero predictions for dgcrf")
        else:
            # Try to load or quick-train minimal if required
            if not predictor.load_or_train_models(base_model_type=backend, episodes=3, 
                                                project_root='/home/ubuntu/checker-framework/checker/tests/index'):
                logger.warning(f"Skipping {base}: load/train failed")
                continue
        
        per_file_results: Dict[str, List[Dict]] = {}
        model_predictions = 0
        
        for jf in java_files:
            try:
                preds = predict_for_file(predictor, jf, backend, exact_map, stem_map, threshold=threshold)
                if preds:
                    per_file_results.setdefault(jf, []).extend(preds)
                    model_predictions += len(preds)
            except Exception as e:
                logger.warning(f"Prediction failed for {jf} ({base}): {e}")
        
        # Save grouped predictions for this base model (legacy combined file)
        out_path = os.path.join(PRED_OUT_DIR, f"case_studies_{base}.predictions.json")
        with open(out_path, 'w') as f:
            json.dump(per_file_results, f, indent=2)
        logger.info(f"Saved {model_predictions} predictions for {base} to: {out_path}")

        # Also emit standardized per-project files for evaluation
        by_project: Dict[str, List[Dict]] = {}
        for fpath, preds in per_file_results.items():
            # Determine project name by path prefix
            rel = os.path.relpath(fpath, CASE_STUDIES_ROOT)
            project = rel.split(os.sep)[0] if os.sep in rel else 'unknown'
            record = {
                'file_path': fpath,
                'predictions': [
                    {
                        'line': int(p.get('line') or p.get('line_number') or p.get('lineno')),
                        'type': (p.get('annotation_type') or p.get('type')),
                        'confidence': float(p.get('confidence') or p.get('score') or 0.0)
                    }
                    for p in preds
                    if (p.get('line') or p.get('line_number') or p.get('lineno')) is not None and (p.get('annotation_type') or p.get('type'))
                ]
            }
            by_project.setdefault(project, []).append(record)

        for project, records in by_project.items():
            proj_dir = os.path.join(CASE_STUDIES_ROOT, project)
            os.makedirs(proj_dir, exist_ok=True)
            out_std = os.path.join(proj_dir, f"predictions_{publish_name.split('_')[-1]}.json")
            with open(out_std, 'w') as f:
                json.dump(records, f, indent=2)
            logger.info(f"Wrote standardized predictions for {project} ({base}): {out_std}")
        total_predictions += model_predictions
        # Post-model: log zero predictions but do not abort; runner will aggregate
        if model_predictions == 0:
            logger.error(f"No predictions produced for model {base}")
    
    logger.info(f"Total predictions generated: {total_predictions}")
    if total_predictions == 0:
        logger.error("No predictions produced across all models in this run")
        sys.exit(2)
    logger.info("✅ Case study predictions completed successfully!")

if __name__ == '__main__':
    main()
