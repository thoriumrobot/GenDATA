#!/usr/bin/env python3
"""
Run ablation prediction metrics: compare no augmentation (aug0) vs augmentation (aug1)
by generating case-study predictions for all 7 models and computing metrics.

Usage:
  python studies/run_ablation_prediction_metrics.py \
    [--models-aug0 PATH_TO_MODELS_AUG0] \
    [--models-aug1 PATH_TO_MODELS_AUG1]

If paths are not provided, the current models directory will be used for both levels.
Outputs:
  - studies/ablation_prediction_metrics.log (human-readable)
  - studies/ablation_prediction_metrics_{ts}.json (structured)
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path


def run(cmd, env=None):
    print("$ "+" ".join(cmd))
    return subprocess.run(cmd, check=False, env=env).returncode


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def summarize_prediction_counts(cs_root: Path, models):
    projects = ['guava', 'jfreechart', 'plume-lib']
    summary = {}
    for proj in projects:
        counts = {}
        for m in models:
            p = cs_root / proj / f'predictions_{m}.json'
            try:
                data = json.loads(p.read_text()) if p.exists() else []
                counts[m] = sum(len(d.get('predictions', [])) for d in data)
            except Exception:
                counts[m] = 0
        summary[proj] = counts
    return summary


def main():
    root = Path.cwd()
    cs_root = root / 'case_studies'
    log_path = root / 'studies' / 'ablation_prediction_metrics.log'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = root / 'studies' / f'ablation_prediction_metrics_{ts}.json'
    models = ['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf']

    # Args
    args = sys.argv[1:]
    def get_arg(flag):
        if flag in args:
            i = args.index(flag)
            if i+1 < len(args):
                return args[i+1]
        return None
    models_aug0 = get_arg('--models-aug0')
    models_aug1 = get_arg('--models-aug1')
    default_models = str(root / 'models_annotation_types')
    if not models_aug0:
        models_aug0 = default_models
    if not models_aug1:
        models_aug1 = default_models

    results = { 'aug0': {}, 'aug1': {} }

    for lvl, models_dir in [(0, models_aug0), (1, models_aug1)]:
        env = os.environ.copy()
        # Only set MODELS_DIR if it's different from default and the directory exists
        default_models = str(root / 'models_annotation_types')
        if models_dir != default_models and os.path.exists(models_dir):
            env['MODELS_DIR'] = models_dir
        elif 'MODELS_DIR' in env:
            # Clear MODELS_DIR if it points to wrong location
            del env['MODELS_DIR']
        # Predictions across all models
        rc = run([sys.executable, 'studies/run_annotation_type_predictions.py'], env=env)
        if rc != 0:
            print(f'ERROR: predictions failed for aug{lvl} (rc={rc})')
        # Summary counts
        counts = summarize_prediction_counts(cs_root, models)
        results[f'aug{lvl}']['prediction_counts'] = counts
        # Compute metrics
        run([sys.executable, 'studies/compute_case_study_metrics.py'], env=env)
        run([sys.executable, 'studies/case_study_metrics_collector.py'], env=env)
        # Read aggregate metrics
        agg = read_json(cs_root / 'evaluation_results' / 'aggregate_metrics.json') or {}
        results[f'aug{lvl}']['aggregate_metrics'] = agg

    # Write log
    lines = []
    lines.append(f"=== Ablation Prediction Metrics ({ts}) ===")
    lines.append(f"Models aug0: {models_aug0}")
    lines.append(f"Models aug1: {models_aug1}")
    for lvl in ['aug0','aug1']:
        lines.append(f"\n-- {lvl} prediction counts --")
        for proj, counts in results[lvl]['prediction_counts'].items():
            lines.append(f"{proj}: {counts}")
        lines.append(f"-- {lvl} aggregate metrics --")
        agg = results[lvl]['aggregate_metrics']
        for m, metrics in agg.items():
            lines.append(f"{m}: acc={metrics.get('accuracy_exact',0):.3f}, acc*={metrics.get('accuracy_partial',0):.3f}, f1w={metrics.get('f1_weighted',0):.3f}, cov={metrics.get('coverage',0):.3f}")

    # Simple side-by-side for key models
    lines.append("\n== 0 vs 1 (accuracy_partial) ==")
    models_cmp = models
    agg0 = results['aug0']['aggregate_metrics']
    agg1 = results['aug1']['aggregate_metrics']
    for m in models_cmp:
        a0 = (agg0.get(m) or {}).get('accuracy_partial', 0.0)
        a1 = (agg1.get(m) or {}).get('accuracy_partial', 0.0)
        lines.append(f"{m}: {a0:.3f} → {a1:.3f}")

    log_path.write_text("\n".join(lines))
    out_json.write_text(json.dumps(results, indent=2))
    print(f"WROTE: {log_path}")
    print(f"WROTE: {out_json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())


