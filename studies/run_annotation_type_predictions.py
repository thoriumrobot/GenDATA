#!/usr/bin/env python3
"""
Run annotation-type predictions across all 7 base models for the case studies,
emit standardized per-project prediction files consumable by metrics scripts.
"""

import os
import subprocess
import sys
import json
from pathlib import Path


def run(cmd):
    print("$ "+" ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main():
    root = Path.cwd()
    cs_root = root / 'case_studies'
    cfg_root = root / 'case_study_cfg_output'

    # 1) Ensure CFGs exist for case studies
    rc = run([sys.executable, 'generate_case_study_cfgs.py'])
    if rc != 0:
        print('WARN: CFG generation returned non-zero')

    # 2) Run predictions per model to ensure correct backend dispatch
    projects = ['guava', 'jfreechart', 'plume-lib']
    models = ['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf']
    per_model_rc = {}
    for m in models:
        rc = run([sys.executable, 'predict_case_studies_fixed.py', '--model', m])
        per_model_rc[m] = rc
        if rc != 0:
            print(f'ERROR: prediction run failed for model {m}')

    # 3) Sanity: count standardized outputs
    projects = ['guava', 'jfreechart', 'plume-lib']
    models = ['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf']
    summary = {}
    for proj in projects:
        proj_dir = cs_root / proj
        counts = {}
        for m in models:
            p = proj_dir / f'predictions_{m}.json'
            try:
                data = json.loads(p.read_text()) if p.exists() else []
                counts[m] = sum(len(d.get('predictions',[])) for d in data)
            except Exception:
                counts[m] = 0
        summary[proj] = counts
    print(json.dumps(summary, indent=2))
    # Success if at least one model produced predictions; else non-zero
    any_preds = any(sum(summary[p].get(m,0) for p in projects) > 0 for m in models)
    if not any_preds:
        print('ERROR: No predictions produced across all models')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())


