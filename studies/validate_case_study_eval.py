#!/usr/bin/env python3
"""
Validation helper: summarize GT/pred coverage and top missing CFGs.
"""

import os
import json
from pathlib import Path
from typing import Dict


def main():
    root = Path.cwd()
    cs_root = root / 'case_studies'
    cfg_root = root / 'case_study_cfg_output'
    projects = ['guava', 'jfreechart', 'plume-lib']
    models = ['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf']

    # Build set of basenames that have cfg.json
    cfg_basenames = set()
    if cfg_root.exists():
        for p in cfg_root.iterdir():
            if p.is_dir() and (p / 'cfg.json').exists():
                cfg_basenames.add(p.name)

    summary: Dict[str, Dict] = {}
    for proj in projects:
        proj_dir = cs_root / proj
        gt_path = proj_dir / 'ground_truth.json'
        gt = []
        try:
            if gt_path.exists():
                gt = json.loads(gt_path.read_text())
        except Exception:
            gt = []
        gt_count = sum(len(r.get('annotations', [])) for r in gt)

        preds = {}
        for m in models:
            pred_path = proj_dir / f'predictions_{m}.json'
            try:
                data = json.loads(pred_path.read_text()) if pred_path.exists() else []
            except Exception:
                data = []
            preds[m] = sum(len(d.get('predictions', [])) for d in data)

        # Sample files missing CFGs (by basename)
        missing_cfg = []
        for r in gt:
            fp = Path(r.get('file_path',''))
            base = fp.stem
            if base and base not in cfg_basenames:
                missing_cfg.append(base)
        missing_cfg = list(dict.fromkeys(missing_cfg))[:20]

        summary[proj] = {
            'gt': gt_count,
            'preds': preds,
            'missing_cfg_samples': missing_cfg,
        }

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


