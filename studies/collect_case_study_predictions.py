#!/usr/bin/env python3
"""
Collect and standardize model predictions for case study projects.

This script consolidates predictions produced by various predictors into a
uniform schema so they can be evaluated against ground truth.

Input sources (if present):
- predictions_annotation_types/*.predictions.json (global dumps)
- predictions_manual_inspection/{project}/{model}.json (if created by runners)

Output per project:
- case_studies/{project}/predictions_{model}.json

Schema per file:
[
  {
    "file_path": "/abs/path/Foo.java",
    "predictions": [
      {"line": 42, "type": "@Positive", "confidence": 0.72}
    ]
  },
  ...
]
"""

import json
from pathlib import Path
from typing import Dict, List


KNOWN_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']


def load_if_exists(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


def normalize_predictions(raw: Dict) -> Dict:
    """Try to normalize a single-file prediction payload into the target schema."""
    out = {
        'file_path': raw.get('file_path') or raw.get('java_file') or raw.get('path'),
        'predictions': []
    }
    preds = raw.get('predictions') or raw.get('placements') or []
    for p in preds:
        line = p.get('line') or p.get('line_number') or p.get('lineno')
        atype = p.get('type') or p.get('annotation') or p.get('label')
        conf = p.get('confidence') or p.get('score') or p.get('prob')
        if out['file_path'] and line and atype:
            out['predictions'].append({'line': int(line), 'type': str(atype), 'confidence': conf})
    return out


def write_project_model_predictions(project_dir: Path, model: str) -> Path:
    # Search known prediction locations and select entries for this project
    project_root = str(project_dir.resolve())
    collected: Dict[str, Dict] = {}

    # 1) predictions_manual_inspection/{project}/{model}.json
    manual = Path('predictions_manual_inspection') / project_dir.name / f'{model}.json'
    mdata = load_if_exists(manual)
    if isinstance(mdata, dict) and 'predictions' in mdata and 'file_path' in mdata:
        norm = normalize_predictions(mdata)
        if norm['file_path'] and norm['file_path'].startswith(project_root):
            collected[norm['file_path']] = norm

    # 2) predictions_annotation_types/*.predictions.json (per-file or batch files)
    pad = Path('predictions_annotation_types')
    if pad.exists():
        for p in pad.glob('*.json'):
            pdata = load_if_exists(p)
            if not isinstance(pdata, dict):
                continue
            # Case A: per-file schema { 'file': <path>, 'predictions': [ { 'annotation_type': ... } ] }
            if 'file' in pdata and 'predictions' in pdata:
                fp = pdata.get('file')
                if isinstance(fp, str) and fp.startswith(project_root):
                    entry = {
                        'file_path': fp,
                        'predictions': [
                            {
                                'line': int(pr.get('line')) if isinstance(pr.get('line'), int) else pr.get('line'),
                                'type': pr.get('annotation_type') or pr.get('type'),
                                'confidence': pr.get('confidence') or pr.get('score')
                            }
                            for pr in pdata.get('predictions', [])
                            if pr.get('line') is not None and (pr.get('annotation_type') or pr.get('type'))
                        ]
                    }
                    prev = collected.get(entry['file_path'])
                    if not prev or len(entry['predictions']) > len(prev.get('predictions', [])):
                        collected[entry['file_path']] = entry
                continue
            # Case B: batch schema {'files': [ { 'file_path': ..., 'predictions': [...] } ]}
            files = pdata.get('files') or pdata.get('prediction_files') or []
            for entry in files:
                try:
                    norm = normalize_predictions(entry)
                    if norm['file_path'] and norm['file_path'].startswith(project_root):
                        prev = collected.get(norm['file_path'])
                        if not prev or len(norm['predictions']) > len(prev.get('predictions', [])):
                            collected[norm['file_path']] = norm
                except Exception:
                    continue

    # Write output
    out_list: List[Dict] = list(collected.values())
    out_path = project_dir / f'predictions_{model}.json'
    out_path.write_text(json.dumps(out_list, indent=2))
    return out_path


def main():
    base = Path('case_studies')
    projects = [p for p in base.iterdir() if p.is_dir() and p.name in ['guava', 'jfreechart', 'plume-lib']]
    for proj in projects:
        for model in KNOWN_MODELS:
            out = write_project_model_predictions(proj, model)
            print(f"WROTE: {out}")


if __name__ == '__main__':
    main()


