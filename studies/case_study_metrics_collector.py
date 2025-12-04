#!/usr/bin/env python3
"""
Aggregate case study evaluation metrics across projects and models.

Inputs: case_studies/evaluation_results/{project}_{model}_metrics.json
Outputs:
- case_studies/evaluation_results/per_project_metrics.json
- case_studies/evaluation_results/aggregate_metrics.json
"""

import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


PROJECTS = ['guava', 'jfreechart', 'plume-lib']
MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']


def main():
    base = Path('case_studies') / 'evaluation_results'
    per_project: Dict[str, Dict[str, Dict]] = {}
    aggregates: Dict[str, Dict] = {}
    for proj in PROJECTS:
        per_project[proj] = {}
        for model in MODELS:
            p = base / f'{proj}_{model}_metrics.json'
            data = load_json(p) or {}
            per_project[proj][model] = data
    # Compute simple aggregates (mean across projects) for key metrics
    key_fields = ['accuracy_exact', 'accuracy_partial', 'f1_macro', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'coverage']
    for model in MODELS:
        accum = {k: 0.0 for k in key_fields}
        count = 0
        for proj in PROJECTS:
            m = per_project[proj].get(model) or {}
            if not m:
                continue
            count += 1
            for k in key_fields:
                v = m.get(k)
                accum[k] += float(v) if isinstance(v, (int, float)) else 0.0
        if count > 0:
            aggregates[model] = {k: (accum[k] / count) for k in key_fields}
        else:
            aggregates[model] = {k: None for k in key_fields}
    (base / 'per_project_metrics.json').write_text(json.dumps(per_project, indent=2))
    (base / 'aggregate_metrics.json').write_text(json.dumps(aggregates, indent=2))
    print(f"WROTE: {(base / 'per_project_metrics.json')}")
    print(f"WROTE: {(base / 'aggregate_metrics.json')}")


if __name__ == '__main__':
    main()


