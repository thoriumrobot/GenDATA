#!/usr/bin/env python3
"""
Generate a human-readable comparison report for case study evaluation metrics.

Reads case_studies/evaluation_results/aggregate_metrics.json and prints a table
summarizing accuracy (exact/partial), F1 (macro/weighted), and coverage.
"""

import json
from pathlib import Path


def main():
    agg_path = Path('case_studies') / 'evaluation_results' / 'aggregate_metrics.json'
    if not agg_path.exists():
        print(f"Error: {agg_path} not found. Run compute_case_study_metrics.py and case_study_metrics_collector.py first.")
        return
    data = json.loads(agg_path.read_text())
    models = sorted(data.keys())
    print('=' * 86)
    print('CASE STUDY EVALUATION (Aggregate Across Projects)')
    print('=' * 86)
    header = f"{'MODEL':10} | {'ACC':>6} | {'ACC*':>6} | {'F1M':>6} | {'F1W':>6} | {'PREC':>6} | {'REC':>6} | {'COV':>6}"
    print(header)
    print('-' * len(header))
    for m in models:
        met = data[m] or {}
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else '  N/A'
        row = f"{m.upper():10} | {fmt(met.get('accuracy_exact')):>6} | {fmt(met.get('accuracy_partial')):>6} | {fmt(met.get('f1_macro')):>6} | {fmt(met.get('f1_weighted')):>6} | {fmt(met.get('precision_weighted')):>6} | {fmt(met.get('recall_weighted')):>6} | {fmt(met.get('coverage')):>6}"
        print(row)
    print('=' * 86)


if __name__ == '__main__':
    main()


