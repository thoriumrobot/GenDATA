#!/usr/bin/env python3
"""
Analyze confidence distributions from case_studies/*/predictions_*.json and
print a simple threshold recommendation (median and 75th percentile).
"""

import json
import glob
from pathlib import Path


def main():
    root = Path.cwd() / 'case_studies'
    files = list(root.glob('*/predictions_*.json'))
    confs = []
    for fp in files:
        try:
            data = json.loads(fp.read_text())
            for rec in data:
                for p in rec.get('predictions', []):
                    c = p.get('confidence')
                    if isinstance(c, (int, float)):
                        confs.append(float(c))
        except Exception:
            continue
    if not confs:
        print('No confidences found')
        return 1
    confs.sort()
    n = len(confs)
    median = confs[n//2]
    p75 = confs[int(0.75*n) if int(0.75*n) < n else n-1]
    print(f'total={n} median={median:.3f} p75={p75:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Analyze prediction confidences across case_studies to suggest a threshold.
Outputs summary stats and simple histogram buckets per model.
"""
import json
from pathlib import Path
from collections import defaultdict

MODELS = ['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf']
PROJECTS = ['guava','jfreechart','plume-lib']


def main():
    cs = Path('case_studies')
    confs = {m: [] for m in MODELS}
    for proj in PROJECTS:
        for m in MODELS:
            p = cs / proj / f'predictions_{m}.json'
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            for rec in data:
                for pred in rec.get('predictions', []):
                    c = pred.get('confidence')
                    if isinstance(c, (int, float)):
                        confs[m].append(float(c))
    # Summaries
    out = {}
    for m, arr in confs.items():
        if not arr:
            out[m] = { 'count': 0 }
            continue
        arr_sorted = sorted(arr)
        def pct(p):
            idx = int(p * (len(arr_sorted)-1))
            return arr_sorted[idx]
        out[m] = {
            'count': len(arr),
            'min': min(arr),
            'p25': pct(0.25),
            'p50': pct(0.50),
            'p75': pct(0.75),
            'p90': pct(0.90),
            'max': max(arr),
            'suggested_threshold': pct(0.75)
        }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
