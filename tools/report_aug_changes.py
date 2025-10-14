#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def list_changed_variants(out_dir: str):
    out = Path(out_dir)
    if not out.exists():
        print(f"Output dir not found: {out}")
        return 1
    java_files = list(out.glob('**/*_variant_*.java'))
    print(f"Found {len(java_files)} variant files")
    # Group by base name
    groups = {}
    for f in java_files:
        base = f.name.split('_variant_')[0]
        groups.setdefault((f.parent, base), []).append(f)
    changed = 0
    for (parent, base), files in groups.items():
        # Compare first two variants if exist
        files = sorted(files)
        if len(files) < 2:
            continue
        a, b = files[0], files[1]
        with open(a, 'r') as fa, open(b, 'r') as fb:
            la = fa.read().splitlines()[4:]
            lb = fb.read().splitlines()[4:]
            if la != lb:
                print(f"DIFF: {a} vs {b}")
                changed += 1
    print(f"Changed groups: {changed}/{len(groups)}")
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: report_aug_changes.py <augmented_output_dir>")
        return 2
    return list_changed_variants(sys.argv[1])


if __name__ == '__main__':
    raise SystemExit(main())


