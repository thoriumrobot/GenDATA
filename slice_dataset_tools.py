#!/usr/bin/env python3
"""
Slice Dataset Tools
- resolve_slice_dir: choose soot or cf slices based on presence of .java
- normalize_slices: copy only .java files into a flat directory
- counters: count_java, count_cfg
"""
import os
import shutil
from pathlib import Path
from typing import Optional


def has_java(dir_path: Path) -> bool:
    if not dir_path.exists():
        return False
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.java'):
                return True
    return False


def resolve_slice_dir(base: Path) -> Optional[Path]:
    soot_dir = base / 'slices' / 'slices_soot'
    cf_dir = base / 'slices' / 'slices_cf'
    if has_java(soot_dir):
        return soot_dir
    if has_java(cf_dir):
        return cf_dir
    return None


def normalize_slices(src: Path, dst: Path) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith('.java'):
                continue
            shutil.copy2(Path(root) / f, dst / f)
            copied += 1
    return copied


def count_java(dir_path: Path) -> int:
    n = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.java'):
                n += 1
    return n


def count_cfg(dir_path: Path) -> int:
    n = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.json'):
                n += 1
    return n


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', help='Base experiment directory (contains slices/)')
    ap.add_argument('--normalize_to', help='Destination directory for normalized slices')
    args = ap.parse_args()
    if args.base and args.normalize_to:
        base = Path(args.base)
        dst = Path(args.normalize_to)
        src = resolve_slice_dir(base)
        if not src:
            print('NO_SLICES')
        else:
            copied = normalize_slices(src, dst)
            print(f'COPIED:{copied}')
