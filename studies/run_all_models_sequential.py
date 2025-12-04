#!/usr/bin/env python3
"""
Sequential runner to execute no-augmentation vs augmented ablations for all models
within a strict wall-clock deadline.

Assumptions:
- study_orchestrator.py has generated study_plan_all_models.json and per-model scripts.
- Each variant contains a scripts directory with step scripts prefixed by two-digit order.
- We run level 0 (no augmentation) and the highest available augmentation level.

This runner enforces a global deadline (default 6h) and stops when time is up.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict


def load_plan(plan_path: Path) -> Dict:
    with open(plan_path, 'r') as f:
        return json.load(f)


def list_step_scripts(scripts_dir: Path) -> List[Path]:
    if not scripts_dir.exists():
        return []
    files = [p for p in scripts_dir.iterdir() if p.is_file() and p.stat().st_mode & 0o111]
    # Ensure ordered execution by filename (e.g., 01_*, 02_* ...)
    files.sort(key=lambda p: p.name)
    return files


def time_left(deadline_ts: float) -> float:
    return max(0.0, deadline_ts - time.monotonic())


def run_script(path: Path, deadline_ts: float) -> int:
    if time_left(deadline_ts) <= 0:
        return 124  # timed out
    env = os.environ.copy()
    # Prefer Soot by default if available
    env.setdefault('SOOT_SLICE_CLI', str(Path.cwd() / 'tools' / 'soot_slicer.sh'))
    env.setdefault('SOOT_JAR', str(Path.cwd() / 'build' / 'libs' / 'GenDATA-all.jar'))
    try:
        proc = subprocess.run([str(path)], env=env)
        return proc.returncode
    except FileNotFoundError:
        return 127


def run_variant(variant: Dict, deadline_ts: float) -> bool:
    scripts_dir = Path(variant['scripts_dir'])
    steps = list_step_scripts(scripts_dir)
    for step in steps:
        if time_left(deadline_ts) <= 0:
            print(f"DEADLINE: Skipping remaining steps for {scripts_dir}")
            return False
        print(f"RUN: {step}")
        rc = run_script(step, deadline_ts)
        if rc != 0:
            print(f"FAIL({rc}): {step}")
            return False
    return True


def select_variants_for_model(model_entry: Dict) -> List[Dict]:
    # model_entry['variants'] is a list of level entries; choose level 0 and max level
    variants = model_entry.get('variants', [])
    if not variants:
        return []
    # Extract by level
    try:
        levels = [(v['level'], v) for v in variants]
        levels.sort(key=lambda t: t[0])
    except KeyError:
        return variants[:1]
    chosen: List[Dict] = []
    # Level 0
    for lvl, v in levels:
        if lvl == 0:
            chosen.append(v)
            break
    # Max level (may equal 0)
    max_lvl, max_v = levels[-1]
    if max_v not in chosen:
        chosen.append(max_v)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='studies/all_models_ablation/study_plan_all_models.json')
    ap.add_argument('--hours', type=float, default=6.0)
    args = ap.parse_args()

    plan_path = Path(args.plan)
    plan = load_plan(plan_path)
    models = plan.get('models', [])

    start = time.monotonic()
    deadline_ts = start + args.hours * 3600.0
    print(f"START: Sequential run with deadline in {args.hours} hours")

    for model_entry in models:
        if not model_entry.get('supported', False):
            print(f"SKIP: Unsupported model entry {model_entry.get('model')}")
            continue
        remaining = time_left(deadline_ts)
        if remaining <= 0:
            print("DEADLINE: Stopping before starting next model")
            break
        model_name = model_entry.get('model', 'unknown')
        print(f"MODEL: {model_name} (time left: {int(remaining)}s)")
        variants = select_variants_for_model(model_entry)
        for v in variants:
            level = v.get('level')
            print(f"\n== MODEL {model_name} LEVEL {level} ==")
            ok = run_variant(v, deadline_ts)
            if not ok:
                print(f"WARN: Variant level {level} failed or timed out; continuing to next")
        print(f"DONE: {model_name}")

    total_elapsed = time.monotonic() - start
    print(f"FINISH: Elapsed {int(total_elapsed)}s")
    # Aggregate metrics at the end
    try:
        agg = Path.cwd() / 'studies' / 'metrics_aggregate.py'
        if agg.exists():
            print('AGGREGATE: Running metrics aggregator')
            subprocess.run([str(agg)], check=False)
        else:
            print('AGGREGATE: metrics_aggregate.py not found; skipping')
    except Exception as e:
        print(f'AGGREGATE: failed: {e}')


if __name__ == '__main__':
    main()


