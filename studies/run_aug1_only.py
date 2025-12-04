#!/usr/bin/env python3
"""
Run aug1 (level 1 augmentation) training for all models.
Executes augmentation, CFG generation, and training steps for aug1 variants only.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def load_plan(plan_path: Path) -> Dict:
    with open(plan_path, 'r') as f:
        return json.load(f)


def list_step_scripts(scripts_dir: Path) -> List[Path]:
    if not scripts_dir.exists():
        return []
    files = [p for p in scripts_dir.iterdir() if p.is_file() and p.stat().st_mode & 0o111]
    files.sort(key=lambda p: p.name)
    return files


def run_script(path: Path) -> int:
    env = os.environ.copy()
    env.setdefault('SOOT_SLICE_CLI', str(Path.cwd() / 'tools' / 'soot_slicer.sh'))
    env.setdefault('SOOT_JAR', str(Path.cwd() / 'build' / 'libs' / 'GenDATA-all.jar'))
    try:
        proc = subprocess.run([str(path)], env=env)
        return proc.returncode
    except FileNotFoundError:
        return 127


def run_variant(variant: Dict) -> bool:
    """Run all steps for a variant (aug1 only)."""
    scripts_dir = Path(variant['scripts_dir'])
    steps = list_step_scripts(scripts_dir)
    # Filter to only aug steps (skip slice and normalize - should already exist)
    aug_steps = [s for s in steps if 'augment' in s.name or 'cfg' in s.name or 'train' in s.name]
    
    if not aug_steps:
        print(f"WARN: No augmentation/CFG/train steps found for {scripts_dir}")
        return False
    
    for step in aug_steps:
        print(f"RUN: {step}")
        rc = run_script(step)
        if rc != 0:
            print(f"FAIL({rc}): {step}")
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='studies/all_models_ablation/study_plan_all_models.json')
    args = ap.parse_args()

    plan_path = Path(args.plan)
    plan = load_plan(plan_path)
    models = plan.get('models', [])

    print(f"START: Running aug1 training for all models")

    for model_entry in models:
        if not model_entry.get('supported', False):
            print(f"SKIP: Unsupported model entry {model_entry.get('model')}")
            continue
        
        model_name = model_entry.get('model', 'unknown')
        variants = model_entry.get('variants', [])
        
        # Find aug1 variant
        aug1_variant = None
        for v in variants:
            if v.get('level') == 1:
                aug1_variant = v
                break
        
        if not aug1_variant:
            print(f"SKIP: No aug1 variant found for {model_name}")
            continue
        
        print(f"\n== MODEL {model_name} LEVEL 1 ==")
        ok = run_variant(aug1_variant)
        if not ok:
            print(f"WARN: aug1 training failed for {model_name}; continuing")
        else:
            print(f"DONE: {model_name} aug1")

    print(f"FINISH: aug1 training complete")


if __name__ == '__main__':
    main()

