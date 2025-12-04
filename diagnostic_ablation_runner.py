#!/usr/bin/env python3
"""
Diagnostic Ablation Runner with Comprehensive Logging

Steps:
  1) Sample Java files (real data) with seed=42
  2) Augment (simple/enhanced) with per-variant timeout and detailed logs
  3) Slice augmented variants (soot/specimin/cf) with command logging
  4) Generate CFGs with per-file timing
  5) Train lightweight GCN briefly to verify pipeline

Emits a JSON summary report at the end.
"""

import os
import sys
import time
import json
import glob
import argparse
import logging
import random
import subprocess
from typing import List, Dict, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def now() -> float:
    return time.monotonic()


def run_cmd(cmd: List[str], cwd: str | None = None, timeout: int | None = None) -> Dict[str, Any]:
    start = now()
    logger.info(f"$ {' '.join(cmd)} (cwd={cwd or os.getcwd()}, timeout={timeout})")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        dur = now() - start
        stdout = (res.stdout or '')
        stderr = (res.stderr or '')
        if res.returncode != 0:
            logger.error(f"Command failed rc={res.returncode} in {dur:.2f}s; stderr(head):\n{stderr[:1000]}")
        else:
            logger.info(f"Command ok rc=0 in {dur:.2f}s; stdout(head):\n{stdout[:500]}")
        return {
            'returncode': res.returncode,
            'stdout': stdout,
            'stderr': stderr,
            'duration_sec': dur,
        }
    except subprocess.TimeoutExpired as e:
        dur = now() - start
        logger.error(f"Command timeout after {dur:.2f}s: {' '.join(cmd)}")
        return {
            'returncode': -1,
            'stdout': e.stdout or '',
            'stderr': e.stderr or 'timeout',
            'duration_sec': dur,
            'timeout': True,
        }


def sample_java_files(project_root: str, max_files: int, seed: int = 42) -> List[str]:
    files = glob.glob(os.path.join(project_root, '**', '*.java'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    random.Random(seed).shuffle(files)
    sampled = files[:max_files]
    logger.info(f"Sampled {len(sampled)} files from {len(files)} total. Example: {sampled[:3]}")
    return sampled


def augment_files(files: List[str], project_root: str, out_root: str, augment_mode: str, max_variants: int,
                  variant_timeout_sec: int, deadline_ts: float, log_every: int) -> Dict[str, Any]:
    from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline

    pipe = SimpleAnnotationTypePipeline(
        project_root=project_root,
        warnings_file=os.path.join(os.path.dirname(__file__), 'index1.small.subset.out'),
        cfwr_root=out_root,
        mode='train',
        device='cpu',
        augment_first=True,
        disable_random_walk=(augment_mode == 'simple'),
        run_checker_on_target=False,
    )
    # Respect caps and timeouts
    pipe.max_files_to_process = len(files)
    pipe.max_variants_per_file = max_variants
    pipe.time_limit_deadline = deadline_ts
    pipe.log_interval = log_every
    # Variant-level timeout used by optimizer
    pipe.variant_timeout_sec = variant_timeout_sec

    # Monkey-patch file list to use our sampled set
    def iter_files_override():
        for f in files:
            yield f

    import types
    pipe._iter_project_files = types.MethodType(lambda self: iter_files_override(), pipe)

    # Execute only augmentation step with detailed timings by calling the private method
    start = now()
    ok = pipe._augment_original_code()
    dur = now() - start
    return {
        'success': bool(ok),
        'duration_sec': dur,
        'augmented_dir': getattr(pipe, 'augmented_code_dir', os.path.join(out_root, 'augmented_code_unified')),
    }


def slice_augmented(augmented_dir: str, cfwr_root: str, project_root: str, warnings_file: str,
                    slicer: str, out_slices_dir: str, deadline_ts: float, log_every: int) -> Dict[str, Any]:
    from pipeline import run_slicing
    os.makedirs(out_slices_dir, exist_ok=True)
    start = now()
    # Ensure Soot wrapper and jar are visible when using soot
    if slicer == 'soot':
        soot_cli = os.path.join(cfwr_root, 'tools', 'soot_slicer.sh')
        if os.path.isfile(soot_cli):
            os.environ['SOOT_SLICE_CLI'] = soot_cli
        soot_jar = os.path.join(cfwr_root, 'build', 'libs', 'GenDATA-all.jar')
        if os.path.isfile(soot_jar):
            os.environ['SOOT_JAR'] = soot_jar
    # Run slicer once over the augmented directory root; pass base dir and slicer
    try:
        run_slicing(project_root=project_root, warnings_file=warnings_file, cfwr_root=cfwr_root,
                    base_slices_dir=out_slices_dir, slicer_type=slicer)
        dur = now() - start
        # Count slices
        count = 0
        for root, _, files in os.walk(out_slices_dir):
            for f in files:
                if f.endswith('.java'):
                    count += 1
        logger.info(f"Slicing complete with {count} .java slices in {dur:.2f}s")
        if slicer == 'soot' and count == 0:
            logger.error("Soot produced zero slices; verify wrapper/JAR and member/line mapping. Falling back to CF.")
            # Attempt CF fallback in the same directory
            start_cf = now()
            run_slicing(project_root=project_root, warnings_file=warnings_file, cfwr_root=cfwr_root,
                        base_slices_dir=out_slices_dir, slicer_type='cf')
            dur_cf = now() - start_cf
            cf_count = 0
            for root, _, files in os.walk(out_slices_dir):
                for f in files:
                    if f.endswith('.java'):
                        cf_count += 1
            logger.info(f"CF fallback produced {cf_count} slices in {dur_cf:.2f}s")
            return {'success': cf_count > 0, 'duration_sec': dur + dur_cf, 'slice_count': cf_count, 'fallback': 'cf'}
        return {'success': True, 'duration_sec': dur, 'slice_count': count}
    except SystemExit as e:
        dur = now() - start
        logger.error(f"Slicing system exit: {e}")
        return {'success': False, 'duration_sec': dur, 'error': str(e)}
    except Exception as e:
        dur = now() - start
        logger.error(f"Slicing error: {e}")
        return {'success': False, 'duration_sec': dur, 'error': str(e)}


def generate_cfgs(slices_dir: str, cfg_out_dir: str) -> Dict[str, Any]:
    from pipeline import run_cfg_generation
    os.makedirs(cfg_out_dir, exist_ok=True)
    start = now()
    try:
        run_cfg_generation(slices_dir, cfg_out_dir)
        dur = now() - start
        # Count CFG jsons
        count = 0
        for root, _, files in os.walk(cfg_out_dir):
            for f in files:
                if f.endswith('.json'):
                    count += 1
        logger.info(f"CFG generation complete with {count} JSONs in {dur:.2f}s")
        return {'success': True, 'duration_sec': dur, 'cfg_count': count}
    except Exception as e:
        dur = now() - start
        logger.error(f"CFG generation error: {e}")
        return {'success': False, 'duration_sec': dur, 'error': str(e)}


def quick_train_gcn(cfg_dir: str, models_dir: str) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    # Train minimal epochs to validate path
    cmd = [sys.executable, 'gcn_train.py', '--cfg_dir', cfg_dir, '--out_dir', os.path.join(models_dir, 'gcn'), '--epochs', '2']
    res = run_cmd(cmd, cwd=os.getcwd(), timeout=900)
    return {'success': res.get('returncode') == 0, 'duration_sec': res.get('duration_sec', 0), 'rc': res.get('returncode'), 'stderr_head': (res.get('stderr') or '')[:1000]}


def main():
    ap = argparse.ArgumentParser(description='Diagnostic ablation runner with comprehensive logging')
    ap.add_argument('--project_root', required=True)
    ap.add_argument('--warnings_file', required=True)
    ap.add_argument('--output_dir', default='diagnostic_output')
    ap.add_argument('--max_files', type=int, default=20)
    ap.add_argument('--max_variants', type=int, default=3)
    ap.add_argument('--time_limit_hours', type=float, default=1.0)
    ap.add_argument('--augment_mode', choices=['simple', 'enhanced'], default='simple')
    ap.add_argument('--variant_timeout_sec', type=int, default=10)
    ap.add_argument('--slicer', choices=['soot', 'specimin', 'cf'], default='soot')
    ap.add_argument('--log_every', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report: Dict[str, Any] = {'args': vars(args), 'stages': {}, 'start_ts': time.time()}
    deadline_ts = time.time() + args.time_limit_hours * 3600

    # Stage 1: sampling
    sampled = sample_java_files(args.project_root, args.max_files)
    report['stages']['sampling'] = {'count': len(sampled)}
    if time.time() > deadline_ts:
        logger.error('Deadline reached after sampling; aborting')
        pass

    # Stage 2: augmentation
    aug = augment_files(sampled, args.project_root, args.output_dir, args.augment_mode, args.max_variants,
                        args.variant_timeout_sec, deadline_ts, args.log_every)
    report['stages']['augmentation'] = aug
    if not aug.get('success'):
        logger.error('Augmentation failed; stopping early')
        with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        return 2
    if time.time() > deadline_ts:
        logger.error('Deadline reached after augmentation; aborting')
        with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        return 3

    # Stage 3: slicing
    slices_dir = os.path.join(args.output_dir, 'diagnostic_slices')
    slicing = slice_augmented(aug['augmented_dir'], os.getcwd(), args.project_root, args.warnings_file,
                              args.slicer, slices_dir, deadline_ts, args.log_every)
    report['stages']['slicing'] = slicing
    if not slicing.get('success'):
        logger.error('Slicing failed; stopping early')
        with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        return 4
    if time.time() > deadline_ts:
        logger.error('Deadline reached after slicing; aborting')
        with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        return 5

    # Stage 4: CFG generation
    cfg_out = os.path.join(args.output_dir, 'diagnostic_cfg')
    cfg = generate_cfgs(slices_dir, cfg_out)
    report['stages']['cfg'] = cfg
    if not cfg.get('success'):
        logger.error('CFG generation failed; stopping early')
        with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        return 6
    if time.time() > deadline_ts:
        logger.error('Deadline reached after cfg; aborting')
        with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        return 7

    # Stage 5: quick train
    train = quick_train_gcn(cfg_out, os.path.join(args.output_dir, 'models'))
    report['stages']['train'] = train

    report['end_ts'] = time.time()
    report['duration_sec'] = report['end_ts'] - report['start_ts']
    with open(os.path.join(args.output_dir, 'diagnostic_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Diagnostic complete in {report['duration_sec']:.2f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())


