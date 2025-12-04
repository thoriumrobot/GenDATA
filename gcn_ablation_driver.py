#!/usr/bin/env python3
"""
GCN Ablation Driver: generate plans to run GCN with increasing augmentation.
- Uses Soot slicing by default
- Outputs a plan JSON and shell scripts per setting
- By default, does NOT execute (dry-run)
"""
import os
import json
import argparse
from pathlib import Path

AUG_LEVELS_DEFAULT = [0, 1, 2]


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def gen_commands(project_root: str, warnings_file: str, out_root: str, epochs: int,
                 layers: int, hidden: int, dropout: float, patience: int,
                 aug_levels: list[int]) -> dict:
    plan = {
        'project_root': project_root,
        'warnings_file': warnings_file,
        'out_root': out_root,
        'slicer': 'soot',
        'gcns': []
    }
    cfwr_root = os.getcwd()
    for lvl in aug_levels:
        tag = f'aug{lvl}'
        out_dir = Path(out_root) / tag
        slices_dir = out_dir / 'slices'
        cfg_dir = out_dir / 'cfg'
        models_dir = out_dir / 'models'
        scripts_dir = out_dir / 'scripts'
        norm_dir = out_dir / 'normalized_slices'
        # Slicing (Soot), environment hints
        slice_cmd = (
            f"export SOOT_SLICE_CLI=\"{cfwr_root}/tools/soot_slicer.sh\"; "
            f"export SOOT_JAR=\"{cfwr_root}/build/libs/GenDATA-all.jar\"; "
            f"python -c 'from pipeline import run_slicing; run_slicing(\"{project_root}\", \"{warnings_file}\", \"{cfwr_root}\", \"{slices_dir}\", \"soot\")'"
        )
        # Resolve + normalize script
        resolve_normalize = (
            f"python - <<'PY'\n"
            f"from pathlib import Path\n"
            f"from slice_dataset_tools import resolve_slice_dir, normalize_slices, count_java\n"
            f"base = Path(\"{out_dir}\")\n"
            f"src = resolve_slice_dir(base)\n"
            f"if not src or count_java(src)==0:\n"
            f"  from pipeline import run_slicing\n"
            f"  run_slicing(\"{project_root}\", \"{warnings_file}\", \"{cfwr_root}\", \"{slices_dir}\", \"cf\")\n"
            f"  src = resolve_slice_dir(base)\n"
            f"if not src or count_java(src)==0:\n"
            f"  raise SystemExit(\"ERROR: No slices (.java) found after CF fallback\")\n"
            f"dst = Path(\"{norm_dir}\")\n"
            f"copied = normalize_slices(src, dst)\n"
            f"print(\"NORMALIZED:\" + str(copied))\n"
            f"PY"
        )
        # Augmentation level: 0 means use normalized slices; >0 means augment per-slice with N variants
        if lvl == 0:
            aug_cmd = None
            train_cfg_dir = str(cfg_dir)
            cfg_src_dir = str(norm_dir)
        else:
            aug_slices_dir = out_dir / 'aug_slices'
            aug_cmd = (
                f"python enhanced_semantic_augment_slices.py {norm_dir} {aug_slices_dir} "
                f"--variants {lvl} --sequence-len 2 --max-depth 3 --min-diff 0.03 "
                f"--disabled switch_statement variable_operation string_concatenation numeric_literal --focus-nodes control dataflow"
            )
            train_cfg_dir = str(out_dir / 'aug_cfg')
            cfg_src_dir = str(aug_slices_dir)
        # CFG generation with guard
        cfg_cmd = (
            f"python - <<'PY'\n"
            f"from pathlib import Path\n"
            f"from pipeline import run_cfg_generation\n"
            f"from slice_dataset_tools import count_cfg\n"
            f"src=\"{cfg_src_dir}\"\n"
            f"dst=\"{train_cfg_dir}\"\n"
            f"run_cfg_generation(src, dst)\n"
            f"if count_cfg(Path(dst))==0:\n"
            f"  raise SystemExit(\"ERROR: No CFGs generated\")\n"
            f"PY"
        )
        # Train GCN (only if CFGs exist)
        train_cmd = (
            f"python - <<'PY'\n"
            f"import json, sys, os\n"
            f"from pathlib import Path\n"
            f"from slice_dataset_tools import count_cfg\n"
            f"cfg=Path(\"{train_cfg_dir}\")\n"
            f"if count_cfg(cfg)==0:\n"
            f"  print('SKIP: No CFGs; not training'); sys.exit(0)\n"
            f"os.system(\"python gcn_train.py --cfg_dir {train_cfg_dir} --out_dir {models_dir}/gcn --epochs {epochs} --layers {layers} --hidden {hidden} --dropout {dropout} --early_stop_patience {patience} --metrics_path {models_dir}/gcn/metrics.json\")\n"
            f"PY"
        )
        scripts = []
        scripts.append(('01_slice.sh', slice_cmd))
        scripts.append(('02_resolve_normalize.sh', resolve_normalize))
        if aug_cmd:
            scripts.append(('03_augment.sh', aug_cmd))
            scripts.append(('04_cfg.sh', cfg_cmd))
        else:
            scripts.append(('03_cfg.sh', cfg_cmd))
        scripts.append(('05_train.sh', train_cmd))
        # Write scripts
        for name, cmd in scripts:
            write_file(scripts_dir / name, f"#!/usr/bin/env bash\nset -euo pipefail\ncd {cfwr_root}\n{cmd}\n")
            os.chmod(scripts_dir / name, 0o755)
        plan['gcns'].append({
            'level': lvl,
            'scripts_dir': str(scripts_dir),
            'slices_dir': str(slices_dir),
            'normalized_dir': str(norm_dir),
            'cfg_dir': str(cfg_dir),
            'augmented_cfg_dir': train_cfg_dir if lvl > 0 else None,
            'models_dir': str(models_dir),
        })
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project_root', required=True)
    ap.add_argument('--warnings_file', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--aug_levels', type=int, nargs='+', default=AUG_LEVELS_DEFAULT)
    ap.add_argument('--dry_run', action='store_true', default=True)
    args = ap.parse_args()

    plan = gen_commands(args.project_root, args.warnings_file, args.out_root, args.epochs,
                        args.layers, args.hidden, args.dropout, args.patience, args.aug_levels)
    plan_path = Path(args.out_root) / 'gcn_ablation_plan.json'
    write_file(plan_path, json.dumps(plan, indent=2))
    print(f'Wrote GCN plan: {plan_path}')
    if not args.dry_run:
        print('Note: execution is disabled by default. Re-run scripts manually if needed.')


if __name__ == '__main__':
    main()
