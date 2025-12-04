#!/usr/bin/env python3
"""
Study Orchestrator: generate multi-model study plans (no execution).
- Soot slicing is the default; CF fallback happens in pipeline if needed.
- Supports GCN, HGT, GBT, Causal, GCSN, DG2N, DGCRF as available.
- Writes a consolidated plan JSON and per-model shell scripts to run later.
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

SUPPORTED_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
MODEL_NAME_MAP = {
    'HGT': 'hgt',
    'GBT': 'gbt',
    'Causal': 'causal',
    'CausalEnhanced': 'causal',
    'GCN': 'gcn',
    'GCSN': 'gcsn',
    'DG2N': 'dg2n',
    'GraphCausal': 'dgcrf',
}


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def plan_for_model(model: str, project_root: str, warnings_file: str, out_root: str,
                   epochs: int, aug_levels: List[int], shared_aug_dirs: Dict[int, Path]) -> Dict:
    cfwr_root = os.getcwd()
    plan: Dict = {'model': model, 'variants': []}
    for lvl in aug_levels:
        tag = f'{model}_aug{lvl}'
        base = Path(out_root) / tag
        slices_dir = base / 'slices'
        cfg_dir = base / 'cfg'
        scripts_dir = base / 'scripts'
        models_dir = base / 'models'
        norm_dir = base / 'normalized_slices'
        # Slice (Soot default with env hints)
        slice_cmd = (
            f"export SOOT_SLICE_CLI=\"{cfwr_root}/tools/soot_slicer.sh\"; "
            f"export SOOT_JAR=\"{cfwr_root}/build/libs/GenDATA-all.jar\"; "
            f"python -c 'from pipeline import run_slicing; run_slicing(\"{project_root}\", \"{warnings_file}\", \"{cfwr_root}\", \"{slices_dir}\", \"soot\")'"
        )
        # Resolve + normalize (CF fallback if Soot empty)
        # For shared augmentation, we need a shared normalized_slices directory too
        shared_norm_dir = Path(out_root) / 'shared_normalized_slices'
        resolve_normalize = (
            f"python - <<'PY'\n"
            f"from pathlib import Path\n"
            f"from slice_dataset_tools import resolve_slice_dir, normalize_slices, count_java\n"
            f"base = Path(\"{base}\")\n"
            f"src = resolve_slice_dir(base)\n"
            f"if not src or count_java(src)==0:\n"
            f"  from pipeline import run_slicing\n"
            f"  run_slicing(\"{project_root}\", \"{warnings_file}\", \"{cfwr_root}\", \"{slices_dir}\", \"cf\")\n"
            f"  src = resolve_slice_dir(base)\n"
            f"if not src or count_java(src)==0:\n"
            f"  raise SystemExit(\"ERROR: No slices (.java) found after CF fallback\")\n"
            f"# Normalize to shared directory (created once, used by all)\n"
            f"shared_norm = Path(\"{shared_norm_dir}\")\n"
            f"if not shared_norm.exists() or count_java(shared_norm)==0:\n"
            f"  n = normalize_slices(src, shared_norm)\n"
            f"  print('NORMALIZED:' + str(n) + ' (shared)')\n"
            f"else:\n"
            f"  print('USING_EXISTING_SHARED_NORMALIZED (' + str(count_java(shared_norm)) + ' files)')\n"
            f"# Also copy to model-specific for backward compatibility\n"
            f"dst = Path(\"{norm_dir}\")\n"
            f"n = normalize_slices(src, dst)\n"
            f"print(\"NORMALIZED:\" + str(n))\n"
            f"PY"
        )
        # Augmentation - use shared directories
        if lvl == 0:
            aug_cmd = None
            train_cfg_dir = str(cfg_dir)
            cfg_source = str(shared_norm_dir)  # Use shared normalized slices
        else:
            # Use shared augmentation directory
            shared_aug_dir = shared_aug_dirs[lvl]
            # Augmentation command - only generate if directory doesn't exist or is empty
            aug_cmd = (
                f"python - <<'PY'\n"
                f"from pathlib import Path\n"
                f"from slice_dataset_tools import count_java\n"
                f"shared_aug = Path(\"{shared_aug_dir}\")\n"
                f"shared_norm = Path(\"{shared_norm_dir}\")\n"
                f"if not shared_aug.exists() or count_java(shared_aug)==0:\n"
                f"  import subprocess\n"
                f"  subprocess.run([\n"
                f"    'python', 'enhanced_semantic_augment_slices.py',\n"
                f"    str(shared_norm), str(shared_aug),\n"
                f"    '--variants', '{lvl}',\n"
                f"    '--sequence-len', '2',\n"
                f"    '--max-depth', '3',\n"
                f"    '--min-diff', '0.03',\n"
                f"    '--disabled', 'switch_statement', 'variable_operation', 'string_concatenation', 'numeric_literal',\n"
                f"    '--focus-nodes', 'control', 'dataflow'\n"
                f"  ], check=True)\n"
                f"  print('GENERATED_SHARED_AUG_L{lvl}:' + str(count_java(shared_aug)))\n"
                f"else:\n"
                f"  print('USING_EXISTING_SHARED_AUG_L{lvl} (' + str(count_java(shared_aug)) + ' files)')\n"
                f"PY"
            )
            train_cfg_dir = str(base / 'aug_cfg')
            cfg_source = str(shared_aug_dir)  # Use shared augmented slices
        # CFG generation with guard
        cfg_cmd = (
            f"python - <<'PY'\n"
            f"from pathlib import Path\n"
            f"from pipeline import run_cfg_generation\n"
            f"from slice_dataset_tools import count_cfg\n"
            f"src=\"{cfg_source}\"\n"
            f"dst=\"{train_cfg_dir if lvl>0 else cfg_dir}\"\n"
            f"run_cfg_generation(src, dst)\n"
            f"if count_cfg(Path(dst))==0:\n"
            f"  raise SystemExit(\"ERROR: No CFGs generated\")\n"
            f"PY"
        )
        # Train command per model with env vars and metrics output
        metrics_json = f"{models_dir}/{model}/metrics.json"
        if model == 'gcn':
            train_cmd = (
                f"python gcn_train.py --cfg_dir {train_cfg_dir} --out_dir {models_dir}/gcn "
                f"--epochs {epochs} --layers 3 --hidden 256 --dropout 0.2 --early_stop_patience 5 --metrics_path {metrics_json}"
            )
        elif model == 'hgt':
            # HGT uses env vars and doesn't emit metrics; wrapper will generate it
            train_cmd = (
                f"export CFG_OUTPUT_DIR=\"{train_cfg_dir}\"; "
                f"export MODELS_DIR=\"{models_dir}/{model}\"; "
                f"export SLICES_DIR=\"{cfg_source}\"; "
                f"python - <<'PYEOF'\n"
                f"import os\n"
                f"import subprocess\n"
                f"import json\n"
                f"import time\n"
                f"import sys\n"
                f"start_ts = time.time()\n"
                f"try:\n"
                f"  proc = subprocess.run(['python', 'hgt.py'], capture_output=True, text=True, timeout=3600)\n"
                f"  rc = proc.returncode\n"
                f"  # Parse val loss from output if available\n"
                f"  val_loss = None\n"
                f"  for line in proc.stdout.split('\\n'):\n"
                f"    if 'Val Loss:' in line:\n"
                f"      try:\n"
                f"        val_loss = float(line.split('Val Loss:')[1].strip().split()[0])\n"
                f"      except: pass\n"
                f"except subprocess.TimeoutExpired:\n"
                f"  rc = 124\n"
                f"  val_loss = None\n"
                f"except Exception as e:\n"
                f"  rc = 1\n"
                f"  val_loss = None\n"
                f"end_ts = time.time()\n"
                f"metrics = {{\n"
                f"  'start_ts': start_ts,\n"
                f"  'cfg_dir': os.environ.get('CFG_OUTPUT_DIR', ''),\n"
                f"  'out_dir': os.environ.get('MODELS_DIR', ''),\n"
                f"  'epochs': {epochs},\n"
                f"  'exit_code': rc,\n"
                f"  'best_val_loss': val_loss,\n"
                f"  'end_ts': end_ts,\n"
                f"  'duration_sec': end_ts - start_ts\n"
                f"}}\n"
                f"os.makedirs(os.path.dirname('{metrics_json}'), exist_ok=True)\n"
                f"with open('{metrics_json}', 'w') as f:\n"
                f"  json.dump(metrics, f, indent=2)\n"
                f"sys.exit(rc)\n"
                f"PYEOF"
            )
        elif model == 'gbt':
            # GBT uses env vars and doesn't emit metrics; wrapper will generate it
            train_cmd = (
                f"export CFG_OUTPUT_DIR=\"{train_cfg_dir}\"; "
                f"export MODELS_DIR=\"{models_dir}/{model}\"; "
                f"export SLICES_DIR=\"{cfg_source}\"; "
                f"python - <<'PYEOF'\n"
                f"import os\n"
                f"import subprocess\n"
                f"import json\n"
                f"import time\n"
                f"import sys\n"
                f"import re\n"
                f"start_ts = time.time()\n"
                f"try:\n"
                f"  proc = subprocess.run(['python', 'gbt.py'], capture_output=True, text=True, timeout=3600)\n"
                f"  rc = proc.returncode\n"
                f"  # Parse accuracy from output\n"
                f"  accuracy = None\n"
                f"  for line in proc.stdout.split('\\n') + proc.stderr.split('\\n'):\n"
                f"    if 'accuracy' in line.lower():\n"
                f"      try:\n"
                f"        m = re.search(r'([0-9.]+)', line)\n"
                f"        if m:\n"
                f"          accuracy = float(m.group(1))\n"
                f"      except: pass\n"
                f"except subprocess.TimeoutExpired:\n"
                f"  rc = 124\n"
                f"  accuracy = None\n"
                f"except Exception as e:\n"
                f"  rc = 1\n"
                f"  accuracy = None\n"
                f"end_ts = time.time()\n"
                f"metrics = {{\n"
                f"  'start_ts': start_ts,\n"
                f"  'cfg_dir': os.environ.get('CFG_OUTPUT_DIR', ''),\n"
                f"  'out_dir': os.environ.get('MODELS_DIR', ''),\n"
                f"  'epochs': {epochs},\n"
                f"  'exit_code': rc,\n"
                f"  'test_accuracy': accuracy,\n"
                f"  'end_ts': end_ts,\n"
                f"  'duration_sec': end_ts - start_ts\n"
                f"}}\n"
                f"os.makedirs(os.path.dirname('{metrics_json}'), exist_ok=True)\n"
                f"with open('{metrics_json}', 'w') as f:\n"
                f"  json.dump(metrics, f, indent=2)\n"
                f"sys.exit(rc)\n"
                f"PYEOF"
            )
        elif model == 'causal':
            train_cmd = (
                f"python - <<'PYEOF'\n"
                f"import os\n"
                f"import subprocess\n"
                f"import json\n"
                f"import time\n"
                f"import sys\n"
                f"start_ts = time.time()\n"
                f"try:\n"
                f"  proc = subprocess.run(['python', 'causal_model.py'], capture_output=True, text=True, timeout=3600, env={{**os.environ, 'CFG_OUTPUT_DIR': '{train_cfg_dir}', 'MODELS_DIR': '{models_dir}/{model}'}})\n"
                f"  rc = proc.returncode\n"
                f"except subprocess.TimeoutExpired:\n"
                f"  rc = 124\n"
                f"except Exception as e:\n"
                f"  rc = 1\n"
                f"end_ts = time.time()\n"
                f"metrics = {{\n"
                f"  'start_ts': start_ts,\n"
                f"  'cfg_dir': '{train_cfg_dir}',\n"
                f"  'out_dir': '{models_dir}/{model}',\n"
                f"  'epochs': {epochs},\n"
                f"  'exit_code': rc,\n"
                f"  'end_ts': end_ts,\n"
                f"  'duration_sec': end_ts - start_ts\n"
                f"}}\n"
                f"os.makedirs(os.path.dirname('{metrics_json}'), exist_ok=True)\n"
                f"with open('{metrics_json}', 'w') as f:\n"
                f"  json.dump(metrics, f, indent=2)\n"
                f"sys.exit(rc)\n"
                f"PYEOF"
            )
        elif model == 'gcsn':
            train_cmd = (
                f"python gcsn_adapter.py --cfg_dir {train_cfg_dir} --out_dir gcsn_data && "
                f"python - <<'PYEOF'\n"
                f"import os\n"
                f"import subprocess\n"
                f"import json\n"
                f"import time\n"
                f"import sys\n"
                f"start_ts = time.time()\n"
                f"try:\n"
                f"  proc = subprocess.run(['python', 'gcsn/train_gcsn.py', '--data_dir', 'gcsn_data', '--out_dir', '{models_dir}/gcsn'], capture_output=True, text=True, timeout=3600)\n"
                f"  rc = proc.returncode\n"
                f"except subprocess.TimeoutExpired:\n"
                f"  rc = 124\n"
                f"except Exception as e:\n"
                f"  rc = 1\n"
                f"end_ts = time.time()\n"
                f"metrics = {{\n"
                f"  'start_ts': start_ts,\n"
                f"  'cfg_dir': '{train_cfg_dir}',\n"
                f"  'out_dir': '{models_dir}/gcsn',\n"
                f"  'epochs': {epochs},\n"
                f"  'exit_code': rc,\n"
                f"  'end_ts': end_ts,\n"
                f"  'duration_sec': end_ts - start_ts\n"
                f"}}\n"
                f"os.makedirs(os.path.dirname('{metrics_json}'), exist_ok=True)\n"
                f"with open('{metrics_json}', 'w') as f:\n"
                f"  json.dump(metrics, f, indent=2)\n"
                f"sys.exit(rc)\n"
                f"PYEOF"
            )
        elif model == 'dg2n':
            train_cmd = (
                f"python dg2n_adapter.py --cfg_dir {train_cfg_dir} --out_dir dg2n_data && "
                f"python - <<'PYEOF'\n"
                f"import os\n"
                f"import subprocess\n"
                f"import json\n"
                f"import time\n"
                f"import sys\n"
                f"start_ts = time.time()\n"
                f"try:\n"
                f"  proc = subprocess.run(['python', 'dg2n/train_dg2n.py', '--data_dir', 'dg2n_data', '--out_dir', '{models_dir}/dg2n'], capture_output=True, text=True, timeout=3600)\n"
                f"  rc = proc.returncode\n"
                f"except subprocess.TimeoutExpired:\n"
                f"  rc = 124\n"
                f"except Exception as e:\n"
                f"  rc = 1\n"
                f"end_ts = time.time()\n"
                f"metrics = {{\n"
                f"  'start_ts': start_ts,\n"
                f"  'cfg_dir': '{train_cfg_dir}',\n"
                f"  'out_dir': '{models_dir}/dg2n',\n"
                f"  'epochs': {epochs},\n"
                f"  'exit_code': rc,\n"
                f"  'end_ts': end_ts,\n"
                f"  'duration_sec': end_ts - start_ts\n"
                f"}}\n"
                f"os.makedirs(os.path.dirname('{metrics_json}'), exist_ok=True)\n"
                f"with open('{metrics_json}', 'w') as f:\n"
                f"  json.dump(metrics, f, indent=2)\n"
                f"sys.exit(rc)\n"
                f"PYEOF"
            )
        elif model == 'dgcrf':
            train_cmd = (
                f"python dg2n_adapter.py --cfg_dir {train_cfg_dir} --out_dir dgcrf_data && "
                f"python - <<'PYEOF'\n"
                f"import os\n"
                f"import subprocess\n"
                f"import json\n"
                f"import time\n"
                f"import sys\n"
                f"start_ts = time.time()\n"
                f"try:\n"
                f"  proc = subprocess.run(['python', 'train_dgcrf.py', '--data_dir', 'dgcrf_data', '--out_dir', '{models_dir}/dgcrf'], capture_output=True, text=True, timeout=3600)\n"
                f"  rc = proc.returncode\n"
                f"except subprocess.TimeoutExpired:\n"
                f"  rc = 124\n"
                f"except Exception as e:\n"
                f"  rc = 1\n"
                f"end_ts = time.time()\n"
                f"metrics = {{\n"
                f"  'start_ts': start_ts,\n"
                f"  'cfg_dir': '{train_cfg_dir}',\n"
                f"  'out_dir': '{models_dir}/dgcrf',\n"
                f"  'epochs': {epochs},\n"
                f"  'exit_code': rc,\n"
                f"  'end_ts': end_ts,\n"
                f"  'duration_sec': end_ts - start_ts\n"
                f"}}\n"
                f"os.makedirs(os.path.dirname('{metrics_json}'), exist_ok=True)\n"
                f"with open('{metrics_json}', 'w') as f:\n"
                f"  json.dump(metrics, f, indent=2)\n"
                f"sys.exit(rc)\n"
                f"PYEOF"
            )
        else:
            train_cmd = "# unsupported model"
        # Scripts
        steps = [('01_slice.sh', slice_cmd), ('02_resolve_normalize.sh', resolve_normalize)]
        if aug_cmd:
            steps.append(('03_augment.sh', aug_cmd))
            steps.append(('04_cfg.sh', cfg_cmd))
        else:
            steps.append(('03_cfg.sh', cfg_cmd))
        steps.append(('05_train.sh', train_cmd))
        for name, cmd in steps:
            write(scripts_dir / name, f"#!/usr/bin/env bash\nset -euo pipefail\ncd {cfwr_root}\n{cmd}\n")
            os.chmod(scripts_dir / name, 0o755)
        plan['variants'].append({
            'level': lvl,
            'scripts_dir': str(scripts_dir),
            'cfg_dir': train_cfg_dir if lvl>0 else str(cfg_dir),
            'models_dir': str(models_dir)
        })
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project_root', required=True)
    ap.add_argument('--warnings_file', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--models', nargs='+', default=['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf'])
    ap.add_argument('--aug_levels', nargs='+', type=int, default=[0,1,2])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--dry_run', action='store_true', default=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Create shared augmentation directories for each augmentation level > 0
    shared_aug_dirs: Dict[int, Path] = {}
    for lvl in args.aug_levels:
        if lvl > 0:
            shared_aug_dirs[lvl] = out_root / f'shared_aug_slices_l{lvl}'
    
    plan_all = {
        'project_root': args.project_root,
        'warnings_file': args.warnings_file,
        'out_root': str(out_root),
        'slicer': 'soot',
        'models': []
    }
    for m in args.models:
        if m not in SUPPORTED_MODELS:
            plan_all['models'].append({'model': m, 'supported': False})
            continue
        plan = plan_for_model(m, args.project_root, args.warnings_file, str(out_root), args.epochs, args.aug_levels, shared_aug_dirs)
        plan['supported'] = True
        plan_all['models'].append(plan)
    plan_path = out_root / 'study_plan_all_models.json'
    write(plan_path, json.dumps(plan_all, indent=2))
    print(f'Wrote study plan: {plan_path}')
    if not args.dry_run:
        print('Execution is not performed automatically. Use the generated scripts.')


if __name__ == '__main__':
    main()
