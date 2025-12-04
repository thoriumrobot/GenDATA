#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
export CFG_OUTPUT_DIR="/home/ubuntu/GenDATA/studies/all_models_ablation/hgt_aug2/aug_cfg"; export MODELS_DIR="/home/ubuntu/GenDATA/studies/all_models_ablation/hgt_aug2/models/hgt"; export SLICES_DIR="/home/ubuntu/GenDATA/studies/all_models_ablation/shared_aug_slices_l2"; python - <<'PYEOF'
import os
import subprocess
import json
import time
import sys
start_ts = time.time()
try:
  proc = subprocess.run(['python', 'hgt.py'], capture_output=True, text=True, timeout=3600)
  rc = proc.returncode
  # Parse val loss from output if available
  val_loss = None
  for line in proc.stdout.split('\n'):
    if 'Val Loss:' in line:
      try:
        val_loss = float(line.split('Val Loss:')[1].strip().split()[0])
      except: pass
except subprocess.TimeoutExpired:
  rc = 124
  val_loss = None
except Exception as e:
  rc = 1
  val_loss = None
end_ts = time.time()
metrics = {
  'start_ts': start_ts,
  'cfg_dir': os.environ.get('CFG_OUTPUT_DIR', ''),
  'out_dir': os.environ.get('MODELS_DIR', ''),
  'epochs': 25,
  'exit_code': rc,
  'best_val_loss': val_loss,
  'end_ts': end_ts,
  'duration_sec': end_ts - start_ts
}
os.makedirs(os.path.dirname('/home/ubuntu/GenDATA/studies/all_models_ablation/hgt_aug2/models/hgt/metrics.json'), exist_ok=True)
with open('/home/ubuntu/GenDATA/studies/all_models_ablation/hgt_aug2/models/hgt/metrics.json', 'w') as f:
  json.dump(metrics, f, indent=2)
sys.exit(rc)
PYEOF
