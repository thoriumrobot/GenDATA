#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
export CFG_OUTPUT_DIR="/home/ubuntu/GenDATA/studies/all_models_ablation/gbt_aug0/cfg"; export MODELS_DIR="/home/ubuntu/GenDATA/studies/all_models_ablation/gbt_aug0/models/gbt"; export SLICES_DIR="/home/ubuntu/GenDATA/studies/all_models_ablation/shared_normalized_slices"; python - <<'PYEOF'
import os
import subprocess
import json
import time
import sys
import re
start_ts = time.time()
try:
  proc = subprocess.run(['python', 'gbt.py'], capture_output=True, text=True, timeout=3600)
  rc = proc.returncode
  # Parse accuracy from output
  accuracy = None
  for line in proc.stdout.split('\n') + proc.stderr.split('\n'):
    if 'accuracy' in line.lower():
      try:
        m = re.search(r'([0-9.]+)', line)
        if m:
          accuracy = float(m.group(1))
      except: pass
except subprocess.TimeoutExpired:
  rc = 124
  accuracy = None
except Exception as e:
  rc = 1
  accuracy = None
end_ts = time.time()
metrics = {
  'start_ts': start_ts,
  'cfg_dir': os.environ.get('CFG_OUTPUT_DIR', ''),
  'out_dir': os.environ.get('MODELS_DIR', ''),
  'epochs': 25,
  'exit_code': rc,
  'test_accuracy': accuracy,
  'end_ts': end_ts,
  'duration_sec': end_ts - start_ts
}
os.makedirs(os.path.dirname('/home/ubuntu/GenDATA/studies/all_models_ablation/gbt_aug0/models/gbt/metrics.json'), exist_ok=True)
with open('/home/ubuntu/GenDATA/studies/all_models_ablation/gbt_aug0/models/gbt/metrics.json', 'w') as f:
  json.dump(metrics, f, indent=2)
sys.exit(rc)
PYEOF
