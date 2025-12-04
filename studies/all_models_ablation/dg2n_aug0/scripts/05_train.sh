#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python dg2n_adapter.py --cfg_dir /home/ubuntu/GenDATA/studies/all_models_ablation/dg2n_aug0/cfg --out_dir dg2n_data && python - <<'PYEOF'
import os
import subprocess
import json
import time
import sys
start_ts = time.time()
try:
  proc = subprocess.run(['python', 'dg2n/train_dg2n.py', '--data_dir', 'dg2n_data', '--out_dir', '/home/ubuntu/GenDATA/studies/all_models_ablation/dg2n_aug0/models/dg2n'], capture_output=True, text=True, timeout=3600)
  rc = proc.returncode
except subprocess.TimeoutExpired:
  rc = 124
except Exception as e:
  rc = 1
end_ts = time.time()
metrics = {
  'start_ts': start_ts,
  'cfg_dir': '/home/ubuntu/GenDATA/studies/all_models_ablation/dg2n_aug0/cfg',
  'out_dir': '/home/ubuntu/GenDATA/studies/all_models_ablation/dg2n_aug0/models/dg2n',
  'epochs': 25,
  'exit_code': rc,
  'end_ts': end_ts,
  'duration_sec': end_ts - start_ts
}
os.makedirs(os.path.dirname('/home/ubuntu/GenDATA/studies/all_models_ablation/dg2n_aug0/models/dg2n/metrics.json'), exist_ok=True)
with open('/home/ubuntu/GenDATA/studies/all_models_ablation/dg2n_aug0/models/dg2n/metrics.json', 'w') as f:
  json.dump(metrics, f, indent=2)
sys.exit(rc)
PYEOF
