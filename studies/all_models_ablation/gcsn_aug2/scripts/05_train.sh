#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python gcsn_adapter.py --cfg_dir /home/ubuntu/GenDATA/studies/all_models_ablation/gcsn_aug2/aug_cfg --out_dir gcsn_data && python - <<'PYEOF'
import os
import subprocess
import json
import time
import sys
start_ts = time.time()
try:
  proc = subprocess.run(['python', 'gcsn/train_gcsn.py', '--data_dir', 'gcsn_data', '--out_dir', '/home/ubuntu/GenDATA/studies/all_models_ablation/gcsn_aug2/models/gcsn'], capture_output=True, text=True, timeout=3600)
  rc = proc.returncode
except subprocess.TimeoutExpired:
  rc = 124
except Exception as e:
  rc = 1
end_ts = time.time()
metrics = {
  'start_ts': start_ts,
  'cfg_dir': '/home/ubuntu/GenDATA/studies/all_models_ablation/gcsn_aug2/aug_cfg',
  'out_dir': '/home/ubuntu/GenDATA/studies/all_models_ablation/gcsn_aug2/models/gcsn',
  'epochs': 25,
  'exit_code': rc,
  'end_ts': end_ts,
  'duration_sec': end_ts - start_ts
}
os.makedirs(os.path.dirname('/home/ubuntu/GenDATA/studies/all_models_ablation/gcsn_aug2/models/gcsn/metrics.json'), exist_ok=True)
with open('/home/ubuntu/GenDATA/studies/all_models_ablation/gcsn_aug2/models/gcsn/metrics.json', 'w') as f:
  json.dump(metrics, f, indent=2)
sys.exit(rc)
PYEOF
