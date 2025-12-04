#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python - <<'PY'
import json, sys, os
from pathlib import Path
from slice_dataset_tools import count_cfg
cfg=Path("/home/ubuntu/GenDATA/studies/gcn_aug_probe/aug1/aug_cfg")
if count_cfg(cfg)==0:
  print('SKIP: No CFGs; not training'); sys.exit(0)
os.system("python gcn_train.py --cfg_dir /home/ubuntu/GenDATA/studies/gcn_aug_probe/aug1/aug_cfg --out_dir /home/ubuntu/GenDATA/studies/gcn_aug_probe/aug1/models/gcn --epochs 20 --layers 3 --hidden 256 --dropout 0.2 --early_stop_patience 5 --metrics_path /home/ubuntu/GenDATA/studies/gcn_aug_probe/aug1/models/gcn/metrics.json")
PY
