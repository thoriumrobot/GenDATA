#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python - <<'PY'
from pathlib import Path
from pipeline import run_cfg_generation
from slice_dataset_tools import count_cfg
src="/home/ubuntu/GenDATA/studies/gcn_aug_probe/aug0/normalized_slices"
dst="/home/ubuntu/GenDATA/studies/gcn_aug_probe/aug0/cfg"
run_cfg_generation(src, dst)
if count_cfg(Path(dst))==0:
  raise SystemExit("ERROR: No CFGs generated")
PY
