#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python - <<'PY'
from pathlib import Path
from pipeline import run_cfg_generation
from slice_dataset_tools import count_cfg
src="/home/ubuntu/GenDATA/studies/all_models_ablation/shared_normalized_slices"
dst="/home/ubuntu/GenDATA/studies/all_models_ablation/gbt_aug0/cfg"
run_cfg_generation(src, dst)
if count_cfg(Path(dst))==0:
  raise SystemExit("ERROR: No CFGs generated")
PY
