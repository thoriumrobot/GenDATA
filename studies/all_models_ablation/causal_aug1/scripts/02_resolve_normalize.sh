#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python - <<'PY'
from pathlib import Path
from slice_dataset_tools import resolve_slice_dir, normalize_slices, count_java
base = Path("/home/ubuntu/GenDATA/studies/all_models_ablation/causal_aug1")
src = resolve_slice_dir(base)
if not src or count_java(src)==0:
  from pipeline import run_slicing
  run_slicing("/home/ubuntu/checker-framework/checker/tests/index", "/home/ubuntu/GenDATA/index1.out", "/home/ubuntu/GenDATA", "/home/ubuntu/GenDATA/studies/all_models_ablation/causal_aug1/slices", "cf")
  src = resolve_slice_dir(base)
if not src or count_java(src)==0:
  raise SystemExit("ERROR: No slices (.java) found after CF fallback")
# Normalize to shared directory (created once, used by all)
shared_norm = Path("/home/ubuntu/GenDATA/studies/all_models_ablation/shared_normalized_slices")
if not shared_norm.exists() or count_java(shared_norm)==0:
  n = normalize_slices(src, shared_norm)
  print('NORMALIZED:' + str(n) + ' (shared)')
else:
  print('USING_EXISTING_SHARED_NORMALIZED (' + str(count_java(shared_norm)) + ' files)')
# Also copy to model-specific for backward compatibility
dst = Path("/home/ubuntu/GenDATA/studies/all_models_ablation/causal_aug1/normalized_slices")
n = normalize_slices(src, dst)
print("NORMALIZED:" + str(n))
PY
