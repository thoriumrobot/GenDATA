#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python - <<'PY'
from pathlib import Path
from slice_dataset_tools import resolve_slice_dir, normalize_slices, count_java
base = Path("/home/ubuntu/GenDATA/studies/gcn_aug_probe/aug2")
src = resolve_slice_dir(base)
if not src or count_java(src)==0:
  from pipeline import run_slicing
  run_slicing("/home/ubuntu/checker-framework/checker/tests/index", "/home/ubuntu/GenDATA/index1.out", "/home/ubuntu/GenDATA", "/home/ubuntu/GenDATA/studies/gcn_aug_probe/aug2/slices", "cf")
  src = resolve_slice_dir(base)
if not src or count_java(src)==0:
  raise SystemExit("ERROR: No slices (.java) found after CF fallback")
dst = Path("/home/ubuntu/GenDATA/studies/gcn_aug_probe/aug2/normalized_slices")
copied = normalize_slices(src, dst)
print("NORMALIZED:" + str(copied))
PY
