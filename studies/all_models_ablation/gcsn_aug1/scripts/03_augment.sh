#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python - <<'PY'
from pathlib import Path
from slice_dataset_tools import count_java
shared_aug = Path("/home/ubuntu/GenDATA/studies/all_models_ablation/shared_aug_slices_l1")
shared_norm = Path("/home/ubuntu/GenDATA/studies/all_models_ablation/shared_normalized_slices")
if not shared_aug.exists() or count_java(shared_aug)==0:
  import subprocess
  subprocess.run([
    'python', 'enhanced_semantic_augment_slices.py',
    str(shared_norm), str(shared_aug),
    '--variants', '1',
    '--sequence-len', '2',
    '--max-depth', '3',
    '--min-diff', '0.03',
    '--disabled', 'switch_statement', 'variable_operation', 'string_concatenation', 'numeric_literal',
    '--focus-nodes', 'control', 'dataflow'
  ], check=True)
  print('GENERATED_SHARED_AUG_L1:' + str(count_java(shared_aug)))
else:
  print('USING_EXISTING_SHARED_AUG_L1 (' + str(count_java(shared_aug)) + ' files)')
PY
