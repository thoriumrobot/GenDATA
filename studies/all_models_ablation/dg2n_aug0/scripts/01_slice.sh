#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
export SOOT_SLICE_CLI="/home/ubuntu/GenDATA/tools/soot_slicer.sh"; export SOOT_JAR="/home/ubuntu/GenDATA/build/libs/GenDATA-all.jar"; python -c 'from pipeline import run_slicing; run_slicing("/home/ubuntu/checker-framework/checker/tests/index", "/home/ubuntu/GenDATA/index1.out", "/home/ubuntu/GenDATA", "/home/ubuntu/GenDATA/studies/all_models_ablation/dg2n_aug0/slices", "soot")'
