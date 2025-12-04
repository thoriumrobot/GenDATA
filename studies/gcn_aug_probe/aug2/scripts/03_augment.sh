#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python enhanced_semantic_augment_slices.py /home/ubuntu/GenDATA/studies/gcn_aug_probe/aug2/normalized_slices /home/ubuntu/GenDATA/studies/gcn_aug_probe/aug2/aug_slices --variants 2 --sequence-len 2 --max-depth 3 --min-diff 0.03 --disabled switch_statement variable_operation string_concatenation numeric_literal --focus-nodes control dataflow
