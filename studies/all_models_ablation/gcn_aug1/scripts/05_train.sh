#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/GenDATA
python gcn_train.py --cfg_dir /home/ubuntu/GenDATA/studies/all_models_ablation/gcn_aug1/aug_cfg --out_dir /home/ubuntu/GenDATA/studies/all_models_ablation/gcn_aug1/models/gcn --epochs 25 --layers 3 --hidden 256 --dropout 0.2 --early_stop_patience 5 --metrics_path /home/ubuntu/GenDATA/studies/all_models_ablation/gcn_aug1/models/gcn/metrics.json
