#!/usr/bin/env python3
"""
End-to-end retraining and evaluation runner using graph-based CFG inputs.

Steps:
1) Slice with Specimin, augment, generate CFGs
2) Train all graph/non-graph models (pipeline 'all')
3) Run simplified pipeline prediction to generate model-based predictions
4) Print a short summary path to predictions
"""

import os
import sys
import subprocess


def run(cmd):
    print("$ "+" ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    project_root = os.environ.get('PROJECT_ROOT', '/home/ubuntu/checker-framework/checker/tests/index')
    warnings_file = os.environ.get('WARNINGS_FILE', '/home/ubuntu/GenDATA/index1.out')
    cfwr_root = os.environ.get('CFWR_ROOT', '/home/ubuntu/GenDATA')

    # 1) Slice + augment + CFG
    run([sys.executable, 'pipeline.py', '--steps', 'all', '--slicer', 'specimin', '--project_root', project_root, '--warnings_file', warnings_file, '--cfwr_root', cfwr_root])

    # 2) Train all models that pipeline.py supports via 'train' (already invoked in 'all')
    # If you need to re-run train explicitly:
    # run([sys.executable, 'pipeline.py', '--steps', 'train', '--model', 'all'])

    # 3) Run prediction over original (optional) or use simple pipeline
    run([sys.executable, 'simple_annotation_type_pipeline.py', '--mode', 'predict'])

    print('Done. See predictions in predictions_annotation_types/.')


if __name__ == '__main__':
    main()


