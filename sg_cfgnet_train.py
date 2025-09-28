#!/usr/bin/env python3
import argparse
import json
import os
import logging
from typing import List, Dict, Any

from sg_cfgnet import SGCFGNetTrainer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_cfg_dir(cfg_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for root, _, files in os.walk(cfg_dir):
        for fn in files:
            if not fn.endswith('.json'):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, 'r') as f:
                    cfg = json.load(f)
                items.append({'file': path, 'method': cfg.get('method_name', 'm'), 'data': cfg})
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_cfg_dir', default='test_results/pf_dataset/train/cfg_output')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--parameter_free', action='store_true', default=True)
    args = ap.parse_args()

    if not os.path.isdir(args.train_cfg_dir):
        logger.error(f"Train CFG dir not found: {args.train_cfg_dir}")
        return

    cfg_files = load_cfg_dir(args.train_cfg_dir)
    trainer = SGCFGNetTrainer(parameter_free=args.parameter_free)
    info = trainer.train(cfg_files, epochs=args.epochs, lr=args.lr)
    logger.info(f"Training finished: {info}")

if __name__ == '__main__':
    main()


