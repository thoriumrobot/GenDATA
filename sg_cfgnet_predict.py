#!/usr/bin/env python3
import argparse
import json
import os
import logging
import torch
from typing import List, Dict, Any

from sg_cfgnet import SGCFGNetTrainer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_cfg_dir(cfg_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for fn in os.listdir(cfg_dir):
        if fn.endswith('.json'):
            with open(os.path.join(cfg_dir, fn), 'r') as f:
                cfg = json.load(f)
            items.append({'file': fn, 'method': cfg.get('method_name', 'm'), 'data': cfg})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_cfg_dir', default='test_results/pf_dataset/train/cfg_output')
    ap.add_argument('--test_cfg_dir', default='test_results/pf_dataset/test/cfg_output')
    ap.add_argument('--parameter_free', action='store_true', default=True)
    args = ap.parse_args()

    if not os.path.isdir(args.train_cfg_dir) or not os.path.isdir(args.test_cfg_dir):
        logger.error('CFG directories not found.')
        return

    train_items = load_cfg_dir(args.train_cfg_dir)
    test_items = load_cfg_dir(args.test_cfg_dir)

    trainer = SGCFGNetTrainer(parameter_free=args.parameter_free)
    info = trainer.train(train_items, epochs=40, lr=1e-3)
    if not info.get('success'):
        logger.error('Training failed.')
        return

    # Predict on test set
    results: List[Dict[str, Any]] = []
    for cf in test_items:
        cfg = cf['data']
        for node in cfg.get('nodes', []):
            from node_level_models import NodeClassifier
            if NodeClassifier.is_annotation_target(node):
                label, conf = trainer.predict_node(node, cfg)
                results.append({'file': cf['file'], 'node_id': node.get('id'), 'prediction': label, 'confidence': conf})

    out_path = 'test_results/sg_cfgnet_predictions.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved predictions to {out_path} ({len(results)} entries)")

if __name__ == '__main__':
    main()


