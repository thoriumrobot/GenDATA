#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict

import torch

from dgcrf_model import DGCRFClassifier, build_hard_class_mask


def load_graph_pt(graph_pt: str) -> Dict:
    return torch.load(graph_pt, map_location='cpu')


def predict(graph_pt: str, ckpt_path: str, out_json: str) -> None:
    obj = load_graph_pt(graph_pt)
    x = obj['x']
    mask = obj.get('mask', None)
    pf_classes = tuple(obj.get('pf_classes', ["NO_ANNOTATION", "@Positive", "@NonNegative", "@GTENegativeOne"]))

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = DGCRFClassifier(num_features=ckpt['num_features'], num_classes=ckpt['num_classes'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    with torch.no_grad():
        is_target = mask if mask is not None else torch.ones(x.size(0), dtype=torch.bool)
        hard_mask = build_hard_class_mask(x, pf_classes, is_target)
        out = model(x, hard_mask, deterministic_gates=True)
        logits = out['logits']
        pred = torch.argmax(logits, dim=1).tolist()

    result = {
        'pred_indices': pred,
        'pf_classes': list(pf_classes),
        'pred_labels': [pf_classes[i] for i in pred],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Wrote predictions to {out_json}")


def main():
    ap = argparse.ArgumentParser(description='Predict with DG-CRF-lite model')
    ap.add_argument('--graph_pt', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_json', required=True)
    args = ap.parse_args()
    predict(args.graph_pt, args.ckpt, args.out_json)


if __name__ == '__main__':
    main()


