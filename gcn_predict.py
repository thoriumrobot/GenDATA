#!/usr/bin/env python3
"""
Simple GCN prediction: given a Java file, ensure CFGs exist, convert to homograph, run node predictions,
and write predictions to JSON (node lines predicted positive).
"""

import os
import json
import argparse
import torch
from cfg import generate_control_flow_graphs, save_cfgs
from gcn_train import cfg_to_homograph


def ensure_cfgs(java_file: str, cfg_output_dir: str):
    base = os.path.splitext(os.path.basename(java_file))[0]
    out_dir = os.path.join(cfg_output_dir, base)
    if not os.path.exists(out_dir) or not any(n.endswith('.json') for n in os.listdir(out_dir)):
        cfgs = generate_control_flow_graphs(java_file, cfg_output_dir)
        save_cfgs(cfgs, out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--java_file', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--cfg_output_dir', default=os.environ.get('CFG_OUTPUT_DIR', 'cfg_output'))
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    cfg_dir = ensure_cfgs(args.java_file, args.cfg_output_dir)

    # Load model
    ckpt = torch.load(args.model_path, map_location='cpu')
    from gcn_train import SimpleGCN
    model = SimpleGCN(in_dim=ckpt['in_dim'], hidden=ckpt['hidden'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    predictions = []
    for name in os.listdir(cfg_dir):
        if not name.endswith('.json'):
            continue
        cfg_path = os.path.join(cfg_dir, name)
        with open(cfg_path, 'r') as fp:
            cfg = json.load(fp)
        data = cfg_to_homograph(cfg)
        if data.x.numel() == 0:
            continue
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            nodes = cfg.get('nodes', [])
            for i, p in enumerate(probs.tolist()):
                if p >= args.threshold and i < len(nodes):
                    line = nodes[i].get('line')
                    if line is not None:
                        predictions.append({'method': cfg.get('method_name'), 'line': line, 'score': p})

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump({'predictions': predictions}, f, indent=2)
    print('Wrote', args.out_path)


if __name__ == '__main__':
    main()


