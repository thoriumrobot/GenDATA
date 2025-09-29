#!/usr/bin/env python3
"""
HGT prediction: given a Java file, ensure CFGs exist, build HeteroData from CFG graphs (cfg_graph),
run node-level predictions with the trained HGT model, and write predictions to JSON.
"""

import os
import json
import argparse
import torch
from torch_geometric.data import HeteroData

from cfg import generate_control_flow_graphs, save_cfgs
from cfg_graph import load_cfg_as_pyg


def ensure_cfgs(java_file: str, cfg_output_dir: str):
    base = os.path.splitext(os.path.basename(java_file))[0]
    out_dir = os.path.join(cfg_output_dir, base)
    if not os.path.exists(out_dir) or not any(n.endswith('.json') for n in os.listdir(out_dir)):
        cfgs = generate_control_flow_graphs(java_file, cfg_output_dir)
        save_cfgs(cfgs, out_dir)
    return out_dir


def pyg_to_hetero(pyg) -> HeteroData:
    data = HeteroData()
    data['node'].x = pyg.x
    data['node', 'to', 'node'].edge_index = pyg.edge_index
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--java_file', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--cfg_output_dir', default=os.environ.get('CFG_OUTPUT_DIR', 'cfg_output'))
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    cfg_dir = ensure_cfgs(args.java_file, args.cfg_output_dir)

    # Rebuild HGT model skeleton based on checkpoint metadata
    ckpt = torch.load(args.model_path, map_location='cpu')
    # Expecting a plain state_dict from hgt.py's HGTModel; we reconstruct a compatible module
    from hgt import HGTModel
    # Minimal metadata for single node/edge type
    metadata = (['node'], [('node', 'to', 'node')])
    # Infer hidden/in_channels from state_dict shapes
    # Fallback defaults
    in_channels = ckpt.get('in_channels', 64)
    hidden = ckpt.get('hidden_channels', 64)
    out_channels = ckpt.get('out_channels', 2)
    try:
        model = HGTModel(in_channels=in_channels, hidden_channels=hidden, out_channels=out_channels, num_heads=2, num_layers=2, metadata=metadata)
        model.load_state_dict(ckpt if isinstance(ckpt, dict) and 'state_dict' not in ckpt else ckpt.get('state_dict', {}), strict=False)
    except Exception:
        # If hgt.py saved only state_dict
        model = HGTModel(in_channels=in_channels, hidden_channels=hidden, out_channels=out_channels, num_heads=2, num_layers=2, metadata=metadata)
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    predictions = []
    for name in os.listdir(cfg_dir):
        if not name.endswith('.json'):
            continue
        cfg_path = os.path.join(cfg_dir, name)
        pyg = load_cfg_as_pyg(cfg_path)
        if pyg.x is None or pyg.x.numel() == 0:
            continue
        data = pyg_to_hetero(pyg)
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            probs = torch.softmax(out, dim=-1)[:, 1]
        with open(cfg_path, 'r') as fp:
            cfg = json.load(fp)
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


