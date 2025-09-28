#!/usr/bin/env python3
"""
Adapter to convert CFWR CFG JSONs into PyG Data objects for GCSN.

Output: three files under out_dir:
  - train_all.pt  (list[Data])
  - val_all.pt    (list[Data])
  - test_all.pt   (list[Data])

Edges: concatenate control_edges and dataflow_edges into a single undirected edge_index.
Labels/masks: use y from PF remapping similar to dg2n_adapter; mark targets as train/val/test via graph-level split.
"""

import os
import re
import json
import argparse
from typing import List, Dict, Tuple

import torch
from torch_geometric.data import Data


PF_CLASSES = [
    "NO_ANNOTATION",
    "@Positive",
    "@NonNegative",
    "@GTENegativeOne",
]


def is_annotation_target(node: Dict) -> bool:
    label = (node.get('label') or '').lower()
    node_type = (node.get('node_type') or '').lower()
    if any(k in label for k in ['methoddeclaration', 'constructordeclaration', 'method(']):
        return True
    if any(k in label for k in ['fielddeclaration', 'variabledeclarator', 'localvariabledeclaration']):
        return True
    if any(k in label for k in ['formalparameter', 'parameter']):
        return True
    if node_type in ('method', 'field', 'parameter', 'variable'):
        return True
    return False


def _pf_to_index(label: str) -> int:
    try:
        return PF_CLASSES.index(label)
    except ValueError:
        return 0


def _remap_to_pf_label(node: Dict, cfg_data: Dict) -> str:
    if not is_annotation_target(node):
        return "NO_ANNOTATION"
    label = (node.get('label') or '').lower()
    txt = label
    has_len = (".length" in txt) or ("length()" in txt) or ("size()" in txt)
    is_param = any(k in txt for k in ["formalparameter", "parameter", "@", "] "])
    has_index_pat = bool(re.search(r"\bindex|\bidx|\[(?:[^\]]+)\]", txt))
    has_array_access = "[" in txt and "]" in txt
    has_numeric = bool(re.search(r"\bint\b|\blong\b|[0-9]+", txt))
    has_comp_or_loop = any(k in txt for k in ["if (", "while (", "for (", ">", "<", ">=", "<="])
    if is_param and has_len:
        return "@Positive"
    if has_index_pat and not has_array_access:
        return "@GTENegativeOne"
    if has_numeric and has_comp_or_loop:
        return "@NonNegative"
    h = (len(txt) + int(node.get('line') or 0)) % 4
    return PF_CLASSES[h]


def extract_features(node: Dict, cfg_data: Dict) -> List[float]:
    label = node.get('label', '')
    node_id = node.get('id', 0)
    control_edges = cfg_data.get('control_edges', [])
    dataflow_edges = cfg_data.get('dataflow_edges', [])
    in_degree = sum(1 for e in control_edges if e.get('target') == node_id)
    out_degree = sum(1 for e in control_edges if e.get('source') == node_id)
    df_in = sum(1 for e in dataflow_edges if e.get('target') == node_id)
    df_out = sum(1 for e in dataflow_edges if e.get('source') == node_id)
    is_control = 1.0 if (node.get('node_type', 'control') == 'control') else 0.0
    return [
        float(len(label)),
        1.0 if is_annotation_target(node) else 0.0,
        float(node.get('line', 0) or 0),
        float(in_degree), float(out_degree), float(df_in), float(df_out), is_control
    ]


def cfg_to_data(cfg_json: str) -> Data:
    with open(cfg_json, 'r') as f:
        data = json.load(f)
    nodes = data.get('nodes', [])
    N = len(nodes)
    x = torch.tensor([extract_features(n, data) for n in nodes], dtype=torch.float)
    y = torch.tensor([_pf_to_index(_remap_to_pf_label(n, data)) for n in nodes], dtype=torch.long)
    target_mask = torch.tensor([bool(is_annotation_target(n)) for n in nodes], dtype=torch.bool)
    # edges: concatenate control and dataflow, make undirected by adding reverse edges
    pairs: List[Tuple[int, int]] = []
    for lst in (data.get('control_edges', []), data.get('dataflow_edges', [])):
        for e in lst:
            s = int(e.get('source', -1)); t = int(e.get('target', -1))
            if 0 <= s < N and 0 <= t < N:
                pairs.append((s, t)); pairs.append((t, s))
    if len(pairs) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src = torch.tensor([s for s, _ in pairs], dtype=torch.long)
        dst = torch.tensor([t for _, t in pairs], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    g = Data(x=x, edge_index=edge_index, y=y)
    # carry masks; train/val/test will be assigned at graph-level split; use target_mask to focus supervision
    g.target_mask = target_mask
    return g


def build_dataset(cfg_dir: str) -> List[Data]:
    graphs: List[Data] = []
    for root, _, files in os.walk(cfg_dir):
        for f in files:
            if f.endswith('.json'):
                try:
                    graphs.append(cfg_to_data(os.path.join(root, f)))
                except Exception:
                    pass
    return graphs


def split_graphs(graphs: List[Data], val_ratio: float = 0.1, test_ratio: float = 0.1):
    import random
    idx = list(range(len(graphs)))
    random.shuffle(idx)
    n = len(idx)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idx = set(idx[:n_val])
    test_idx = set(idx[n_val:n_val + n_test])
    train, val, test = [], [], []
    for i, g in enumerate(graphs):
        # build node-level masks using target_mask to supervise only targets
        tm = getattr(g, 'target_mask', torch.ones(g.y.size(0), dtype=torch.bool))
        train_mask = torch.zeros_like(tm)
        val_mask = torch.zeros_like(tm)
        test_mask = torch.zeros_like(tm)
        if i in val_idx:
            val_mask = tm.clone()
            val.append(Data(x=g.x, edge_index=g.edge_index, y=g.y, val_mask=val_mask))
        elif i in test_idx:
            test_mask = tm.clone()
            test.append(Data(x=g.x, edge_index=g.edge_index, y=g.y, test_mask=test_mask))
        else:
            train_mask = tm.clone()
            train.append(Data(x=g.x, edge_index=g.edge_index, y=g.y, train_mask=train_mask))
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description='Convert CFG JSONs to PyG Data for GCSN')
    ap.add_argument('--cfg_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    graphs = build_dataset(args.cfg_dir)
    if len(graphs) == 0:
        print(f"No graphs found in {args.cfg_dir}")
        return
    train, val, test = split_graphs(graphs, args.val_ratio, args.test_ratio)
    torch.save(train, os.path.join(args.out_dir, 'train_all.pt'))
    torch.save(val, os.path.join(args.out_dir, 'val_all.pt'))
    torch.save(test, os.path.join(args.out_dir, 'test_all.pt'))
    # also store class mapping
    with open(os.path.join(args.out_dir, 'pf_classes.json'), 'w') as f:
        json.dump({'pf_classes': PF_CLASSES}, f, indent=2)
    print(f"GCSN adapter: wrote {len(train)} train, {len(val)} val, {len(test)} test graphs to {args.out_dir}")


if __name__ == '__main__':
    main()


