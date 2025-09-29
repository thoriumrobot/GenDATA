#!/usr/bin/env python3
"""
CFG -> Graph tensor conversion utilities.

This module loads CFG JSON files produced by cfg.py and converts them into
PyTorch Geometric Data objects with rich node/edge features:
- Node features: node type one-hots, degree, line number (normalized),
  Laplacian positional encodings (k eigenvectors), random-walk structural encodings (RWSE)
- Edge features: one-hot edge type for control vs dataflow

Assumptions about CFG JSON schema (as produced by cfg.py):
- cfg_data = { 'nodes': [{ 'id': str|int, 'label': str, 'node_type': str, 'line': int }...],
               'edges': [{ 'source': id, 'target': id, 'type': 'control'|'dataflow' }...] }

If fields are missing, reasonable defaults are applied.
"""

import os
import json
from typing import Dict, Tuple, List

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree


DEFAULT_K_LPE = 8
DEFAULT_RW_STEPS = 4


def _build_id_maps(nodes: List[Dict]) -> Dict:
    id_map: Dict[str, int] = {}
    for idx, n in enumerate(nodes):
        nid = str(n.get('id', idx))
        id_map[nid] = idx
    return id_map


def _node_type_vocab(nodes: List[Dict]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for n in nodes:
        t = str(n.get('node_type', 'statement')).lower()
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def _one_hot(index: int, size: int) -> Tensor:
    t = torch.zeros(size, dtype=torch.float32)
    if 0 <= index < size:
        t[index] = 1.0
    return t


def _compute_laplacian_pe(edge_index: Tensor, num_nodes: int, k: int) -> Tensor:
    # Compute symmetric normalized Laplacian eigenvectors using torch.linalg.eigh
    # Construct adjacency
    device = edge_index.device
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    A[edge_index[1], edge_index[0]] = 1.0
    # Degree and D^{-1/2}
    d = torch.sum(A, dim=1)
    d_clamped = torch.clamp(d, min=1.0)
    D_inv_sqrt = torch.diag(torch.pow(d_clamped, -0.5))
    # L = I - D^{-1/2} A D^{-1/2}
    I = torch.eye(num_nodes, device=device)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    # Compute k smallest non-trivial eigenvectors
    try:
        evals, evecs = torch.linalg.eigh(L)
    except RuntimeError:
        # Fallback to CPU
        evals, evecs = torch.linalg.eigh(L.cpu())
        evecs = evecs.to(device)
    # Skip the first eigenvector (constant); take next k
    start = 1 if evecs.shape[1] > 1 else 0
    end = min(start + k, evecs.shape[1])
    pe = evecs[:, start:end]
    if pe.shape[1] < k:
        # pad
        pad = torch.zeros((num_nodes, k - pe.shape[1]), device=device)
        pe = torch.cat([pe, pad], dim=1)
    return pe


def _compute_rwse(edge_index: Tensor, num_nodes: int, steps: int) -> Tensor:
    # Simple k-step random-walk landing probabilities approximation via power iteration on transition matrix
    device = edge_index.device
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    # Row-normalize
    row_sum = torch.clamp(torch.sum(A, dim=1, keepdim=True), min=1.0)
    P = A / row_sum
    # Start from uniform distribution at each node (one-hot rows)
    rw_feats = []
    X = torch.eye(num_nodes, device=device)
    cur = X
    for _ in range(steps):
        cur = cur @ P  # next-step distribution
        rw_feats.append(cur.diag().unsqueeze(1))  # landing prob on self after t steps
    rw = torch.cat(rw_feats, dim=1) if rw_feats else torch.zeros((num_nodes, 0), device=device)
    return rw


def load_cfg_as_pyg(cfg_file: str, k_lpe: int = DEFAULT_K_LPE, rw_steps: int = DEFAULT_RW_STEPS) -> Data:
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    nodes = cfg.get('nodes', [])
    edges = cfg.get('edges', [])

    if not nodes:
        # Return empty graph
        return Data(x=torch.zeros((0, 1), dtype=torch.float32))

    id_map = _build_id_maps(nodes)
    node_type_to_idx = _node_type_vocab(nodes)

    # Build edge_index and edge_attr (edge type one-hot: control, dataflow)
    edge_list: List[Tuple[int, int]] = []
    edge_types: List[int] = []
    for e in edges:
        src = id_map.get(str(e.get('source')))
        dst = id_map.get(str(e.get('target')))
        if src is None or dst is None:
            continue
        edge_list.append((src, dst))
        t = str(e.get('type', 'control')).lower()
        edge_types.append(0 if t == 'control' else 1)

    if not edge_list:
        # Avoid empty edge index by creating self-loops
        edge_index = torch.arange(len(nodes)).unsqueeze(0).repeat(2, 1)
        edge_types_tensor = torch.zeros((edge_index.shape[1], 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        et = torch.tensor(edge_types, dtype=torch.long)
        # duplicate edge_types for the reversed edges introduced by undirected conversion
        et = torch.cat([et, et], dim=0)
        edge_types_tensor = torch.nn.functional.one_hot(et, num_classes=2).to(torch.float32)

    num_nodes = len(nodes)
    # Base node features
    node_feats: List[Tensor] = []
    # node type one-hot
    num_types = len(node_type_to_idx)
    for n in nodes:
        t = str(n.get('node_type', 'statement')).lower()
        node_feats.append(_one_hot(node_type_to_idx[t], num_types))
    X_type = torch.stack(node_feats, dim=0)

    # degree
    deg = degree(edge_index[0], num_nodes=num_nodes).unsqueeze(1)
    # line number normalized
    lines = torch.tensor([float(n.get('line', 0)) for n in nodes]).unsqueeze(1)
    if lines.max() > 0:
        lines = lines / (lines.max() + 1e-6)

    # Laplacian PE and RWSE
    pe = _compute_laplacian_pe(edge_index, num_nodes, k=k_lpe)
    rw = _compute_rwse(edge_index, num_nodes, steps=rw_steps)

    x = torch.cat([X_type, deg, lines, pe, rw], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_types_tensor)
    data.num_nodes = num_nodes
    data.node_types = torch.tensor([node_type_to_idx[str(n.get('node_type', 'statement')).lower()] for n in nodes], dtype=torch.long)
    return data


def load_cfg_dir_graph(cfg_dir: str, file_basename: str) -> Data:
    cfg_file = os.path.join(cfg_dir, file_basename, 'cfg.json')
    return load_cfg_as_pyg(cfg_file)


