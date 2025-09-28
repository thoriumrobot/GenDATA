#!/usr/bin/env python3
"""
Simple GCN training on CFG JSONs combining control and dataflow edges.
Saves model to MODELS_DIR/gcn/best_gcn.pth
"""

import os
import json
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


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


def cfg_to_homograph(cfg_data: Dict) -> Data:
    nodes = cfg_data.get('nodes', [])
    control_edges = cfg_data.get('control_edges', [])
    dataflow_edges = cfg_data.get('dataflow_edges', [])

    x = torch.tensor([extract_features(n, cfg_data) for n in nodes], dtype=torch.float)
    y = torch.tensor([1 if is_annotation_target(n) else 0 for n in nodes], dtype=torch.long)

    edges = control_edges + dataflow_edges
    if edges:
        src = torch.tensor([e['source'] for e in edges], dtype=torch.long)
        dst = torch.tensor([e['target'] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.lin(x)


def load_cfg_dataset(cfg_dir: str) -> List[Data]:
    dataset: List[Data] = []
    for root, _, files in os.walk(cfg_dir):
        for f in files:
            if not f.endswith('.json'):
                continue
            with open(os.path.join(root, f), 'r') as fp:
                cfg = json.load(fp)
            g = cfg_to_homograph(cfg)
            if g.x.numel() == 0:
                continue
            dataset.append(g)
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_dir', required=True)
    ap.add_argument('--out_dir', default=os.path.join(os.environ.get('MODELS_DIR', 'models'), 'gcn'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = load_cfg_dataset(args.cfg_dir)
    if not dataset:
        print('No graphs found in cfg_dir')
        return

    # Simple split
    split = int(0.8 * len(dataset))
    train_ds = dataset[:split]
    val_ds = dataset[split:]
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    in_dim = dataset[0].x.size(-1)
    model = SimpleGCN(in_dim=in_dim, hidden=args.hidden)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = crit(logits, batch.y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        # val
        model.eval();
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                loss = crit(logits, batch.y)
                val_loss += float(loss.item())
        print(f"Epoch {epoch:03d} | train_loss={total:.4f} | val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(), 'in_dim': in_dim, 'hidden': args.hidden}, os.path.join(args.out_dir, 'best_gcn.pth'))

    print('Training complete')


if __name__ == '__main__':
    main()


