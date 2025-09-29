#!/usr/bin/env python3
"""
Graph encoder for CFG graphs.

Primary: Graph Transformer with edge encodings and global attention pooling.
Fallbacks: PNA and GAT. Produces a fixed-D graph embedding for downstream classifiers.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, PNAConv, global_add_pool, global_mean_pool
from torch_geometric.nn import TransformerConv


class GlobalAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, batch):
        gate_scores = self.gate(x).squeeze(-1)
        # softmax within batch segments
        max_per_batch = torch.zeros_like(gate_scores)
        max_per_batch = torch.scatter_reduce(torch.zeros_like(gate_scores), 0, batch, gate_scores, reduce="amax")
        # subtract max for stability
        weights = torch.exp(gate_scores - max_per_batch[batch])
        sum_per_batch = torch.scatter_add(torch.zeros_like(weights), 0, batch, weights)
        weights = weights / (sum_per_batch[batch] + 1e-9)
        x_weighted = x * weights.unsqueeze(-1)
        out = torch.scatter_add(torch.zeros((batch.max().item() + 1, x.size(-1)), device=x.device, dtype=x.dtype), 0, batch.unsqueeze(-1).expand_as(x_weighted), x_weighted)
        return out


class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int = 2, hidden_dim: int = 128, layers: int = 4, heads: int = 4, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        convs = []
        for _ in range(layers):
            convs.append(TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout, beta=True))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.act = nn.ReLU()
        self.pool = GlobalAttentionPool(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, getattr(data, 'batch', torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device))
        h = self.input(x)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = self.act(h)
            h = h + h_res
        g = self.pool(h, batch)
        return self.proj(g)


class PNAGATEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int = 2, hidden_dim: int = 128, layers: int = 4, out_dim: int = 256, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        convs = []
        for i in range(layers):
            if i % 2 == 0:
                convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim))
            else:
                convs.append(PNAConv(in_channels=hidden_dim, out_channels=hidden_dim, aggregators=['mean','max','min','std','sum'], scalers=['identity','amplification','attenuation'], edge_dim=edge_dim))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.act = nn.ReLU()
        self.pool = GlobalAttentionPool(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, getattr(data, 'batch', torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device))
        h = self.input(x)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            if isinstance(conv, GATv2Conv):
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = self.act(h)
            h = h + h_res
        g = self.pool(h, batch)
        return self.proj(g)


def build_graph_encoder(in_dim: int, edge_dim: int = 2, out_dim: int = 256, variant: str = 'transformer') -> nn.Module:
    if variant == 'transformer':
        return GraphTransformerEncoder(in_dim=in_dim, edge_dim=edge_dim, out_dim=out_dim)
    if variant == 'pna_gat':
        return PNAGATEncoder(in_dim=in_dim, edge_dim=edge_dim, out_dim=out_dim)
    # default
    return GraphTransformerEncoder(in_dim=in_dim, edge_dim=edge_dim, out_dim=out_dim)


