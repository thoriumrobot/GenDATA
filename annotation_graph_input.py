#!/usr/bin/env python3
"""
Utilities to load CFG JSONs and produce graph embeddings for annotation-type models.
Uses cfg_graph.load_cfg_as_pyg and graph_encoder.build_graph_encoder.
"""

import os
import json
from typing import List

import torch

from cfg_graph import load_cfg_as_pyg
from graph_encoder import build_graph_encoder


class GraphEmbeddingProvider:
    def __init__(self, out_dim: int = 256, variant: str = 'transformer', device: str = 'cpu'):
        self.out_dim = out_dim
        self.variant = variant
        self.device = device
        self._enc_cache = {}

    def _get_encoder(self, in_dim: int, edge_dim: int) -> torch.nn.Module:
        key = (in_dim, edge_dim, self.variant, self.out_dim)
        if key not in self._enc_cache:
            enc = build_graph_encoder(in_dim=in_dim, edge_dim=edge_dim, out_dim=self.out_dim, variant=self.variant)
            enc = enc.to(self.device)
            self._enc_cache[key] = enc
        return self._enc_cache[key]

    def embed_cfg_file(self, cfg_file: str) -> torch.Tensor:
        data = load_cfg_as_pyg(cfg_file)
        if data.x is None or data.x.numel() == 0:
            return torch.zeros(self.out_dim)
        in_dim = int(data.x.size(1))
        edge_dim = int(data.edge_attr.size(1)) if getattr(data, 'edge_attr', None) is not None else 0
        enc = self._get_encoder(in_dim, edge_dim)
        enc.eval()
        with torch.no_grad():
            emb = enc(data)
        if isinstance(emb, torch.Tensor) and emb.dim() == 2 and emb.size(0) == 1:
            emb = emb.squeeze(0)
        return emb

    def embed_cfg_dir(self, cfg_dir_for_file: str) -> torch.Tensor:
        """Aggregate embeddings over multiple cfg.json files for a Java file.
        Strategy: mean pooling across methods.
        """
        embs: List[torch.Tensor] = []
        if not os.path.isdir(cfg_dir_for_file):
            return torch.zeros(self.out_dim)
        for root, _, files in os.walk(cfg_dir_for_file):
            for f in files:
                if not f.endswith('.json'):
                    continue
                emb = self.embed_cfg_file(os.path.join(root, f))
                if emb.numel() > 0:
                    embs.append(emb)
        if not embs:
            return torch.zeros(self.out_dim)
        return torch.stack(embs, dim=0).mean(dim=0)


