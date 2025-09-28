#!/usr/bin/env python3
"""
Deterministically-Gated Factor Model (DG-CRF-lite)

Minimal implementation tailored to PF multi-class over CFG nodes using
feature gates (hard-concrete) plus hard projection constraints.

Data format: expects graphs produced by `dg2n_adapter.py`:
  - x: [N, F] float features
  - y: [N] long labels in PF class index space
  - mask: [N] bool supervision mask
  - edge_index_dict: optional dict of relations (unused by the base model)

Notes:
  - This is a lightweight variant without full pairwise inference; instead,
    we implement hard constraints by masking illegal classes per-node before
    the softmax. This enforces 0-probability for disallowed assignments.
  - Feature selection is done via hard-concrete gates per feature dimension.
    At evaluation we determinize gates to 0/1.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteGate(nn.Module):
    """
    Hard-concrete gate per feature dimension (Louizos et al., 2018).
    Exposes expected L0 for sparsity regularization during training.
    """

    def __init__(self, num_feats: int, log_alpha_init: float = -2.0, beta: float = 2.0,
                 gamma: float = -0.1, zeta: float = 1.1, temperature: float = 0.33):
        super().__init__()
        self.num_feats = num_feats
        self.log_alpha = nn.Parameter(torch.full((num_feats,), float(log_alpha_init)))
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.temperature = temperature

    def _stretch(self, s: torch.Tensor) -> torch.Tensor:
        return s * (self.zeta - self.gamma) + self.gamma

    def _hard_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)

    def expected_L0(self) -> torch.Tensor:
        # P(z > 0) for hard-concrete
        logits = self.log_alpha
        probs = torch.sigmoid(logits - self.beta * torch.log(torch.tensor(-self.gamma / self.zeta)))
        return probs.sum()

    def forward(self, x: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [N, F]
        if deterministic:
            gate_probs = torch.sigmoid(self.log_alpha)
            z = (gate_probs > 0.5).float()
            return x * z, z
        # sample u ~ Uniform(0,1)
        u = torch.rand_like(x)
        # Compute s = sigmoid((logu - log1-u + log_alpha)/temp)
        logu = torch.log(u + 1e-8)
        log1_u = torch.log(1 - u + 1e-8)
        s = torch.sigmoid((self.log_alpha + (logu - log1_u)) / self.temperature)
        s = self._stretch(s)
        z = self._hard_sigmoid(s)
        return x * z, z


class DGCRFClassifier(nn.Module):
    """
    Node-level gated classifier with per-node hard constraint projection.
    """

    def __init__(self, num_features: int, num_classes: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gates = HardConcreteGate(num_features)
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def project_logits(self, logits: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
        # hard_mask: [N, C] with 1 for allowed classes, 0 for disallowed
        minus_inf = torch.finfo(logits.dtype).min
        projected = logits.masked_fill(hard_mask == 0, minus_inf)
        return projected

    def forward(self, x: torch.Tensor, hard_mask: torch.Tensor, deterministic_gates: bool) -> Dict[str, torch.Tensor]:
        x_gated, z = self.gates(x, deterministic=deterministic_gates)
        logits = self.encoder(x_gated)
        logits = self.project_logits(logits, hard_mask)
        return {"logits": logits, "gate_mask": z}


def build_hard_class_mask(x: torch.Tensor, pf_classes: Tuple[str, ...], is_target: torch.Tensor) -> torch.Tensor:
    """
    Produce a [N, C] mask of allowed classes given simple per-node heuristics.
    Hard constraints implemented:
      - If not a target, only NO_ANNOTATION is allowed.
      - If the label suggests index-like usage, prefer {@GTENegativeOne} (but still allow NO_ANNOTATION).
    """
    device = x.device
    num_nodes = x.size(0)
    num_classes = len(pf_classes)
    mask = torch.ones((num_nodes, num_classes), device=device, dtype=torch.bool)

    # Class indices
    class_to_idx = {c: i for i, c in enumerate(pf_classes)}
    no_ann = class_to_idx.get("NO_ANNOTATION", 0)
    gte_neg1 = class_to_idx.get("@GTENegativeOne", None)

    # Non-target nodes: force NO_ANNOTATION
    non_target = ~is_target
    if non_target.any():
        mask[non_target, :] = False
        mask[non_target, no_ann] = True

    # Heuristic index-like: if feature[0] (label_length) small and feature[2] (line_number) > 0,
    # we avoid enforcing too many constraints; here use a lightweight signal from features only.
    if gte_neg1 is not None:
        likely_index_like = (x[:, 0] < 25) & (x[:, 2] > 0)
        rows = torch.where(likely_index_like & is_target)[0]
        if rows.numel() > 0:
            mask[rows, :] = False
            mask[rows, no_ann] = True
            mask[rows, gte_neg1] = True

    return mask


def compute_loss(outputs: Dict[str, torch.Tensor], y: torch.Tensor, mask: torch.Tensor, gate_module: HardConcreteGate,
                 l0_lambda: float = 1e-4) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    # Only supervised nodes contribute
    if mask is not None:
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            ce = torch.tensor(0.0, device=logits.device)
        else:
            ce = F.cross_entropy(logits[idx], y[idx])
    else:
        ce = F.cross_entropy(logits, y)

    l0 = gate_module.expected_L0()
    loss = ce + l0_lambda * l0
    return loss, {"ce": float(ce.item()), "l0": float(l0.item())}


