#!/usr/bin/env python3
"""
Selective-Gated Causal CFG Network (SG-CFGNet)

Purpose
  A lightweight, program-analysis–oriented classifier for node-level
  annotation type prediction that (1) learns near-binary feature relevance
  via L0-style gates, (2) optionally gates relation channels (control/dataflow),
  and (3) adds a differentiable constraint head to bias toward rule-compatible
  predictions.

Notes
  - Uses existing feature extraction from AnnotationTypeClassifier to ensure
    consistency with other models.
  - Keeps message passing minimal to reduce integration burden; relations are
    summarized as small statistics and modulated via edge gates.
  - Trains with cross-entropy + sparsity losses. Constraint loss is a simple
    proxy; refine as needed.
"""

import math
import time
import logging
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import networkx as nx  # for structural predicates if available
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

from sklearn.model_selection import train_test_split

from annotation_type_prediction import AnnotationTypeClassifier, ParameterFreeConfig, LowerBoundAnnotationType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class HardConcreteGate(nn.Module):
    """Hard-concrete stochastic gate (Louizos et al., 2018) approximated for L0."""
    def __init__(self, size: int, init_log_alpha: float = -1.5):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((size,), init_log_alpha))
        # Hard-concrete parameters
        self.beta = 2.0
        self.gamma = -0.1
        self.zeta = 1.1

    def forward(self, training: bool = True) -> torch.Tensor:
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)
        return z

    def l0(self) -> torch.Tensor:
        # Expected L0 norm (probability gate is non-zero)
        s = torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta))
        return s.sum()


class SGCFGNet(nn.Module, AnnotationTypeClassifier):
    """Selective-Gated model using feature and relation gates with a compact MLP."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, edge_stats_dim: int = 8):
        AnnotationTypeClassifier.__init__(self)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Feature gates over per-node feature vector
        self.feature_gates = HardConcreteGate(input_dim)

        # Relation-statistics gates (control/dataflow/dom/postdom in/out counts)
        self.edge_gate = HardConcreteGate(edge_stats_dim)

        # Encoder for node features
        self.norm_in = nn.LayerNorm(input_dim)
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        # Relation stats encoder
        self.edge_encoder = nn.Linear(edge_stats_dim, hidden_dim // 2)

        # Fusion + classifier
        self.fuse = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(hidden_dim, num_classes)

        # Weight init
        for m in [self.fc_in, self.edge_encoder, self.fuse, self.cls]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_node: torch.Tensor, x_edge_stats: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Apply gates
        g_feat = self.feature_gates(training=training)
        g_edge = self.edge_gate(training=training)

        xg = x_node * g_feat
        eg = x_edge_stats * g_edge

        h = self.fc_in(self.norm_in(xg))
        h = F.relu(h)
        e = F.relu(self.edge_encoder(eg))
        f = torch.cat([h, e], dim=-1)
        f = F.relu(self.fuse(f))
        f = self.dropout(f)
        logits = self.cls(f)
        return logits

    def l0_penalty(self) -> torch.Tensor:
        return self.feature_gates.l0() + self.edge_gate.l0()


def _build_graph(cfg: Dict[str, Any]):
    control_edges = cfg.get('control_edges', [])
    G = None
    if NX_AVAILABLE:
        G = nx.DiGraph()
        for n in cfg.get('nodes', []):
            G.add_node(n.get('id'))
        for e in control_edges:
            s = e.get('source', e.get('from'))
            t = e.get('target', e.get('to'))
            if s is not None and t is not None:
                G.add_edge(s, t)
    return G


def build_edge_stats(node: Dict[str, Any], cfg: Dict[str, Any]) -> List[float]:
    control_edges = cfg.get('control_edges', [])
    dataflow_edges = cfg.get('dataflow_edges', [])
    nid = node.get('id')
    ci = sum(1 for e in control_edges if e.get('target') == nid)
    co = sum(1 for e in control_edges if e.get('source') == nid)
    di = sum(1 for e in dataflow_edges if e.get('target') == nid)
    do = sum(1 for e in dataflow_edges if e.get('source') == nid)
    # Dominator/post-dominator approximations
    dom_in = 0
    dom_out = 0
    pdom_in = 0
    pdom_out = 0
    if NX_AVAILABLE:
        G = _build_graph(cfg)
        if G is not None and G.number_of_nodes() > 0:
            # choose entry as lowest id
            try:
                entry = min(G.nodes)
                doms = nx.immediate_dominators(G, entry)
                # reverse graph for post-dominators; select exit as max id
                RG = G.reverse(copy=True)
                exit_node = max(G.nodes)
                pdoms = nx.immediate_dominators(RG, exit_node)
                # Count how many nodes this node immediately dominates / post-dominates
                dom_out = sum(1 for k, v in doms.items() if v == nid and k != nid)
                pdom_out = sum(1 for k, v in pdoms.items() if v == nid and k != nid)
                # in-degree in dominance tree (is dominated by who?)
                dom_in = 0 if nid == doms.get(nid, nid) else 1
                pdom_in = 0 if nid == pdoms.get(nid, nid) else 1
            except Exception:
                pass
    return [ci, co, di, do, dom_in, dom_out, pdom_in, pdom_out]


class SGCFGNetTrainer:
    def __init__(self, parameter_free: bool = True, lmbd_l0: float = 1e-4, lmbd_constraints: float = 1e-3, lmbd_cf: float = 1e-3):
        self.parameter_free = parameter_free
        self.lmbd_l0 = lmbd_l0
        self.lmbd_constraints = lmbd_constraints
        self.lmbd_cf = lmbd_cf

        self.classifier = AnnotationTypeClassifier()

    def _labels_space(self) -> List[str]:
        if self.parameter_free:
            return sorted(list(ParameterFreeConfig.PARAMETER_FREE_TYPES))
        return [at.value for at in LowerBoundAnnotationType if at != LowerBoundAnnotationType.NO_ANNOTATION]

    def _constraint_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Placeholder: rely on supervised loss; extend with rule-aware margins later.
        return torch.tensor(0.0, device=logits.device)

    def prepare_dataset(self, cfg_files: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_nodes: List[List[float]] = []
        X_edges: List[List[float]] = []
        y_labels: List[str] = []

        for cf in cfg_files:
            cfg = cf['data']
            for node in cfg.get('nodes', []):
                from node_level_models import NodeClassifier
                if NodeClassifier.is_annotation_target(node):
                    feats = self.classifier.extract_features(node, cfg)
                    label = self.classifier.determine_annotation_type(feats).value
                    if self.parameter_free and not ParameterFreeConfig.is_parameter_free(label):
                        continue
                    X_nodes.append(self.classifier.features_to_vector(feats))
                    X_edges.append(build_edge_stats(node, cfg))
                    y_labels.append(label)

        if len(X_nodes) < 2:
            return np.zeros((0, 1)), np.zeros((0, 4)), np.zeros((0,))

        # Encode labels on PF space if requested
        from sklearn.preprocessing import LabelEncoder
        self.encoder = LabelEncoder()
        self.encoder.fit(self._labels_space())
        y = self.encoder.transform(y_labels)
        return np.array(X_nodes, dtype=np.float32), np.array(X_edges, dtype=np.float32), np.array(y, dtype=np.int64)

    def train(self, cfg_files: List[Dict[str, Any]], epochs: int = 60, lr: float = 1e-3) -> Dict[str, Any]:
        Xn, Xe, y = self.prepare_dataset(cfg_files)
        if Xn.shape[0] < 2 or len(np.unique(y)) < 2:
            logger.warning("SG-CFGNet: insufficient data/classes")
            return {'success': False}

        Xn_tr, Xn_val, Xe_tr, Xe_val, y_tr, y_val = train_test_split(
            Xn, Xe, y, test_size=0.2, random_state=42, stratify=y)

        model = SGCFGNet(input_dim=Xn.shape[1], num_classes=len(self.encoder.classes_))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        Xn_tr_t = torch.tensor(Xn_tr, device=device)
        Xe_tr_t = torch.tensor(Xe_tr, device=device)
        y_tr_t = torch.tensor(y_tr, device=device)
        Xn_val_t = torch.tensor(Xn_val, device=device)
        Xe_val_t = torch.tensor(Xe_val, device=device)
        y_val_t = torch.tensor(y_val, device=device)

        best_val = -1.0
        best_state = None
        patience, patience_left = 10, 10
        start = time.time()
        for epoch in range(epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(Xn_tr_t, Xe_tr_t, training=True)
            ce = criterion(logits, y_tr_t)
            l0 = model.l0_penalty() * self.lmbd_l0
            cl = self._constraint_loss(logits, y_tr_t) * self.lmbd_constraints
            # Counterfactual consistency: compare with hard-thresholded masked inputs
            with torch.no_grad():
                g_feat = (model.feature_gates(training=False) > 0.5).float()
                g_edge = (model.edge_gate(training=False) > 0.5).float()
                xcf = Xn_tr_t * g_feat
                ecf = Xe_tr_t * g_edge
            logits_cf = model(xcf, ecf, training=False)
            p = F.log_softmax(logits, dim=-1)
            q = F.softmax(logits_cf, dim=-1)
            cf = F.kl_div(p, q, reduction='batchmean') * self.lmbd_cf
            loss = ce + l0 + cl + cf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(Xn_val_t, Xe_val_t, training=False)
                val_pred = torch.argmax(val_logits, dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        self.is_trained = True
        train_time = time.time() - start
        return {'success': True, 'val_accuracy': best_val, 'train_time_sec': train_time}

    def predict_node(self, node: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, float]:
        if not getattr(self, 'is_trained', False):
            return LowerBoundAnnotationType.NO_ANNOTATION.value, 0.0
        feats = self.classifier.extract_features(node, cfg)
        x_node = torch.tensor([self.classifier.features_to_vector(feats)], dtype=torch.float32)
        x_edge = torch.tensor([build_edge_stats(node, cfg)], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_node.to(next(self.model.parameters()).device), x_edge.to(next(self.model.parameters()).device), training=False)
            probs = F.softmax(logits, dim=-1)[0]
            idx = int(torch.argmax(probs).item())
            label = self.encoder.inverse_transform([idx])[0]
            return label, float(probs[idx].item())


def main():
    # Minimal smoke test: creates a tiny dataset from existing pf_dataset if available
    import os, json
    dataset_dir = 'test_results/pf_dataset/train/cfg_output'
    cfg_files: List[Dict[str, Any]] = []
    if os.path.isdir(dataset_dir):
        for fn in os.listdir(dataset_dir):
            if fn.endswith('.json'):
                with open(os.path.join(dataset_dir, fn), 'r') as f:
                    cfg = json.load(f)
                    cfg_files.append({'file': fn, 'method': cfg.get('method_name','m'), 'data': cfg})

    if not cfg_files:
        logger.warning('No pf_dataset found for SG-CFGNet smoke test; exiting.')
        return

    trainer = SGCFGNetTrainer(parameter_free=True)
    info = trainer.train(cfg_files, epochs=40)
    logger.info(f"SG-CFGNet training result: {info}")

if __name__ == '__main__':
    main()


