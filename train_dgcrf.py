#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from dgcrf_model import DGCRFClassifier, build_hard_class_mask, compute_loss


class GraphDataset(Dataset):
    def __init__(self, data_dir: str):
        self.paths: List[str] = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.pt'):
                    self.paths.append(os.path.join(root, f))
        self.paths.sort()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        obj = torch.load(self.paths[idx], map_location='cpu')
        return obj


def split_indices(n: int, val_ratio: float = 0.1) -> Tuple[List[int], List[int]]:
    import random
    idx = list(range(n))
    random.shuffle(idx)
    v = max(1, int(n * val_ratio)) if n > 1 else 0
    return idx[v:], idx[:v]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ds = GraphDataset(args.data_dir)
    if len(ds) == 0:
        print(f"No .pt graphs found in {args.data_dir}")
        return

    # Infer dimensions from first example
    sample = ds[0]
    x = sample['x']
    y = sample['y']
    pf_classes = tuple(sample.get('pf_classes', ["NO_ANNOTATION", "@Positive", "@NonNegative", "@GTENegativeOne"]))
    num_features = x.size(1)
    num_classes = len(pf_classes)

    model = DGCRFClassifier(num_features=num_features, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Split train/val by graphs
    train_idx, val_idx = split_indices(len(ds), val_ratio=args.val_ratio)
    train_dl = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=1, shuffle=True)
    val_dl = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=1, shuffle=False)

    best_val = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, 'dgcrf_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({"pf_classes": list(pf_classes)}, f)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            x = batch['x'][0].to(device)
            y = batch['y'][0].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask[0].to(device)
            # Build hard class mask
            is_target = mask if mask is not None else torch.ones(x.size(0), dtype=torch.bool, device=device)
            hard_mask = build_hard_class_mask(x, pf_classes, is_target)

            out = model(x, hard_mask, deterministic_gates=False)
            loss, logs = compute_loss(out, y, mask, model.gates, l0_lambda=args.l0_lambda)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        # Validation (deterministic gates)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                x = batch['x'][0].to(device)
                y = batch['y'][0].to(device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask[0].to(device)
                is_target = mask if mask is not None else torch.ones(x.size(0), dtype=torch.bool, device=device)
                hard_mask = build_hard_class_mask(x, pf_classes, is_target)
                out = model(x, hard_mask, deterministic_gates=True)
                loss, _ = compute_loss(out, y, mask, model.gates, l0_lambda=args.l0_lambda)
                val_loss += float(loss.item())

        print(f"Epoch {epoch}: train_loss={total_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.out_dir, 'best_dgcrf.pt')
            torch.save({
                'model_state': model.state_dict(),
                'num_features': num_features,
                'num_classes': num_classes,
                'pf_classes': pf_classes,
            }, ckpt_path)
            print(f"Saved {ckpt_path}")


def main():
    ap = argparse.ArgumentParser(description='Train DG-CRF-lite model on DG2N graphs')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--l0_lambda', type=float, default=1e-4)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()


