#!/usr/bin/env python3
"""
Schema-agnostic metrics aggregator for ablation studies.
- Scans study directories for models/*/metrics.json
- Extracts best_val_loss from top-level or from epochs_log/epochs/history
- Writes per-model comparison JSONs and overall summary JSON
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def extract_metric(m: Dict[str, Any], prefer_accuracy: bool = False) -> Optional[float]:
    """Extract best metric from metrics dict.
    For GBT and similar: prefer test_accuracy (higher is better).
    For others: prefer best_val_loss (lower is better).
    """
    if not m:
        return None
    # For GBT and similar models, use test_accuracy (higher is better)
    if prefer_accuracy and "test_accuracy" in m and isinstance(m["test_accuracy"], (int, float)):
        return float(m["test_accuracy"])
    # For most models, use best_val_loss (lower is better)
    if "best_val_loss" in m and isinstance(m["best_val_loss"], (int, float)):
        return float(m["best_val_loss"])
    epochs: List[Dict[str, Any]] = m.get("epochs_log") or m.get("epochs") or m.get("history") or []
    if isinstance(epochs, list):
        best = None
        for e in epochs:
            if isinstance(e, dict) and isinstance(e.get("val_loss"), (int, float)):
                vl = float(e["val_loss"])
                best = vl if best is None else min(best, vl)
        return best
    return None


def aggregate_gcn_probe(root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lvl in [0, 1, 2]:
        mp = root / f"aug{lvl}" / "models" / "gcn" / "metrics.json"
        out[f"aug{lvl}"] = extract_metric(load_json(mp))
    return {"gcn": out}


def aggregate_all_models(root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    # Model dirs are named like <model>_aug<level>
    for model_dir in root.iterdir():
        if not model_dir.is_dir() or "_aug" not in model_dir.name:
            continue
        try:
            model, lvl_str = model_dir.name.split("_aug", 1)
            lvl = int(lvl_str)
        except Exception:
            continue
        models_dir = model_dir / "models"
        if not models_dir.exists():
            continue
        best = None
        prefer_acc = model.lower() == "gbt"  # GBT uses test_accuracy
        for mp in models_dir.rglob("metrics.json"):
            m = load_json(mp)
            if not m:
                continue
            cand = extract_metric(m, prefer_accuracy=prefer_acc)
            if cand is not None:
                # For accuracy: max is better; for loss: min is better
                if prefer_acc:
                    best = cand if best is None else max(best, cand)
                else:
                    best = cand if best is None else min(best, cand)
            # If no metric but has exit_code, note it
            elif m.get("exit_code") == 0 and best is None:
                # Model ran successfully but no performance metric captured
                best = None  # Keep None to indicate missing metric
        result.setdefault(model, {})[str(lvl)] = best
    return result


def aggregate_comparison(root: Path, level_a: int, level_b: int) -> Dict[str, Any]:
    """Compare metrics between two specific augmentation levels."""
    result: Dict[str, Any] = {}
    
    # First aggregate all models (includes both levels)
    all_metrics = aggregate_all_models(root)
    
    # Extract comparison for each model
    for model, levels_dict in all_metrics.items():
        metric_a = levels_dict.get(str(level_a))
        metric_b = levels_dict.get(str(level_b))
        result[model] = {
            f"aug{level_a}": metric_a,
            f"aug{level_b}": metric_b
        }
        
        # Determine which is better
        if metric_a is None and metric_b is None:
            result[model]["best"] = None
            result[model]["improvement_pct"] = None
        elif metric_a is None:
            result[model]["best"] = f"aug{level_b}"
            result[model]["improvement_pct"] = None
        elif metric_b is None:
            result[model]["best"] = f"aug{level_a}"
            result[model]["improvement_pct"] = None
        else:
            # GBT uses accuracy (higher is better), others use loss (lower is better)
            prefer_acc = model.lower() == "gbt"
            if prefer_acc:
                if metric_b > metric_a:
                    result[model]["best"] = f"aug{level_b}"
                    result[model]["improvement_pct"] = ((metric_b - metric_a) / metric_a) * 100
                else:
                    result[model]["best"] = f"aug{level_a}"
                    result[model]["improvement_pct"] = ((metric_a - metric_b) / metric_b) * 100
            else:
                if metric_b < metric_a:
                    result[model]["best"] = f"aug{level_b}"
                    result[model]["improvement_pct"] = ((metric_a - metric_b) / metric_a) * 100
                else:
                    result[model]["best"] = f"aug{level_a}"
                    result[model]["improvement_pct"] = ((metric_b - metric_a) / metric_b) * 100
    
    return result


def main():
    base = Path.cwd()
    # GCN probe summary
    gcn_summary_path = base / "studies" / "gcn_aug_probe" / "summary.json"
    gcn_summary = aggregate_gcn_probe(gcn_summary_path.parent)
    gcn_summary_path.write_text(json.dumps(gcn_summary, indent=2))
    print(f"WROTE {gcn_summary_path}")
    # All models summary
    am_root = base / "studies" / "all_models_ablation"
    overall = aggregate_all_models(am_root)
    overall_path = am_root / "ablation_overall_summary.json"
    overall_path.write_text(json.dumps(overall, indent=2))
    print(f"WROTE {overall_path}")
    # Comparison: aug0 vs aug1
    comparison = aggregate_comparison(am_root, 0, 1)
    comparison_path = am_root / "ablation_comparison_0_vs_1.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"WROTE {comparison_path}")


if __name__ == "__main__":
    main()


