#!/usr/bin/env python3
"""
Generate a formatted comparison report between augmentation levels.
"""
import json
from pathlib import Path


def main():
    comparison_path = Path("studies/all_models_ablation/ablation_comparison_0_vs_1.json")
    
    if not comparison_path.exists():
        print(f"Error: {comparison_path} not found")
        return
    
    with open(comparison_path) as f:
        data = json.load(f)
    
    print("=" * 70)
    print("ABLATION STUDY RESULTS: No Augmentation vs Level 1 Augmentation")
    print("=" * 70)
    print()
    
    for model in sorted(data.keys()):
        metrics = data[model]
        aug0 = metrics.get("aug0")
        aug1 = metrics.get("aug1")
        best = metrics.get("best")
        improvement = metrics.get("improvement_pct")
        
        if aug0 is None and aug1 is None:
            print(f"{model.upper():10} | No metrics available (failed or no metric captured)")
            continue
        
        if model.lower() == "gbt":
            # Higher is better for accuracy
            metric_name = "Test Accuracy"
            if aug0 is not None and aug1 is not None:
                if best == "aug1":
                    print(f"{model.upper():10} | NO-AUG: {aug0:.4f} | L1-AUG: {aug1:.4f} | Best: L1-AUG ({improvement:+.1f}%)")
                else:
                    print(f"{model.upper():10} | NO-AUG: {aug0:.4f} | L1-AUG: {aug1:.4f} | Best: NO-AUG ({improvement:+.1f}%)")
            elif aug0 is not None:
                print(f"{model.upper():10} | NO-AUG: {aug0:.4f} | L1-AUG: N/A | Incomplete")
            elif aug1 is not None:
                print(f"{model.upper():10} | NO-AUG: N/A | L1-AUG: {aug1:.4f} | Incomplete")
        else:
            # Lower is better for loss
            metric_name = "Validation Loss"
            if aug0 is not None and aug1 is not None:
                if best == "aug1":
                    print(f"{model.upper():10} | NO-AUG: {aug0:.4f} | L1-AUG: {aug1:.4f} | Best: L1-AUG ({improvement:+.1f}% better)")
                else:
                    print(f"{model.upper():10} | NO-AUG: {aug0:.4f} | L1-AUG: {aug1:.4f} | Best: NO-AUG ({improvement:+.1f}% worse)")
            elif aug0 is not None:
                print(f"{model.upper():10} | NO-AUG: {aug0:.4f} | L1-AUG: N/A | Incomplete")
            elif aug1 is not None:
                print(f"{model.upper():10} | NO-AUG: N/A | L1-AUG: {aug1:.4f} | Incomplete")
    
    print()
    print("=" * 70)
    
    # Summary statistics
    improved = sum(1 for m in data.values() if m.get("best") == "aug1")
    degraded = sum(1 for m in data.values() if m.get("best") == "aug0")
    no_data = sum(1 for m in data.values() if m.get("aug0") is None and m.get("aug1") is None)
    
    print(f"Summary: {improved} model(s) improved with L1 augmentation, {degraded} degraded, {no_data} no data")
    print("=" * 70)


if __name__ == "__main__":
    main()

