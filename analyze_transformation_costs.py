#!/usr/bin/env python3
"""
Analyze and present transformation cost ablation results.

Shows the accuracy cost of dropping each transformation in a clear format.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def analyze_transformation_costs(results_file: str):
    """Analyze and present transformation cost results"""
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    data = json.load(open(results_path))
    
    baseline = data.get('baseline', {})
    costs = data.get('transformation_costs', {})
    summary = data.get('summary', {})
    
    print("=" * 80)
    print("TRANSFORMATION COST ABLATION RESULTS")
    print("=" * 80)
    
    # Baseline summary
    baseline_acc = {}
    for key, result in baseline.items():
        if result.get('success'):
            stats = result.get('training_stats', {})
            val_acc = stats.get('val_accuracy')
            if val_acc is not None:
                baseline_acc[key] = val_acc
    
    if baseline_acc:
        baseline_avg = sum(baseline_acc.values()) / len(baseline_acc.values())
        print(f"\nBaseline (All Transformations Enabled):")
        print(f"  Average Validation Accuracy: {baseline_avg:.4f} ({baseline_avg*100:.2f}%)")
        print(f"  Models: {len(baseline_acc)}")
    
    # Transformation costs
    if costs:
        print(f"\n{'=' * 80}")
        print("ACCURACY COST OF DROPPING EACH TRANSFORMATION")
        print(f"{'=' * 80}\n")
        
        # Sort by cost (highest first)
        sorted_costs = sorted(costs.items(), 
                            key=lambda x: x[1].get('average_cost', 0), 
                            reverse=True)
        
        print(f"{'Rank':<6} {'Transformation':<30} {'Cost':<12} {'% Cost':<10} {'Ablated Avg':<12}")
        print("-" * 80)
        
        for rank, (transform, data) in enumerate(sorted_costs, 1):
            cost = data.get('average_cost', 0)
            percent_cost = data.get('percent_cost', 0)
            ablated_avg = data.get('ablated_avg', 0)
            
            print(f"{rank:<6} {transform:<30} {cost:>10.4f}  {percent_cost:>7.2f}%  {ablated_avg:>10.4f}")
        
        # Summary statistics
        print(f"\n{'=' * 80}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 80}")
        
        all_costs = [d.get('average_cost', 0) for d in costs.values()]
        if all_costs:
            print(f"Total Transformations Tested: {len(costs)}")
            print(f"Average Cost: {sum(all_costs) / len(all_costs):.4f}")
            print(f"Max Cost: {max(all_costs):.4f}")
            print(f"Min Cost: {min(all_costs):.4f}")
        
        # Most critical
        most_critical = summary.get('most_critical_transformations', [])
        if most_critical:
            print(f"\nTop 5 Most Critical Transformations:")
            for i, item in enumerate(most_critical[:5], 1):
                print(f"  {i}. {item['transformation']}: {item['average_cost']:.4f} ({item['percent_cost']:.2f}%)")
        
        # Least critical
        least_critical = summary.get('least_critical_transformations', [])
        if least_critical:
            print(f"\nTop 5 Least Critical Transformations:")
            for i, item in enumerate(least_critical[:5], 1):
                print(f"  {i}. {item['transformation']}: {item['average_cost']:.4f} ({item['percent_cost']:.2f}%)")
        
        # Per-model breakdown for top transformations
        print(f"\n{'=' * 80}")
        print("DETAILED BREAKDOWN: Top 3 Most Critical Transformations")
        print(f"{'=' * 80}")
        
        for transform, data in sorted_costs[:3]:
            print(f"\n{transform} (Cost: {data.get('average_cost', 0):.4f}):")
            per_model = data.get('per_model_costs', {})
            if per_model:
                print(f"  {'Model':<30} {'Baseline':<12} {'Ablated':<12} {'Cost':<12} {'% Cost':<10}")
                print("  " + "-" * 78)
                for key, model_data in sorted(per_model.items(), 
                                             key=lambda x: x[1].get('cost', 0), 
                                             reverse=True):
                    print(f"  {key:<30} {model_data['baseline']:>10.4f}  "
                          f"{model_data['ablated']:>10.4f}  {model_data['cost']:>10.4f}  "
                          f"{model_data['percent_cost']:>7.2f}%")
    else:
        print("\nNo transformation cost data available.")
        print("Run the transformation cost ablation study first.")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze transformation cost ablation results'
    )
    parser.add_argument(
        '--results_file',
        type=str,
        default='ablation_transformation_costs/transformation_cost_results.json',
        help='Path to transformation cost results JSON file'
    )
    
    args = parser.parse_args()
    
    analyze_transformation_costs(args.results_file)
    
    return 0


if __name__ == '__main__':
    exit(main())

