#!/usr/bin/env python3
"""
Generate Balanced Training Metrics Report

This script generates a comprehensive metrics report for balanced training
across all checkers (Lower Bound, SQL Quotes, Signature String).
"""

import os
import sys
import json
import glob
import torch
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add GenDATA root to path
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
sys.path.insert(0, str(GEN_DATA_ROOT))

from checker_evaluation_config import CHECKER_CONFIGS, get_checker_config

def load_model_statistics(model_file: Path) -> Optional[Dict[str, Any]]:
    """Load training statistics from a model file"""
    try:
        # Use weights_only=False for our own model files
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        stats = checkpoint.get('training_stats', {})
        if not stats:
            # Try to extract from checkpoint directly
            stats = {
                'best_accuracy': checkpoint.get('best_accuracy', 0.0),
                'final_metrics': checkpoint.get('final_metrics', {}),
                'annotation_type': checkpoint.get('annotation_type', ''),
                'model_type': checkpoint.get('model_type', ''),
                'training_history': checkpoint.get('training_history', {})
            }
        return stats
    except Exception as e:
        # Fallback: try a companion *_stats.json file (common for non-torch models like GBT)
        stats_path = model_file.with_name(f"{model_file.stem}_stats.json")
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    return stats
            except Exception as e_json:
                print(f"Warning: Could not load statistics from {stats_path}: {e_json}")
        print(f"Warning: Could not load statistics from {model_file}: {e}")
        return None

def load_dataset_statistics(dataset_dir: Path, checker_name: str) -> Dict[str, Any]:
    """Load balanced dataset generation statistics"""
    stats_file = dataset_dir / 'real_generation_statistics.json'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load dataset statistics: {e}")
    return {}

def aggregate_metrics_for_checker(checker_name: str, models_dir: Path, dataset_dir: Path) -> Dict[str, Any]:
    """Aggregate metrics for a specific checker"""
    config = get_checker_config(checker_name)
    if not config:
        return {}
    
    annotation_types = config.get('annotation_types', [])
    base_models = config.get('base_models', [])
    
    # Find all model files (balanced suffix preferred; fall back to legacy names for lower_bound)
    model_files = []
    for ann_type in annotation_types:
        ann_name = ann_type.replace('@', '').lower()
        for base_model in base_models:
            patterns = [f"{ann_name}_{base_model}_balanced_model.pth"]
            if checker_name == 'lower_bound':
                # Legacy files may not have the _balanced suffix
                patterns.append(f"{ann_name}_{base_model}_model.pth")
            for pattern in patterns:
            files = list(models_dir.glob(pattern))
            model_files.extend(files)
    
    # Load statistics from each model
    model_metrics = []
    for model_file in model_files:
        stats = load_model_statistics(model_file)
            # Extract model name from file
            model_name = model_file.stem.replace('_balanced_model', '')
            
        if stats:
            # Normalize accuracy values (handle both percentage and decimal formats)
            best_acc = stats.get('best_accuracy', 0.0)
            if best_acc > 1.0:  # If stored as percentage, convert to decimal
                best_acc = best_acc / 100.0
            
            model_metrics.append({
                'model_name': model_name,
                'model_file': str(model_file),
                'best_accuracy': best_acc,
                'final_metrics': stats.get('final_metrics', {}),
                'annotation_type': stats.get('annotation_type', ''),
                'training_history': stats.get('training_history', {})
            })
        else:
            # Still record presence to reach expected counts even if stats unavailable
            model_metrics.append({
                'model_name': model_name,
                'model_file': str(model_file),
                'best_accuracy': 0.0,
                'final_metrics': {},
                'annotation_type': '',
                'training_history': {}
            })
    
    # Load dataset statistics
    dataset_stats = load_dataset_statistics(dataset_dir, checker_name)
    
    # Aggregate metrics
    if model_metrics:
        accuracies = [m['best_accuracy'] for m in model_metrics if m.get('best_accuracy', 0) > 0]
        final_accuracies = []
        for m in model_metrics:
            final_metrics = m.get('final_metrics', {})
            if isinstance(final_metrics, dict):
                acc = final_metrics.get('accuracy', 0.0)
                # Normalize accuracy (handle both percentage and decimal formats)
                if acc > 1.0:  # If stored as percentage, convert to decimal
                    acc = acc / 100.0
                if acc > 0:
                    final_accuracies.append(acc)
        
        return {
            'checker_name': checker_name,
            'display_name': config.get('name', checker_name),
            'total_models': len(model_metrics),
            'expected_models': len(annotation_types) * len(base_models),
            'model_metrics': model_metrics,
            'aggregate_metrics': {
                'best_accuracy': {
                    'mean': statistics.mean(accuracies) if accuracies else 0.0,
                    'median': statistics.median(accuracies) if accuracies else 0.0,
                    'min': min(accuracies) if accuracies else 0.0,
                    'max': max(accuracies) if accuracies else 0.0,
                    'count': len(accuracies)
                },
                'final_accuracy': {
                    'mean': statistics.mean(final_accuracies) if final_accuracies else 0.0,
                    'median': statistics.median(final_accuracies) if final_accuracies else 0.0,
                    'min': min(final_accuracies) if final_accuracies else 0.0,
                    'max': max(final_accuracies) if final_accuracies else 0.0,
                    'count': len(final_accuracies)
                }
            },
            'dataset_statistics': dataset_stats
        }
    
    return {
        'checker_name': checker_name,
        'display_name': config.get('name', checker_name),
        'total_models': 0,
        'expected_models': len(annotation_types) * len(base_models),
        'model_metrics': [],
        'aggregate_metrics': {},
        'dataset_statistics': dataset_stats
    }

def generate_markdown_report(all_metrics: Dict[str, Dict[str, Any]]) -> str:
    """Generate markdown report from aggregated metrics"""
    report_lines = []
    
    report_lines.append("# Balanced Training Metrics Report")
    report_lines.append("")
    report_lines.append("This report contains training and validation metrics for all checkers trained on balanced datasets.")
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append("")
    
    # Summary table
    report_lines.append("| Checker | Expected Models | Trained Models | Best Accuracy (Mean) | Final Accuracy (Mean) |")
    report_lines.append("|---------|----------------|----------------|----------------------|----------------------|")
    
    for checker_name in ['lower_bound', 'sql_quotes', 'signature_string']:
        metrics = all_metrics.get(checker_name, {})
        display_name = metrics.get('display_name', checker_name)
        expected = metrics.get('expected_models', 0)
        trained = metrics.get('total_models', 0)
        best_acc = metrics.get('aggregate_metrics', {}).get('best_accuracy', {}).get('mean', 0.0)
        final_acc = metrics.get('aggregate_metrics', {}).get('final_accuracy', {}).get('mean', 0.0)
        
        report_lines.append(f"| {display_name} | {expected} | {trained} | {best_acc*100:.2f}% | {final_acc*100:.2f}% |")
    
    report_lines.append("")
    report_lines.append("## Per-Checker Details")
    report_lines.append("")
    
    # Detailed metrics for each checker
    for checker_name in ['lower_bound', 'sql_quotes', 'signature_string']:
        metrics = all_metrics.get(checker_name, {})
        if not metrics:
            continue
        
        display_name = metrics.get('display_name', checker_name)
        report_lines.append(f"### {display_name}")
        report_lines.append("")
        
        # Dataset statistics
        dataset_stats = metrics.get('dataset_statistics', {})
        if dataset_stats:
            report_lines.append("#### Dataset Statistics")
            report_lines.append("")
            total_examples = dataset_stats.get('total_examples', 0)
            positive_examples = dataset_stats.get('positive_examples', 0)
            negative_examples = dataset_stats.get('negative_examples', 0)
            
            report_lines.append(f"- **Total Examples**: {total_examples:,}")
            report_lines.append(f"- **Positive Examples**: {positive_examples:,}")
            report_lines.append(f"- **Negative Examples**: {negative_examples:,}")
            if total_examples > 0:
                balance_ratio = positive_examples / total_examples
                report_lines.append(f"- **Balance Ratio**: {balance_ratio:.3f} ({balance_ratio*100:.1f}% positive)")
            report_lines.append("")
            
            # Per-annotation type statistics
            ann_type_counts = dataset_stats.get('annotation_type_counts', {})
            if ann_type_counts:
                report_lines.append("**Per-Annotation Type Statistics:**")
                report_lines.append("")
                for ann_type, counts in ann_type_counts.items():
                    pos = counts.get('positive', 0)
                    neg = counts.get('negative', 0)
                    total = pos + neg
                    if total > 0:
                        ratio = pos / total
                        report_lines.append(f"- {ann_type}: {pos} positive, {neg} negative (balance: {ratio:.3f})")
                report_lines.append("")
        
        # Aggregate training metrics
        agg_metrics = metrics.get('aggregate_metrics', {})
        if agg_metrics:
            report_lines.append("#### Aggregate Training Metrics")
            report_lines.append("")
            
            best_acc = agg_metrics.get('best_accuracy', {})
            if best_acc.get('count', 0) > 0:
                report_lines.append("**Best Validation Accuracy:**")
                report_lines.append(f"- Mean: {best_acc['mean']*100:.2f}%")
                report_lines.append(f"- Median: {best_acc['median']*100:.2f}%")
                report_lines.append(f"- Min: {best_acc['min']*100:.2f}%")
                report_lines.append(f"- Max: {best_acc['max']*100:.2f}%")
                report_lines.append("")
            
            final_acc = agg_metrics.get('final_accuracy', {})
            if final_acc.get('count', 0) > 0:
                report_lines.append("**Final Validation Accuracy:**")
                report_lines.append(f"- Mean: {final_acc['mean']*100:.2f}%")
                report_lines.append(f"- Median: {final_acc['median']*100:.2f}%")
                report_lines.append(f"- Min: {final_acc['min']*100:.2f}%")
                report_lines.append(f"- Max: {final_acc['max']*100:.2f}%")
                report_lines.append("")
        
        # Per-model metrics
        model_metrics = metrics.get('model_metrics', [])
        if model_metrics:
            report_lines.append("#### Per-Model Metrics")
            report_lines.append("")
            report_lines.append("| Model | Annotation Type | Best Accuracy | Final Accuracy |")
            report_lines.append("|-------|-----------------|----------------|----------------|")
            
            for model_metric in sorted(model_metrics, key=lambda x: x.get('model_name', '')):
                model_name = model_metric.get('model_name', 'unknown')
                ann_type = model_metric.get('annotation_type', 'unknown')
                best_acc = model_metric.get('best_accuracy', 0.0)
                final_metrics = model_metric.get('final_metrics', {})
                final_acc = final_metrics.get('accuracy', 0.0) if isinstance(final_metrics, dict) else 0.0
                
                report_lines.append(f"| {model_name} | {ann_type} | {best_acc*100:.2f}% | {final_acc*100:.2f}% |")
            
            report_lines.append("")
    
    report_lines.append("## Notes")
    report_lines.append("")
    report_lines.append("- All models were trained on balanced datasets (50% positive, 50% negative examples)")
    report_lines.append("- Best accuracy refers to the highest validation accuracy achieved during training")
    report_lines.append("- Final accuracy refers to the validation accuracy at the end of training")
    report_lines.append("- Models are saved with `_balanced` suffix to distinguish from non-balanced models")
    report_lines.append("")
    
    return "\n".join(report_lines)

def main():
    """Generate comprehensive metrics report"""
    print("=" * 80)
    print("Generating Balanced Training Metrics Report")
    print("=" * 80)
    
    all_metrics = {}
    
    # Process each checker
    for checker_name in ['lower_bound', 'sql_quotes', 'signature_string']:
        print(f"\nProcessing {checker_name}...")
        
        # Determine directories
        if checker_name == 'lower_bound':
            models_dir = GEN_DATA_ROOT / 'models_annotation_types'
            dataset_dir = GEN_DATA_ROOT / 'balanced_datasets'
        else:
            models_dir = GEN_DATA_ROOT / f'models_annotation_types_{checker_name}'
            dataset_dir = GEN_DATA_ROOT / f'balanced_datasets_{checker_name}'
        
        # Check if directories exist
        if not models_dir.exists():
            print(f"Warning: Models directory not found: {models_dir}")
            continue
        
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
        
        # Aggregate metrics
        metrics = aggregate_metrics_for_checker(checker_name, models_dir, dataset_dir)
        all_metrics[checker_name] = metrics
        
        print(f"  Found {metrics.get('total_models', 0)}/{metrics.get('expected_models', 0)} models")
    
    # Generate markdown report
    report_content = generate_markdown_report(all_metrics)
    
    # Save report
    report_file = GEN_DATA_ROOT / 'BALANCED_TRAINING_METRICS_REPORT.md'
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print("\n" + "=" * 80)
    print(f"Report generated: {report_file}")
    print("=" * 80)
    
    # Print summary
    print("\nSummary:")
    for checker_name, metrics in all_metrics.items():
        display_name = metrics.get('display_name', checker_name)
        trained = metrics.get('total_models', 0)
        expected = metrics.get('expected_models', 0)
        best_acc = metrics.get('aggregate_metrics', {}).get('best_accuracy', {}).get('mean', 0.0)
        print(f"  {display_name}: {trained}/{expected} models, Best Accuracy: {best_acc*100:.2f}%")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

