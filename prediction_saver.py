#!/usr/bin/env python3
"""
Prediction Saver Utility
Saves model predictions to separate files for manual inspection
"""

import json
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionSaver:
    def __init__(self, output_dir='predictions_manual_inspection'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_predictions(self, model_name, project_name, predictions, metadata=None):
        """Save predictions to a structured file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{project_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data structure
        data = {
            'model_name': model_name,
            'project_name': project_name,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'predictions': predictions,
            'summary': {
                'total_predictions': len(predictions),
                'prediction_types': self._analyze_predictions(predictions)
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(predictions)} predictions to {filepath}")
        return filepath
    
    def _analyze_predictions(self, predictions):
        """Analyze prediction types and patterns"""
        if not predictions:
            return {'no_predictions': True}
        
        analysis = {
            'total_count': len(predictions),
            'line_numbers': [p.get('line') for p in predictions if p.get('line')],
            'confidence_scores': [p.get('confidence', 0) for p in predictions],
            'node_types': {},
            'avg_confidence': 0
        }
        
        # Count node types
        for pred in predictions:
            node_type = pred.get('node_type', 'unknown')
            analysis['node_types'][node_type] = analysis['node_types'].get(node_type, 0) + 1
        
        # Calculate average confidence
        if analysis['confidence_scores']:
            analysis['avg_confidence'] = sum(analysis['confidence_scores']) / len(analysis['confidence_scores'])
        
        return analysis
    
    def save_model_comparison(self, model_predictions, project_name):
        """Save predictions from multiple models for comparison"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{project_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        comparison_data = {
            'project_name': project_name,
            'timestamp': timestamp,
            'models': {},
            'comparison_summary': {}
        }
        
        # Collect predictions from all models
        for model_name, predictions in model_predictions.items():
            comparison_data['models'][model_name] = {
                'predictions': predictions,
                'summary': {
                    'total_predictions': len(predictions),
                    'prediction_types': self._analyze_predictions(predictions)
                }
            }
        
        # Generate comparison summary
        comparison_data['comparison_summary'] = self._generate_comparison_summary(model_predictions)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Saved model comparison for {project_name} to {filepath}")
        return filepath
    
    def _generate_comparison_summary(self, model_predictions):
        """Generate summary comparing predictions across models"""
        summary = {
            'model_counts': {name: len(preds) for name, preds in model_predictions.items()},
            'consensus_predictions': [],
            'unique_predictions': {},
            'overall_stats': {
                'total_unique_lines': set(),
                'avg_predictions_per_model': 0
            }
        }
        
        # Collect all unique line numbers
        all_lines = set()
        for model_name, predictions in model_predictions.items():
            model_lines = set()
            for pred in predictions:
                if pred.get('line'):
                    line = pred['line']
                    all_lines.add(line)
                    model_lines.add(line)
            summary['unique_predictions'][model_name] = list(model_lines)
        
        summary['overall_stats']['total_unique_lines'] = len(all_lines)
        summary['overall_stats']['avg_predictions_per_model'] = sum(summary['model_counts'].values()) / len(summary['model_counts'])
        
        # Find consensus predictions (lines predicted by multiple models)
        line_counts = {}
        for model_name, predictions in model_predictions.items():
            for pred in predictions:
                line = pred.get('line')
                if line:
                    line_counts[line] = line_counts.get(line, 0) + 1
        
        summary['consensus_predictions'] = [line for line, count in line_counts.items() if count > 1]
        
        return summary
    
    def create_readable_report(self, json_file):
        """Create a human-readable report from prediction JSON"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        report_file = json_file.replace('.json', '_report.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"PREDICTION REPORT\n")
            f.write(f"================\n\n")
            f.write(f"Model: {data['model_name']}\n")
            f.write(f"Project: {data['project_name']}\n")
            f.write(f"Timestamp: {data['timestamp']}\n\n")
            
            summary = data['summary']
            f.write(f"SUMMARY\n")
            f.write(f"-------\n")
            f.write(f"Total Predictions: {summary['total_predictions']}\n")
            f.write(f"Average Confidence: {summary['prediction_types'].get('avg_confidence', 0):.3f}\n\n")
            
            if 'node_types' in summary['prediction_types']:
                f.write(f"PREDICTION BREAKDOWN BY NODE TYPE\n")
                f.write(f"---------------------------------\n")
                for node_type, count in summary['prediction_types']['node_types'].items():
                    f.write(f"{node_type}: {count}\n")
                f.write("\n")
            
            f.write(f"DETAILED PREDICTIONS\n")
            f.write(f"-------------------\n")
            for i, pred in enumerate(data['predictions'], 1):
                f.write(f"{i}. Line {pred.get('line', 'N/A')}: {pred.get('label', 'N/A')[:80]}...\n")
                f.write(f"   Type: {pred.get('node_type', 'N/A')}, Confidence: {pred.get('confidence', 'N/A')}\n\n")
        
        logger.info(f"Created readable report: {report_file}")
        return report_file

def main():
    parser = argparse.ArgumentParser(description='Prediction Saver Utility')
    parser.add_argument('--output_dir', default='predictions_manual_inspection', 
                       help='Output directory for saved predictions')
    parser.add_argument('--create_reports', action='store_true',
                       help='Create human-readable reports from JSON files')
    
    args = parser.parse_args()
    
    saver = PredictionSaver(args.output_dir)
    
    if args.create_reports:
        # Find all JSON files in output directory and create reports
        json_files = [f for f in os.listdir(args.output_dir) if f.endswith('.json')]
        for json_file in json_files:
            saver.create_readable_report(os.path.join(args.output_dir, json_file))

if __name__ == "__main__":
    main()
