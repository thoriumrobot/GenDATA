#!/usr/bin/env python3
"""
Script to display model predictions with their code context.

This script reads prediction files and shows:
1. The actual code where predictions were made
2. The model predictions with confidence scores
3. The line numbers and context around each prediction
"""

import os
import json
import argparse
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

class PredictionContextViewer:
    """View predictions with their code context"""
    
    def __init__(self, predictions_dir: str = "predictions_annotation_types"):
        self.predictions_dir = predictions_dir
        
    def load_predictions(self, prediction_file: str) -> Dict[str, Any]:
        """Load predictions from a JSON file"""
        try:
            with open(prediction_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {prediction_file}: {e}")
            return {}
    
    def read_java_file(self, java_file_path: str) -> Optional[List[str]]:
        """Read Java file and return lines"""
        try:
            with open(java_file_path, 'r') as f:
                return f.readlines()
        except Exception as e:
            print(f"Error reading {java_file_path}: {e}")
            return None
    
    def extract_line_number(self, prediction: Dict[str, Any]) -> Optional[int]:
        """Extract line number from prediction"""
        # Try different possible keys for line number
        line_keys = ['line_number', 'line', 'line_num', 'position']
        for key in line_keys:
            if key in prediction:
                try:
                    return int(prediction[key])
                except (ValueError, TypeError):
                    continue
        
        # Try to extract from context or location
        if 'context' in prediction:
            context = prediction['context']
            # Look for line numbers in context
            line_match = re.search(r'line\s+(\d+)', context, re.IGNORECASE)
            if line_match:
                return int(line_match.group(1))
        
        return None
    
    def is_meaningful_prediction(self, prediction: Dict[str, Any], java_lines: Optional[List[str]] = None, 
                                line_num: Optional[int] = None, confidence_threshold: float = 0.6) -> bool:
        """Check if this is a meaningful prediction worth showing"""
        # Filter out predictions on comments, whitespace, and imports
        if java_lines and line_num and line_num <= len(java_lines):
            line_content = java_lines[line_num - 1].strip()
            
            # Skip empty lines
            if not line_content:
                return False
            
            # Skip comment lines
            if line_content.startswith('//') or line_content.startswith('*') or line_content.startswith('/*'):
                return False
            
            # Skip package and import statements
            if line_content.startswith('package ') or line_content.startswith('import '):
                return False
            
            # Skip class/interface declarations (unless they have parameters)
            if line_content.startswith('public class ') or line_content.startswith('private class '):
                return False
            
            # Skip lines that only contain annotations (like @Positive)
            if line_content.strip().startswith('@') and len(line_content.strip()) < 50:
                return False
            
            # Skip lines that are just whitespace with annotations
            if line_content.strip() == '@Positive' or line_content.strip() == '@NonNegative' or line_content.strip() == '@GTENegativeOne':
                return False
        
        # Filter by confidence - only show high confidence predictions
        confidence = prediction.get('confidence', prediction.get('score', 0))
        if confidence < confidence_threshold:
            return False
        
        return True
    
    def get_code_context(self, lines: List[str], line_num: int, context_lines: int = 3) -> Dict[str, Any]:
        """Get code context around a specific line"""
        if not lines or line_num < 1 or line_num > len(lines):
            return {'before': [], 'target': '', 'after': [], 'line_number': line_num}
        
        start_line = max(1, line_num - context_lines)
        end_line = min(len(lines), line_num + context_lines)
        
        return {
            'before': lines[start_line-1:line_num-1],
            'target': lines[line_num-1].rstrip() if line_num <= len(lines) else '',
            'after': lines[line_num:end_line],
            'line_number': line_num
        }
    
    def format_prediction(self, prediction: Dict[str, Any]) -> str:
        """Format a single prediction for display"""
        model_type = prediction.get('model_type', 'unknown')
        annotation_type = prediction.get('annotation_type', 'unknown')
        confidence = prediction.get('confidence', prediction.get('score', 0))
        
        # Try to extract the actual code element being predicted
        target = prediction.get('target', prediction.get('element', prediction.get('variable', 'unknown')))
        
        # Try to get method or class context
        method = prediction.get('method', prediction.get('function', ''))
        class_name = prediction.get('class', prediction.get('class_name', ''))
        
        context_info = []
        if class_name:
            context_info.append(f"Class: {class_name}")
        if method:
            context_info.append(f"Method: {method}")
        
        context_str = f" ({', '.join(context_info)})" if context_info else ""
        
        return f"  🤖 {model_type} → {annotation_type} on '{target}'{context_str} (confidence: {confidence:.3f})"
    
    def show_predictions_for_file(self, prediction_file: str, show_context: bool = True, context_lines: int = 3, confidence_threshold: float = 0.6):
        """Show predictions for a specific file with code context"""
        predictions_data = self.load_predictions(prediction_file)
        if not predictions_data:
            return
        
        java_file = predictions_data.get('file', '')
        predictions = predictions_data.get('predictions', [])
        
        # Read the Java file for context
        java_lines = None
        if show_context and java_file and os.path.exists(java_file):
            java_lines = self.read_java_file(java_file)
        
        # Group predictions by line number for better display
        predictions_by_line = {}
        meaningful_predictions = 0
        for pred in predictions:
            line_num = self.extract_line_number(pred)
            if line_num:
                # Filter for meaningful predictions
                if self.is_meaningful_prediction(pred, java_lines, line_num, confidence_threshold):
                    if line_num not in predictions_by_line:
                        predictions_by_line[line_num] = []
                    predictions_by_line[line_num].append(pred)
                    meaningful_predictions += 1
        
        print(f"\n{'='*80}")
        print(f"📄 FILE: {java_file}")
        print(f"🎯 PREDICTIONS: {len(predictions)} total, {meaningful_predictions} meaningful")
        print(f"{'='*80}")
        
        if not predictions_by_line:
            print("No meaningful predictions found in this file.")
            print(f"(Predictions filtered by: confidence > {confidence_threshold*100:.0f}%, excluding comments/imports/empty lines)")
            return
        
        # Sort by line number
        for line_num in sorted(predictions_by_line.keys()):
            line_predictions = predictions_by_line[line_num]
            
            print(f"\n📍 LINE {line_num}")
            print("-" * 40)
            
            # Show code context if available
            if show_context and java_lines:
                context = self.get_code_context(java_lines, line_num, context_lines)
                
                # Show before lines
                for i, line in enumerate(context['before']):
                    print(f"  {line_num - len(context['before']) + i:3d}: {line.rstrip()}")
                
                # Show target line with highlighting
                target_line = context['target']
                print(f"→ {line_num:3d}: {target_line}")  # Arrow to highlight target
                
                # Show after lines
                for i, line in enumerate(context['after']):
                    print(f"  {line_num + i + 1:3d}: {line.rstrip()}")
            else:
                # Just show the line number if we can't read the file
                print(f"  Line {line_num} (file not accessible for context)")
            
            # Show predictions for this line
            print("\n🎯 PREDICTIONS:")
            for pred in line_predictions:
                print(self.format_prediction(pred))
    
    def show_all_predictions(self, show_context: bool = True, context_lines: int = 3, 
                           filter_model: Optional[str] = None, filter_annotation: Optional[str] = None,
                           confidence_threshold: float = 0.6):
        """Show predictions from all files in the predictions directory"""
        if not os.path.exists(self.predictions_dir):
            print(f"Predictions directory {self.predictions_dir} not found!")
            return
        
        prediction_files = []
        for root, dirs, files in os.walk(self.predictions_dir):
            for file in files:
                if file.endswith('.predictions.json') and not file.startswith('pipeline_summary'):
                    prediction_files.append(os.path.join(root, file))
        
        if not prediction_files:
            print(f"No prediction files found in {self.predictions_dir}")
            return
        
        print(f"🔍 Found {len(prediction_files)} prediction files")
        
        total_predictions = 0
        for pred_file in sorted(prediction_files):
            predictions_data = self.load_predictions(pred_file)
            predictions = predictions_data.get('predictions', [])
            
            # Apply filters if specified
            if filter_model or filter_annotation:
                filtered_predictions = []
                for pred in predictions:
                    if filter_model and pred.get('model_type', '') != filter_model:
                        continue
                    if filter_annotation and pred.get('annotation_type', '') != filter_annotation:
                        continue
                    filtered_predictions.append(pred)
                
                if not filtered_predictions:
                    continue  # Skip files with no matching predictions
                
                predictions_data['predictions'] = filtered_predictions
            
            self.show_predictions_for_file(pred_file, show_context, context_lines, confidence_threshold)
            total_predictions += len(predictions_data.get('predictions', []))
        
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY: {total_predictions} total predictions across {len(prediction_files)} files")
        print(f"{'='*80}")
    
    def show_prediction_summary(self):
        """Show a summary of all predictions by model and annotation type"""
        if not os.path.exists(self.predictions_dir):
            print(f"Predictions directory {self.predictions_dir} not found!")
            return
        
        model_counts = {}
        annotation_counts = {}
        file_counts = {}
        
        prediction_files = []
        for root, dirs, files in os.walk(self.predictions_dir):
            for file in files:
                if file.endswith('.predictions.json') and not file.startswith('pipeline_summary'):
                    prediction_files.append(os.path.join(root, file))
        
        for pred_file in prediction_files:
            predictions_data = self.load_predictions(pred_file)
            predictions = predictions_data.get('predictions', [])
            java_file = predictions_data.get('file', 'Unknown')
            
            file_counts[java_file] = len(predictions)
            
            for pred in predictions:
                model_type = pred.get('model_type', 'unknown')
                annotation_type = pred.get('annotation_type', 'unknown')
                
                model_counts[model_type] = model_counts.get(model_type, 0) + 1
                annotation_counts[annotation_type] = annotation_counts.get(annotation_type, 0) + 1
        
        print(f"\n📊 PREDICTION SUMMARY")
        print(f"{'='*50}")
        
        print(f"\n🤖 Predictions by Model Type:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} predictions")
        
        print(f"\n🏷️  Predictions by Annotation Type:")
        for annotation, count in sorted(annotation_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {annotation}: {count} predictions")
        
        print(f"\n📄 Files with Most Predictions:")
        for file, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {os.path.basename(file)}: {count} predictions")


def main():
    parser = argparse.ArgumentParser(description='Show model predictions with code context')
    parser.add_argument('--predictions_dir', default='predictions_annotation_types',
                       help='Directory containing prediction files')
    parser.add_argument('--file', help='Show predictions for a specific file only')
    parser.add_argument('--no-context', action='store_true', help='Don\'t show code context')
    parser.add_argument('--context-lines', type=int, default=3, help='Number of context lines to show')
    parser.add_argument('--filter-model', help='Filter predictions by model type')
    parser.add_argument('--filter-annotation', help='Filter predictions by annotation type')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, 
                       help='Minimum confidence threshold for showing predictions (default: 0.6)')
    parser.add_argument('--summary', action='store_true', help='Show prediction summary only')
    
    args = parser.parse_args()
    
    viewer = PredictionContextViewer(args.predictions_dir)
    
    if args.summary:
        viewer.show_prediction_summary()
    elif args.file:
        viewer.show_predictions_for_file(args.file, not args.no_context, args.context_lines, args.confidence_threshold)
    else:
        viewer.show_all_predictions(
            show_context=not args.no_context,
            context_lines=args.context_lines,
            filter_model=args.filter_model,
            filter_annotation=args.filter_annotation,
            confidence_threshold=args.confidence_threshold
        )


if __name__ == '__main__':
    main()
