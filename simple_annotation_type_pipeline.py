#!/usr/bin/env python3
"""
Simplified Annotation Type Pipeline Script
Direct training and testing of annotation-specific models.
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import torch
import logging
from pathlib import Path
import time
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAnnotationTypePipeline:
    """Simplified pipeline for training and testing annotation types"""
    
    def __init__(self, project_root, warnings_file, cfwr_root, mode='train'):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        self.mode = mode
        
        # Set up directories
        self.models_dir = os.path.join(cfwr_root, 'models_annotation_types')
        self.predictions_dir = os.path.join(cfwr_root, 'predictions_annotation_types')
        
        # Create directories
        for dir_path in [self.models_dir, self.predictions_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Annotation types
        self.annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        self.script_mapping = {
            '@Positive': 'annotation_type_rl_positive.py',
            '@NonNegative': 'annotation_type_rl_nonnegative.py',
            '@GTENegativeOne': 'annotation_type_rl_gtenegativeone.py'
        }
    
    def run_training_pipeline(self, episodes=50, base_model='gcn'):
        """Run the training pipeline for annotation type models"""
        logger.info("Starting simplified annotation type training pipeline")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Episodes: {episodes}")
        logger.info(f"Base model: {base_model}")
        
        # Train annotation type models
        logger.info("Training annotation type models")
        if not self._train_annotation_type_models(episodes, base_model):
            logger.error("Failed to train annotation type models")
            return False
        
        logger.info("Training pipeline completed successfully")
        return True
    
    def run_prediction_pipeline(self, target_file=None):
        """Run the prediction pipeline"""
        logger.info("Starting simplified annotation type prediction pipeline")
        
        # Use target file or default project files
        if target_file:
            target_files = [target_file]
        else:
            # Find Java files in project root
            target_files = glob.glob(os.path.join(self.project_root, '**/*.java'), recursive=True)
        
        if not target_files:
            logger.error("No target files found for prediction")
            return False
        
        logger.info(f"Found {len(target_files)} target files")
        
        # Predict and place annotations
        logger.info("Predicting and placing annotations")
        if not self._predict_and_place_annotations(target_files):
            logger.error("Failed to predict and place annotations")
            return False
        
        logger.info("Prediction pipeline completed successfully")
        return True
    
    def _train_annotation_type_models(self, episodes, base_model):
        """Train models for each annotation type"""
        success_count = 0
        
        for annotation_type in self.annotation_types:
            logger.info(f"Training model for {annotation_type}")
            
            script_name = self.script_mapping[annotation_type]
            cmd = [
                'python', script_name,
                '--project_root', self.project_root,
                '--warnings_file', self.warnings_file,
                '--cfwr_root', self.cfwr_root,
                '--episodes', str(episodes),
                '--base_model', base_model,
                '--device', 'cpu'
            ]
            
            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
                
                if result.returncode == 0:
                    logger.info(f"{annotation_type} model training completed successfully")
                    success_count += 1
                else:
                    logger.error(f"{annotation_type} model training failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"{annotation_type} model training timed out")
            except Exception as e:
                logger.error(f"Error training {annotation_type} model: {e}")
        
        logger.info(f"Successfully trained {success_count}/{len(self.annotation_types)} annotation type models")
        return success_count > 0
    
    def _predict_and_place_annotations(self, target_files):
        """Predict and place annotations using trained models"""
        try:
            logger.info("Predicting and placing annotations")
            
            # Load trained models info
            models = {}
            for annotation_type in self.annotation_types:
                model_file = os.path.join(self.models_dir, f"{annotation_type.replace('@', '').lower()}_model.pth")
                stats_file = os.path.join(self.models_dir, f"{annotation_type.replace('@', '').lower()}_stats.json")
                
                if os.path.exists(model_file) and os.path.exists(stats_file):
                    models[annotation_type] = {
                        'model_file': model_file,
                        'stats_file': stats_file
                    }
                    logger.info(f"Found trained model for {annotation_type}")
                else:
                    logger.warning(f"Model not found for {annotation_type}")
            
            if not models:
                logger.error("No trained models found")
                return False
            
            # Process each Java file
            total_predictions = 0
            processed_files = 0
            
            for java_file in target_files:
                try:
                    predictions = self._predict_annotations_for_file(java_file, models)
                    if predictions:
                        self._place_annotations_in_file(java_file, predictions)
                        total_predictions += len(predictions)
                        processed_files += 1
                        
                        logger.info(f"Processed {java_file}: {len(predictions)} predictions")
                except Exception as e:
                    logger.warning(f"Error processing {java_file}: {e}")
            
            logger.info(f"Made {total_predictions} annotation predictions across {processed_files} files")
            return True
            
        except Exception as e:
            logger.error(f"Error predicting and placing annotations: {e}")
            return False
    
    def _predict_annotations_for_file(self, java_file, models):
        """Predict annotations for a single Java file"""
        # Mock prediction logic - in real implementation, this would use the trained models
        # to analyze the file and predict annotation placements
        
        predictions = []
        
        # Simple heuristic-based prediction for demonstration
        with open(java_file, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Predict @Positive for count/size/length variables
            if any(keyword in line_lower for keyword in ['count', 'size', 'length']) and 'int' in line_lower:
                predictions.append({
                    'line': i,
                    'annotation_type': '@Positive',
                    'confidence': 0.8,
                    'reason': 'count/size/length variable'
                })
            
            # Predict @NonNegative for index variables
            elif any(keyword in line_lower for keyword in ['index', 'offset', 'position']) and 'int' in line_lower:
                predictions.append({
                    'line': i,
                    'annotation_type': '@NonNegative',
                    'confidence': 0.7,
                    'reason': 'index/offset variable'
                })
            
            # Predict @GTENegativeOne for capacity variables
            elif any(keyword in line_lower for keyword in ['capacity', 'limit', 'bound']) and 'int' in line_lower:
                predictions.append({
                    'line': i,
                    'annotation_type': '@GTENegativeOne',
                    'confidence': 0.6,
                    'reason': 'capacity/limit variable'
                })
        
        return predictions
    
    def _place_annotations_in_file(self, java_file, predictions):
        """Place annotations in a Java file"""
        try:
            # Create backup
            backup_file = java_file + '.backup'
            shutil.copy2(java_file, backup_file)
            
            with open(java_file, 'r') as f:
                lines = f.readlines()
            
            # Sort predictions by line number in descending order to avoid line shift issues
            predictions.sort(key=lambda x: x['line'], reverse=True)
            
            annotations_placed = 0
            for pred in predictions:
                line_num = pred['line'] - 1  # Convert to 0-based index
                annotation = pred['annotation_type']
                
                if 0 <= line_num < len(lines):
                    # Add annotation before the line
                    lines.insert(line_num, f"    {annotation}\n")
                    annotations_placed += 1
            
            # Write back to file
            with open(java_file, 'w') as f:
                f.writelines(lines)
                
            logger.info(f"Placed {annotations_placed} annotations in {java_file}")
            
            # Save prediction report
            report_file = os.path.join(self.predictions_dir, f"{os.path.basename(java_file)}.predictions.json")
            with open(report_file, 'w') as f:
                json.dump({
                    'file': java_file,
                    'predictions': predictions,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error placing annotations in {java_file}: {e}")
    
    def generate_summary_report(self):
        """Generate a summary report of all training and prediction results"""
        logger.info("Generating summary report")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': self.mode,
            'project_root': self.project_root,
            'models_trained': [],
            'predictions_made': [],
            'overall_success': True
        }
        
        # Check training results
        for annotation_type in self.annotation_types:
            model_file = os.path.join(self.models_dir, f"{annotation_type.replace('@', '').lower()}_model.pth")
            stats_file = os.path.join(self.models_dir, f"{annotation_type.replace('@', '').lower()}_stats.json")
            
            if os.path.exists(model_file) and os.path.exists(stats_file):
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    report['models_trained'].append({
                        'annotation_type': annotation_type,
                        'model_file': model_file,
                        'episodes': len(stats.get('episodes', [])),
                        'final_reward': stats.get('rewards', [0])[-1] if stats.get('rewards') else 0,
                        'success': True
                    })
                except Exception as e:
                    logger.warning(f"Error reading stats for {annotation_type}: {e}")
                    report['models_trained'].append({
                        'annotation_type': annotation_type,
                        'success': False,
                        'error': str(e)
                    })
                    report['overall_success'] = False
            else:
                report['models_trained'].append({
                    'annotation_type': annotation_type,
                    'success': False,
                    'error': 'Model files not found'
                })
                report['overall_success'] = False
        
        # Check prediction results
        prediction_files = glob.glob(os.path.join(self.predictions_dir, '*.predictions.json'))
        for pred_file in prediction_files:
            try:
                with open(pred_file, 'r') as f:
                    pred_data = json.load(f)
                
                report['predictions_made'].append({
                    'file': pred_data['file'],
                    'predictions_count': len(pred_data['predictions']),
                    'timestamp': pred_data['timestamp']
                })
            except Exception as e:
                logger.warning(f"Error reading prediction file {pred_file}: {e}")
        
        # Save report
        report_file = os.path.join(self.predictions_dir, 'pipeline_summary_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {report_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description='Simplified Annotation Type Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='train',
                       help='Pipeline mode: train, predict, or both')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index',
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/CFWR/index1.small.out',
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/CFWR',
                       help='Root directory of CFWR project')
    parser.add_argument('--target_file', 
                       help='Specific Java file to process (for prediction mode)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes (for training mode)')
    parser.add_argument('--base_model', default='gcn', choices=['gcn', 'gbt', 'causal'],
                       help='Base model type (for training mode)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SimpleAnnotationTypePipeline(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        mode=args.mode
    )
    
    # Run pipeline
    success = True
    
    if args.mode in ['train', 'both']:
        logger.info("=== TRAINING PHASE ===")
        success &= pipeline.run_training_pipeline(
            episodes=args.episodes,
            base_model=args.base_model
        )
    
    if args.mode in ['predict', 'both']:
        logger.info("=== PREDICTION PHASE ===")
        success &= pipeline.run_prediction_pipeline(target_file=args.target_file)
    
    # Generate summary report
    logger.info("=== SUMMARY REPORT ===")
    report = pipeline.generate_summary_report()
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*50}")
    
    for model_info in report['models_trained']:
        status = "✓ SUCCESS" if model_info['success'] else "✗ FAILED"
        logger.info(f"{model_info['annotation_type']}: {status}")
        if model_info['success']:
            logger.info(f"  Episodes: {model_info['episodes']}, Final reward: {model_info['final_reward']:.3f}")
    
    logger.info(f"\nPredictions made: {len(report['predictions_made'])}")
    for pred in report['predictions_made']:
        logger.info(f"  {pred['file']}: {pred['predictions_count']} predictions")
    
    if success and report['overall_success']:
        logger.info("\n✓ Pipeline completed successfully!")
        return 0
    else:
        logger.error("\n✗ Pipeline completed with errors")
        return 1

if __name__ == '__main__':
    exit(main())
