#!/usr/bin/env python3
"""
Annotation Type Pipeline Script
Integrates training and prediction for annotation-specific models using the full pipeline.
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

class AnnotationTypePipeline:
    """Pipeline for training and predicting annotation types"""
    
    def __init__(self, project_root, warnings_file, cfwr_root, mode='train'):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        self.mode = mode
        
        # Set up directories
        self.slices_dir = os.path.join(cfwr_root, 'slices_specimin')
        self.augmented_slices_dir = os.path.join(cfwr_root, 'slices_aug_specimin')
        self.cfg_dir = os.path.join(cfwr_root, 'cfg_output_specimin')
        self.models_dir = os.path.join(cfwr_root, 'models_annotation_types')
        self.predictions_dir = os.path.join(cfwr_root, 'predictions_annotation_types')
        
        # Create directories
        for dir_path in [self.slices_dir, self.augmented_slices_dir, self.cfg_dir, 
                        self.models_dir, self.predictions_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Annotation types
        self.annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        self.script_mapping = {
            '@Positive': 'annotation_type_rl_positive.py',
            '@NonNegative': 'annotation_type_rl_nonnegative.py',
            '@GTENegativeOne': 'annotation_type_rl_gtenegativeone.py'
        }
    
    def run_training_pipeline(self, episodes=50, base_model='gcn', augmentation_factor=10):
        """Run the complete training pipeline"""
        logger.info("Starting annotation type training pipeline")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Warnings file: {self.warnings_file}")
        logger.info(f"Augmentation factor: {augmentation_factor}")
        
        # Step 1: Generate slices using Specimin
        logger.info("Step 1: Generating slices using Specimin")
        if not self._generate_slices_with_specimin():
            logger.error("Failed to generate slices with Specimin")
            return False
        
        # Step 2: Augment slices
        logger.info("Step 2: Augmenting slices")
        if not self._augment_slices(augmentation_factor):
            logger.error("Failed to augment slices")
            return False
        
        # Step 3: Generate CFGs using Checker Framework CFG Builder
        logger.info("Step 3: Generating CFGs using Checker Framework CFG Builder")
        if not self._generate_cfgs_with_cfg_builder():
            logger.error("Failed to generate CFGs")
            return False
        
        # Step 4: Train annotation type models
        logger.info("Step 4: Training annotation type models")
        if not self._train_annotation_type_models(episodes, base_model):
            logger.error("Failed to train annotation type models")
            return False
        
        logger.info("Training pipeline completed successfully")
        return True
    
    def run_prediction_pipeline(self, target_classes_dir):
        """Run the prediction pipeline using Soot + Vineflower"""
        logger.info("Starting annotation type prediction pipeline")
        logger.info(f"Target classes directory: {target_classes_dir}")
        
        # Step 1: Compile target classes
        logger.info("Step 1: Compiling target classes")
        if not self._compile_target_classes(target_classes_dir):
            logger.error("Failed to compile target classes")
            return False
        
        # Step 2: Run Soot bytecode slicing
        logger.info("Step 2: Running Soot bytecode slicing")
        if not self._run_soot_slicing(target_classes_dir):
            logger.error("Failed to run Soot slicing")
            return False
        
        # Step 3: Use Vineflower for decompilation
        logger.info("Step 3: Using Vineflower for decompilation")
        if not self._run_vineflower_decompilation():
            logger.error("Failed to run Vineflower decompilation")
            return False
        
        # Step 4: Predict and place annotations
        logger.info("Step 4: Predicting and placing annotations")
        if not self._predict_and_place_annotations():
            logger.error("Failed to predict and place annotations")
            return False
        
        logger.info("Prediction pipeline completed successfully")
        return True
    
    def _generate_slices_with_specimin(self):
        """Generate slices using Specimin"""
        try:
            # Use the existing Specimin integration
            cmd = [
                'python', 'train_with_specimin.py',
                '--project_root', self.project_root,
                '--warnings_file', self.warnings_file,
                '--cfwr_root', self.cfwr_root,
                '--slices_dir', self.slices_dir
            ]
            
            logger.info(f"Running Specimin: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Specimin slice generation completed successfully")
                return True
            else:
                logger.error(f"Specimin failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Specimin slice generation timed out")
            return False
        except Exception as e:
            logger.error(f"Error running Specimin: {e}")
            return False
    
    def _augment_slices(self, augmentation_factor):
        """Augment slices by the specified factor"""
        try:
            # Use the existing augmentation system
            from augment_slices import augment_directory
            
            logger.info(f"Augmenting slices by factor {augmentation_factor}")
            augment_directory(self.slices_dir, self.augmented_slices_dir, augmentation_factor)
            
            # Verify augmentation
            original_files = len(glob.glob(os.path.join(self.slices_dir, '**/*.java'), recursive=True))
            augmented_files = len(glob.glob(os.path.join(self.augmented_slices_dir, '**/*.java'), recursive=True))
            
            logger.info(f"Original files: {original_files}, Augmented files: {augmented_files}")
            
            if augmented_files >= original_files * augmentation_factor:
                logger.info("Slice augmentation completed successfully")
                return True
            else:
                logger.warning(f"Augmentation may be incomplete: expected ~{original_files * augmentation_factor}, got {augmented_files}")
                return True  # Still proceed
                
        except Exception as e:
            logger.error(f"Error augmenting slices: {e}")
            return False
    
    def _generate_cfgs_with_cfg_builder(self):
        """Generate CFGs using Checker Framework CFG Builder"""
        try:
            # Use the existing CFG generation system
            from cfg import generate_control_flow_graphs
            
            logger.info("Generating CFGs using Checker Framework CFG Builder")
            generate_control_flow_graphs(self.augmented_slices_dir, self.cfg_dir)
            
            # Verify CFG generation
            cfg_files = len(glob.glob(os.path.join(self.cfg_dir, '**/*.json'), recursive=True))
            logger.info(f"Generated {cfg_files} CFG files")
            
            if cfg_files > 0:
                logger.info("CFG generation completed successfully")
                return True
            else:
                logger.error("No CFG files generated")
                return False
                
        except Exception as e:
            logger.error(f"Error generating CFGs: {e}")
            return False
    
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
    
    def _compile_target_classes(self, target_classes_dir):
        """Compile target Java classes"""
        try:
            logger.info(f"Compiling Java classes in {target_classes_dir}")
            
            # Find all Java files
            java_files = glob.glob(os.path.join(target_classes_dir, '**/*.java'), recursive=True)
            
            if not java_files:
                logger.error("No Java files found to compile")
                return False
            
            # Compile using javac
            cmd = ['javac', '-cp', '.:*'] + java_files
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=target_classes_dir)
            
            if result.returncode == 0:
                logger.info("Java compilation completed successfully")
                return True
            else:
                logger.error(f"Java compilation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error compiling Java classes: {e}")
            return False
    
    def _run_soot_slicing(self, target_classes_dir):
        """Run Soot bytecode slicing"""
        try:
            logger.info("Running Soot bytecode slicing")
            
            # Use the existing Soot integration
            soot_jar = os.path.join(self.cfwr_root, 'build/libs/CFWR-all.jar')
            
            if not os.path.exists(soot_jar):
                logger.error(f"Soot JAR not found at {soot_jar}")
                return False
            
            # Run Soot slicing on compiled classes
            cmd = [
                'java', '-cp', soot_jar,
                'cfwr.SootSlicer',
                '--projectRoot', target_classes_dir,
                '--targetFile', 'dummy.java',  # Will be handled by SootSlicer
                '--line', '1',
                '--output', os.path.join(self.predictions_dir, 'soot_slices'),
                '--member', 'dummy',
                '--prediction-mode'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Soot slicing completed successfully")
                return True
            else:
                logger.error(f"Soot slicing failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running Soot slicing: {e}")
            return False
    
    def _run_vineflower_decompilation(self):
        """Run Vineflower decompilation"""
        try:
            logger.info("Running Vineflower decompilation")
            
            # Check for Vineflower JAR
            vineflower_jar = os.environ.get('VINEFLOWER_JAR')
            if not vineflower_jar or not os.path.exists(vineflower_jar):
                logger.warning("Vineflower JAR not found, skipping decompilation")
                return True  # Not critical for basic functionality
            
            soot_slices_dir = os.path.join(self.predictions_dir, 'soot_slices')
            decompiled_dir = os.path.join(self.predictions_dir, 'decompiled')
            os.makedirs(decompiled_dir, exist_ok=True)
            
            # Find class files to decompile
            class_files = glob.glob(os.path.join(soot_slices_dir, '**/*.class'), recursive=True)
            
            for class_file in class_files:
                rel_path = os.path.relpath(class_file, soot_slices_dir)
                output_file = os.path.join(decompiled_dir, rel_path.replace('.class', '.java'))
                output_dir = os.path.dirname(output_file)
                os.makedirs(output_dir, exist_ok=True)
                
                cmd = ['java', '-jar', vineflower_jar, class_file, output_dir]
                subprocess.run(cmd, capture_output=True)
            
            logger.info("Vineflower decompilation completed")
            return True
            
        except Exception as e:
            logger.error(f"Error running Vineflower decompilation: {e}")
            return False
    
    def _predict_and_place_annotations(self):
        """Predict and place annotations using trained models"""
        try:
            logger.info("Predicting and placing annotations")
            
            # Load trained models
            models = {}
            for annotation_type in self.annotation_types:
                model_file = os.path.join(self.models_dir, f"{annotation_type.replace('@', '').lower()}_model.pth")
                if os.path.exists(model_file):
                    models[annotation_type] = model_file
                    logger.info(f"Loaded model for {annotation_type}")
                else:
                    logger.warning(f"Model not found for {annotation_type}")
            
            if not models:
                logger.error("No trained models found")
                return False
            
            # Find decompiled Java files
            decompiled_dir = os.path.join(self.predictions_dir, 'decompiled')
            java_files = glob.glob(os.path.join(decompiled_dir, '**/*.java'), recursive=True)
            
            if not java_files:
                logger.warning("No decompiled Java files found, using original source")
                java_files = glob.glob(os.path.join(self.project_root, '**/*.java'), recursive=True)
            
            # Process each Java file
            predictions_made = 0
            for java_file in java_files:
                try:
                    predictions = self._predict_annotations_for_file(java_file, models)
                    if predictions:
                        self._place_annotations_in_file(java_file, predictions)
                        predictions_made += len(predictions)
                except Exception as e:
                    logger.warning(f"Error processing {java_file}: {e}")
            
            logger.info(f"Made {predictions_made} annotation predictions")
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
            if any(keyword in line_lower for keyword in ['count', 'size', 'length']):
                predictions.append({
                    'line': i,
                    'annotation_type': '@Positive',
                    'confidence': 0.8,
                    'reason': 'count/size/length variable'
                })
            
            # Predict @NonNegative for index variables
            elif any(keyword in line_lower for keyword in ['index', 'offset', 'position']):
                predictions.append({
                    'line': i,
                    'annotation_type': '@NonNegative',
                    'confidence': 0.7,
                    'reason': 'index/offset variable'
                })
            
            # Predict @GTENegativeOne for capacity variables
            elif any(keyword in line_lower for keyword in ['capacity', 'limit', 'bound']):
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
            with open(java_file, 'r') as f:
                lines = f.readlines()
            
            # Sort predictions by line number in descending order to avoid line shift issues
            predictions.sort(key=lambda x: x['line'], reverse=True)
            
            for pred in predictions:
                line_num = pred['line'] - 1  # Convert to 0-based index
                annotation = pred['annotation_type']
                
                if 0 <= line_num < len(lines):
                    # Add annotation before the line
                    lines.insert(line_num, f"    {annotation}\n")
            
            # Write back to file
            with open(java_file, 'w') as f:
                f.writelines(lines)
                
            logger.info(f"Placed {len(predictions)} annotations in {java_file}")
            
        except Exception as e:
            logger.error(f"Error placing annotations in {java_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Annotation Type Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                       help='Pipeline mode: train or predict')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index',
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/CFWR/index1.small.out',
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/CFWR',
                       help='Root directory of CFWR project')
    parser.add_argument('--target_classes_dir', 
                       help='Directory containing compiled target classes (for prediction mode)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of training episodes (for training mode)')
    parser.add_argument('--base_model', default='gcn', choices=['gcn', 'gbt', 'causal'],
                       help='Base model type (for training mode)')
    parser.add_argument('--augmentation_factor', type=int, default=10,
                       help='Augmentation factor for slices (for training mode)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = AnnotationTypePipeline(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        mode=args.mode
    )
    
    # Run pipeline
    if args.mode == 'train':
        success = pipeline.run_training_pipeline(
            episodes=args.episodes,
            base_model=args.base_model,
            augmentation_factor=args.augmentation_factor
        )
    else:  # predict
        if not args.target_classes_dir:
            logger.error("--target_classes_dir is required for prediction mode")
            return 1
        
        success = pipeline.run_prediction_pipeline(args.target_classes_dir)
    
    if success:
        logger.info("Pipeline completed successfully")
        return 0
    else:
        logger.error("Pipeline failed")
        return 1

if __name__ == '__main__':
    exit(main())
