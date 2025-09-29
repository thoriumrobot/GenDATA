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
    
    def __init__(self, project_root, warnings_file, cfwr_root, mode='train', no_auto_train=False):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        self.mode = mode
        self.no_auto_train = no_auto_train
        
        # Set up directories
        self.slices_dir = os.path.join(cfwr_root, 'slices_specimin')
        self.cfg_dir = os.path.join(cfwr_root, 'cfg_output_specimin')
        self.models_dir = os.path.join(cfwr_root, 'models_annotation_types')
        self.predictions_dir = os.path.join(cfwr_root, 'predictions_annotation_types')
        
        # Create directories
        for dir_path in [self.slices_dir, self.cfg_dir, self.models_dir, self.predictions_dir]:
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
        
        # Step 1: Generate slices using Specimin
        logger.info("Step 1: Generating slices using Specimin")
        if not self._generate_slices_with_specimin():
            logger.error("Failed to generate slices with Specimin")
            return False
        
        # Step 2: Generate CFGs from slices
        logger.info("Step 2: Generating CFGs from slices")
        if not self._generate_cfgs_from_slices():
            logger.error("Failed to generate CFGs from slices")
            return False
        
        # Step 3: Train annotation type models using real CFG data
        logger.info("Step 3: Training annotation type models using real CFG data")
        if not self._train_annotation_type_models_with_cfgs(episodes, base_model):
            logger.error("Failed to train annotation type models")
            return False
        
        logger.info("Training pipeline completed successfully")
        return True
    
    def run_prediction_pipeline(self, target_file=None):
        """Run the prediction pipeline"""
        logger.info("Starting simplified annotation type prediction pipeline")
        
        # Step 1: Generate slices for prediction files
        logger.info("Step 1: Generating slices for prediction")
        if not self._generate_slices_for_prediction(target_file):
            logger.error("Failed to generate slices for prediction")
            return False
        
        # Step 2: Generate CFGs from prediction slices
        logger.info("Step 2: Generating CFGs from prediction slices")
        if not self._generate_cfgs_for_prediction():
            logger.error("Failed to generate CFGs for prediction")
            return False
        
        # Step 3: Predict and place annotations using real CFG data
        logger.info("Step 3: Predicting and placing annotations using real CFG data")
        if not self._predict_and_place_annotations_with_cfgs(target_file):
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
    
    def _generate_slices_with_specimin(self):
        """Generate slices using Specimin and augment them"""
        try:
            # Use the existing pipeline slicing functionality
            from pipeline import run_slicing
            
            logger.info("Generating slices using Specimin")
            # Use the correct warnings file path
            warnings_file = os.path.join(self.cfwr_root, 'index1.out')
            run_slicing(self.project_root, warnings_file, self.cfwr_root, 
                       os.path.dirname(self.slices_dir), 'specimin')
            
            # Augment slices using augment_slices.py
            logger.info("Augmenting slices with augment_slices.py")
            augmented_dir = os.path.join(self.cfwr_root, 'slices_augmented')
            os.makedirs(augmented_dir, exist_ok=True)
            
            # Run augmentation
            import subprocess
            augment_cmd = [
                'python', 'augment_slices.py',
                '--slices_dir', self.slices_dir,
                '--out_dir', augmented_dir,
                '--variants_per_file', '10'
            ]
            result = subprocess.run(augment_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Slice augmentation failed: {result.stderr}")
            else:
                logger.info("Slice augmentation completed successfully")
                # Update slices_dir to use augmented slices
                self.slices_dir = augmented_dir
            
            # Verify slices were generated (check both specimin and cf directories)
            specimin_slices = glob.glob(os.path.join(self.slices_dir, '**/*.java'), recursive=True)
            cf_slices_dir = os.path.join(self.cfwr_root, 'slices_cf')
            cf_slices = glob.glob(os.path.join(cf_slices_dir, '**/*.java'), recursive=True) if os.path.exists(cf_slices_dir) else []
            
            total_slices = len(specimin_slices) + len(cf_slices)
            logger.info(f"Generated {total_slices} slice files ({len(specimin_slices)} specimin, {len(cf_slices)} cf)")
            
            if total_slices > 0:
                logger.info("Slice generation completed successfully")
                return True
            else:
                logger.error("No slice files generated")
                return False
                
        except Exception as e:
            logger.error(f"Error generating slices: {e}")
            return False
    
    def _generate_cfgs_from_slices(self):
        """Generate CFGs from slices"""
        try:
            # Use the existing CFG generation functionality
            from pipeline import run_cfg_generation
            
            logger.info("Generating CFGs from slices")
            run_cfg_generation(self.slices_dir, self.cfg_dir)
            
            # Verify CFGs were generated (check both specimin and cf directories)
            specimin_cfgs = glob.glob(os.path.join(self.cfg_dir, '**/*.json'), recursive=True)
            cf_cfgs_dir = os.path.join(self.cfwr_root, 'slices_cf')
            cf_cfgs = glob.glob(os.path.join(cf_cfgs_dir, '**/*.json'), recursive=True) if os.path.exists(cf_cfgs_dir) else []
            
            total_cfgs = len(specimin_cfgs) + len(cf_cfgs)
            logger.info(f"Generated {total_cfgs} CFG files ({len(specimin_cfgs)} specimin, {len(cf_cfgs)} cf)")
            
            if total_cfgs > 0:
                logger.info("CFG generation completed successfully")
                return True
            else:
                logger.error("No CFG files generated")
                return False
                
        except Exception as e:
            logger.error(f"Error generating CFGs: {e}")
            return False
    
    def _train_annotation_type_models_with_cfgs(self, episodes, base_model):
        """Train models for each annotation type using real CFG data"""
        success_count = 0
        
        for annotation_type in self.annotation_types:
            logger.info(f"Training model for {annotation_type} using real CFG data")
            
            script_name = self.script_mapping[annotation_type]
            cmd = [
                'python', script_name,
                '--project_root', self.project_root,
                '--warnings_file', self.warnings_file,
                '--cfwr_root', self.cfwr_root,
                '--slices_dir', self.slices_dir,
                '--cfg_dir', self.cfg_dir,
                '--episodes', str(episodes),
                '--base_model', base_model,
                '--device', 'cpu',
                '--use_real_cfg_data'
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
    
    def _generate_slices_for_prediction(self, target_file):
        """Generate slices for prediction files using Soot and Vineflower"""
        try:
            # For prediction, we use Soot for bytecode slicing and Vineflower for decompilation
            logger.info("Generating slices for prediction using Soot and Vineflower")

            # Create prediction slices directory
            pred_slices_dir = os.path.join(self.cfwr_root, 'prediction_slices')
            os.makedirs(pred_slices_dir, exist_ok=True)

            # Determine target files: if not provided, scan case_studies for .java files
            if target_file:
                target_files = [target_file]
            else:
                case_studies_root = os.path.join(self.cfwr_root, 'case_studies')
                if os.path.isdir(case_studies_root):
                    target_files = glob.glob(os.path.join(case_studies_root, '**/*.java'), recursive=True)
                else:
                    # Fallback to project_root
                    target_files = glob.glob(os.path.join(self.project_root, '**/*.java'), recursive=True)

            if not target_files:
                logger.warning("No target files found for Soot prediction slicing")
                return True

            # Use SootSlicer for bytecode slicing with Vineflower decompilation
            soot_slicer_jar = os.path.join(self.cfwr_root, 'build/libs/GenDATA-all.jar')
            vineflower_jar = os.path.join(self.cfwr_root, 'tools/vineflower.jar')

            if os.path.exists(soot_slicer_jar) and os.path.exists(vineflower_jar):
                successes = 0
                for tf in target_files:
                    cmd = [
                        'java', '-cp', soot_slicer_jar,
                        'cfwr.SootSlicer',
                        '--projectRoot', os.path.dirname(tf),
                        '--targetFile', tf,
                        '--output', pred_slices_dir,
                        '--decompiler', vineflower_jar,
                        '--prediction-mode'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        successes += 1
                    else:
                        logger.debug(f"Soot slicing failed for {tf}: {result.stderr}")

                if successes > 0:
                    logger.info(f"Soot slicing with Vineflower completed for {successes}/{len(target_files)} files")
                    return True
                else:
                    logger.warning("Soot slicing produced no slices; proceeding with existing slices if any")
            else:
                logger.warning("SootSlicer or Vineflower jar not found; proceeding with existing slices if any")

            # Fallback to existing slices if Soot slicing fails
            if os.path.exists(self.slices_dir):
                import shutil
                shutil.copytree(self.slices_dir, pred_slices_dir, dirs_exist_ok=True)
                logger.info("Using existing slices for prediction")
                return True
            else:
                logger.warning("No existing slices found, skipping slice generation for prediction")
                return True

        except Exception as e:
            logger.error(f"Error generating slices for prediction: {e}")
            return False
    
    def _generate_cfgs_for_prediction(self):
        """Generate CFGs for prediction"""
        try:
            # Use existing CFG generation
            pred_slices_dir = os.path.join(self.cfwr_root, 'prediction_slices')
            pred_cfg_dir = os.path.join(self.cfwr_root, 'prediction_cfg_output')
            os.makedirs(pred_cfg_dir, exist_ok=True)
            
            if os.path.exists(pred_slices_dir):
                from pipeline import run_cfg_generation
                run_cfg_generation(pred_slices_dir, pred_cfg_dir)
                logger.info("Generated CFGs for prediction")
                return True
            else:
                logger.warning("No prediction slices found, using existing CFGs")
                return True
                
        except Exception as e:
            logger.error(f"Error generating CFGs for prediction: {e}")
            return False
    
    def _predict_and_place_annotations_with_cfgs(self, target_file):
        """Predict and place annotations using real CFG data"""
        try:
            logger.info("Predicting and placing annotations using real CFG data")
            
            # Use target file or, by default, Java files produced by Soot/Vineflower slices
            if target_file:
                target_files = [target_file]
            else:
                # Prefer prediction_slices generated by SootSlicer
                pred_slices_dir = os.path.join(self.cfwr_root, 'prediction_slices')
                target_files = []
                if os.path.isdir(pred_slices_dir):
                    for root, _, files in os.walk(pred_slices_dir):
                        for f in files:
                            if f.endswith('.java'):
                                target_files.append(os.path.join(root, f))
                
                # Fallback to project root if no sliced files were found
                if not target_files:
                    target_files = glob.glob(os.path.join(self.project_root, '**/*.java'), recursive=True)
            
            if not target_files:
                logger.error("No target files found for prediction")
                return False
            
            logger.info(f"Found {len(target_files)} target files")
            
            # Load trained models info (check for any base model type)
            models = {}
            base_model_types = ['enhanced_causal', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
            
            for annotation_type in self.annotation_types:
                model_name = annotation_type.replace('@', '').lower()
                found_model = False
                
                for base_model_type in base_model_types:
                    model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
                    stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
                    
                    if os.path.exists(model_file) and os.path.exists(stats_file):
                        models[annotation_type] = {
                            'model_file': model_file,
                            'stats_file': stats_file,
                            'base_model_type': base_model_type
                        }
                        logger.info(f"Found trained model for {annotation_type} ({base_model_type})")
                        found_model = True
                        break
                
                if not found_model:
                    logger.info(f"No trained model found for {annotation_type}")
            
            if not models and self.no_auto_train:
                logger.error("No trained models found and auto-training is disabled")
                return False
            elif not models:
                logger.info("No trained models found, but auto-training is enabled - will train models as needed")
            
            # Process each Java file using real CFG data
            total_predictions = 0
            processed_files = 0
            
            for java_file in target_files:
                try:
                    # Pass models dict (empty if auto-training) to the prediction method
                    predictions = self._predict_annotations_for_file_with_cfg(java_file, models if models else {})
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
    
    def _predict_annotations_for_file_with_cfg(self, java_file, models):
        """Predict annotations for a single Java file using real CFG data"""
        try:
            # Import the model-based predictor
            from model_based_predictor import ModelBasedPredictor
            
            # Create predictor with auto-training enabled (unless disabled via command line)
            auto_train = not getattr(self, 'no_auto_train', False)
            predictor = ModelBasedPredictor(models_dir=self.models_dir, auto_train=auto_train)
            
            # Iterate all base model types to produce predictions for all 21 combinations
            base_model_types = ['enhanced_causal', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
            all_predictions = []

            prediction_cfg_dir = os.path.join(self.cfwr_root, 'prediction_cfg_output')

            for base_model_type in base_model_types:
                if not predictor.load_or_train_models(base_model_type=base_model_type, episodes=10, project_root='/home/ubuntu/checker-framework/checker/tests/index'):
                    logger.warning(f"Skipping base model type {base_model_type}: load/train failed")
                    continue

                logger.info(f"✅ Using trained models with base model type: {base_model_type}")
                preds = predictor.predict_annotations_for_file_with_cfg(java_file, prediction_cfg_dir, threshold=0.3)
                if preds:
                    # Tag predictions with base model type to distinguish outputs
                    for p in preds:
                        p['model_type'] = base_model_type
                    all_predictions.extend(preds)
                else:
                    logger.warning(f"No predictions generated by trained models for {base_model_type}")

            if all_predictions:
                logger.info(f"Generated {len(all_predictions)} predictions across base models using real CFG data")
                return all_predictions
            else:
                logger.warning("No predictions generated by any base model")
                return []
                
        except Exception as e:
            logger.error(f"Error using trained models: {e}")
            raise Exception(f"Model-based prediction failed: {e}")
    
    def _predict_and_place_annotations(self, target_files):
        """Predict and place annotations using trained models"""
        try:
            logger.info("Predicting and placing annotations")
            
            # Load trained models info (check for any base model type)
            models = {}
            base_model_types = ['enhanced_causal', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
            
            for annotation_type in self.annotation_types:
                model_name = annotation_type.replace('@', '').lower()
                found_model = False
                
                for base_model_type in base_model_types:
                    model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
                    stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
                    
                    if os.path.exists(model_file) and os.path.exists(stats_file):
                        models[annotation_type] = {
                            'model_file': model_file,
                            'stats_file': stats_file,
                            'base_model_type': base_model_type
                        }
                        logger.info(f"Found trained model for {annotation_type} ({base_model_type})")
                        found_model = True
                        break
                
                if not found_model:
                    logger.info(f"No trained model found for {annotation_type}")
            
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
        """Predict annotations for a single Java file using trained models"""
        try:
            # Import the model-based predictor
            from model_based_predictor import ModelBasedPredictor
            
            # Create predictor with auto-training enabled (unless disabled via command line)
            auto_train = not getattr(self, 'no_auto_train', False)
            predictor = ModelBasedPredictor(models_dir=self.models_dir, auto_train=auto_train)
            
            # Try to load or train models with different base model types
            base_model_types = ['enhanced_causal', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
            models_loaded = False
            
            for base_model_type in base_model_types:
                if predictor.load_or_train_models(base_model_type=base_model_type, episodes=10, project_root='/home/ubuntu/checker-framework/checker/tests/index'):
                    logger.info(f"✅ Using trained models with base model type: {base_model_type}")
                    models_loaded = True
                    break
            
            if not models_loaded:
                logger.error("❌ Failed to load or train any models - this should not happen with auto-training enabled")
                raise Exception("No models available and auto-training failed")
            
            # Use trained models for prediction
            predictions = predictor.predict_annotations_for_file(java_file, threshold=0.3)
            
            if predictions:
                logger.info(f"Generated {len(predictions)} predictions using trained models")
                return predictions
            else:
                logger.warning("No predictions generated by trained models")
                return []
                
        except Exception as e:
            logger.error(f"Error using trained models: {e}")
            raise Exception(f"Model-based prediction failed: {e}")
    
    # Note: Heuristic fallback removed - the pipeline now auto-trains missing models
    # to ensure evaluation focuses purely on model performance, not heuristics
    
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
        
        # Check training results - only fail if we're in training mode
        for annotation_type in self.annotation_types:
            # Check for models with any base model type
            model_found = False
            base_model_types = ['enhanced_causal', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
            
            for base_model_type in base_model_types:
                model_name = annotation_type.replace('@', '').lower()
                model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
                stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
                
                if os.path.exists(model_file) and os.path.exists(stats_file):
                    try:
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        report['models_trained'].append({
                            'annotation_type': annotation_type,
                            'base_model_type': base_model_type,
                            'model_file': model_file,
                            'episodes': len(stats.get('episodes', [])),
                            'final_reward': stats.get('rewards', [0])[-1] if stats.get('rewards') else 0,
                            'success': True
                        })
                        model_found = True
                        break
                    except Exception as e:
                        logger.warning(f"Error reading stats for {annotation_type} ({base_model_type}): {e}")
            
            if not model_found:
                # Only fail overall success if we're in training mode
                if self.mode in ['train', 'both']:
                    report['models_trained'].append({
                        'annotation_type': annotation_type,
                        'success': False,
                        'error': 'Model files not found'
                    })
                    report['overall_success'] = False
                else:
                    # In predict-only mode, just note that no models were found
                    report['models_trained'].append({
                        'annotation_type': annotation_type,
                        'success': False,
                        'error': 'No trained models found (predict mode)',
                        'note': 'Models may be auto-trained during prediction'
                    })
        
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
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='predict',
                       help='Pipeline mode: train, predict, or both (default: predict)')
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
    parser.add_argument('--base_model', default='enhanced_causal', choices=['gcn', 'gbt', 'causal', 'enhanced_causal'],
                       help='Base model type (for training mode, default: enhanced_causal)')
    parser.add_argument('--no_auto_train', action='store_true',
                       help='Disable automatic training of missing models (default: auto-train enabled)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SimpleAnnotationTypePipeline(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        mode=args.mode,
        no_auto_train=args.no_auto_train
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
