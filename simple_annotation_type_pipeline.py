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
    
    def __init__(self, project_root, warnings_file, cfwr_root, mode='train', no_auto_train=False, device='auto'):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        self.mode = mode
        self.no_auto_train = no_auto_train
        self.device = device
        
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
                '--variants_per_file', '100'
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
        """Predict and place annotations using improved balanced models"""
        try:
            logger.info("Predicting and placing annotations using improved balanced models")
            
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
            
            # Check for improved balanced models first (these are the default)
            models = {}
            balanced_models_found = 0
            
            # Check for balanced models (trained on real code examples)
            for annotation_type in self.annotation_types:
                model_name = annotation_type.replace('@', '').lower()
                balanced_model_file = os.path.join(self.models_dir, f"{model_name}_real_balanced_model.pth")
                
                if os.path.exists(balanced_model_file):
                    models[annotation_type] = {
                        'model_file': balanced_model_file,
                        'base_model_type': 'improved_balanced_causal',
                        'training_type': 'balanced_real_code'
                    }
                    balanced_models_found += 1
                    logger.info(f"Found balanced model for {annotation_type} (trained on real code examples)")
            
            # If balanced models found, use them exclusively
            if balanced_models_found > 0:
                logger.info(f"Using {balanced_models_found} balanced models (trained on real code examples)")
            else:
                # Fallback to legacy models
                logger.warning("No balanced models found, falling back to legacy models")
                base_model_types = ['enhanced_hybrid', 'enhanced_gcn', 'enhanced_gat', 'enhanced_transformer', 'enhanced_causal', 'enhanced_graph_causal', 'graph_causal', 'graphite', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
                
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
                                'base_model_type': base_model_type,
                                'training_type': 'legacy'
                            }
                            logger.info(f"Found legacy model for {annotation_type} ({base_model_type})")
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
                        # Save predictions report (always save predictions)
                        self._save_predictions_report(java_file, predictions)
                        
                        # Only place annotations if not in predict-only mode
                        # For now, we'll always save predictions but not place annotations
                        # This can be controlled by a command line flag if needed
                        
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
        """Predict annotations for a single Java file using balanced models"""
        try:
            # Check if we have balanced models
            balanced_models_found = any(model.get('training_type') == 'balanced_real_code' for model in models.values())
            
            if balanced_models_found:
                logger.info("Using improved balanced models (trained on real code examples)")
                return self._predict_with_balanced_models(java_file, models)
            else:
                logger.info("Using legacy models (fallback)")
                return self._predict_with_legacy_models(java_file, models)
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return []
    
    def _predict_with_balanced_models(self, java_file, models):
        """Predict using improved balanced models"""
        try:
            # Import required modules
            import torch
            from enhanced_graph_predictor import EnhancedGraphPredictor
            
            # Create predictor
            predictor = EnhancedGraphPredictor(models_dir=self.models_dir, device=self.device, auto_train=False)
            
            # Load balanced models
            balanced_models = {}
            for annotation_type, model_info in models.items():
                if model_info.get('training_type') == 'balanced_real_code':
                    try:
                        # Load the balanced model
                        checkpoint = torch.load(model_info['model_file'], map_location=predictor.device)
                        
                        # Create model architecture
                        from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeModel
                        model = ImprovedBalancedAnnotationTypeModel(
                            input_dim=21,  # Balanced model uses 21 features (from training)
                            hidden_dims=[512, 256, 128, 64],
                            dropout_rate=0.4
                        )
                        
                        # Load state dict
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        model = model.to(predictor.device)
                        
                        balanced_models[annotation_type] = model
                        logger.info(f"Loaded balanced model for {annotation_type}")
                        
                    except Exception as e:
                        logger.error(f"Error loading balanced model for {annotation_type}: {e}")
                        continue
            
            if not balanced_models:
                logger.error("No balanced models loaded successfully")
                return []
            
            # Find CFG file
            cfg_file = self._find_cfg_file_for_java(java_file)
            if not cfg_file:
                logger.warning(f"No CFG file found for {java_file}")
                return []
            
            # Load CFG data
            from cfg_graph import load_cfg_as_pyg
            cfg_data = load_cfg_as_pyg(cfg_file)
            if not hasattr(cfg_data, 'batch') or cfg_data.batch is None:
                cfg_data.batch = torch.zeros(cfg_data.x.shape[0], dtype=torch.long)
            
            cfg_data = cfg_data.to(predictor.device)
            
            predictions = []
            
            # Convert graph data to tabular format for balanced models
            # Extract node features and create a single representative vector
            node_features = cfg_data.x.cpu().numpy()
            
            # Create a representative feature vector (mean of all node features)
            if node_features.shape[0] > 0:
                representative_features = torch.tensor(node_features.mean(axis=0), dtype=torch.float32).unsqueeze(0)
                representative_features = representative_features.to(predictor.device)
                
                # Ensure we have the right number of features (pad or truncate to 21)
                if representative_features.shape[1] < 21:
                    # Pad with zeros
                    padding = torch.zeros(representative_features.shape[0], 21 - representative_features.shape[1], device=representative_features.device)
                    representative_features = torch.cat([representative_features, padding], dim=1)
                elif representative_features.shape[1] > 21:
                    # Truncate
                    representative_features = representative_features[:, :21]
            else:
                # Fallback: create zero features
                representative_features = torch.zeros(1, 21, device=predictor.device)
            
            # Predict with each balanced model
            for annotation_type, model in balanced_models.items():
                try:
                    with torch.no_grad():
                        # Get model prediction using tabular features
                        logits = model(representative_features)
                        probabilities = torch.softmax(logits, dim=1)
                        prediction = torch.argmax(logits, dim=1)
                        confidence = probabilities.gather(1, prediction.unsqueeze(1)).squeeze(1)
                        
                        # Only add predictions above threshold
                        if prediction.item() == 1 and confidence.item() > 0.3:
                            prediction_dict = {
                                'line': 1,  # Default line number
                                'annotation_type': annotation_type,
                                'confidence': confidence.item(),
                                'features': representative_features.cpu().numpy().flatten().tolist(),
                                'reason': f"{annotation_type} predicted by balanced model with {confidence.item():.3f} confidence (real code training)",
                                'model_type': 'improved_balanced_causal'
                            }
                            predictions.append(prediction_dict)
                            
                except Exception as e:
                    logger.error(f"Error predicting with balanced {annotation_type} model: {e}")
                    continue
            
            logger.info(f"Generated {len(predictions)} predictions using balanced models")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with balanced models: {e}")
            return []
    
    def _predict_with_legacy_models(self, java_file, models):
        """Predict using legacy models (fallback)"""
        try:
            # Import the model-based predictor
            from enhanced_graph_predictor import EnhancedGraphPredictor as ModelBasedPredictor
            
            # Create predictor with auto-training enabled (unless disabled via command line)
            auto_train = not getattr(self, 'no_auto_train', False)
            predictor = ModelBasedPredictor(models_dir=self.models_dir, device=self.device, auto_train=auto_train)
            
            # Use the specific model types found
            all_predictions = []
            prediction_cfg_dir = os.path.join(self.cfwr_root, 'prediction_cfg_output')

            for annotation_type, model_info in models.items():
                base_model_type = model_info.get('base_model_type', 'enhanced_causal')
                
                if not predictor.load_or_train_models(base_model_type=base_model_type, epochs=10):
                    logger.warning(f"Skipping base model type {base_model_type}: load/train failed")
                    continue

                logger.info(f"✅ Using legacy model with base model type: {base_model_type}")
                preds = predictor.predict_annotations_for_file_with_cfg(java_file, prediction_cfg_dir, threshold=0.3)
                if preds:
                    # Tag predictions with base model type to distinguish outputs
                    for p in preds:
                        p['model_type'] = base_model_type
                        p['training_type'] = 'legacy'
                    all_predictions.extend(preds)
                else:
                    logger.warning(f"No predictions generated by legacy model for {base_model_type}")

            if all_predictions:
                logger.info(f"Generated {len(all_predictions)} predictions using legacy models")
                return all_predictions
            else:
                logger.warning("No predictions generated by any legacy model")
                return []
                
        except Exception as e:
            logger.error(f"Error using legacy models: {e}")
            return []
    
    def _find_cfg_file_for_java(self, java_file):
        """Find CFG file corresponding to a Java file"""
        try:
            java_basename = os.path.splitext(os.path.basename(java_file))[0]
            
            # Try different CFG file locations
            cfg_file_candidates = [
                os.path.join(self.cfg_dir, f"{java_basename}.cfg.json"),
                os.path.join(self.cfg_dir, java_basename, "cfg.json"),
                os.path.join(self.cfg_dir, f"{java_basename}_slice", "cfg.json"),
                os.path.join(self.cfwr_root, 'prediction_cfg_output', f"{java_basename}.cfg.json"),
                os.path.join(self.cfwr_root, 'prediction_cfg_output', java_basename, "cfg.json"),
            ]
            
            for candidate in cfg_file_candidates:
                if os.path.exists(candidate):
                    return candidate
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding CFG file for {java_file}: {e}")
            return None
    
    def _predict_and_place_annotations(self, target_files):
        """Predict and place annotations using trained models"""
        try:
            logger.info("Predicting and placing annotations")
            
            # Load trained models info (check for any base model type)
            models = {}
            base_model_types = ['enhanced_hybrid', 'enhanced_gcn', 'enhanced_gat', 'enhanced_transformer', 'enhanced_causal', 'enhanced_graph_causal', 'graph_causal', 'graphite', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
            
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
            from enhanced_graph_predictor import EnhancedGraphPredictor as ModelBasedPredictor
            
            # Create predictor with auto-training enabled (unless disabled via command line)
            auto_train = not getattr(self, 'no_auto_train', False)
            predictor = ModelBasedPredictor(models_dir=self.models_dir, device=self.device, auto_train=auto_train)
            
            # Try to load or train models with different base model types
            base_model_types = ['enhanced_hybrid', 'enhanced_gcn', 'enhanced_gat', 'enhanced_transformer', 'enhanced_causal', 'enhanced_graph_causal', 'graph_causal', 'graphite', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
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
            self._save_predictions_report(java_file, predictions)
            
        except Exception as e:
            logger.error(f"Error placing annotations in {java_file}: {e}")
    
    def _save_predictions_report(self, java_file, predictions):
        """Save predictions report to JSON file with enhanced metadata"""
        try:
            java_basename = os.path.splitext(os.path.basename(java_file))[0]
            
            # Determine if we're using balanced models
            using_balanced_models = any(p.get('model_type') == 'improved_balanced_causal' for p in predictions)
            
            if using_balanced_models:
                report_file = os.path.join(self.predictions_dir, f"{java_basename}_balanced.predictions.json")
                model_type = "improved_balanced_causal"
                training_info = "real_code_examples"
                balance_info = "50% positive, 50% negative"
            else:
                report_file = os.path.join(self.predictions_dir, f"{java_basename}.predictions.json")
                model_type = "legacy_models"
                training_info = "legacy_training"
                balance_info = "unknown"
            
            # Enhanced metadata
            metadata = {
                'file': java_file,
                'predictions': predictions,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': model_type,
                'training_data': training_info,
                'balance_ratio': balance_info,
                'feature_dimensions': 21 if using_balanced_models else 15,
                'pipeline_version': 'improved_balanced_pipeline' if using_balanced_models else 'legacy_pipeline',
                'total_predictions': len(predictions),
                'annotation_types': list(set(p.get('annotation_type') for p in predictions))
            }
            
            with open(report_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {len(predictions)} predictions to {report_file} (using {model_type})")
        except Exception as e:
            logger.error(f"Error saving predictions report: {e}")
    
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
            base_model_types = ['enhanced_hybrid', 'enhanced_gcn', 'enhanced_gat', 'enhanced_transformer', 'enhanced_causal', 'enhanced_graph_causal', 'graph_causal', 'graphite', 'causal', 'hgt', 'gcn', 'gbt', 'gcsn', 'dg2n']
            
            for base_model_type in base_model_types:
                model_name = annotation_type.replace('@', '').lower()
                # Try the new naming pattern first (without base_model_type suffix)
                model_file = os.path.join(self.models_dir, f"{model_name}_model.pth")
                stats_file = os.path.join(self.models_dir, f"{model_name}_stats.json")
                
                # If not found, try the old pattern with base_model_type suffix
                if not (os.path.exists(model_file) and os.path.exists(stats_file)):
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
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training/inference (default: auto)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SimpleAnnotationTypePipeline(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        mode=args.mode,
        no_auto_train=args.no_auto_train,
        device=args.device
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
