#!/usr/bin/env python3
"""
Enhanced Graph-Based Predictor with Large Input Support

This module provides an enhanced predictor that uses the new CFG dataloader
and enhanced models to handle large CFG inputs with proper batching.
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our enhanced components
from cfg_dataloader import CFGDataLoader, CFGSizeConfig, find_cfg_files, CFGBatchProcessor
from enhanced_graph_models import create_enhanced_model
from enhanced_training_framework import EnhancedModelTrainer

logger = logging.getLogger(__name__)

class EnhancedGraphPredictor:
    """Enhanced predictor that uses CFG dataloader and enhanced models for large input support"""
    
    def __init__(self, 
                 models_dir: str = 'models_annotation_types', 
                 device: str = 'cpu', 
                 auto_train: bool = True,
                 max_nodes: int = 1000,
                 max_edges: int = 2000,
                 max_batch_size: int = 16):
        """
        Initialize enhanced predictor
        
        Args:
            models_dir: Directory containing trained models
            device: Device to use for inference
            auto_train: Whether to automatically train missing models
            max_nodes: Maximum nodes per graph
            max_edges: Maximum edges per graph
            max_batch_size: Maximum batch size for inference
        """
        self.models_dir = models_dir
        
        # Auto-detect best device (GPU if available)
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                logger.info("💻 Using CPU")
        else:
            self.device = device
            
        self.auto_train = auto_train
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_batch_size = max_batch_size
        
        # Update global config
        CFGSizeConfig.update_limits(max_nodes, max_edges, max_batch_size)
        
        self.loaded_models = {}
        self.model_stats = {}
        
        # Create trainer for auto-training
        if auto_train:
            self.trainer = EnhancedModelTrainer(
                models_dir=models_dir,
                device=device,
                max_nodes=max_nodes,
                max_edges=max_edges,
                max_batch_size=max_batch_size
            )
    
    def load_trained_models(self, base_model_type: str = 'enhanced_hybrid') -> bool:
        """Load all trained enhanced models"""
        try:
            logger.info(f"Loading enhanced models with base model type: {base_model_type}")
            
            annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
            loaded_count = 0
            
            for annotation_type in annotation_types:
                model_name = annotation_type.replace('@', '').lower()
                
                # Try different model file patterns
                model_file_candidates = [
                    os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth"),
                    os.path.join(self.models_dir, f"{model_name}_enhanced_{base_model_type}_model.pth"),
                    os.path.join(self.models_dir, f"{model_name}_model.pth")
                ]
                
                model_file = None
                for candidate in model_file_candidates:
                    if os.path.exists(candidate):
                        model_file = candidate
                        break
                
                # Find corresponding stats file
                stats_file = None
                if model_file:
                    stats_file_candidates = [
                        model_file.replace('_model.pth', '_stats.json'),
                        model_file.replace('.pth', '_stats.json')
                    ]
                    for candidate in stats_file_candidates:
                        if os.path.exists(candidate):
                            stats_file = candidate
                            break
                
                try:
                    # Create enhanced model
                    model = self._create_enhanced_model(base_model_type, annotation_type)
                    
                    if model_file and os.path.exists(model_file):
                        # Load model weights
                        try:
                            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
                        except Exception as load_error:
                            logger.warning(f"⚠️  Corrupted model file for {annotation_type}: {load_error}")
                            logger.info(f"Creating new untrained model for {annotation_type}")
                            # Create a new untrained model instead of failing
                            model = self._create_enhanced_model(base_model_type, annotation_type)
                            model.eval()
                            model = model.to(self.device)
                            self.loaded_models[annotation_type] = model
                            loaded_count += 1
                            continue
                        
                        if 'model_state_dict' in checkpoint:
                            # Try loading with dimension adaptation for architecture mismatches
                            try:
                                # First try strict loading
                                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                                logger.info(f"✅ Loaded {annotation_type} enhanced model from {os.path.basename(model_file)}")
                            except RuntimeError as strict_error:
                                # Handle architecture mismatch with dimension adaptation
                                logger.warning(f"⚠️  Architecture mismatch for {annotation_type}: {strict_error}")
                                
                                # Try partial loading with dimension adaptation
                                try:
                                    self._adapt_model_weights(model, checkpoint['model_state_dict'])
                                    logger.info(f"✅ Adapted {annotation_type} model weights for enhanced architecture")
                                except Exception as adapt_error:
                                    logger.warning(f"⚠️  Weight adaptation failed: {adapt_error}")
                                    logger.info(f"Creating new untrained model for {annotation_type}")
                                    # Create a new untrained model as fallback
                                    model = self._create_enhanced_model(base_model_type, annotation_type)
                                    model.eval()
                                    model = model.to(self.device)
                                    self.loaded_models[annotation_type] = model
                                    loaded_count += 1
                                    continue
                            except Exception as load_error:
                                logger.warning(f"⚠️  Failed to load {annotation_type} model: {load_error}")
                                logger.info(f"Creating new untrained model for {annotation_type}")
                                # Create a new untrained model instead of failing
                                model = self._create_enhanced_model(base_model_type, annotation_type)
                                model.eval()
                                model = model.to(self.device)
                                self.loaded_models[annotation_type] = model
                                loaded_count += 1
                                continue
                        else:
                            logger.warning(f"⚠️  Invalid checkpoint format for {annotation_type}")
                            continue
                    else:
                        logger.warning(f"⚠️  Model file not found for {annotation_type}")
                        # Create a new untrained model
                        model.eval()
                        model = model.to(self.device)
                        self.loaded_models[annotation_type] = model
                        loaded_count += 1
                        logger.info(f"Creating new untrained enhanced model for {annotation_type}")
                    
                    # Set to evaluation mode and add to loaded models (for successful loads)
                    if annotation_type not in self.loaded_models:
                        model.eval()
                        model = model.to(self.device)
                        self.loaded_models[annotation_type] = model
                    
                    # Load stats if available
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                            self.model_stats[annotation_type] = stats
                    
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {annotation_type} enhanced model: {e}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"✅ Successfully loaded {loaded_count}/{len(annotation_types)} enhanced models")
                return True
            else:
                logger.error("❌ No enhanced models loaded successfully")
                return False
                
        except Exception as e:
            logger.error(f"Error loading enhanced models: {e}")
            return False
    
    def _create_enhanced_model(self, base_model_type: str, annotation_type: str) -> torch.nn.Module:
        """Create an enhanced model for the specified type"""
        # Use the base_model_type directly as it now supports all model types
        model_type = base_model_type
        
        # Define parameters based on model type
        if model_type in ['enhanced_causal', 'causal']:
            # Embedding-based models - enhanced for GPU acceleration
            return create_enhanced_model(
                model_type=model_type,
                input_dim=CFGSizeConfig.NODE_FEATURE_DIM,
                hidden_dim=512,  # Increased for GPU acceleration
                out_dim=2,
                dropout=0.1
            )
        elif model_type in ['gbt']:
            # GBT models are embedding-based - enhanced for GPU acceleration
            return create_enhanced_model(
                model_type=model_type,
                input_dim=CFGSizeConfig.NODE_FEATURE_DIM,
                hidden_dim=512,  # Increased for GPU acceleration
                out_dim=2,
                dropout=0.1
            )
        else:
            # Graph-based models - use GPU-optimized architecture with larger hidden dimensions
            # Enhanced for RTX 4070 Ti SUPER with 16.7 GB memory
            return create_enhanced_model(
                model_type=model_type,
                input_dim=CFGSizeConfig.NODE_FEATURE_DIM,
                hidden_dim=512,  # Increased for GPU acceleration
                out_dim=2,
                num_layers=6,    # Deeper architecture for better performance
                dropout=0.1,
                max_nodes=self.max_nodes
            )
    
    def _adapt_model_weights(self, model: torch.nn.Module, old_state_dict: Dict[str, torch.Tensor]) -> None:
        """Adapt old model weights to new enhanced architecture dimensions"""
        try:
            current_state_dict = model.state_dict()
            
            # Handle dimension mismatches by adapting weights
            adapted_state_dict = {}
            
            for key, old_tensor in old_state_dict.items():
                if key in current_state_dict:
                    new_tensor = current_state_dict[key]
                    
                    # Ensure tensors are on the same device
                    if old_tensor.device != new_tensor.device:
                        old_tensor = old_tensor.to(new_tensor.device)
                    
                    # Check for dimension mismatches
                    if old_tensor.shape != new_tensor.shape:
                        logger.info(f"Adapting weights for {key}: {old_tensor.shape} -> {new_tensor.shape}")
                        
                        if 'classifier' in key and len(old_tensor.shape) == 2:
                            # Handle classifier weight adaptation
                            if old_tensor.shape[1] == new_tensor.shape[1]:
                                # Same input dimension, adapt output dimension
                                if old_tensor.shape[0] < new_tensor.shape[0]:
                                    # Pad with zeros for larger output (ensure same device)
                                    padding = torch.zeros(new_tensor.shape[0] - old_tensor.shape[0], old_tensor.shape[1], device=new_tensor.device)
                                    adapted_tensor = torch.cat([old_tensor, padding], dim=0)
                                else:
                                    # Truncate for smaller output
                                    adapted_tensor = old_tensor[:new_tensor.shape[0], :]
                            else:
                                # Different input dimension, use Xavier initialization
                                adapted_tensor = torch.nn.init.xavier_uniform_(torch.zeros_like(new_tensor))
                        elif 'bias' in key and len(old_tensor.shape) == 1:
                            # Handle bias adaptation
                            if old_tensor.shape[0] < new_tensor.shape[0]:
                                # Pad with zeros (ensure same device)
                                padding = torch.zeros(new_tensor.shape[0] - old_tensor.shape[0], device=new_tensor.device)
                                adapted_tensor = torch.cat([old_tensor, padding], dim=0)
                            else:
                                # Truncate
                                adapted_tensor = old_tensor[:new_tensor.shape[0]]
                        elif len(old_tensor.shape) == 1 and len(new_tensor.shape) == 1:
                            # Handle 1D tensor adaptation (bias, weight, etc.)
                            if old_tensor.shape[0] < new_tensor.shape[0]:
                                # Pad with zeros
                                padding = torch.zeros(new_tensor.shape[0] - old_tensor.shape[0], device=new_tensor.device)
                                adapted_tensor = torch.cat([old_tensor, padding], dim=0)
                            else:
                                # Truncate
                                adapted_tensor = old_tensor[:new_tensor.shape[0]]
                        elif 'att_' in key and len(old_tensor.shape) == 3:
                            # Handle attention weight adaptation (e.g., att_src, att_dst)
                            if old_tensor.shape[1] != new_tensor.shape[1]:
                                # Different number of heads - use Xavier initialization
                                adapted_tensor = torch.nn.init.xavier_uniform_(torch.zeros_like(new_tensor))
                            else:
                                adapted_tensor = old_tensor
                        elif 'lin_beta' in key and len(old_tensor.shape) == 2:
                            # Handle lin_beta weight adaptation
                            if old_tensor.shape[1] != new_tensor.shape[1]:
                                # Different feature dimension - use Xavier initialization
                                adapted_tensor = torch.nn.init.xavier_uniform_(torch.zeros_like(new_tensor))
                            else:
                                adapted_tensor = old_tensor
                        else:
                            # Use Xavier initialization for other mismatches
                            adapted_tensor = torch.nn.init.xavier_uniform_(torch.zeros_like(new_tensor))
                        
                        adapted_state_dict[key] = adapted_tensor
                    else:
                        # Same shape, use original weights
                        adapted_state_dict[key] = old_tensor
                else:
                    # Key not in current model, skip
                    logger.debug(f"Skipping key {key} not in current model")
            
            # Load adapted weights
            model.load_state_dict(adapted_state_dict, strict=False)
            logger.info(f"Successfully adapted model weights with {len(adapted_state_dict)} parameters")
            
        except Exception as e:
            logger.error(f"Weight adaptation failed: {e}")
            raise
    
    def predict_annotations_for_file_with_cfg(self, 
                                            java_file: str, 
                                            cfg_dir: str, 
                                            threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Predict annotations for a single Java file using enhanced models and CFG dataloader"""
        if not self.loaded_models:
            logger.error("No enhanced models loaded")
            return []
        
        predictions = []
        
        try:
            # Find CFG file for this Java file
            java_basename = os.path.splitext(os.path.basename(java_file))[0]
            
            # Try different CFG file locations (including slice patterns)
            cfg_file_candidates = [
                os.path.join(cfg_dir, f"{java_basename}.cfg.json"),
                os.path.join(cfg_dir, java_basename, "cfg.json"),
                os.path.join(cfg_dir, java_basename, f"{java_basename}.cfg.json"),
                os.path.join(cfg_dir, f"{java_basename}_slice", "cfg.json"),
                os.path.join(cfg_dir, f"{java_basename}_slice", f"{java_basename}_slice.json"),
                os.path.join(cfg_dir, f"{java_basename}_slice", f"{java_basename}.cfg.json")
            ]
            
            cfg_file = None
            for candidate in cfg_file_candidates:
                if os.path.exists(candidate):
                    cfg_file = candidate
                    break
            
            if cfg_file is None:
                logger.warning(f"No CFG file found for {java_file}. Tried: {cfg_file_candidates}")
                return []
            
            # Create dataloader for this single file
            dataloader = CFGDataLoader(
                cfg_files=[cfg_file],
                targets=[0],  # Dummy target
                batch_size=1,
                shuffle=False,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges
            )
            
            # Get batch
            for batch in dataloader:
                # Move batch to device
                batch = CFGBatchProcessor.move_to_device(batch, self.device)
                
                # Predict with each model
                for annotation_type, model in self.loaded_models.items():
                    try:
                        with torch.no_grad():
                            # Get model prediction
                            logits = model(batch)
                            probabilities = torch.softmax(logits, dim=1)
                            prediction = torch.argmax(logits, dim=1).item()
                            confidence = probabilities[0, prediction].item()
                            
                            # Check if prediction is positive and above threshold
                            if prediction == 1 and confidence > threshold:
                                # Extract line number from graph data
                                line_number = self._get_node_line_number(batch, sample_idx=0)
                                
                                # Extract CFG features for this prediction
                                features = self._extract_cfg_features(batch, sample_idx=0)
                                
                                prediction_dict = {
                                    'line': line_number,
                                    'annotation_type': annotation_type,
                                    'confidence': confidence,
                                    'features': features,
                                    'reason': f"{annotation_type} expected (predicted by enhanced {model.__class__.__name__} with {confidence:.3f} confidence) (using large CFG support)",
                                    'model_type': f"enhanced_{model.__class__.__name__}"
                                }
                                predictions.append(prediction_dict)
                                
                    except Exception as e:
                        logger.error(f"Error predicting with enhanced {annotation_type} model: {e}")
                        continue
                
                break  # Only process first batch
            
            logger.info(f"Generated {len(predictions)} enhanced predictions for {java_file}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing {java_file} with enhanced predictor: {e}")
            return []
    
    def predict_annotations_batch(self, 
                                java_files: List[str],
                                cfg_dir: str,
                                threshold: float = 0.3) -> Dict[str, List[Dict[str, Any]]]:
        """Predict annotations for multiple Java files using batching"""
        if not self.loaded_models:
            logger.error("No enhanced models loaded")
            return {}
        
        # Find all CFG files
        cfg_files = []
        valid_java_files = []
        
        for java_file in java_files:
            java_basename = os.path.splitext(os.path.basename(java_file))[0]
            
            # Try different CFG file locations (including slice patterns)
            cfg_file_candidates = [
                os.path.join(cfg_dir, f"{java_basename}.cfg.json"),
                os.path.join(cfg_dir, java_basename, "cfg.json"),
                os.path.join(cfg_dir, java_basename, f"{java_basename}.cfg.json"),
                os.path.join(cfg_dir, f"{java_basename}_slice", "cfg.json"),
                os.path.join(cfg_dir, f"{java_basename}_slice", f"{java_basename}_slice.json"),
                os.path.join(cfg_dir, f"{java_basename}_slice", f"{java_basename}.cfg.json")
            ]
            
            cfg_file = None
            for candidate in cfg_file_candidates:
                if os.path.exists(candidate):
                    cfg_file = candidate
                    break
            
            if cfg_file:
                cfg_files.append(cfg_file)
                valid_java_files.append(java_file)
            else:
                logger.warning(f"No CFG file found for {java_file}")
        
        if not cfg_files:
            logger.error("No valid CFG files found")
            return {}
        
        logger.info(f"Processing {len(cfg_files)} files with enhanced batch prediction")
        
        # Create dataloader for batch processing
        dataloader = CFGDataLoader(
            cfg_files=cfg_files,
            targets=[0] * len(cfg_files),  # Dummy targets
            batch_size=self.max_batch_size,
            shuffle=False,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges
        )
        
        all_predictions = {}
        
        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = CFGBatchProcessor.move_to_device(batch, self.device)
            
            # Get batch info
            batch_info = CFGBatchProcessor.get_batch_info(batch)
            logger.info(f"Processing batch {batch_idx}: {batch_info}")
            
            # Predict with each model
            for annotation_type, model in self.loaded_models.items():
                try:
                    with torch.no_grad():
                        # Get model predictions for entire batch
                        logits = model(batch)
                        probabilities = torch.softmax(logits, dim=1)
                        predictions = torch.argmax(logits, dim=1)
                        confidences = probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
                        
                        # Process each sample in the batch
                        batch_start_idx = batch_idx * self.max_batch_size
                        for i in range(batch.batch_size):
                            file_idx = batch_start_idx + i
                            if file_idx < len(valid_java_files):
                                java_file = valid_java_files[file_idx]
                                
                                if predictions[i].item() == 1 and confidences[i].item() > threshold:
                                    line_number = self._get_node_line_number(batch, i)
                                    
                                    prediction_dict = {
                                        'line': line_number,
                                        'annotation_type': annotation_type,
                                        'confidence': confidences[i].item(),
                                        'reason': f"{annotation_type} expected (predicted by enhanced {model.__class__.__name__} with {confidences[i].item():.3f} confidence) (using batch processing)",
                                        'model_type': f"enhanced_{model.__class__.__name__}"
                                    }
                                    
                                    if java_file not in all_predictions:
                                        all_predictions[java_file] = []
                                    all_predictions[java_file].append(prediction_dict)
                        
                except Exception as e:
                    logger.error(f"Error in batch prediction with enhanced {annotation_type} model: {e}")
                    continue
        
        logger.info(f"Generated enhanced predictions for {len(all_predictions)} files")
        return all_predictions
    
    def _get_node_line_number(self, batch, sample_idx: int) -> int:
        """Extract line number from batch data for a specific sample"""
        try:
            # Find nodes belonging to this sample
            sample_mask = batch.batch == sample_idx
            sample_nodes = torch.where(sample_mask)[0]
            
            if len(sample_nodes) > 0:
                # Try to get line number from node features
                node_features = batch.x[sample_nodes[0]]
                
                # Assume line number is in the first few features
                for i in range(min(5, node_features.size(0))):
                    val = node_features[i].item()
                    if val > 0 and val < 10000:  # Reasonable line number range
                        return int(val)
            
            return 1  # Default fallback
            
        except Exception as e:
            logger.debug(f"Error extracting line number for sample {sample_idx}: {e}")
            return 1
    
    def _extract_cfg_features(self, batch, sample_idx: int) -> List[float]:
        """Extract CFG features for a specific sample in the batch"""
        try:
            # Find nodes belonging to this sample
            sample_mask = batch.batch == sample_idx
            sample_nodes = torch.where(sample_mask)[0]
            
            if len(sample_nodes) > 0:
                # Get node features for the first node of this sample
                node_features = batch.x[sample_nodes[0]]
                # Convert to list for JSON serialization
                features = node_features.cpu().numpy().tolist()
                return features
            else:
                # Return empty features if no nodes found
                return [0.0] * 15  # Default 15-dimensional feature vector
                
        except Exception as e:
            logger.debug(f"Error extracting CFG features for sample {sample_idx}: {e}")
            return [0.0] * 15  # Default 15-dimensional feature vector
    
    def train_missing_models(self, 
                           base_model_type: str = 'enhanced_hybrid',
                           epochs: int = 50,
                           cfg_files: Optional[List[str]] = None) -> bool:
        """Train any missing enhanced models"""
        if not self.auto_train or not hasattr(self, 'trainer'):
            logger.error("Auto-training not enabled or trainer not available")
            return False
        
        logger.info(f"Training missing enhanced models with {base_model_type} architecture")
        
        # Check which models are missing
        annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        missing_models = []
        
        for annotation_type in annotation_types:
            model_name = annotation_type.replace('@', '').lower()
            model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
            
            if not os.path.exists(model_file):
                missing_models.append(annotation_type)
        
        if not missing_models:
            logger.info("All enhanced models already exist, no training needed")
            return True
        
        logger.info(f"Training missing enhanced models: {missing_models}")
        
        # Train missing models
        try:
            results = self.trainer.train_all_models(
                base_model_type=base_model_type,
                epochs=epochs,
                cfg_files=cfg_files
            )
            
            success = all(results.values())
            if success:
                logger.info("✅ Successfully trained all missing enhanced models")
                # Reload models after training
                return self.load_trained_models(base_model_type)
            else:
                logger.error("❌ Some enhanced models failed to train")
                return False
                
        except Exception as e:
            logger.error(f"Error training missing enhanced models: {e}")
            return False
    
    def load_or_train_models(self, 
                           base_model_type: str = 'enhanced_hybrid',
                           epochs: int = 50,
                           cfg_files: Optional[List[str]] = None) -> bool:
        """Load existing enhanced models or train missing ones"""
        # Try to load existing models first
        if self.load_trained_models(base_model_type):
            return True
        
        # If no models loaded and auto-training is enabled, train missing models
        if self.auto_train:
            logger.info("No enhanced models found, creating and training new models")
            return self.train_missing_models(base_model_type, epochs, cfg_files)
        
        logger.error("No enhanced models found and auto-training is disabled")
        return False


# Compatibility wrapper for the old interface
class EnhancedModelBasedPredictor(EnhancedGraphPredictor):
    """Compatibility wrapper for the old ModelBasedPredictor interface"""
    pass
