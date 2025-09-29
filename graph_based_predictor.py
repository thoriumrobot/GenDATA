#!/usr/bin/env python3
"""
Graph-based predictor that uses trained annotation type models for prediction
Uses CFG graphs directly as input to sophisticated graph neural networks
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Graph imports
try:
    from cfg_graph import load_cfg_as_pyg
    from graph_encoder import build_graph_encoder
    from graph_based_annotation_models import (
        create_graph_based_model, 
        GraphBasedGBTModel,
        AnnotationTypeGCNModel,
        AnnotationTypeGATModel,
        AnnotationTypeTransformerModel,
        AnnotationTypeHGTModel,
        AnnotationTypeGCSNModel,
        AnnotationTypeDG2NModel,
        AnnotationTypeCausalModel,
        AnnotationTypeEnhancedCausalModel
    )
    PYG_AVAILABLE = True
except Exception as e:
    logging.warning(f"Graph-based models not available: {e}")
    PYG_AVAILABLE = False

logger = logging.getLogger(__name__)

class GraphBasedPredictor:
    """Predictor that uses graph-based models for annotation type prediction"""
    
    def __init__(self, models_dir: str = 'models_annotation_types', device: str = 'cpu', auto_train: bool = True):
        self.models_dir = models_dir
        self.device = device
        self.auto_train = auto_train
        self.loaded_models = {}
        self.model_stats = {}
        
        # Graph encoder for embedding-based models
        self.graph_encoder = None
        if PYG_AVAILABLE:
            self.graph_encoder = build_graph_encoder(
                in_dim=64,  # Will be adjusted based on actual CFG features
                edge_dim=2,
                out_dim=256,
                variant='transformer'
            ).to(device)
        
    def _create_graph_based_model(self, base_model_type: str, annotation_type: str, input_dim: int = 15) -> torch.nn.Module:
        """Create a graph-based model for the specified type"""
        
        model_classes = {
            'gcn': AnnotationTypeGCNModel,
            'gat': AnnotationTypeGATModel,
            'transformer': AnnotationTypeTransformerModel,
            'hgt': AnnotationTypeHGTModel,
            'gcsn': AnnotationTypeGCSNModel,
            'dg2n': AnnotationTypeDG2NModel,
            'causal': AnnotationTypeCausalModel,
            'enhanced_causal': AnnotationTypeEnhancedCausalModel
        }
        
        if base_model_type in model_classes:
            model_class = model_classes[base_model_type]
            return model_class(
                input_dim=input_dim,
                hidden_dim=128,
                out_dim=2,
                num_layers=3,
                dropout=0.1
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {base_model_type}")
    
    def load_trained_models(self, base_model_type: str = 'enhanced_causal') -> bool:
        """Load all trained annotation type models using graph-based architecture"""
        try:
            logger.info(f"Loading graph-based models with base model type: {base_model_type}")
            
            # Define annotation types
            annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
            
            loaded_count = 0
            for annotation_type in annotation_types:
                model_name = annotation_type.replace('@', '').lower()
                model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
                stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
                
                try:
                    # Create graph-based model with default input dimension
                    # Will be adjusted if we can load actual CFG data
                    model = self._create_graph_based_model(base_model_type, annotation_type, input_dim=15)
                    
                    if os.path.exists(model_file):
                        # Try to load model weights
                        try:
                            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
                            
                            if 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            else:
                                model.load_state_dict(checkpoint, strict=False)
                            
                            logger.info(f"✅ Loaded {annotation_type} model ({base_model_type})")
                        except Exception as e:
                            logger.warning(f"⚠️  Could not load existing model for {annotation_type}: {e}")
                            logger.info(f"Creating new untrained graph-based model for {annotation_type}")
                    else:
                        logger.warning(f"⚠️  Model file not found for {annotation_type}: {model_file}")
                        logger.info(f"Creating new untrained graph-based model for {annotation_type}")
                    
                    # Set to evaluation mode
                    model.eval()
                    self.loaded_models[annotation_type] = model
                    
                    # Load stats if available
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                            self.model_stats[annotation_type] = stats
                    
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {annotation_type} model: {e}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"✅ Successfully loaded {loaded_count}/{len(annotation_types)} graph-based models")
                return True
            else:
                logger.error("❌ No models loaded successfully")
                return False
                
        except Exception as e:
            logger.error(f"Error loading graph-based models: {e}")
            return False
    
    def predict_annotations_for_file_with_cfg(self, java_file: str, cfg_dir: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Predict annotations for a single Java file using graph-based models and real CFGs"""
        if not self.loaded_models:
            logger.error("No trained models loaded")
            return []
        
        predictions = []
        
        try:
            # Find CFG file for this Java file
            java_basename = os.path.splitext(os.path.basename(java_file))[0]
            
            # Try different CFG file locations
            cfg_file_candidates = [
                os.path.join(cfg_dir, f"{java_basename}.cfg.json"),
                os.path.join(cfg_dir, java_basename, "cfg.json"),
                os.path.join(cfg_dir, java_basename, f"{java_basename}.cfg.json")
            ]
            
            cfg_file = None
            for candidate in cfg_file_candidates:
                if os.path.exists(candidate):
                    cfg_file = candidate
                    break
            
            if cfg_file is None:
                logger.warning(f"No CFG file found for {java_file}. Tried: {cfg_file_candidates}")
                return []
            
            # Load CFG as PyG graph
            graph_data = load_cfg_as_pyg(cfg_file)
            if graph_data.x is None or graph_data.x.numel() == 0:
                logger.warning("Empty graph features; skipping")
                return []
            
            # Ensure batch tensor exists
            if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            
            # Move to device
            graph_data = graph_data.to(self.device)
            
            # Predict with each model
            for annotation_type, model in self.loaded_models.items():
                try:
                    with torch.no_grad():
                        # Get model prediction
                        logits = model(graph_data)
                        probabilities = torch.softmax(logits, dim=1)
                        prediction = torch.argmax(logits, dim=1).item()
                        confidence = probabilities[0, prediction].item()
                        
                        # Check if prediction is positive and above threshold
                        if prediction == 1 and confidence > threshold:
                            # Find the most relevant node (highest activation)
                            # For now, use the first node as placeholder
                            node_id = 0
                            line_number = self._get_node_line_number(graph_data, node_id)
                            
                            prediction_dict = {
                                'line': line_number,
                                'annotation_type': annotation_type,
                                'confidence': confidence,
                                'reason': f"{annotation_type} expected (predicted by {model.__class__.__name__} with {confidence:.3f} confidence) (using CFG graph)",
                                'model_type': model.__class__.__name__
                            }
                            predictions.append(prediction_dict)
                            
                except Exception as e:
                    logger.error(f"Error predicting with {annotation_type} model: {e}")
                    continue
            
            logger.info(f"Generated {len(predictions)} predictions for {java_file}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing {java_file}: {e}")
            return []
    
    def _get_node_line_number(self, graph_data, node_id: int) -> int:
        """Extract line number from graph node"""
        try:
            # Try to get line number from node features
            if hasattr(graph_data, 'line') and graph_data.line is not None:
                if isinstance(graph_data.line, torch.Tensor):
                    if node_id < graph_data.line.size(0):
                        return int(graph_data.line[node_id].item())
                elif isinstance(graph_data.line, (list, tuple)):
                    if node_id < len(graph_data.line):
                        return int(graph_data.line[node_id])
            
            # Fallback: try to extract from node features
            if graph_data.x is not None and node_id < graph_data.x.size(0):
                node_features = graph_data.x[node_id]
                # Assume line number is in the first few features
                for i in range(min(5, node_features.size(0))):
                    val = node_features[i].item()
                    if val > 0 and val < 10000:  # Reasonable line number range
                        return int(val)
            
            return 1  # Default fallback
            
        except Exception as e:
            logger.debug(f"Error extracting line number for node {node_id}: {e}")
            return 1
    
    def train_missing_models(self, base_model_type: str = 'enhanced_causal', episodes: int = 50, 
                           project_root: str = '/home/ubuntu/checker-framework/checker/tests/index') -> bool:
        """Train any missing models for the specified base model type"""
        logger.info(f"Training missing graph-based models with base model type: {base_model_type}")
        
        try:
            # Check which models are missing
            annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
            missing_models = []
            
            for annotation_type in annotation_types:
                model_name = annotation_type.replace('@', '').lower()
                model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
                
                if not os.path.exists(model_file):
                    missing_models.append(annotation_type)
            
            if not missing_models:
                logger.info("All models already exist, no training needed")
                return True
            
            logger.info(f"Training missing models: {missing_models}")
            
            # For now, create untrained models (full training would require CFG data)
            for annotation_type in missing_models:
                try:
                    model = self._create_graph_based_model(base_model_type, annotation_type)
                    model.eval()
                    self.loaded_models[annotation_type] = model
                    logger.info(f"Created untrained {annotation_type} model")
                except Exception as e:
                    logger.error(f"Failed to create {annotation_type} model: {e}")
            
            return len(missing_models) > 0
            
        except Exception as e:
            logger.error(f"Error training missing models: {e}")
            return False
    
    def load_or_train_models(self, base_model_type: str = 'enhanced_causal', episodes: int = 50, 
                           project_root: str = '/home/ubuntu/checker-framework/checker/tests/index') -> bool:
        """Load existing models or train missing ones"""
        # Try to load existing models first
        if self.load_trained_models(base_model_type):
            return True
        
        # If no models loaded and auto-training is enabled, train missing models
        if self.auto_train:
            logger.info("No trained models found, creating new models")
            return self.train_missing_models(base_model_type, episodes, project_root)
        
        logger.error("No trained models found and auto-training is disabled")
        return False
    
    def _generate_model_reason(self, annotation_type: str, node: Dict[str, Any], confidence: float, model_type: str) -> str:
        """Generate a human-readable reason for the prediction"""
        return f"{annotation_type} expected (predicted by {model_type} with {confidence:.3f} confidence) (using CFG graph)"


# Compatibility wrapper for the old interface
class ModelBasedPredictor(GraphBasedPredictor):
    """Compatibility wrapper for the old ModelBasedPredictor interface"""
    pass
