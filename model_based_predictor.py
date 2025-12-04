#!/usr/bin/env python3
"""
Model-based predictor that uses trained annotation type models for prediction
Now uses graph-based models that process CFG graphs directly
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
    logger.warning(f"Graph-based models not available: {e}")
    PYG_AVAILABLE = False

# Import the annotation type trainers for compatibility
try:
    from annotation_type_rl_positive import AnnotationTypeTrainer as PositiveTrainer
    from annotation_type_rl_nonnegative import AnnotationTypeTrainer as NonNegativeTrainer
    from annotation_type_rl_gtenegativeone import AnnotationTypeTrainer as GTENegativeOneTrainer
except ImportError:
    logger.warning("Legacy trainers not available, using graph-based models only")

# Import enhanced causal model if available
try:
    from enhanced_causal_model import extract_enhanced_causal_features
    ENHANCED_CAUSAL_AVAILABLE = True
except ImportError:
    ENHANCED_CAUSAL_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelBasedPredictor:
    """Predictor that uses trained models for annotation type prediction"""
    
    def __init__(self, models_dir: str = 'models_annotation_types', device: str = 'cpu', auto_train: bool = True):
        self.models_dir = models_dir
        self.device = device
        self.auto_train = auto_train
        self.loaded_models = {}
        self.model_stats = {}
        self._encoder_cache: Dict[str, Any] = {}
        
    def load_trained_models(self, base_model_type: str = 'enhanced_causal') -> bool:
        """Load all trained annotation type models"""
        try:
            logger.info(f"Loading trained models with base model type: {base_model_type}")
            
            # Define annotation types and their corresponding trainers
            annotation_configs = [
                ('@Positive', PositiveTrainer),
                ('@NonNegative', NonNegativeTrainer),
                ('@GTENegativeOne', GTENegativeOneTrainer)
            ]
            
            loaded_count = 0
            for annotation_type, trainer_class in annotation_configs:
                model_name = annotation_type.replace('@', '').lower()
                model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
                stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
                
                if os.path.exists(model_file) and os.path.exists(stats_file):
                    try:
                        # Create trainer instance
                        trainer = trainer_class(
                            annotation_type=annotation_type,
                            base_model_type=base_model_type,
                            device=self.device
                        )
                        
                        # Load model based on type
                        if base_model_type == 'gbt':
                            # For GBT models, load with joblib
                            import joblib
                            checkpoint = joblib.load(model_file)
                            if 'model' in checkpoint:
                                trainer.model = checkpoint['model']
                            else:
                                trainer.model = checkpoint
                        else:
                            # For PyTorch models, load with torch
                            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
                            if hasattr(trainer.model, 'load_state_dict'):
                                # Extract model state from checkpoint
                                if 'model_state_dict' in checkpoint:
                                    trainer.model.load_state_dict(checkpoint['model_state_dict'])
                                else:
                                    trainer.model.load_state_dict(checkpoint)
                            else:
                                # For non-PyTorch models
                                if 'model_state_dict' in checkpoint:
                                    trainer.model = checkpoint['model_state_dict']
                                else:
                                    trainer.model = checkpoint
                        
                        # Only call eval() for PyTorch models
                        if hasattr(trainer.model, 'eval'):
                            trainer.model.eval()
                        
                        # Load stats
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        self.loaded_models[annotation_type] = trainer
                        self.model_stats[annotation_type] = stats
                        
                        logger.info(f"✅ Loaded {annotation_type} model ({base_model_type})")
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load {annotation_type} model: {e}")
                else:
                    logger.warning(f"⚠️ Model files not found for {annotation_type}")
            
            logger.info(f"Successfully loaded {loaded_count}/{len(annotation_configs)} models")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            return False

    def train_missing_models(self, base_model_type: str = 'enhanced_causal', episodes: int = 50, project_root: str = '/home/ubuntu/checker-framework/checker/tests/index') -> bool:
        """Train any missing models for the specified base model type"""
        logger.info(f"Training missing models with base model type: {base_model_type}")
        
        # Define annotation types and their corresponding trainers
        annotation_configs = [
            ('@Positive', PositiveTrainer),
            ('@NonNegative', NonNegativeTrainer),
            ('@GTENegativeOne', GTENegativeOneTrainer)
        ]
        
        trained_count = 0
        for annotation_type, trainer_class in annotation_configs:
            model_name = annotation_type.replace('@', '').lower()
            model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
            stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
            
            # Check if model already exists
            if os.path.exists(model_file) and os.path.exists(stats_file):
                logger.info(f"Model already exists for {annotation_type} ({base_model_type}), skipping training")
                continue
            
            # Train the missing model
            logger.info(f"Training missing model: {annotation_type} ({base_model_type})")
            try:
                trainer = trainer_class(
                    annotation_type=annotation_type,
                    base_model_type=base_model_type,
                    device=self.device
                )
                
                # Train the model
                training_stats = trainer.train(
                    project_root=project_root,
                    warnings_file='/home/ubuntu/checker-framework/checker/tests/index/index1.out',
                    cfwr_root='/home/ubuntu/GenDATA',
                    num_episodes=episodes
                )
                
                if training_stats:
                    logger.info(f"✅ Successfully trained {annotation_type} ({base_model_type}) model")
                    trained_count += 1
                else:
                    logger.error(f"❌ Failed to train {annotation_type} ({base_model_type}) model")
                    
            except Exception as e:
                logger.error(f"❌ Error training {annotation_type} ({base_model_type}) model: {e}")
        
        logger.info(f"Trained {trained_count} missing models for {base_model_type}")
        return trained_count > 0

    def load_or_train_models(self, base_model_type: str = 'enhanced_causal', episodes: int = 50, project_root: str = '/home/ubuntu/checker-framework/checker/tests/index') -> bool:
        """Load existing models or train missing ones"""
        # First try to load existing models
        if self.load_trained_models(base_model_type):
            logger.info(f"✅ Successfully loaded all models for {base_model_type}")
            return True
        
        # If auto_train is enabled and some models are missing, train them
        if self.auto_train:
            logger.info(f"Some models missing for {base_model_type}, training missing models...")
            if self.train_missing_models(base_model_type, episodes, project_root):
                # Try loading again after training
                if self.load_trained_models(base_model_type):
                    logger.info(f"✅ Successfully loaded all models for {base_model_type} after training")
                    return True
                else:
                    logger.error(f"❌ Failed to load models for {base_model_type} even after training")
                    return False
            else:
                logger.error(f"❌ Failed to train missing models for {base_model_type}")
                return False
        else:
            logger.warning(f"Models missing for {base_model_type} and auto_train is disabled")
            return False
    
    def predict_annotations_for_file(self, java_file: str, threshold: float = 0.3, cfg_dir: str = None) -> List[Dict[str, Any]]:
        """Predict annotations for a single Java file using trained models and real CFGs only."""
        if not self.loaded_models:
            logger.error("No trained models loaded")
            return []
        
        if not cfg_dir:
            logger.error("CFG directory is required for prediction; mock fallback is disabled")
            return []

        return self.predict_annotations_for_file_with_cfg(java_file, cfg_dir, threshold)
    
    def _create_mock_node(self, line: str, line_number: int) -> Dict[str, Any]:
        """Deprecated: mock node generator (retained for compatibility). Not used."""
        return {
            'id': f"node_{line_number}",
            'label': '',
            'node_type': 'statement',
            'line': line_number,
            'is_annotation_target': False
        }
    
    def _create_mock_cfg_data(self, lines: List[str]) -> Dict[str, Any]:
        """Deprecated: mock CFG generator (removed - use real pipeline data)."""
        logger.warning("Mock CFG data generation is deprecated. Use real pipeline data instead.")
        return {'nodes': [], 'edges': []}
    
    def _extract_basic_features(self, node: Dict[str, Any], cfg_data: Dict[str, Any]) -> List[float]:
        """Extract basic features for prediction"""
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        line = node.get('line', 0)
        
        # Basic feature extraction
        try:
            features = [
                float(len(label)),  # label_length
                float(line if line is not None else 0),  # line_number
                float('method' in node_type.lower()),  # is_method
                float('field' in node_type.lower()),  # is_field
                float('parameter' in node_type.lower()),  # is_parameter
                float('variable' in node_type.lower()),  # is_variable
                float('positive' in label.lower()),  # contains_positive
                float('negative' in label.lower()),  # contains_negative
                float('count' in label.lower()),  # is_count_variable
                float('size' in label.lower()),  # is_size_variable
                float('length' in label.lower()),  # is_length_variable
                float('index' in label.lower()),  # is_index_variable
                float('offset' in label.lower()),  # is_offset_variable
                float('capacity' in label.lower()),  # is_capacity_variable
            ]
        except Exception as e:
            logger.error(f"Error extracting features for node {node}: {e}")
            logger.error(f"Label: {label}, Line: {line}, Node type: {node_type}")
            raise
        
        # Pad to expected dimension if using enhanced causal
        if ENHANCED_CAUSAL_AVAILABLE:
            # Pad to 32 dimensions for enhanced causal model
            while len(features) < 32:
                features.append(0.0)
        
        return features
    
    def _get_model_prediction(self, trainer, features: List[float], annotation_type: str, node: Dict[str, Any]) -> tuple:
        """Get prediction from a trained model"""
        try:
            if hasattr(trainer.model, 'forward'):
                # PyTorch model
                feature_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    outputs = trainer.model(feature_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0, prediction].item()
                
                # Determine if this is a positive prediction for the annotation type
                if prediction == 1 and confidence > 0.3:  # Assuming 1 is positive class
                    reason = self._generate_model_reason(annotation_type, node, confidence, trainer.base_model_type)
                    return True, confidence, reason
                else:
                    return False, confidence, "Model predicted no annotation needed"
                    
            else:
                # Non-PyTorch model (e.g., GBT)
                feature_array = np.array([features])
                probabilities = trainer.model.predict_proba(feature_array)[0]
                # For GBT models, use argmax to get prediction
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction] if len(probabilities) > prediction else 0.5
                
                if prediction == 1 and confidence > 0.3:
                    reason = self._generate_model_reason(annotation_type, node, confidence, trainer.base_model_type)
                    return True, confidence, reason
                else:
                    return False, confidence, "Model predicted no annotation needed"
                    
        except Exception as e:
            logger.debug(f"Error in model prediction: {e}")
            return False, 0.0, f"Prediction error: {e}"
    
    def _generate_model_reason(self, annotation_type: str, node: Dict[str, Any], confidence: float, model_type: str) -> str:
        """Generate explanation for model prediction based on model inference"""
        # Generate pure model-based reasons without heuristic keyword matching
        if annotation_type == '@Positive':
            return f"positive value expected (predicted by {model_type.upper()} model with {confidence:.3f} confidence)"
                
        elif annotation_type == '@NonNegative':
            return f"non-negative value expected (predicted by {model_type.upper()} model with {confidence:.3f} confidence)"
                
        elif annotation_type == '@GTENegativeOne':
            return f"value >= -1 expected (predicted by {model_type.upper()} model with {confidence:.3f} confidence)"
        
        return f"model prediction (predicted by {model_type.upper()} model)"
    
    def predict_annotations_for_file_with_cfg(self, java_file, cfg_dir, threshold=0.3, cfg_file_override=None):
        """Predict annotations for a Java file using real CFG graphs (PyG)."""
        try:
            # Use override if provided, otherwise construct path
            if cfg_file_override and os.path.exists(cfg_file_override):
                cfg_file = cfg_file_override
            else:
                # Find CFG data for this Java file
                java_basename = os.path.splitext(os.path.basename(java_file))[0]
                cfg_file = os.path.join(cfg_dir, java_basename, 'cfg.json')
            
            if not os.path.exists(cfg_file):
                logger.warning(f"No CFG file found for {java_file}; skipping (mock disabled)")
                return []
            
            if not PYG_AVAILABLE:
                logger.error("PyTorch Geometric not available; cannot process CFG graphs")
                return []
            
            # Load CFG as PyG graph with rich features
            graph_data = load_cfg_as_pyg(cfg_file)
            if graph_data.x is None or graph_data.x.numel() == 0:
                logger.warning("Empty graph features; skipping")
                return []
            
            # Add batch tensor for graph encoder compatibility
            if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                # All nodes belong to the same graph (batch 0)
                graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            
            predictions = []
            for annotation_type in ['@Positive', '@NonNegative', '@GTENegativeOne']:
                if annotation_type in self.loaded_models:
                    # loaded_models maps to the trainer instance directly
                    trainer = self.loaded_models[annotation_type]
                    base_model_type = getattr(trainer, 'base_model_type', 'unknown')
                    graph_models = {'hgt', 'gcn', 'gcsn', 'dg2n'}

                    try:
                        logger.info(f"Processing {annotation_type} with base_model_type={base_model_type}")
                        # Load CFG JSON to access node data and line numbers
                        # If cfg_file points to a directory, load all CFG JSON files and merge
                        nodes = []
                        cfg_json = {}
                        try:
                            if os.path.isdir(cfg_file):
                                # Load all JSON files in the directory and merge nodes
                                cfg_dir = cfg_file
                                all_nodes = []
                                all_edges = []
                                all_control_edges = []
                                all_dataflow_edges = []
                                for json_file in os.listdir(cfg_file):
                                    if json_file.endswith('.json'):
                                        json_path = os.path.join(cfg_file, json_file)
                                        with open(json_path, 'r') as f:
                                            cfg_data = json.load(f)
                                            if isinstance(cfg_data, dict):
                                                if 'nodes' in cfg_data:
                                                    all_nodes.extend(cfg_data['nodes'])
                                                if 'edges' in cfg_data:
                                                    all_edges.extend(cfg_data.get('edges', []))
                                                if 'control_edges' in cfg_data:
                                                    all_control_edges.extend(cfg_data.get('control_edges', []))
                                                if 'dataflow_edges' in cfg_data:
                                                    all_dataflow_edges.extend(cfg_data.get('dataflow_edges', []))
                                # Build combined cfg_json
                                cfg_json = {
                                    'nodes': all_nodes,
                                    'edges': all_edges,
                                    'control_edges': all_control_edges,
                                    'dataflow_edges': all_dataflow_edges,
                                    'java_file': cfg_file
                                }
                                nodes = all_nodes
                            elif os.path.isfile(cfg_file):
                                # Load single CFG file
                                with open(cfg_file, 'r') as f:
                                    cfg_json = json.load(f)
                                    if isinstance(cfg_json, dict) and 'nodes' in cfg_json:
                                        nodes = cfg_json['nodes']
                                    # Also try loading other CFG files in the same directory and merge
                                    cfg_dir = os.path.dirname(cfg_file)
                                    if os.path.isdir(cfg_dir):
                                        all_nodes = list(nodes) if nodes else []
                                        all_edges = list(cfg_json.get('edges', []))
                                        all_control_edges = list(cfg_json.get('control_edges', []))
                                        all_dataflow_edges = list(cfg_json.get('dataflow_edges', []))
                                        for json_file in os.listdir(cfg_dir):
                                            if json_file.endswith('.json') and json_file != os.path.basename(cfg_file):
                                                json_path = os.path.join(cfg_dir, json_file)
                                                try:
                                                    with open(json_path, 'r') as f2:
                                                        cfg_data = json.load(f2)
                                                        if isinstance(cfg_data, dict):
                                                            if 'nodes' in cfg_data:
                                                                all_nodes.extend(cfg_data['nodes'])
                                                            if 'edges' in cfg_data:
                                                                all_edges.extend(cfg_data.get('edges', []))
                                                            if 'control_edges' in cfg_data:
                                                                all_control_edges.extend(cfg_data.get('control_edges', []))
                                                            if 'dataflow_edges' in cfg_data:
                                                                all_dataflow_edges.extend(cfg_data.get('dataflow_edges', []))
                                                except Exception:
                                                    pass
                                        # Update cfg_json with merged data
                                        cfg_json['nodes'] = all_nodes
                                        cfg_json['edges'] = all_edges
                                        cfg_json['control_edges'] = all_control_edges
                                        cfg_json['dataflow_edges'] = all_dataflow_edges
                                        nodes = all_nodes
                        except Exception as e:
                            logger.warning(f"Failed to load CFG JSON for line numbers: {cfg_file}, error: {e}")
                        
                        if base_model_type in graph_models and hasattr(trainer.model, 'forward'):
                            # Try graph models first: some (hgt, gcsn, dg2n) might be true graph models
                            trainer.model.eval() if hasattr(trainer.model, 'eval') else None
                            try:
                                with torch.no_grad():
                                    out = trainer.model(graph_data)
                                # Check if output is per-node [num_nodes, num_classes]
                                if isinstance(out, torch.Tensor) and out.dim() == 2 and out.size(0) == graph_data.x.size(0) and out.size(1) >= 2:
                                    probs = torch.softmax(out, dim=1)
                                    preds = torch.argmax(out, dim=1)
                                    # Extract line numbers from CFG JSON nodes in order
                                    node_lines = [int(n.get('line', 0)) if isinstance(n.get('line'), int) else 0 for n in nodes]
                                    
                                    # LOCALIZATION FIX: Try to adjust line numbers backwards for parameter/declaration nodes
                                    # CFG nodes often point to statement lines, but annotations are on parameter declarations
                                    # Look for parameter/variable declaration patterns and adjust line numbers backwards
                                    adjusted_lines = []
                                    for i, (node, orig_line) in enumerate(zip(nodes, node_lines)):
                                        if orig_line > 0:
                                            label = node.get('label', '').lower()
                                            node_type = node.get('node_type', '').lower()
                                            
                                            # Check if this is a statement node that might follow a parameter declaration
                                            # If the previous node(s) have no line or are much earlier, this might be a statement
                                            # following a parameter declaration on the same or previous line
                                            if i > 0 and ('parameter' in label or 'variable' in label or 'declaration' in label):
                                                # This is likely a declaration node - keep original line
                                                adjusted_lines.append(orig_line)
                                            elif i > 0 and orig_line > 0:
                                                # Check previous nodes to see if we should adjust backwards
                                                prev_line = node_lines[i-1] if i > 0 else 0
                                                # If there's a gap, this might be a statement following a declaration
                                                # Try adjusting back by 1-2 lines for better alignment with GT
                                                if prev_line > 0 and orig_line - prev_line > 2:
                                                    # Large gap suggests this is a statement, try -1 line
                                                    adjusted_lines.append(max(1, orig_line - 1))
                                                else:
                                                    adjusted_lines.append(orig_line)
                                            else:
                                                adjusted_lines.append(orig_line)
                                        else:
                                            adjusted_lines.append(0)
                                    
                                    # Fill zeros with forward/backward propagation
                                    last = 0
                                    for i in range(len(adjusted_lines)):
                                        if adjusted_lines[i] == 0 and last > 0:
                                            adjusted_lines[i] = last
                                        elif adjusted_lines[i] > 0:
                                            last = adjusted_lines[i]
                                    last = 0
                                    for i in range(len(adjusted_lines)-1, -1, -1):
                                        if adjusted_lines[i] == 0 and last > 0:
                                            adjusted_lines[i] = last
                                        elif adjusted_lines[i] > 0:
                                            last = adjusted_lines[i]
                                    node_lines = [ln if ln > 0 else 1 for ln in adjusted_lines]
                                    # Emit per-node predictions
                                    for idx in range(min(int(preds.size(0)), len(node_lines))):
                                        pred_label = int(preds[idx].item())
                                        conf_val = float(probs[idx, pred_label].item())
                                        if pred_label == 1 and conf_val >= threshold:
                                            line_num = node_lines[idx]
                                            if line_num <= 0:
                                                continue
                                            reason = self._generate_model_reason(annotation_type, {}, conf_val, base_model_type)
                                            predictions.append({
                                                'line': int(line_num),
                                                'annotation_type': annotation_type,
                                                'confidence': conf_val,
                                                'reason': f"{reason} (node {idx})",
                                                'model_type': base_model_type
                                            })
                                    continue  # Successfully processed as graph model
                            except Exception as e:
                                logger.debug(f"Graph model forward failed (likely feature-based model): {e}")
                                # Fall through to feature extraction path
                        
                        # Feature-based models (gcn, causal, gbt, or graph models that failed): extract per-node features
                        if hasattr(trainer, '_extract_annotation_type_features') and nodes:
                            logger.info(f"Using feature-based prediction for {annotation_type} ({base_model_type}) with {len(nodes)} nodes")
                            # Extract features for each node and run model
                            trainer.model.eval() if hasattr(trainer.model, 'eval') else None
                            node_pred_count = 0
                            with torch.no_grad():
                                for idx, node in enumerate(nodes):
                                    try:
                                        # Extract features using trainer's method
                                        feat_vec = trainer._extract_annotation_type_features(node, cfg_json)
                                        # Convert to tensor and ensure correct shape
                                        if isinstance(feat_vec, (list, tuple)):
                                            feat_vec = torch.tensor(feat_vec, dtype=torch.float32)
                                        elif isinstance(feat_vec, np.ndarray):
                                            feat_vec = torch.from_numpy(feat_vec).float()
                                        else:
                                            feat_vec = torch.tensor([feat_vec], dtype=torch.float32)
                                        # Ensure 2D: [1, feature_dim]
                                        if feat_vec.dim() == 1:
                                            feat_vec = feat_vec.unsqueeze(0)
                                        # Run model
                                        out = trainer.model(feat_vec.to(self.device))
                                        if out.dim() == 2 and out.size(1) >= 2:
                                            probs = torch.softmax(out, dim=1)
                                            pred_label = int(torch.argmax(out, dim=1).item())
                                            conf_val = float(probs[0, pred_label].item())
                                            if pred_label == 1 and conf_val >= threshold:
                                                line_num = int(node.get('line', 0)) if isinstance(node.get('line'), int) else 0
                                                if line_num <= 0:
                                                    continue
                                                reason = self._generate_model_reason(annotation_type, {}, conf_val, base_model_type)
                                                predictions.append({
                                                    'line': line_num,
                                                    'annotation_type': annotation_type,
                                                    'confidence': conf_val,
                                                    'reason': f"{reason} (node {idx})",
                                                    'model_type': base_model_type
                                                })
                                                node_pred_count += 1
                                    except Exception as e:
                                        logger.debug(f"Feature extraction/prediction failed for node {idx}: {e}")
                                        continue
                            logger.info(f"Feature-based prediction for {annotation_type}: {node_pred_count} positive predictions")
                            continue  # Successfully processed as feature-based model
                        else:
                            # Non-graph models: build/apply a graph encoder to get a fixed-length embedding
                            logger.info(f"Using non-graph model approach for {annotation_type}")
                            emb = self._encode_graph(graph_data, base_model_type)
                            logger.info(f"Generated embedding with shape: {emb.shape}")
                            
                            # Fix dimension mismatch: enhanced causal models expect 32 features
                            if base_model_type == 'enhanced_causal' and emb.shape[0] != 32:
                                if emb.shape[0] > 32:
                                    # Truncate to 32 dimensions
                                    emb = emb[:32]
                                    logger.info(f"Truncated embedding to shape: {emb.shape}")
                                else:
                                    # Pad to 32 dimensions with zeros
                                    padding = torch.zeros(32 - emb.shape[0])
                                    emb = torch.cat([emb, padding])
                                    logger.info(f"Padded embedding to shape: {emb.shape}")
                            
                            prediction, conf, reason = self._predict_with_embedding(trainer, emb, annotation_type, base_model_type)
                            is_positive = prediction and conf >= threshold
                            logger.info(f"Non-graph model prediction: {annotation_type} - pred={prediction}, conf={conf:.3f}, threshold={threshold}, is_positive={is_positive}")

                        # Non-graph models emit at most one prediction
                        if 'is_positive' in locals() and is_positive:
                            # For non-graph path above
                            predictions.append({
                                'line': 1,
                                'annotation_type': annotation_type,
                                'confidence': conf,
                                'reason': f"{reason} (graph embedding)",
                                'model_type': base_model_type
                            })
                    except Exception as e:
                        logger.debug(f"Prediction error for {annotation_type} ({base_model_type}): {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting annotations for {java_file} with CFG: {e}")
            return []

    def _encode_graph(self, data, base_model_type: str) -> torch.Tensor:
        """Encode a PyG graph into a fixed-length embedding using the configured encoder."""
        key = f"encoder:{base_model_type}:{int(data.x.size(1))}"
        if key not in self._encoder_cache:
            encoder = build_graph_encoder(in_dim=int(data.x.size(1)), edge_dim=int(data.edge_attr.size(1)) if getattr(data, 'edge_attr', None) is not None else 0, out_dim=256, variant='transformer')
            encoder = encoder.to(self.device) if hasattr(encoder, 'to') else encoder
            self._encoder_cache[key] = encoder
        encoder = self._encoder_cache[key]
        encoder.eval() if hasattr(encoder, 'eval') else None
        with torch.no_grad():
            emb = encoder(data)
        # emb shape: [batch_size(=1), D] or [D]; ensure 1D numpy for scikit models
        if isinstance(emb, torch.Tensor):
            if emb.dim() == 2 and emb.size(0) == 1:
                emb = emb.squeeze(0)
        return emb

    def _predict_with_embedding(self, trainer, emb: torch.Tensor, annotation_type: str, base_model_type: str):
        """Run classification using an embedding with either torch or sklearn model."""
        logger.info(f"_predict_with_embedding called for {annotation_type}")
        if hasattr(trainer.model, 'forward'):
            logger.info(f"Model has forward method, using PyTorch path")
            try:
                # Adjust embedding dimension to match model expected input if needed
                expected_in = None
                try:
                    # Try to infer expected input dimension from first Linear layer
                    for m in trainer.model.modules():
                        if hasattr(m, 'weight') and hasattr(m, 'bias') and hasattr(m, 'in_features'):
                            expected_in = int(m.in_features)
                            break
                except Exception:
                    expected_in = None
                if expected_in is not None and emb.dim() >= 1:
                    cur = int(emb.size(-1))
                    if cur != expected_in:
                        if cur > expected_in:
                            emb = emb[..., :expected_in]
                            logger.info(f"Truncated embedding {cur}→{expected_in}")
                        else:
                            pad = torch.zeros(expected_in - cur, device=emb.device, dtype=emb.dtype)
                            emb = torch.cat([emb, pad], dim=-1)
                            logger.info(f"Padded embedding {cur}→{expected_in}")
                with torch.no_grad():
                    logits = trainer.model(emb.unsqueeze(0) if emb.dim() == 1 else emb)
                    probs = torch.softmax(logits, dim=1)
                    pred = int(torch.argmax(logits, dim=1).item())
                    conf = float(probs[0, pred].item())
                logger.info(f"PyTorch prediction: pred={pred}, conf={conf}")
                if pred == 1:
                    reason = self._generate_model_reason(annotation_type, {}, conf, base_model_type)
                    return True, conf, reason
                else:
                    return False, conf, "Model predicted no annotation needed"
            except Exception as e:
                logger.error(f"Error in PyTorch model forward pass: {e}")
                return False, 0.0, f"Model error: {e}"
        else:
            # scikit-learn like (e.g., GBT)
            import numpy as np
            vec = emb.detach().cpu().numpy().reshape(1, -1)
            try:
                proba = trainer.model.predict_proba(vec)[0]
                pred = int(np.argmax(proba))
                conf = float(proba[pred]) if len(proba) > pred else 0.5
            except Exception:
                # fallback to decision_function or predict
                if hasattr(trainer.model, 'decision_function'):
                    score = float(trainer.model.decision_function(vec).squeeze())
                    pred = 1 if score > 0 else 0
                    conf = float(1 / (1 + np.exp(-score)))
                else:
                    pred = int(trainer.model.predict(vec).squeeze())
                    conf = 0.5
            if pred == 1:
                reason = self._generate_model_reason(annotation_type, {}, conf, base_model_type)
                return True, conf, reason
            return False, conf, "Model predicted no annotation needed"
    
    def _extract_cfg_features(self, cfg_data, java_content):
        """Extract features from real CFG data"""
        try:
            # Extract features from CFG nodes
            nodes = cfg_data.get('nodes', [])
            if not nodes:
                return None
            
            # Use the first node as representative
            node = nodes[0]
            
            # Extract features similar to mock data but from real CFG
            features = [
                float(len(node.get('label', ''))),  # label_length
                float(node.get('line') if node.get('line') is not None else 0),  # line_number
                float('method' in node.get('node_type', '').lower()),  # is_method
                float('field' in node.get('node_type', '').lower()),  # is_field
                float('parameter' in node.get('node_type', '').lower()),  # is_parameter
                float('variable' in node.get('node_type', '').lower()),  # is_variable
                float('positive' in node.get('label', '').lower()),  # contains_positive
                float('negative' in node.get('label', '').lower()),  # contains_negative
                float('count' in node.get('label', '').lower()),  # is_count_variable
                float('size' in node.get('label', '').lower()),  # is_size_variable
                float('length' in node.get('label', '').lower()),  # is_length_variable
                float('index' in node.get('label', '').lower()),  # is_index_variable
                float('offset' in node.get('label', '').lower()),  # is_offset_variable
                float('capacity' in node.get('label', '').lower()),  # is_capacity_variable
            ]
            
            # Pad to expected dimension if using enhanced causal
            if ENHANCED_CAUSAL_AVAILABLE:
                # Pad to 32 dimensions for enhanced causal model
                while len(features) < 32:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting CFG features: {e}")
            return None
    
    def _deduplicate_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate predictions for the same line, keeping the highest confidence"""
        line_predictions = {}
        
        for pred in predictions:
            line = pred['line']
            if line not in line_predictions or pred['confidence'] > line_predictions[line]['confidence']:
                line_predictions[line] = pred
        
        return list(line_predictions.values())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self.loaded_models.keys()),
            'model_stats': self.model_stats,
            'enhanced_causal_available': ENHANCED_CAUSAL_AVAILABLE
        }
        return info


def main():
    """Test the model-based predictor"""
    logging.basicConfig(level=logging.INFO)
    
    predictor = ModelBasedPredictor()
    
    # Load models
    if predictor.load_trained_models(base_model_type='enhanced_causal'):
        print("✅ Models loaded successfully")
        print("Model info:", predictor.get_model_info())
        
        # Test prediction on a sample file
        sample_files = [
            '/home/ubuntu/checker-framework/checker/tests/index/StringMethods.java',
            '/home/ubuntu/checker-framework/checker/tests/index/IndexSameLen.java'
        ]
        
        for sample_file in sample_files:
            if os.path.exists(sample_file):
                print(f"\n🔍 Testing prediction on {sample_file}")
                predictions = predictor.predict_annotations_for_file(sample_file)
                print(f"Generated {len(predictions)} predictions:")
                for pred in predictions[:5]:  # Show first 5
                    print(f"  Line {pred['line']}: {pred['annotation_type']} (confidence: {pred['confidence']:.3f}) - {pred['reason']}")
                break
    else:
        print("❌ Failed to load models")


if __name__ == "__main__":
    main()
