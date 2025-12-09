#!/usr/bin/env python3
"""
Multi-Checker Predictor with Confidence-Based Selection

This module provides a unified prediction system that supports all checkers
(Lower Bound, SQL Quotes, Signature String) with checker-specific annotation types.
For each location, it runs all relevant models and selects the annotation with
the highest confidence if any model predicts "yes".
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from checker_evaluation_config import get_checker_config, build_model_name, GEN_DATA_ROOT
from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeModel, ImprovedBalancedAnnotationTypeTrainer

# Try to import graph-based models
try:
    from cfg_graph import load_cfg_as_pyg
    from graph_encoder import build_graph_encoder
    GRAPH_MODELS_AVAILABLE = True
except ImportError:
    GRAPH_MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultiCheckerPredictor:
    """
    Unified predictor for all checkers with confidence-based annotation selection.
    
    For each location, runs all annotation type models for the checker and selects
    the annotation with the highest confidence if any model predicts "yes".
    """
    
    def __init__(self, checker_name: str, models_dir: Optional[str] = None, device: str = 'auto'):
        """
        Initialize multi-checker predictor.
        
        Args:
            checker_name: Name of the checker ('lower_bound', 'sql_quotes', 'signature_string')
            models_dir: Directory containing models (defaults to checker-specific directory)
            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        self.checker_name = checker_name.lower()
        self.config = get_checker_config(self.checker_name)
        
        if not self.config:
            raise ValueError(f"Unknown checker: {checker_name}")
        
        # Determine models directory
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            # Use checker-specific directory
            if self.checker_name == 'lower_bound':
                self.models_dir = GEN_DATA_ROOT / 'models_annotation_types'
            else:
                self.models_dir = GEN_DATA_ROOT / f'models_annotation_types_{self.checker_name}'
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Get checker configuration
        self.annotation_types = self.config.get('annotation_types', [])
        self.base_models = self.config.get('base_models', [])
        
        # Loaded models: {annotation_type: {base_model: model}}
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        
        # Model metadata: {annotation_type: {base_model: metadata}}
        self.model_metadata: Dict[str, Dict[str, Dict]] = {}
        
        logger.info(f"Initialized MultiCheckerPredictor for {self.config.get('name', checker_name)}")
        logger.info(f"  Annotation types: {self.annotation_types}")
        logger.info(f"  Base models: {self.base_models}")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  Device: {self.device}")
    
    def load_checker_models(self) -> bool:
        """
        Load all annotation type models for this checker.
        
        Returns:
            True if at least one model was loaded successfully
        """
        logger.info(f"Loading models for {self.checker_name} checker...")
        
        loaded_count = 0
        total_expected = len(self.annotation_types) * len(self.base_models)
        
        for annotation_type in self.annotation_types:
            self.loaded_models[annotation_type] = {}
            self.model_metadata[annotation_type] = {}
            
            for base_model in self.base_models:
                # Build model filename
                ann_normalized = annotation_type.replace('@', '').lower()
                model_filename = f"{ann_normalized}_{base_model}_balanced_model.pth"
                model_path = self.models_dir / model_filename
                
                if not model_path.exists():
                    # Try without _balanced suffix for Lower Bound checker
                    if self.checker_name == 'lower_bound':
                        alt_filename = f"{ann_normalized}_{base_model}_model.pth"
                        alt_path = self.models_dir / alt_filename
                        if alt_path.exists():
                            model_path = alt_path
                        else:
                            logger.debug(f"Model not found: {model_filename} or {alt_filename}")
                            continue
                    else:
                        logger.debug(f"Model not found: {model_filename}")
                        continue
                
                try:
                    # Load model checkpoint
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # Extract metadata
                    input_dim = checkpoint.get('input_dim', 21)  # Default to 21 for balanced models
                    model_type = checkpoint.get('model_type', f'improved_balanced_{base_model}')
                    training_stats = checkpoint.get('training_stats', {})
                    
                    # Create model architecture
                    if 'improved_balanced' in model_type or base_model in ['gbt', 'causal', 'enhanced_causal', 'dg2n']:
                        # Feature-based model
                        model = ImprovedBalancedAnnotationTypeModel(
                            input_dim=input_dim,
                            hidden_dims=[512, 256, 128, 64],
                            dropout_rate=0.4
                        )
                        
                        # Load state dict
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        
                        model.eval()
                        model = model.to(self.device)
                        
                        self.loaded_models[annotation_type][base_model] = {
                            'model': model,
                            'type': 'feature_based',
                            'input_dim': input_dim
                        }
                        
                    else:
                        # Graph-based model (GCN, HGT, GCSN)
                        # For now, skip graph models - they require different loading logic
                        logger.debug(f"Skipping graph model {base_model} for {annotation_type} (requires graph loading)")
                        continue
                    
                    # Store metadata
                    self.model_metadata[annotation_type][base_model] = {
                        'input_dim': input_dim,
                        'model_type': model_type,
                        'training_stats': training_stats,
                        'model_path': str(model_path)
                    }
                    
                    loaded_count += 1
                    logger.debug(f"Loaded {annotation_type} ({base_model})")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {annotation_type} ({base_model}): {e}")
                    continue
        
        logger.info(f"Loaded {loaded_count}/{total_expected} models for {self.checker_name} checker")
        return loaded_count > 0
    
    def _extract_features(self, node: Dict[str, Any], cfg_data: Dict[str, Any], 
                          checker_name: str) -> List[float]:
        """
        Extract features for a node based on checker type.
        
        Args:
            node: CFG node dictionary
            cfg_data: CFG data dictionary
            checker_name: Name of the checker
            
        Returns:
            List of feature values
        """
        try:
            from improved_balanced_dataset_generator import ImprovedBalancedDatasetGenerator
            
            generator = ImprovedBalancedDatasetGenerator(
                target_balance=0.5,
                random_seed=42,
                checker_name=checker_name
            )
            
            # Extract features using the generator
            features = generator.extract_node_features(node, cfg_data, include_checker_patterns=True)
            
            # Validate features
            if not isinstance(features, list):
                logger.warning(f"Feature extraction returned non-list: {type(features)}")
                return []
            
            if not all(isinstance(f, (int, float)) for f in features):
                logger.warning("Feature extraction returned non-numeric values")
                return []
            
            return features
        except Exception as e:
            logger.debug(f"Failed to extract features: {e}")
            return []
    
    def _get_model_prediction(self, model_info: Dict[str, Any], features: List[float], 
                              annotation_type: str, base_model: str) -> Tuple[bool, float, str]:
        """
        Get prediction from a model.
        
        Args:
            model_info: Model information dictionary with 'model', 'type', 'input_dim'
            features: Feature vector
            annotation_type: Annotation type being predicted
            base_model: Base model type
            
        Returns:
            Tuple of (is_positive, confidence, reason)
        """
        try:
            model = model_info['model']
            model_type = model_info['type']
            input_dim = model_info['input_dim']
            
            # Ensure features match input dimension
            if len(features) < input_dim:
                # Pad with zeros
                features = features + [0.0] * (input_dim - len(features))
            elif len(features) > input_dim:
                # Truncate
                features = features[:input_dim]
            
            if model_type == 'feature_based':
                # Feature-based model (PyTorch)
                feature_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    outputs = model(feature_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0, prediction].item()
                
                # Class 1 is positive (needs annotation)
                is_positive = (prediction == 1)
                
                if is_positive:
                    reason = f"{annotation_type} predicted by {base_model.upper()} model (confidence: {confidence:.3f})"
                else:
                    reason = f"No {annotation_type} needed (predicted by {base_model.upper()} model, confidence: {confidence:.3f})"
                
                return is_positive, confidence, reason
            else:
                # Graph-based models not yet supported in this predictor
                return False, 0.0, f"Graph model {base_model} not yet supported"
                
        except Exception as e:
            logger.debug(f"Error in model prediction: {e}")
            return False, 0.0, f"Prediction error: {e}"
    
    def predict_for_location(self, cfg_data: Dict[str, Any], node: Dict[str, Any], 
                             line_number: int, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """
        Predict annotation for a single location using all annotation type models.
        Returns the prediction with highest confidence if any model predicts "yes".
        
        Args:
            cfg_data: CFG data dictionary
            node: CFG node dictionary
            line_number: Line number of the location
            threshold: Confidence threshold for positive predictions
            
        Returns:
            Dict with keys: annotation_type, confidence, model_type, reason, line_number
            None if no models predict "yes"
        """
        if not self.loaded_models:
            logger.warning("No models loaded. Call load_checker_models() first.")
            return None
        
        # Extract features once
        try:
            features = self._extract_features(node, cfg_data, self.checker_name)
        except Exception as e:
            logger.debug(f"Failed to extract features: {e}")
            return None
        
        predictions = []
        
        # Run all annotation type models for this checker
        for annotation_type in self.annotation_types:
            if annotation_type not in self.loaded_models:
                continue
            
            for base_model in self.base_models:
                if base_model not in self.loaded_models[annotation_type]:
                    continue
                
                model_info = self.loaded_models[annotation_type][base_model]
                
                # Get prediction (yes/no, confidence)
                is_positive, confidence, reason = self._get_model_prediction(
                    model_info, features, annotation_type, base_model
                )
                
                if is_positive and confidence >= threshold:
                    predictions.append({
                        'annotation_type': annotation_type,
                        'confidence': confidence,
                        'model_type': base_model,
                        'reason': reason,
                        'line_number': line_number
                    })
        
        # Select highest confidence prediction
        if predictions:
            best_prediction = max(predictions, key=lambda p: p['confidence'])
            logger.debug(f"Selected {best_prediction['annotation_type']} (confidence: {best_prediction['confidence']:.3f}) "
                        f"from {len(predictions)} positive predictions at line {line_number}")
            return best_prediction
        
        return None
    
    def predict_for_file(self, java_file: str, cfg_dir: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Predict annotations for all locations in a file.
        Groups predictions by location and selects highest-confidence annotation per location.
        
        Args:
            java_file: Path to Java file
            cfg_dir: Directory containing CFG files
            threshold: Confidence threshold for positive predictions
            
        Returns:
            List of prediction dictionaries (one per location with highest confidence)
        """
        if not self.loaded_models:
            logger.warning("No models loaded. Call load_checker_models() first.")
            return []
        
        # Find CFG file for this Java file
        cfg_file = self._find_cfg_file(java_file, cfg_dir)
        if not cfg_file:
            logger.debug(f"No CFG file found for {java_file}")
            return []
        
        # Load CFG data
        try:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                cfg_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse CFG JSON file {cfg_file}: {e}")
            return []
        except FileNotFoundError:
            logger.debug(f"CFG file not found: {cfg_file}")
            return []
        except Exception as e:
            logger.warning(f"Failed to load CFG file {cfg_file}: {e}")
            return []
        
        # Validate CFG structure
        if not isinstance(cfg_data, dict):
            logger.warning(f"Invalid CFG structure: expected dict, got {type(cfg_data)}")
            return []
        
        # Get nodes from CFG
        nodes = cfg_data.get('nodes', [])
        if not nodes:
            logger.debug(f"No nodes found in CFG file {cfg_file}")
            return []
        
        if not isinstance(nodes, list):
            logger.warning(f"Invalid CFG structure: 'nodes' should be a list, got {type(nodes)}")
            return []
        
        # Predict for each node/location
        location_predictions: Dict[Tuple[str, int], List[Dict]] = {}
        
        for node in nodes:
            if not isinstance(node, dict):
                logger.debug(f"Skipping invalid node: expected dict, got {type(node)}")
                continue
            
            # Handle both 'line' and 'line_number' keys for compatibility
            line_number = node.get('line_number') or node.get('line')
            if not line_number:
                logger.debug(f"Skipping node without line number: {node.get('id', 'unknown')}")
                continue
            
            # Ensure line_number is an integer
            try:
                line_number = int(line_number)
            except (ValueError, TypeError):
                logger.debug(f"Skipping node with invalid line number: {line_number}")
                continue
            
            # Predict for this location
            prediction = self.predict_for_location(cfg_data, node, line_number, threshold)
            
            if prediction:
                # Group by location (file_path, line_number)
                location_key = (java_file, line_number)
                if location_key not in location_predictions:
                    location_predictions[location_key] = []
                location_predictions[location_key].append(prediction)
        
        # Select highest confidence prediction for each location
        selected_predictions = []
        for (file_path, line_number), preds in location_predictions.items():
            if preds:
                # Select highest confidence
                best_pred = max(preds, key=lambda p: p['confidence'])
                best_pred['file_path'] = file_path
                selected_predictions.append(best_pred)
        
        logger.info(f"Generated {len(selected_predictions)} predictions for {java_file} "
                   f"(from {len(location_predictions)} locations with positive predictions)")
        
        return selected_predictions
    
    def _find_cfg_file(self, java_file: str, cfg_dir: str) -> Optional[str]:
        """
        Find CFG file corresponding to a Java file.
        
        Args:
            java_file: Path to Java file
            cfg_dir: Directory containing CFG files
            
        Returns:
            Path to CFG file, or None if not found
        """
        java_basename = Path(java_file).stem
        
        # Look for CFG file with matching name
        cfg_dir_path = Path(cfg_dir)
        
        # Try direct match
        cfg_file = cfg_dir_path / f"{java_basename}.json"
        if cfg_file.exists():
            return str(cfg_file)
        
        # Try in subdirectories
        for cfg_file in cfg_dir_path.rglob(f"{java_basename}.json"):
            return str(cfg_file)
        
        # Try cfg.json in subdirectory
        for subdir in cfg_dir_path.iterdir():
            if subdir.is_dir():
                cfg_file = subdir / "cfg.json"
                if cfg_file.exists():
                    return str(cfg_file)
        
        return None


def main():
    """Test the multi-checker predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multi-checker predictor')
    parser.add_argument('--checker', choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       required=True, help='Checker name')
    parser.add_argument('--java_file', help='Java file to predict on')
    parser.add_argument('--cfg_dir', help='CFG directory')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = MultiCheckerPredictor(args.checker)
    
    # Load models
    if not predictor.load_checker_models():
        print("Failed to load models")
        return 1
    
    # Predict if file provided
    if args.java_file and args.cfg_dir:
        predictions = predictor.predict_for_file(args.java_file, args.cfg_dir, args.threshold)
        print(f"\nGenerated {len(predictions)} predictions:")
        for pred in predictions:
            print(f"  Line {pred['line_number']}: {pred['annotation_type']} "
                  f"(confidence: {pred['confidence']:.3f}, model: {pred['model_type']})")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

