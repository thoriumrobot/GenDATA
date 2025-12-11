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
                    
                    # Get state dict
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Detect model architecture from state dict keys
                    state_dict_keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
                    has_feature_extractor = any('feature_extractor' in key for key in state_dict_keys)
                    has_network = any('network' in key for key in state_dict_keys)
                    
                    # Extract metadata
                    input_dim = checkpoint.get('input_dim', 21)  # Default to 21
                    model_type = checkpoint.get('model_type', 'unknown')
                    training_stats = checkpoint.get('training_stats', {})
                    
                    # Determine model architecture and create appropriate model
                    model = None
                    
                    if has_feature_extractor:
                        # AnnotationType models with feature_extractor + classifier architecture
                        # Determine actual input_dim from state_dict (feature_extractor.0.weight shape)
                        feature_extractor_0_key = [k for k in state_dict_keys if 'feature_extractor.0.weight' in k]
                        if feature_extractor_0_key:
                            actual_input_dim = state_dict[feature_extractor_0_key[0]].shape[1]
                            if actual_input_dim != input_dim:
                                logger.debug(f"Detected input_dim={actual_input_dim} from state_dict (expected {input_dim})")
                                input_dim = actual_input_dim
                        
                        # Determine hidden_dim from feature_extractor.0 output
                        if feature_extractor_0_key:
                            hidden_dim = state_dict[feature_extractor_0_key[0]].shape[0]
                        else:
                            hidden_dim = 128  # Default
                        
                        # Check classifier structure - single Linear or Sequential
                        has_classifier_sequential = any('classifier.0' in k for k in state_dict_keys)
                        
                        # Import appropriate model class based on base_model
                        if base_model == 'hgt':
                            from annotation_type_rl_positive import AnnotationTypeHGTModel
                            model = AnnotationTypeHGTModel(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                out_dim=2
                            )
                        elif base_model == 'gcsn':
                            from annotation_type_rl_positive import AnnotationTypeGCSNModel
                            model = AnnotationTypeGCSNModel(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                out_dim=2
                            )
                        elif base_model == 'dg2n':
                            from annotation_type_rl_positive import AnnotationTypeDG2NModel
                            model = AnnotationTypeDG2NModel(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                out_dim=2
                            )
                        elif base_model == 'enhanced_causal':
                            from annotation_type_rl_positive import AnnotationTypeEnhancedCausalModel
                            # Enhanced causal uses hidden_dim for feature_extractor, outputs hidden_dim // 2
                            model = AnnotationTypeEnhancedCausalModel(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                out_dim=2
                            )
                        elif 'causal' in base_model:
                            from annotation_type_rl_positive import AnnotationTypeCausalModel
                            # Causal uses hidden_dim for feature_extractor, outputs hidden_dim // 2
                            model = AnnotationTypeCausalModel(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                out_dim=2
                            )
                        else:
                            # Default to GCN model
                            from annotation_type_rl_positive import AnnotationTypeGCNModel
                            model = AnnotationTypeGCNModel(
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                out_dim=2
                            )
                        
                        # Load state dict
                        try:
                            model.load_state_dict(state_dict, strict=True)
                        except RuntimeError as e:
                            logger.warning(f"Strict loading failed for {annotation_type} ({base_model}), trying non-strict: {e}")
                            # Try non-strict loading
                            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                            if missing_keys:
                                logger.warning(f"Missing keys: {missing_keys[:3]}")
                            if unexpected_keys:
                                logger.warning(f"Unexpected keys: {unexpected_keys[:3]}")
                            # If still fails, skip this model
                            if any('weight' in k for k in missing_keys):
                                logger.warning(f"Cannot load {annotation_type} ({base_model}) - critical weights missing")
                                continue
                        
                    elif has_network:
                        # ImprovedBalancedAnnotationTypeModel architecture
                        model = ImprovedBalancedAnnotationTypeModel(
                            input_dim=input_dim,
                            hidden_dims=[512, 256, 128, 64],
                            dropout_rate=0.4
                        )
                        model.load_state_dict(state_dict)
                        
                    else:
                        logger.warning(f"Unknown model architecture for {annotation_type} ({base_model}): {state_dict_keys[:5]}")
                        continue
                    
                    if model is None:
                        continue
                    
                    model.eval()
                    model = model.to(self.device)
                    
                    self.loaded_models[annotation_type][base_model] = {
                        'model': model,
                        'type': 'feature_based',
                        'input_dim': input_dim
                    }
                    
                    # Store metadata
                    self.model_metadata[annotation_type][base_model] = {
                        'input_dim': input_dim,
                        'model_type': model_type,
                        'training_stats': training_stats,
                        'model_path': str(model_path),
                        'architecture': 'feature_extractor' if has_feature_extractor else 'network'
                    }
                    
                    loaded_count += 1
                    logger.debug(f"Loaded {annotation_type} ({base_model}) with {('feature_extractor' if has_feature_extractor else 'network')} architecture")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {annotation_type} ({base_model}): {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
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
        
        # Find all CFG files for this Java file (may have multiple method-level CFGs)
        cfg_files = self._find_all_cfg_files(java_file, cfg_dir)
        if not cfg_files:
            logger.debug(f"No CFG files found for {java_file}")
            return []
        
        # Predict for each node/location across all CFG files
        location_predictions: Dict[Tuple[str, int], List[Dict]] = {}
        
        for cfg_file in cfg_files:
            # Load CFG data
            try:
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse CFG JSON file {cfg_file}: {e}")
                continue
            except FileNotFoundError:
                logger.debug(f"CFG file not found: {cfg_file}")
                continue
            except Exception as e:
                logger.debug(f"Failed to load CFG file {cfg_file}: {e}")
                continue
            
            # Validate CFG structure
            if not isinstance(cfg_data, dict):
                logger.debug(f"Invalid CFG structure: expected dict, got {type(cfg_data)}")
                continue
            
            # Get nodes from CFG
            nodes = cfg_data.get('nodes', [])
            if not nodes:
                continue
            
            if not isinstance(nodes, list):
                continue
            
            # Get Java file path from CFG if available (for per-method CFGs)
            cfg_java_file = cfg_data.get('java_file')
            target_java_file = java_file
            if cfg_java_file:
                # Use the Java file from CFG, but verify it matches or is relative
                if not Path(cfg_java_file).is_absolute():
                    # Try to resolve relative to project
                    cfg_java_file_resolved = str(Path(cfg_file).parent.parent / cfg_java_file)
                    if Path(cfg_java_file_resolved).exists():
                        target_java_file = cfg_java_file_resolved
                elif Path(cfg_java_file).exists():
                    target_java_file = cfg_java_file
            
            # Predict for each node/location in this CFG
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                
                # Handle both 'line' and 'line_number' keys for compatibility
                line_number = node.get('line_number') or node.get('line')
                if not line_number:
                    continue
                
                # Ensure line_number is an integer
                try:
                    line_number = int(line_number)
                except (ValueError, TypeError):
                    continue
                
                # Predict for this location
                prediction = self.predict_for_location(cfg_data, node, line_number, threshold)
                
                if prediction:
                    # Group by location (file_path, line_number)
                    location_key = (target_java_file, line_number)
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
        """Find a single CFG file (for backward compatibility)"""
        cfg_files = self._find_all_cfg_files(java_file, cfg_dir)
        return cfg_files[0] if cfg_files else None
    
    def _find_all_cfg_files(self, java_file: str, cfg_dir: str) -> List[str]:
        """
        Find all CFG files corresponding to a Java file.
        May return multiple files if CFGs are stored per-method.
        
        Args:
            java_file: Path to Java file
            cfg_dir: Directory containing CFG files
            
        Returns:
            List of paths to CFG files
        """
        java_basename = Path(java_file).stem
        java_path = Path(java_file)
        java_name = java_path.name
        
        cfg_dir_path = Path(cfg_dir)
        cfg_files = []
        
        # Try direct match
        cfg_file = cfg_dir_path / f"{java_basename}.json"
        if cfg_file.exists():
            cfg_files.append(str(cfg_file))
        
        # Look for all JSON files in subdirectories
        # Since CFGs may not have java_file field, match by directory structure
        # CFGs are often in subdirectories named after the class/method
        for cfg_file in cfg_dir_path.rglob("*.json"):
            try:
                # Check if CFG file is in a subdirectory that might match the Java file
                # e.g., cfg_dir/ClassName/method.json or cfg_dir/ClassName/cfg.json
                relative_path = cfg_file.relative_to(cfg_dir_path)
                parent_dir = relative_path.parent
                
                # If CFG is in a subdirectory, check if subdirectory name matches Java file
                if parent_dir != Path('.'):
                    # Subdirectory name might match class name
                    if (parent_dir.name == java_basename or 
                        parent_dir.name.lower() == java_basename.lower() or
                        java_basename in parent_dir.name):
                        if str(cfg_file) not in cfg_files:
                            cfg_files.append(str(cfg_file))
                            continue
                
                # Also check if CFG file has java_file field that matches
                with open(cfg_file, 'r') as f:
                    cfg_data = json.load(f)
                    cfg_java_file = cfg_data.get('java_file', '')
                    if cfg_java_file:
                        # Normalize paths for comparison
                        cfg_java_path = Path(cfg_java_file)
                        # Match by filename, stem, or if Java file path contains the CFG's Java file name
                        matches = (
                            cfg_java_path.name == java_name or 
                            cfg_java_path.stem == java_basename or
                            cfg_java_path.name.lower() == java_name.lower() or
                            java_name in cfg_java_file or
                            java_basename in cfg_java_file
                        )
                        if matches:
                            if str(cfg_file) not in cfg_files:
                                cfg_files.append(str(cfg_file))
            except (json.JSONDecodeError, Exception) as e:
                # Skip invalid JSON files
                logger.debug(f"Skipping invalid CFG file {cfg_file}: {e}")
                continue
        
        # Also try subdirectories matching Java file name or class name
        # Extract class name from Java file path (last component before .java)
        class_name = java_basename
        # Also try matching by any part of the path
        path_parts = java_path.parts
        possible_names = {java_basename, class_name, java_name}
        # Add any capitalized words from the path
        for part in path_parts:
            if part.endswith('.java'):
                possible_names.add(part[:-5])  # Remove .java
            elif part[0].isupper():
                possible_names.add(part)
        
        for subdir in cfg_dir_path.iterdir():
            if subdir.is_dir():
                # Match if subdirectory name matches any possible name
                if any(name == subdir.name or name.lower() == subdir.name.lower() 
                       for name in possible_names):
                    for cfg_file in subdir.glob("*.json"):
                        if str(cfg_file) not in cfg_files:
                            cfg_files.append(str(cfg_file))
        
        # If still no matches, try to match by checking all subdirectories
        # and including all CFGs (they might be from the same project)
        if not cfg_files:
            logger.debug(f"No CFG files matched for {java_file}, trying all CFGs in directory")
            # As a fallback, include all CFG files (they might be from the same project)
            for cfg_file in cfg_dir_path.rglob("*.json"):
                if str(cfg_file) not in cfg_files:
                    cfg_files.append(str(cfg_file))
            # Limit to first 50 to avoid processing too many
            if len(cfg_files) > 50:
                cfg_files = cfg_files[:50]
                logger.debug(f"Limited to first 50 CFG files")
        
        return cfg_files


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

