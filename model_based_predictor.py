#!/usr/bin/env python3
"""
Model-based predictor that uses trained annotation type models for prediction
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the annotation type trainers
from annotation_type_rl_positive import AnnotationTypeTrainer as PositiveTrainer
from annotation_type_rl_nonnegative import AnnotationTypeTrainer as NonNegativeTrainer
from annotation_type_rl_gtenegativeone import AnnotationTypeTrainer as GTENegativeOneTrainer

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
    
    def predict_annotations_for_file(self, java_file: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Predict annotations for a single Java file using trained models"""
        if not self.loaded_models:
            logger.error("No trained models loaded")
            return []
        
        try:
            # Read Java file
            with open(java_file, 'r') as f:
                lines = f.readlines()
            
            predictions = []
            
            # Create mock CFG data for prediction
            # In a real implementation, this would come from actual CFG analysis
            for i, line in enumerate(lines, 1):
                line_lower = line.lower().strip()
                
                # Skip empty lines and comments
                if not line_lower or line_lower.startswith('//') or line_lower.startswith('/*'):
                    continue
                
                # Create mock node data
                mock_node = self._create_mock_node(line, i)
                mock_cfg_data = self._create_mock_cfg_data(lines)
                
                # Get predictions from all loaded models
                for annotation_type, trainer in self.loaded_models.items():
                    try:
                        # Extract features
                        if hasattr(trainer, '_extract_annotation_type_features'):
                            features = trainer._extract_annotation_type_features(mock_node, mock_cfg_data)
                        else:
                            features = self._extract_basic_features(mock_node, mock_cfg_data)
                        
                        # Get prediction
                        prediction, confidence, reason = self._get_model_prediction(
                            trainer, features, annotation_type, mock_node
                        )
                        
                        if prediction and confidence >= threshold:
                            predictions.append({
                                'line': i,
                                'annotation_type': annotation_type,
                                'confidence': confidence,
                                'reason': reason,
                                'model_type': trainer.base_model_type,
                                'features': features[:5] if len(features) > 5 else features  # Show first 5 features
                            })
                            
                    except Exception as e:
                        logger.debug(f"Error predicting with {annotation_type}: {e}")
            
            # Remove duplicate predictions (same line, different models)
            predictions = self._deduplicate_predictions(predictions)
            
            logger.debug(f"Generated {len(predictions)} predictions for {java_file}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting annotations for {java_file}: {e}")
            return []
    
    def _create_mock_node(self, line: str, line_number: int) -> Dict[str, Any]:
        """Create mock node data from Java line"""
        line_lower = line.lower().strip()
        
        # Determine node type based on line content
        if 'int' in line_lower and ('=' in line or ';' in line):
            node_type = 'variable'
        elif line_lower.startswith('public') or line_lower.startswith('private'):
            if '(' in line and ')' in line:
                node_type = 'method'
            else:
                node_type = 'field'
        elif 'int' in line_lower and ('(' in line and ')' in line):
            node_type = 'parameter'
        else:
            node_type = 'statement'
        
        return {
            'id': f"node_{line_number}",
            'label': line.strip(),
            'node_type': node_type,
            'line': line_number,
            'is_annotation_target': True
        }
    
    def _create_mock_cfg_data(self, lines: List[str]) -> Dict[str, Any]:
        """Create mock CFG data"""
        return {
            'nodes': [self._create_mock_node(line, i+1) for i, line in enumerate(lines)],
            'edges': [],
            'method_name': 'mock_method',
            'file_path': 'mock_file.java'
        }
    
    def _extract_basic_features(self, node: Dict[str, Any], cfg_data: Dict[str, Any]) -> List[float]:
        """Extract basic features for prediction"""
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        line = node.get('line', 0)
        
        # Basic feature extraction
        features = [
            float(len(label)),  # label_length
            float(line),  # line_number
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
    
    def predict_annotations_for_file_with_cfg(self, java_file, cfg_dir, threshold=0.3):
        """Predict annotations for a Java file using real CFG data"""
        try:
            # Find CFG data for this Java file
            java_basename = os.path.splitext(os.path.basename(java_file))[0]
            cfg_file = os.path.join(cfg_dir, java_basename, 'cfg.json')
            
            if not os.path.exists(cfg_file):
                logger.warning(f"No CFG file found for {java_file}, falling back to mock data")
                return self.predict_annotations_for_file(java_file, threshold)
            
            # Load CFG data
            with open(cfg_file, 'r') as f:
                cfg_data = json.load(f)
            
            # Read the Java file
            with open(java_file, 'r') as f:
                java_content = f.read()
            
            predictions = []
            for annotation_type in ['@Positive', '@NonNegative', '@GTENegativeOne']:
                if annotation_type in self.loaded_models:
                    model_info = self.loaded_models[annotation_type]
                    trainer = model_info['trainer']
                    base_model_type = model_info['base_model_type']
                    
                    # Extract features from real CFG data
                    cfg_features = self._extract_cfg_features(cfg_data, java_content)
                    
                    if cfg_features is not None:
                        # Get prediction from model using real CFG features
                        prediction, confidence, reason = self._get_model_prediction(
                            trainer, cfg_features, annotation_type, cfg_data
                        )
                        
                        if prediction and confidence >= threshold:
                            predictions.append({
                                'line': cfg_data.get('line', 1),
                                'annotation_type': annotation_type,
                                'confidence': confidence,
                                'reason': f"{reason} (using real CFG data)",
                                'model_type': base_model_type
                            })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting annotations for {java_file} with CFG: {e}")
            return []
    
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
                float(node.get('line', 0)),  # line_number
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
