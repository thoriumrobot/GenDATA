#!/usr/bin/env python3
"""
Annotation Type Reinforcement Learning Training Script for @NonNegative
Uses binary RL models to train a separate model for predicting @NonNegative annotation placement.
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
from pathlib import Path
import time
import logging
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from annotation_graph_input import GraphEmbeddingProvider

# Import enhanced causal model
try:
    from enhanced_causal_model import EnhancedCausalModel, extract_enhanced_causal_features
    ENHANCED_CAUSAL_AVAILABLE = True
except ImportError:
    ENHANCED_CAUSAL_AVAILABLE = False

# Import checker-specific modules
try:
    from checker_config import CheckerType
    from checker_specific_models import create_checker_specific_model
    CHECKER_MODULES_AVAILABLE = True
except ImportError:
    CHECKER_MODULES_AVAILABLE = False
    CheckerType = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnnotationTypeTrainer:
    """Trainer for specific annotation types using binary RL models"""
    
    def __init__(self, annotation_type='@NonNegative', base_model_type='gcn', learning_rate=0.001, device='cuda', checker_type=None):
        self.annotation_type = annotation_type
        self.base_model_type = base_model_type
        self.checker_type = checker_type
        # Fall back to CPU if CUDA unavailable
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning(f"CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = device
        self.learning_rate = learning_rate
        
        # Determine base feature dimension for this model (without graph embeddings)
        self.base_feature_dim = self._get_base_feature_dim()
        # Initialize the annotation-specific model
        self.model = self._init_annotation_model()
        
        # Only create optimizer for PyTorch models (not GBT)
        if hasattr(self.model, 'parameters'):
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.optimizer = None
            self.criterion = None
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'annotation_predictions': [],
            'accuracy': []
        }
        self.graph_embedder = GraphEmbeddingProvider(out_dim=256, variant='transformer', device=device)
        self.cfg_root = None
        
    def _init_annotation_model(self):
        """Initialize model for specific annotation type prediction"""
        # Use checker-specific model if available and checker type is specified
        if CHECKER_MODULES_AVAILABLE and self.checker_type is not None:
            base_input_dim = 14
            from checker_config import get_checker_config
            config = get_checker_config(self.checker_type)
            pattern_dim = len(config.get('value_patterns', []))
            input_dim = base_input_dim + pattern_dim
            
            return create_checker_specific_model(
                checker_type=self.checker_type,
                base_model_type=self.base_model_type,
                input_dim=input_dim,
                hidden_dim=128,
                out_dim=2
            )
        
        # Use standard models
        if self.base_model_type == 'gcn':
            return AnnotationTypeGCNModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'gbt':
            return AnnotationTypeGBTModel()
        elif self.base_model_type == 'causal':
            return AnnotationTypeCausalModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'enhanced_causal':
            if not ENHANCED_CAUSAL_AVAILABLE:
                raise ImportError("Enhanced causal model not available. Please ensure enhanced_causal_model.py is present.")
            return AnnotationTypeEnhancedCausalModel(input_dim=32, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'hgt':
            return AnnotationTypeHGTModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'gcsn':
            return AnnotationTypeGCSNModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'dg2n':
            return AnnotationTypeDG2NModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        else:
            raise ValueError(f"Unsupported base model type: {self.base_model_type}")
    
    def extract_annotation_features(self, cfg_data, binary_predictions):
        """Extract features specifically for annotation type prediction"""
        features = []
        targets = []
        
        nodes = cfg_data.get('nodes', [])
        for i, node in enumerate(nodes):
            # Check if this node was predicted by binary model
            is_binary_target = any(pred['line'] == node.get('line') for pred in binary_predictions)
            
            if not is_binary_target:
                continue  # Only consider nodes predicted by binary model
            
            # Extract features for annotation type prediction
            feature_vector = self._extract_annotation_type_features(node, cfg_data)
            java_base = os.path.splitext(os.path.basename(cfg_data.get('java_file','') or ''))[0]
            cfg_dir = None
            if self.cfg_root and java_base:
                cfg_dir = os.path.join(self.cfg_root, java_base)
            if not cfg_dir or not os.path.isdir(cfg_dir):
                env_root = os.environ.get('CFG_OUTPUT_DIR') or os.environ.get('PREDICTION_CFG_DIR') or 'prediction_cfg_output'
                cfg_dir = os.path.join(env_root, java_base) if java_base else ''
            # Only append graph embeddings for models that expect extended feature vectors.
            # Graph-based models (gcn/hgt/gcsn) use input_dim=14; adding 256 dims breaks them.
            if self.base_model_type not in ['gcn', 'hgt', 'gcsn']:
                if os.path.isdir(cfg_dir):
                    emb = self.graph_embedder.embed_cfg_dir(cfg_dir)
                    feature_vector = np.concatenate([feature_vector, emb.cpu().numpy()])
            features.append(feature_vector)
            
            # Determine if this node should have the specific annotation type
            should_have_annotation = self._should_have_annotation_type(node)
            targets.append(1 if should_have_annotation else 0)
        
        return np.array(features), np.array(targets)
    
    def _get_base_feature_dim(self):
        """Compute base feature dimension from a dummy node (no embeddings)."""
        dummy_node = {'label': '', 'node_type': '', 'line': 0, 'id': 0}
        dummy_cfg = {'nodes': [dummy_node]}
        feats = self._extract_annotation_type_features(dummy_node, dummy_cfg)
        return len(feats)

    def _extract_annotation_type_features(self, node, cfg_data):
        """Extract features for annotation type prediction"""
        # Use enhanced causal features if available and model type is enhanced_causal
        if self.base_model_type == 'enhanced_causal' and ENHANCED_CAUSAL_AVAILABLE:
            return extract_enhanced_causal_features(node, cfg_data)
        
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        line = node.get('line') or 0
        
        # Features specific to annotation type prediction
        features = [
            float(len(label)),  # label_length
            float(line),  # line_number
            float('method' in node_type.lower()),  # is_method
            float('field' in node_type.lower()),  # is_field
            float('parameter' in node_type.lower()),  # is_parameter
            float('variable' in node_type.lower()),  # is_variable
            float('positive' in label.lower()),  # contains_positive
            float('negative' in label.lower()),  # contains_negative
        ]
        
        # Add annotation-specific features for @NonNegative
        features.extend([
            float('index' in label.lower()),  # is_index_variable
            float('offset' in label.lower()),  # is_offset_variable
            float('>=' in label),  # has_greater_equal
            float('loop' in label.lower()),  # is_loop_related
            float('array' in label.lower()),  # is_array_related
            float('for' in label.lower()),  # is_for_loop
        ])
        
        # "Could be zero" detection patterns (strong signal for @NonNegative)
        label_lower = label.lower()
        nodes = cfg_data.get('nodes', [])
        current_idx = next((i for i, n in enumerate(nodes) if n.get('id') == node.get('id')), -1)
        
        # Pattern 1: Array index usage (indices can be 0)
        is_used_as_array_index = (
            ('[' in label or ']' in label or 'array[' in label_lower or 'list[' in label_lower) and
            any(var in label_lower for var in ['index', 'i', 'j', 'k', 'idx', 'pos'])
        )
        
        # Pattern 2: Loop iteration variable (often start at 0)
        is_loop_variable = (
            any(pattern in label_lower for pattern in ['for', 'while', 'iterator', 'iter', 'loop']) and
            any(var in label_lower for var in ['i', 'j', 'k', 'idx', 'index', 'counter'])
        )
        
        # Pattern 3: Subtraction result that could be 0
        is_subtraction_result = any(pattern in label_lower for pattern in [
            ' - ', '- ', 'length -', 'size -', 'count -', '.length -', '.size -'
        ])
        
        # Pattern 4: Parameter used in array access context
        is_param_in_array_context = False
        if 'parameter' in node_type.lower() and current_idx >= 0:
            for offset in [-3, -2, -1, 1, 2, 3]:
                idx = current_idx + offset
                if 0 <= idx < len(nodes):
                    nearby_label = nodes[idx].get('label', '').lower()
                    if '[' in nearby_label and ']' in nearby_label:
                        is_param_in_array_context = True
                        break
        
        # Pattern 5: Comparison with length/size
        compared_with_length = any(pattern in label_lower for pattern in [
            '< length', '< size', '<= length', '<= size',
            'length >', 'size >', 'length >=', 'size >=',
            '.length', '.size()'
        ])
        
        # Pattern 6: Initialization to 0
        initialized_to_zero = any(pattern in label_lower for pattern in [
            '= 0', '=0', ':= 0', ':=0', 'equals 0', 'equals zero', 'zero'
        ])
        
        # Pattern 7: Used in >= 0 check
        used_in_nonnegative_check = any(pattern in label_lower for pattern in ['>= 0', '>=0', '>= -1', '>=-1'])
        
        # Pattern 8: Offset/position variable
        is_offset_or_position = any(pattern in label_lower for pattern in [
            'offset', 'position', 'pos', 'start', 'begin', 'beginning'
        ])
        
        # Aggregated "could be zero" score
        could_be_zero_indicators = [
            is_used_as_array_index, is_loop_variable, is_subtraction_result,
            is_param_in_array_context, compared_with_length, initialized_to_zero,
            used_in_nonnegative_check, is_offset_or_position
        ]
        could_be_zero_score = sum(could_be_zero_indicators) / max(len(could_be_zero_indicators), 1)
        
        # Add "could be zero" features (strong signal for @NonNegative)
        features.extend([
            float(is_used_as_array_index) * 2.0,
            float(is_loop_variable) * 2.0,
            float(is_subtraction_result) * 1.5,
            float(is_param_in_array_context) * 2.0,
            float(compared_with_length) * 1.5,
            float(initialized_to_zero) * 2.0,
            float(used_in_nonnegative_check) * 2.0,
            float(is_offset_or_position) * 1.5,
            float(could_be_zero_score) * 3.0,  # Aggregated score, highly emphasized
        ])
        
        return features
    
    def _should_have_annotation_type(self, node):
        """Determine if node should have the specific annotation type"""
        label = node.get('label', '').lower()
        
        # @NonNegative: for values that must be >= 0
        nonnegative_indicators = ['index', 'offset', 'position', 'loop', 'i', 'j', 'k']
        return any(indicator in label for indicator in nonnegative_indicators)
    
    def train_episode(self, cfg_data, binary_predictions, original_warnings):
        """Train on a single episode"""
        try:
            # Extract features and targets for annotation type prediction
            features, targets = self.extract_annotation_features(cfg_data, binary_predictions)
            
            if len(features) == 0:
                logger.info(f"No features extracted for {self.annotation_type}")
                return 0.0
            
            # Train GBT model if needed
            if self.base_model_type == 'gbt' and not self.model.is_trained:
                self.model.fit(features, targets)
            
            # Predict annotation type
            predicted_annotations = self.predict_annotation_type(features)
            
            # Simulate reward based on annotation accuracy
            reward = self.compute_annotation_reward(predicted_annotations, targets, original_warnings)
            
            # Store experience
            experience = {
                'features': features,
                'targets': targets,
                'predicted_annotations': predicted_annotations,
                'reward': reward
            }
            self.experience_buffer.append(experience)
            
            logger.info(f"Episode completed: {self.annotation_type} reward={reward:.3f}, predictions={len(predicted_annotations)}")
            return reward
            
        except Exception as e:
            logger.error(f"Error in training episode: {e}")
            return 0.0
    
    def predict_annotation_type(self, features):
        """Predict annotation type for given features"""
        if self.base_model_type in ['gcn', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']:
            self.model.eval()
            with torch.no_grad():
                X = torch.tensor(features, dtype=torch.float).to(self.device)
                logits = self.model(X)
                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities[:, 1] > 0.5  # Class 1 = needs annotation
                return predictions.cpu().numpy()
        else:  # GBT
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)
                predictions = probabilities[:, 1] > 0.5
                return predictions
            else:
                # Model not trained yet
                return np.zeros(len(features), dtype=bool)
    
    def compute_annotation_reward(self, predictions, targets, original_warnings):
        """Compute reward based on annotation type prediction accuracy"""
        if len(predictions) == 0:
            return 0.0
        
        # Accuracy-based reward
        correct_predictions = np.sum(predictions == targets)
        accuracy = correct_predictions / len(predictions)
        
        # Bonus for predicting positive cases (more important for annotation placement)
        positive_cases = np.sum(targets == 1)
        if positive_cases > 0:
            positive_accuracy = np.sum(predictions[targets == 1] == 1) / positive_cases
            accuracy = 0.7 * accuracy + 0.3 * positive_accuracy
        
        # Reward based on warning reduction (simulate)
        warning_reduction = random.uniform(0.1, 0.3) if accuracy > 0.7 else random.uniform(-0.1, 0.1)
        
        return accuracy + warning_reduction
    
    def _load_cfg_data(self, cfg_dir):
        """Load CFG data from files"""
        cfg_data_list = []
        try:
            if not os.path.exists(cfg_dir):
                logger.error(f"CFG directory does not exist: {cfg_dir}")
                return cfg_data_list
            
            # Count files first
            json_files = list(Path(cfg_dir).rglob('*.json'))
            logger.info(f"Found {len(json_files)} CFG JSON files in {cfg_dir}")
            
            for cfg_file in json_files:
                try:
                    with open(cfg_file, 'r') as f:
                        cfg_data = json.load(f)
                        cfg_data_list.append(cfg_data)
                except Exception as e:
                    logger.warning(f"Error loading CFG file {cfg_file}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(cfg_data_list)} CFG files")
        except Exception as e:
            logger.error(f"Error loading CFG data from {cfg_dir}: {e}")
        return cfg_data_list
    
    def _create_mock_cfg_data(self):
        """Deprecated: mock CFG data generation (removed - use real pipeline data)"""
        logger.warning("Mock CFG data generation is deprecated. Use real pipeline data instead.")
        return {'nodes': [], 'edges': []}
    
    def train(self, project_root, warnings_file, cfwr_root, num_episodes=50, slices_dir=None, cfg_dir=None, use_real_cfg_data=True):
        """Train the annotation type model"""
        logger.info(f"Starting training for {self.annotation_type} annotation type")
        logger.info(f"Base model: {self.base_model_type}")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Use real CFG data: {use_real_cfg_data}")
        
        # Try to get CFG directory from various sources
        if not cfg_dir:
            # Try environment variable
            cfg_dir = os.environ.get('CFG_OUTPUT_DIR') or os.environ.get('PREDICTION_CFG_DIR')
        
        self.cfg_root = cfg_dir if cfg_dir and os.path.exists(cfg_dir) else None

        # Load real CFG data if available
        if use_real_cfg_data:
            if cfg_dir and os.path.exists(cfg_dir):
                logger.info(f"Loading real CFG data for training from {cfg_dir}")
                cfg_data_list = self._load_cfg_data(cfg_dir)
                if cfg_data_list:
                    logger.info(f"Loaded {len(cfg_data_list)} CFG files for training")
                else:
                    logger.error(f"No CFG data found in {cfg_dir}. Please run the pipeline first to generate CFG data.")
                    return
            else:
                logger.error(f"CFG directory not found: {cfg_dir}. Please provide --cfg_dir or set CFG_OUTPUT_DIR environment variable.")
                return
        else:
            logger.error("Mock CFG data is deprecated. Please use real CFG data with --use_real_cfg_data (default).")
            return
        
        # Training loop
        episode_rewards = []
        all_train_accuracies = []
        all_val_accuracies = []
        
        # Split CFG data into train/val (80/20)
        split_idx = int(len(cfg_data_list) * 0.8)
        train_cfg_data = cfg_data_list[:split_idx]
        val_cfg_data = cfg_data_list[split_idx:] if split_idx < len(cfg_data_list) else []
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            # Simulate binary predictions (from binary RL model)
            binary_predictions = [
                {'line': 11, 'confidence': 0.8},
                {'line': 12, 'confidence': 0.7},
                {'line': 13, 'confidence': 0.9}
            ]
            
            # Simulate original warnings
            original_warnings = [f"warning_{i}" for i in range(random.randint(5, 15))]
            
            # Use real CFG data or mock data
            cfg_data = train_cfg_data[episode % len(train_cfg_data)] if train_cfg_data else cfg_data_list[episode % len(cfg_data_list)]
            
            # Train episode
            reward = self.train_episode(cfg_data, binary_predictions, original_warnings)
            episode_rewards.append(reward)
            
            # Compute training accuracy for this episode
            features, targets = self.extract_annotation_features(cfg_data, binary_predictions)
            if len(features) > 0:
                predictions = self.predict_annotation_type(features)
                if len(predictions) > 0 and len(targets) > 0:
                    train_acc = np.mean(predictions == targets)
                    all_train_accuracies.append(train_acc)
            
            # Compute validation accuracy periodically
            if val_cfg_data and (episode + 1) % 5 == 0:
                val_cfg = val_cfg_data[episode % len(val_cfg_data)]
                val_features, val_targets = self.extract_annotation_features(val_cfg, binary_predictions)
                if len(val_features) > 0:
                    val_predictions = self.predict_annotation_type(val_features)
                    if len(val_predictions) > 0 and len(val_targets) > 0:
                        val_acc = np.mean(val_predictions == val_targets)
                        all_val_accuracies.append(val_acc)
            
            # Update training statistics
            self.training_stats['episodes'].append(episode + 1)
            self.training_stats['rewards'].append(reward)
            self.training_stats['annotation_predictions'].append(len(binary_predictions))
            
            # Experience replay training (every 10 episodes)
            if len(self.experience_buffer) >= 16 and (episode + 1) % 10 == 0:
                self._train_from_experience(batch_size=16)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_train_acc = np.mean(all_train_accuracies[-10:]) if all_train_accuracies else 0.0
                avg_val_acc = np.mean(all_val_accuracies[-5:]) if all_val_accuracies else 0.0
                logger.info(f"Episode {episode + 1}: avg_reward={avg_reward:.3f}, Train Acc={avg_train_acc:.3f}, Val Acc={avg_val_acc:.3f}")
        
        # Compute final metrics
        final_train_acc = np.mean(all_train_accuracies) if all_train_accuracies else 0.0
        final_val_acc = np.mean(all_val_accuracies) if all_val_accuracies else 0.0
        best_val_acc = max(all_val_accuracies) if all_val_accuracies else 0.0
        
        # Log final metrics
        logger.info(f"Training completed - Train Acc: {final_train_acc:.4f}, Val Acc: {final_val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
        logger.info(f"Best validation accuracy: {best_val_acc * 100:.2f} percent")
        
        # Update training stats with accuracy metrics
        self.training_stats['train_accuracy'] = float(final_train_acc)
        self.training_stats['val_accuracy'] = float(final_val_acc)
        self.training_stats['best_val_accuracy'] = float(best_val_acc)
        
        # Save model and training statistics
        self.save_model(f'models_annotation_types/{self.annotation_type.replace("@", "").lower()}_{self.base_model_type}_model.pth')
        self.save_training_stats(f'models_annotation_types/{self.annotation_type.replace("@", "").lower()}_{self.base_model_type}_stats.json')
        
        logger.info(f"{self.annotation_type} annotation type training completed")
        return self.training_stats
    
    def _train_from_experience(self, batch_size):
        """Train model using experience replay"""
        if len(self.experience_buffer) < batch_size:
            return
        
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        if self.base_model_type in ['gcn', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']:
            all_features = []
            all_labels = []
            
            for experience in batch:
                all_features.append(experience['features'])
                all_labels.append(experience['targets'])
            
            if all_features:
                X = np.vstack(all_features)
                y = np.hstack(all_labels)
                
                X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
                y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
                
                self.model.train()
                self.optimizer.zero_grad()
                logits = self.model(X_tensor)
                loss = self.criterion(logits, y_tensor)
                loss.backward()
                self.optimizer.step()
                
                logger.info(f"Experience replay training: loss={loss.item():.4f}")
        else:  # GBT
            all_features = []
            all_labels = []
            
            for experience in batch:
                all_features.append(experience['features'])
                all_labels.append(experience['targets'])
            
            if all_features:
                X = np.vstack(all_features)
                y = np.hstack(all_labels)
                
                # Check for class diversity for GBT models
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    logger.warning(f"GBT training skipped: only {len(unique_classes)} class(es) found. Adding synthetic negative examples.")
                    # Add synthetic negative examples to ensure class diversity
                    n_samples = len(X)
                    synthetic_X = X + np.random.normal(0, 0.1, X.shape)  # Add noise
                    synthetic_y = np.zeros(n_samples)  # All negative class
                    
                    # Combine original and synthetic data
                    X_combined = np.vstack([X, synthetic_X])
                    y_combined = np.hstack([y, synthetic_y])
                    
                    self.model.fit(X_combined, y_combined)
                    self.model.is_trained = True
                    logger.info("GBT experience replay training completed with synthetic data")
                else:
                    self.model.fit(X, y)
                    self.model.is_trained = True
                    logger.info("GBT experience replay training completed")
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.base_model_type == 'gbt':
            joblib.dump({
                'model': self.model,
                'annotation_type': self.annotation_type,
                'training_stats': self.training_stats
            }, filepath)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'annotation_type': self.annotation_type,
                'training_stats': self.training_stats
            }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def save_training_stats(self, filepath):
        """Save training statistics"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info(f"Training stats saved to {filepath}")

class AnnotationTypeGCNModel(nn.Module):
    """Neural network model for annotation type prediction"""
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

class AnnotationTypeCausalModel(nn.Module):
    """Causal model for annotation type prediction"""
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, out_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

class AnnotationTypeEnhancedCausalModel(nn.Module):
    """Enhanced causal model for annotation type prediction"""
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, out_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# Import GBT model from standalone module to avoid pickle issues
from gbt_model import AnnotationTypeGBTModel

def main():
    parser = argparse.ArgumentParser(description=f'Training for @NonNegative annotation type')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index', 
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/CFWR/index1.small.out', 
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/CFWR', 
                       help='Root directory of CFWR project')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--base_model', default='gcn', choices=['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n'],
                       help='Base model type to use')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--slices_dir', help='Directory containing slice files')
    parser.add_argument('--cfg_dir', help='Directory containing CFG files')
    parser.add_argument('--use_real_cfg_data', action='store_true', default=True, help='Use real CFG data instead of mock data (default: True)')
    parser.add_argument('--models_dir', help='Directory to save trained models and stats')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AnnotationTypeTrainer(
        annotation_type='@NonNegative',
        base_model_type=args.base_model,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Train the model
    stats = trainer.train(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        num_episodes=args.episodes,
        slices_dir=args.slices_dir,
        cfg_dir=args.cfg_dir,
        use_real_cfg_data=args.use_real_cfg_data
    )
    
    # Save outputs to provided models_dir if specified
    if args.models_dir:
        out_base = os.path.join(args.models_dir, f"{trainer.annotation_type.replace('@', '').lower()}_{trainer.base_model_type}")
        trainer.save_model(out_base + '_model.pth')
        trainer.save_training_stats(out_base + '_stats.json')
    logger.info("@NonNegative annotation type training completed successfully")

class AnnotationTypeHGTModel(nn.Module):
    """HGT-based model for annotation type prediction"""
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(AnnotationTypeHGTModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Enhanced feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

class AnnotationTypeGCSNModel(nn.Module):
    """GCSN-based model for annotation type prediction"""
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(AnnotationTypeGCSNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # GCSN-style feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

class AnnotationTypeDG2NModel(nn.Module):
    """DG2N-based model for annotation type prediction"""
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(AnnotationTypeDG2NModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # DG2N-style feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

if __name__ == '__main__':
    main()
