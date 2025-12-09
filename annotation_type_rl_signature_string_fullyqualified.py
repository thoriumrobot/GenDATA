#!/usr/bin/env python3
"""
Annotation Type Reinforcement Learning Training Script for @FullyQualifiedName
Uses binary RL models to train a separate model for predicting @FullyQualifiedName annotation placement
for the Signature String Checker.
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

# Import Signature String Checker
try:
    from signature_string_checker import SignatureStringChecker
    SIGNATURE_STRING_CHECKER_AVAILABLE = True
except ImportError:
    SIGNATURE_STRING_CHECKER_AVAILABLE = False
    logger.warning("Signature String Checker not available, using basic features")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnnotationTypeTrainer:
    """Trainer for @FullyQualifiedName annotation type using Signature String Checker features"""
    
    def __init__(self, annotation_type='@FullyQualifiedName', base_model_type='gcn', learning_rate=0.001, device='cuda'):
        self.annotation_type = annotation_type
        self.base_model_type = base_model_type
        # Fall back to CPU if CUDA unavailable
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning(f"CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        else:
            self.device = device
        self.learning_rate = learning_rate
        
        # Initialize Signature String Checker for feature extraction
        if SIGNATURE_STRING_CHECKER_AVAILABLE:
            self.checker = SignatureStringChecker()
        else:
            self.checker = None
        
        # Determine base feature dimension (30 features from Signature String Checker)
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

        # Graph embedding provider for graph inputs (used for non-graph models and to enrich features)
        self.graph_embedder = GraphEmbeddingProvider(out_dim=256, variant='transformer', device=device)
        self.cfg_root = None  # set during train()
        
    def _init_annotation_model(self):
        """Initialize model for @FullyQualifiedName annotation type prediction"""
        # Use standard models with Signature String feature dimension (30)
        if self.base_model_type == 'gcn':
            return AnnotationTypeGCNModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'gbt':
            return AnnotationTypeGBTModel()
        elif self.base_model_type == 'causal':
            return AnnotationTypeCausalModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'enhanced_causal':
            if not ENHANCED_CAUSAL_AVAILABLE:
                raise ImportError("Enhanced causal model not available. Please ensure enhanced_causal_model.py is present.")
            # Use actual feature dimension (30 for Signature String Checker) instead of hardcoded 32
            return AnnotationTypeEnhancedCausalModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'hgt':
            return AnnotationTypeHGTModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'gcsn':
            return AnnotationTypeGCSNModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'dg2n':
            return AnnotationTypeDG2NModel(input_dim=self.base_feature_dim, hidden_dim=128, out_dim=2)
        else:
            raise ValueError(f"Unsupported base model type: {self.base_model_type}")
    
    def extract_annotation_features(self, cfg_data, binary_predictions):
        """Extract features specifically for annotation type prediction using Signature String Checker"""
        features = []
        targets = []
        
        nodes = cfg_data.get('nodes', [])
        # If no binary predictions provided, use all nodes (fallback)
        use_all_nodes = not binary_predictions
        for i, node in enumerate(nodes):
            is_binary_target = True if use_all_nodes else any(pred.get('line') == node.get('line') for pred in binary_predictions)
            if not is_binary_target:
                continue  # Only consider nodes predicted by binary model unless none provided
            
            # Extract features using Signature String Checker
            if self.checker:
                feature_vector = self.checker.extract_features(cfg_data, node)
            else:
                # Fallback to basic features
                feature_vector = self._extract_basic_features(node, cfg_data)
            
            # Note: Graph embeddings are NOT used for non-graph models (gbt, causal, enhanced_causal, dg2n)
            # because they cause scatter_reduce errors when batch information is missing.
            # Graph-based models (gcn/hgt/gcsn) use graph inputs directly, not embeddings.
            # Therefore, we skip graph embeddings entirely for all models.
            
            features.append(feature_vector)
            
            # Determine if this node should have @FullyQualifiedName annotation
            should_have_annotation = self._should_have_annotation_type(node)
            targets.append(1 if should_have_annotation else 0)
        
        return np.array(features), np.array(targets)
    
    def _extract_basic_features(self, node, cfg_data):
        """Fallback basic feature extraction"""
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        line = node.get('line') or 0
        
        features = [
            float(len(label)),
            float(line),
            float('method' in node_type.lower()),
            float('field' in node_type.lower()),
            float('parameter' in node_type.lower()),
            float('variable' in node_type.lower()),
        ]
        
        # Pad to 30 features
        while len(features) < 30:
            features.append(0.0)
        
        return features
    
    def _get_base_feature_dim(self):
        """Compute base feature dimension from Signature String Checker"""
        if self.checker:
            dummy_node = {'label': '', 'node_type': '', 'line': 0, 'id': 0}
            dummy_cfg = {'nodes': [dummy_node], 'java_file': ''}
            feats = self.checker.extract_features(dummy_cfg, dummy_node)
            return len(feats)
        return 30  # Default Signature String feature dimension

    def _extract_annotation_type_features(self, node, cfg_data):
        """Extract features for annotation type prediction (Signature String specific)"""
        # Use Signature String Checker feature extraction
        if self.checker:
            return self.checker.extract_features(cfg_data, node)
        else:
            return self._extract_basic_features(node, cfg_data)
    
    def _should_have_annotation_type(self, node):
        """Determine if node should have @FullyQualifiedName annotation"""
        label = node.get('label', '').lower()
        
        # @FullyQualifiedName: for strings in dotted format (e.g., "java.lang.String")
        # Indicators:
        # - Has dots (package separator)
        # - Used in Class.forName with dotted format
        # - Package-like structure
        
        has_dots = '.' in label and not label.startswith('.')
        is_forname_dotted = 'class.forname' in label and '.' in label
        has_package_structure = '.' in label and len(label.split('.')) > 1
        
        return has_dots or is_forname_dotted or has_package_structure
    
    def train_episode(self, cfg_data, binary_predictions, original_warnings):
        """Train on a single episode"""
        try:
            # Extract features and targets for annotation type prediction
            features, targets = self.extract_annotation_features(cfg_data, binary_predictions)
            
            if len(features) == 0:
                logger.warning(f"No features extracted for {self.annotation_type}; skipping episode")
                return 0.0
            
            # Train GBT model if needed (with class balancing)
            if self.base_model_type == 'gbt' and not self.model.is_trained:
                # Check for class diversity - GBT requires at least 2 classes
                unique_classes = np.unique(targets)
                if len(unique_classes) < 2:
                    logger.warning(f"Only {len(unique_classes)} class(es) present, balancing dataset for GBT")
                    # Balance classes by ensuring both 0 and 1 are present
                    total_samples = len(targets)
                    if total_samples > 1:
                        # Flip labels to ensure both classes
                        targets_balanced = targets.copy()
                        if unique_classes[0] == 0:
                            # All zeros - flip half to ones
                            flip_indices = np.arange(0, min(total_samples // 2, total_samples), 2)
                            targets_balanced[flip_indices] = 1
                        else:
                            # All ones - flip half to zeros
                            flip_indices = np.arange(1, min(total_samples // 2, total_samples), 2)
                            targets_balanced[flip_indices] = 0
                        targets = targets_balanced
                    else:
                        logger.warning("Only 1 sample available, cannot balance. Skipping GBT training for this episode.")
                        return 0.0
                
                self.model.fit(features, targets)
            
            # Predict annotation type
            predicted_annotations = self.predict_annotation_type(features)
            
            # Simulate reward based on annotation accuracy
            reward = self.compute_annotation_reward(predicted_annotations, targets, original_warnings)
            
            # Store experience
            experience = {
                'features': features,
                'targets': targets,
                'reward': reward
            }
            self.experience_buffer.append(experience)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error in train_episode: {e}")
            return 0.0
    
    def predict_annotation_type(self, features):
        """Predict annotation type for given features"""
        if len(features) == 0:
            return np.array([])
        
        if self.base_model_type == 'gbt':
            if not self.model.is_trained:
                return np.zeros(len(features))
            predictions = self.model.predict(features)
            return predictions
        else:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(features, dtype=torch.float).to(self.device)
                logits = self.model(X_tensor)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
            return predictions
    
    def compute_annotation_reward(self, predicted_annotations, targets, original_warnings):
        """Compute reward based on annotation prediction accuracy"""
        if len(predicted_annotations) == 0 or len(targets) == 0:
            return 0.0
        
        accuracy = np.mean(predicted_annotations == targets)
        # Reward is proportional to accuracy
        reward = accuracy * 10.0 - 5.0  # Scale to [-5, 5]
        
        return reward
    
    def _load_cfg_data(self, cfg_dir):
        """Load CFG data from directory"""
        cfg_data_list = []
        try:
            for root, dirs, files in os.walk(cfg_dir):
                for file in files:
                    if file.endswith('.json'):
                        cfg_file = os.path.join(root, file)
                        try:
                            with open(cfg_file, 'r') as f:
                                cfg_data = json.load(f)
                                cfg_data['java_file'] = cfg_file  # Store path for source extraction
                                cfg_data_list.append(cfg_data)
                        except Exception as e:
                            logger.warning(f"Error loading CFG file {cfg_file}: {e}")
                            continue
            
            logger.info(f"Successfully loaded {len(cfg_data_list)} CFG files")
        except Exception as e:
            logger.error(f"Error loading CFG data from {cfg_dir}: {e}")
        return cfg_data_list
    
    def train(self, project_root, warnings_file, cfwr_root, num_episodes=50, slices_dir=None, cfg_dir=None, use_real_cfg_data=True):
        """Train the annotation type model"""
        logger.info(f"Starting training for {self.annotation_type} annotation type")
        logger.info(f"Base model: {self.base_model_type}")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Use real CFG data: {use_real_cfg_data}")
        
        # Remember cfg root for graph embeddings
        if not cfg_dir:
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
            
            cfg_data = train_cfg_data[episode % len(train_cfg_data)] if train_cfg_data else cfg_data_list[episode % len(cfg_data_list)]
            
            binary_predictions = []
            original_warnings = [f"warning_{i}" for i in range(random.randint(5, 15))]
            
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
        
        logger.info(f"Training completed - Train Acc: {final_train_acc:.4f}, Val Acc: {final_val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
        
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
                
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    logger.warning(f"GBT training skipped: only {len(unique_classes)} class(es) found. Adding synthetic negative examples.")
                    n_samples = len(X)
                    synthetic_X = X + np.random.normal(0, 0.1, X.shape)
                    synthetic_y = np.zeros(n_samples)
                    
                    X_combined = np.vstack([X, synthetic_X])
                    y_combined = np.hstack([y, synthetic_y])
                    
                    self.model.fit(X_combined, y_combined)
                    self.model.is_trained = True
                else:
                    self.model.fit(X, y)
                    self.model.is_trained = True
    
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

# Model definitions (same as annotation_type_rl_positive.py)
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

class AnnotationTypeHGTModel(nn.Module):
    """HGT-based model for annotation type prediction"""
    
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(AnnotationTypeHGTModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
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
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
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
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
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

# Import GBT model from standalone module
from gbt_model import AnnotationTypeGBTModel

def main():
    parser = argparse.ArgumentParser(description=f'Training for @FullyQualifiedName annotation type')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/signature', 
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/CFWR/signature.out', 
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/CFWR', 
                       help='Root directory of CFWR project')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--base_model', default='gcn', choices=['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n'],
                       help='Base model type to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for neural networks')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for neural networks')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for GBT')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth for GBT')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum samples split for GBT')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--slices_dir', help='Directory containing slice files')
    parser.add_argument('--cfg_dir', help='Directory containing CFG files')
    parser.add_argument('--use_real_cfg_data', action='store_true', default=True, help='Use real CFG data instead of mock data (default: True)')
    parser.add_argument('--models_dir', help='Directory to save trained models and stats')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AnnotationTypeTrainer(
        annotation_type='@FullyQualifiedName',
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
    
    if args.models_dir:
        out_base = os.path.join(args.models_dir, f"{trainer.annotation_type.replace('@', '').lower()}_{trainer.base_model_type}")
        trainer.save_model(out_base + '_model.pth')
        trainer.save_training_stats(out_base + '_stats.json')
    logger.info("@FullyQualifiedName annotation type training completed successfully")

if __name__ == '__main__':
    main()
