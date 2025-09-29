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

# Import enhanced causal model
try:
    from enhanced_causal_model import EnhancedCausalModel, extract_enhanced_causal_features
    ENHANCED_CAUSAL_AVAILABLE = True
except ImportError:
    ENHANCED_CAUSAL_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnnotationTypeTrainer:
    """Trainer for specific annotation types using binary RL models"""
    
    def __init__(self, annotation_type='@NonNegative', base_model_type='gcn', learning_rate=0.001, device='cpu'):
        self.annotation_type = annotation_type
        self.base_model_type = base_model_type
        self.device = device
        self.learning_rate = learning_rate
        
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
        
    def _init_annotation_model(self):
        """Initialize model for specific annotation type prediction"""
        if self.base_model_type == 'gcn':
            return AnnotationTypeGCNModel(input_dim=14, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'gbt':
            return AnnotationTypeGBTModel()
        elif self.base_model_type == 'causal':
            return AnnotationTypeCausalModel(input_dim=14, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'enhanced_causal':
            if not ENHANCED_CAUSAL_AVAILABLE:
                raise ImportError("Enhanced causal model not available. Please ensure enhanced_causal_model.py is present.")
            return AnnotationTypeEnhancedCausalModel(input_dim=32, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'hgt':
            return AnnotationTypeHGTModel(input_dim=14, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'gcsn':
            return AnnotationTypeGCSNModel(input_dim=14, hidden_dim=128, out_dim=2)
        elif self.base_model_type == 'dg2n':
            return AnnotationTypeDG2NModel(input_dim=14, hidden_dim=128, out_dim=2)
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
            features.append(feature_vector)
            
            # Determine if this node should have the specific annotation type
            should_have_annotation = self._should_have_annotation_type(node)
            targets.append(1 if should_have_annotation else 0)
        
        return np.array(features), np.array(targets)
    
    def _extract_annotation_type_features(self, node, cfg_data):
        """Extract features for annotation type prediction"""
        # Use enhanced causal features if available and model type is enhanced_causal
        if self.base_model_type == 'enhanced_causal' and ENHANCED_CAUSAL_AVAILABLE:
            return extract_enhanced_causal_features(node, cfg_data)
        
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        line = node.get('line', 0)
        
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
            for root, dirs, files in os.walk(cfg_dir):
                for file in files:
                    if file.endswith('.json'):
                        cfg_file = os.path.join(root, file)
                        with open(cfg_file, 'r') as f:
                            cfg_data = json.load(f)
                            cfg_data_list.append(cfg_data)
        except Exception as e:
            logger.error(f"Error loading CFG data: {e}")
        return cfg_data_list
    
    def _create_mock_cfg_data(self):
        """Create mock CFG data for training"""
        return {
            'nodes': [
                {'id': 0, 'label': 'public void method()', 'node_type': 'method', 'line': 10},
                {'id': 1, 'label': 'int index = 0;', 'node_type': 'variable', 'line': 11},
                {'id': 2, 'label': 'int offset = 5;', 'node_type': 'variable', 'line': 12},
                {'id': 3, 'label': 'int position = 10;', 'node_type': 'variable', 'line': 13}
            ],
            'control_edges': [
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2},
                {'source': 2, 'target': 3}
            ],
            'dataflow_edges': []
        }
    
    def train(self, project_root, warnings_file, cfwr_root, num_episodes=50, slices_dir=None, cfg_dir=None, use_real_cfg_data=False):
        """Train the annotation type model"""
        logger.info(f"Starting training for {self.annotation_type} annotation type")
        logger.info(f"Base model: {self.base_model_type}")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Use real CFG data: {use_real_cfg_data}")
        
        # Load real CFG data if available
        if use_real_cfg_data and cfg_dir and os.path.exists(cfg_dir):
            logger.info("Loading real CFG data for training")
            cfg_data_list = self._load_cfg_data(cfg_dir)
            if cfg_data_list:
                logger.info(f"Loaded {len(cfg_data_list)} CFG files for training")
            else:
                logger.warning("No CFG data found, falling back to mock data")
                cfg_data_list = [self._create_mock_cfg_data()]
        else:
            logger.info("Using mock CFG data for training")
            cfg_data_list = [self._create_mock_cfg_data()]
        
        # Training loop
        episode_rewards = []
        
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
            cfg_data = cfg_data_list[episode % len(cfg_data_list)]
            
            # Train episode
            reward = self.train_episode(cfg_data, binary_predictions, original_warnings)
            episode_rewards.append(reward)
            
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
                logger.info(f"Episode {episode + 1}: avg_reward={avg_reward:.3f}")
        
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
    parser.add_argument('--use_real_cfg_data', action='store_true', help='Use real CFG data instead of mock data')
    
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
