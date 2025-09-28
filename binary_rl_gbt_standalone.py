#!/usr/bin/env python3
"""
Standalone Binary Reinforcement Learning Training Script for GBT Model
Focuses only on predicting whether an annotation needs to be placed on fields, methods, or parameters.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockCheckerFrameworkEvaluator:
    """Mock Checker Framework evaluator for testing"""
    
    def evaluate_file(self, java_file, checker_type):
        """Mock evaluation that returns simulated results"""
        # Simulate some warning reduction
        original_count = random.randint(5, 15)
        new_count = max(0, original_count - random.randint(0, 3))
        
        return MockEvaluationResult(
            original_warnings=[f"warning_{i}" for i in range(original_count)],
            new_warnings=[f"warning_{i}" for i in range(new_count)],
            warning_count_change=original_count - new_count,
            success=True,
            error_message=""
        )

class MockEvaluationResult:
    """Mock evaluation result"""
    def __init__(self, original_warnings, new_warnings, warning_count_change, success, error_message):
        self.original_warnings = original_warnings
        self.new_warnings = new_warnings
        self.warning_count_change = warning_count_change
        self.success = success
        self.error_message = error_message

class MockAnnotationPlacementManager:
    """Mock annotation placement manager for testing"""
    
    def __init__(self, java_file):
        self.java_file = java_file
    
    def place_annotations(self, line_numbers, annotation_category):
        """Mock annotation placement that always succeeds"""
        logger.info(f"Mock annotation placement on lines {line_numbers} for category {annotation_category}")
        return True

class BinaryGBTTrainer:
    """Binary RL trainer for GBT-based annotation placement prediction"""
    
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=2, device='cpu', checker_type='index'):
        self.device = device
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.checker_type = checker_type
        
        # Initialize GBT classifier for binary classification
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        # Experience replay buffer for better learning
        self.experience_buffer = deque(maxlen=1000)
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'warning_changes': [],
            'accuracy': [],
            'binary_predictions': []
        }
        
        # Mock Checker Framework evaluator
        self.evaluator = MockCheckerFrameworkEvaluator()
        
        # Model training state
        self.is_trained = False
        self.feature_names = None
        
    def extract_node_features(self, cfg_data):
        """Extract features from CFG nodes for binary classification using GBT features"""
        nodes = cfg_data.get('nodes', [])
        features = []
        node_info = []
        
        for node in nodes:
            # Extract GBT-style features
            feature_vector = self._extract_gbt_features(node, cfg_data)
            features.append(feature_vector)
            
            node_info.append({
                'id': node.get('id'),
                'line': node.get('line'),
                'label': node.get('label', ''),
                'node_type': node.get('node_type', '')
            })
        
        return np.array(features), node_info
    
    def _extract_gbt_features(self, node, cfg_data):
        """Extract GBT-style features for a single node"""
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        line = node.get('line', 0)
        
        # Basic features
        features = [
            len(label),  # label_length
            1.0 if line > 0 else 0.0,  # has_line_number
            float('method' in node_type.lower()),  # is_method
            float('field' in node_type.lower()),  # is_field
            float('parameter' in node_type.lower()),  # is_parameter
            float('variable' in node_type.lower()),  # is_variable
            self._is_annotation_target(node),  # is_annotation_target
        ]
        
        # Label-based features
        features.extend([
            float('public' in label.lower()),  # is_public
            float('private' in label.lower()),  # is_private
            float('protected' in label.lower()),  # is_protected
            float('static' in label.lower()),  # is_static
            float('final' in label.lower()),  # is_final
            float('void' in label.lower()),  # is_void
            float('int' in label.lower()),  # is_int
            float('string' in label.lower()),  # is_string
        ])
        
        # CFG context features
        total_nodes = len(cfg_data.get('nodes', []))
        control_edges = len(cfg_data.get('control_edges', []))
        dataflow_edges = len(cfg_data.get('dataflow_edges', []))
        
        features.extend([
            float(total_nodes),  # total_nodes
            float(control_edges),  # control_edges_count
            float(dataflow_edges),  # dataflow_edges_count
            float(control_edges) / max(total_nodes, 1),  # control_edges_density
            float(dataflow_edges) / max(total_nodes, 1),  # dataflow_edges_density
        ])
        
        return features
    
    def _is_annotation_target(self, node):
        """Determine if a node is a valid annotation target"""
        label = node.get('label', '').lower()
        node_type = node.get('node_type', '').lower()
        
        # Check for method declarations
        if any(keyword in label for keyword in ['methoddeclaration', 'constructordeclaration']):
            return 1.0
        
        # Check for field declarations
        if any(keyword in label for keyword in ['fielddeclaration', 'variabledeclarator']):
            return 1.0
        
        # Check for parameter declarations
        if any(keyword in label for keyword in ['formalparameter', 'parameter']):
            return 1.0
        
        # Check for mock data patterns (for testing)
        if any(keyword in label for keyword in ['public void method', 'void method', 'method(']):
            return 1.0
        if any(keyword in label for keyword in ['int variable', 'variable =', 'variable;']):
            return 1.0
        if any(keyword in label for keyword in ['parameter', 'param']):
            return 1.0
        
        # Check node type
        if node_type in ['method', 'field', 'parameter', 'variable']:
            return 1.0
        
        return 0.0
    
    def predict_binary_annotations(self, cfg_data, threshold=0.5):
        """Predict whether annotations are needed (binary classification)"""
        features, node_info = self.extract_node_features(cfg_data)
        
        if len(features) == 0:
            return []
        
        # If not trained yet, make random predictions to bootstrap learning
        if not self.is_trained:
            predicted_lines = []
            for i, node in enumerate(node_info):
                # Random prediction with some bias toward annotation targets
                if self._is_annotation_target({'label': node['label'], 'node_type': node['node_type']}) > 0:
                    # Higher chance of predicting annotation for valid targets
                    if np.random.random() > 0.3:  # 70% chance
                        predicted_lines.append({
                            'line': node['line'],
                            'confidence': np.random.random(),
                            'node_type': node['node_type'],
                            'label': node['label'][:50]
                        })
            return predicted_lines
        
        # Get predictions from GBT model
        probabilities = self.model.predict_proba(features)
        annotation_probs = probabilities[:, 1]  # Probability of "annotation needed"
        predictions = annotation_probs > threshold
        
        # Extract line numbers for nodes that need annotations
        predicted_lines = []
        for i, (pred, node) in enumerate(zip(predictions, node_info)):
            if pred and node.get('line') and self._is_annotation_target({'label': node['label'], 'node_type': node['node_type']}) > 0:
                predicted_lines.append({
                    'line': node['line'],
                    'confidence': annotation_probs[i],
                    'node_type': node['node_type'],
                    'label': node['label'][:50]  # Truncate for readability
                })
        
        return predicted_lines
    
    def place_binary_annotations(self, java_file, predicted_lines):
        """Place binary annotations (generic @NonNull or @IndexFor)"""
        try:
            # Create a copy of the file for annotation
            temp_file = java_file + '.binary_annotated'
            shutil.copy2(java_file, temp_file)
            
            # Use the mock annotation placement manager
            manager = MockAnnotationPlacementManager(temp_file)
            
            # Extract just the line numbers
            line_numbers = [pred['line'] for pred in predicted_lines]
            
            # Determine annotation category based on checker type
            annotation_category = 'index' if self.checker_type == 'index' else 'nullness'
            
            logger.info(f"Placing binary annotations on lines: {line_numbers}")
            logger.info(f"Annotation category: {annotation_category}")
            
            # Place annotations
            success = manager.place_annotations(line_numbers, annotation_category)
            
            logger.info(f"Binary annotation placement success: {success}")
            return temp_file if success else None
            
        except Exception as e:
            logger.error(f"Error placing binary annotations: {e}")
            return None
    
    def evaluate_with_checker_framework(self, java_file):
        """Run Checker Framework evaluation"""
        try:
            result = self.evaluator.evaluate_file(java_file, self.checker_type)
            return result
        except Exception as e:
            logger.error(f"Error running Checker Framework: {e}")
            return MockEvaluationResult(
                original_warnings=[],
                new_warnings=[],
                warning_count_change=0,
                success=False,
                error_message=str(e)
            )
    
    def compute_reward(self, original_warnings, new_warnings):
        """Compute reward based on warning reduction"""
        if not original_warnings:
            return 0.0
        
        reduction = len(original_warnings) - len(new_warnings)
        
        # Normalize reward between -1 and 1
        if reduction > 0:
            return min(reduction / len(original_warnings), 1.0)
        elif reduction < 0:
            return max(reduction / len(original_warnings), -1.0)
        else:
            return 0.0
    
    def train_episode(self, cfg_data, original_warnings, java_file):
        """Train the model on a single episode"""
        try:
            # Predict binary annotation locations
            predicted_lines = self.predict_binary_annotations(cfg_data)
            
            if not predicted_lines:
                logger.info("No binary annotation targets predicted")
                return 0.0, []
            
            # Place binary annotations
            annotated_file = self.place_binary_annotations(java_file, predicted_lines)
            
            if not annotated_file:
                logger.warning("Failed to place binary annotations")
                return 0.0, []
            
            # Evaluate with Checker Framework
            result = self.evaluate_with_checker_framework(annotated_file)
            
            # Compute reward
            reward = self.compute_reward(original_warnings, result.new_warnings)
            
            # Store experience for replay
            experience = {
                'cfg_data': cfg_data,
                'predicted_lines': predicted_lines,
                'reward': reward,
                'original_warnings': len(original_warnings),
                'new_warnings': len(result.new_warnings)
            }
            self.experience_buffer.append(experience)
            
            # Clean up temp file
            if os.path.exists(annotated_file):
                os.remove(annotated_file)
            
            logger.info(f"Episode completed: reward={reward:.3f}, predictions={len(predicted_lines)}")
            return reward, predicted_lines
            
        except Exception as e:
            logger.error(f"Error in training episode: {e}")
            return 0.0, []
    
    def train_from_experience(self, batch_size=32):
        """Train GBT model using experience replay"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch from experience buffer
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        # Prepare training data
        all_features = []
        all_labels = []
        
        for experience in batch:
            cfg_data = experience['cfg_data']
            reward = experience['reward']
            
            features, _ = self.extract_node_features(cfg_data)
            
            if len(features) == 0:
                continue
            
            # Create binary labels based on reward
            # Positive reward -> annotation needed (label 1)
            # Negative/zero reward -> no annotation (label 0)
            labels = np.full(len(features), 1 if reward > 0 else 0)
            
            all_features.append(features)
            all_labels.append(labels)
        
        if not all_features:
            return
        
        # Combine all features and labels
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Train GBT model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate accuracy
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        logger.info(f"Experience replay training: accuracy={accuracy:.4f}")
    
    def train(self, project_root, warnings_file, cfwr_root, num_episodes=50):
        """Train the binary RL GBT model"""
        logger.info(f"Starting binary RL training for GBT model")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Warnings file: {warnings_file}")
        logger.info(f"CFWR root: {cfwr_root}")
        logger.info(f"Episodes: {num_episodes}")
        
        # Create mock CFG data for testing
        mock_cfg_data = {
            'nodes': [
                {'id': 0, 'label': 'public void method()', 'node_type': 'method', 'line': 10},
                {'id': 1, 'label': 'int variable = 5;', 'node_type': 'variable', 'line': 11},
                {'id': 2, 'label': 'return variable;', 'node_type': 'control', 'line': 12}
            ],
            'control_edges': [
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2}
            ],
            'dataflow_edges': [
                {'source': 1, 'target': 2}
            ]
        }
        
        # Create mock Java file
        mock_java_file = os.path.join(cfwr_root, 'mock_test_gbt.java')
        with open(mock_java_file, 'w') as f:
            f.write("""public class MockTestGBT {
    public void method() {
        int variable = 5;
        return variable;
    }
}""")
        
        # Training loop
        episode_rewards = []
        episode_predictions = []
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            # Get original warnings (simulate)
            original_warnings = [f"warning_{i}" for i in range(random.randint(5, 20))]
            
            # Train episode
            reward, predictions = self.train_episode(mock_cfg_data, original_warnings, mock_java_file)
            
            episode_rewards.append(reward)
            episode_predictions.append(len(predictions))
            
            # Update training statistics
            self.training_stats['episodes'].append(episode + 1)
            self.training_stats['rewards'].append(reward)
            self.training_stats['binary_predictions'].append(len(predictions))
            
            # Experience replay training (every 10 episodes)
            if len(self.experience_buffer) >= 32 and (episode + 1) % 10 == 0:
                self.train_from_experience(batch_size=32)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_predictions = np.mean(episode_predictions[-10:])
                logger.info(f"Episode {episode + 1}: avg_reward={avg_reward:.3f}, avg_predictions={avg_predictions:.1f}")
        
        # Clean up mock file
        if os.path.exists(mock_java_file):
            os.remove(mock_java_file)
        
        # Save model and training statistics
        self.save_model('models/binary_rl_gbt_model.joblib')
        self.save_training_stats('models/binary_rl_gbt_stats.json')
        
        logger.info("Binary RL training completed")
        return self.training_stats
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def save_training_stats(self, filepath):
        """Save training statistics"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info(f"Training stats saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Binary RL Training for GBT Model')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index', 
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/CFWR/index1.small.out', 
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/CFWR', 
                       help='Root directory of CFWR project')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum depth')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum samples split')
    parser.add_argument('--checker_type', default='index', choices=['index', 'nullness'], help='Checker type')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to file for manual inspection')
    parser.add_argument('--predictions_output_dir', default='predictions_manual_inspection', help='Directory to save predictions')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BinaryGBTTrainer(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        checker_type=args.checker_type
    )
    
    # Train the model
    stats = trainer.train(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        num_episodes=args.episodes
    )
    
    # Save predictions if requested
    if args.save_predictions:
        from prediction_saver import PredictionSaver
        saver = PredictionSaver(args.predictions_output_dir)
        
        # Generate predictions on mock data
        mock_cfg_data = {
            'nodes': [
                {'id': 0, 'label': 'public void method()', 'node_type': 'method', 'line': 10},
                {'id': 1, 'label': 'int variable = 5;', 'node_type': 'variable', 'line': 11},
                {'id': 2, 'label': 'return variable;', 'node_type': 'control', 'line': 12}
            ],
            'control_edges': [{'source': 0, 'target': 1}, {'source': 1, 'target': 2}],
            'dataflow_edges': [{'source': 1, 'target': 2}]
        }
        
        predictions = trainer.predict_binary_annotations(mock_cfg_data)
        metadata = {
            'model_type': 'GBT',
            'hyperparameters': {
                'learning_rate': args.learning_rate,
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth,
                'min_samples_split': args.min_samples_split,
                'episodes': args.episodes
            }
        }
        
        saver.save_predictions('gbt', 'mock_training_data', predictions, metadata)
        logger.info(f"Predictions saved to {args.predictions_output_dir}")
    
    logger.info("Binary RL training completed successfully")

if __name__ == '__main__':
    main()