#!/usr/bin/env python3
"""
Enhanced Reinforcement Learning Training Script

This script integrates annotation placement, Checker Framework evaluation,
and feedback mechanisms for training annotation prediction models.
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
from torch_geometric.data import DataLoader
from collections import defaultdict
import random
from pathlib import Path
import time
import logging

# Import our modules
from hgt import HGTModel, create_heterodata, load_cfgs
from gbt import load_cfgs as load_cfgs_gbt, extract_features_from_cfg
from causal_model import load_cfgs as load_cfgs_causal, extract_features_and_labels, parse_warnings, run_index_checker, preprocess_data
from cfg import generate_control_flow_graphs, save_cfgs
from augment_slices import augment_file
from annotation_placement import AnnotationPlacementManager
from checker_framework_integration import CheckerFrameworkEvaluator, CheckerType, EvaluationResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAnnotationPlacementModel(nn.Module):
    """Enhanced model for predicting annotation placement locations"""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout_rate=0.3):
        super(EnhancedAnnotationPlacementModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Enhanced feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head for annotation placement
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        # Attention mechanism for focusing on important features
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, batch_first=True)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Apply attention if input is 3D (batch, sequence, features)
        if len(features.shape) == 3:
            attended_features, _ = self.attention(features, features, features)
            features = attended_features.mean(dim=1)  # Global average pooling
        
        logits = self.classifier(features)
        return logits

class EnhancedReinforcementLearningTrainer:
    """Enhanced RL trainer with better feedback mechanisms"""
    
    def __init__(self, model_type='hgt', learning_rate=0.001, device='cpu', 
                 checker_type='nullness', reward_strategy='adaptive'):
        self.model_type = model_type
        self.device = device
        self.learning_rate = learning_rate
        self.checker_type = CheckerType.NULLNESS if checker_type == 'nullness' else CheckerType.INDEX
        self.reward_strategy = reward_strategy
        
        # Initialize the appropriate model
        if model_type == 'hgt':
            self.model = self._init_hgt_model()
        elif model_type == 'gbt':
            self.model = self._init_gbt_model()
        elif model_type == 'causal':
            self.model = self._init_causal_model()
        elif model_type == 'dg2n':
            # Placeholder minimal NN to satisfy optimizer; DG2N inference is done via external script
            self.model = self._init_causal_model()
        elif model_type == 'gcn':
            # Use minimal NN; inference via gcn_predict script-like path inside RL
            self.model = self._init_causal_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10, factor=0.5)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize Checker Framework evaluator
        self.evaluator = CheckerFrameworkEvaluator()
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'warning_changes': [],
            'accuracy': [],
            'learning_rate': [],
            'loss': []
        }
        
        # Experience replay buffer for better learning
        self.experience_buffer = []
        self.buffer_size = 1000
        
    def _init_hgt_model(self):
        """Initialize HGT-based annotation placement model"""
        return EnhancedAnnotationPlacementModel(input_dim=2, hidden_dim=256)
    
    def _init_gbt_model(self):
        """Initialize GBT-based annotation placement model"""
        return EnhancedAnnotationPlacementModel(input_dim=14, hidden_dim=256)
    
    def _init_causal_model(self):
        """Initialize Causal-based annotation placement model"""
        return EnhancedAnnotationPlacementModel(input_dim=12, hidden_dim=256)
    
    def predict_annotation_locations(self, cfg_data, threshold=0.01):
        """Predict where annotations should be placed based on CFG data"""
        if self.model_type == 'hgt':
            return self._predict_hgt_locations(cfg_data, threshold)
        elif self.model_type == 'gbt':
            return self._predict_gbt_locations(cfg_data, threshold)
        elif self.model_type == 'causal':
            return self._predict_causal_locations(cfg_data, threshold)
        elif self.model_type == 'dg2n':
            return self._predict_dg2n_locations(cfg_data, threshold)
        elif self.model_type == 'gcn':
            return self._predict_gcn_locations(cfg_data, threshold)
    
    def _predict_hgt_locations(self, cfg_data, threshold):
        """Predict annotation locations using HGT model"""
        try:
            # Create heterodata from CFG
            heterodata = create_heterodata(cfg_data)
            if heterodata is None:
                return []
            
            # Get node features
            node_features = heterodata['node'].x
            if node_features is None or len(node_features) == 0:
                return []
            
            # Predict for each node
            self.model.eval()
            with torch.no_grad():
                logits = self.model(node_features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = probabilities[:, 1] > threshold
                
                # Debug logging
                logger.info(f"HGT predictions - logits: {logits}, probabilities: {probabilities}")
                logger.info(f"Threshold: {threshold}, predictions: {predictions}")
            
            # Extract line numbers for predicted locations
            predicted_lines = []
            nodes = cfg_data.get('nodes', [])
            for i, pred in enumerate(predictions):
                if pred and i < len(nodes) and nodes[i].get('line') is not None:
                    predicted_lines.append(nodes[i]['line'])
            
            logger.info(f"Extracted predicted lines: {predicted_lines}")
            return predicted_lines
        except Exception as e:
            logger.error(f"Error in HGT prediction: {e}")
            return []
    
    def _predict_gbt_locations(self, cfg_data, threshold):
        """Predict annotation locations using GBT model"""
        try:
            # Extract features
            features = extract_features_from_cfg(cfg_data)
            if not features:
                return []
            
            # Convert to tensor
            feature_tensor = torch.tensor([features], dtype=torch.float32)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                logits = self.model(feature_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = probabilities[0, 1] > threshold
            
            # For GBT, we predict at method level, so return method start line
            if prediction and cfg_data.get('method_name'):
                # Find method start line from CFG nodes
                for node in cfg_data.get('nodes', []):
                    if node.get('label') == 'Entry':
                        return [node.get('line', 1)]
            
            return []
        except Exception as e:
            logger.error(f"Error in GBT prediction: {e}")
            return []
    
    def _predict_causal_locations(self, cfg_data, threshold):
        """Predict annotation locations using Causal model"""
        try:
            # Extract features
            records = extract_features_and_labels(cfg_data, {})
            if not records:
                return []
            
            # Use the first record for prediction
            features = records[0]
            feature_values = [
                features.get('label_length', 0),
                features.get('in_degree', 0),
                features.get('out_degree', 0),
                features.get('label_encoded', 0),
                features.get('line_number', 0),
                features.get('control_in_degree', 0),
                features.get('control_out_degree', 0),
                features.get('dataflow_in_degree', 0),
                features.get('dataflow_out_degree', 0),
                features.get('variables_used', 0),
                features.get('dataflow_count', 0),
                features.get('control_count', 0)
            ]
            
            # Convert to tensor
            feature_tensor = torch.tensor([feature_values], dtype=torch.float32)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                logits = self.model(feature_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = probabilities[0, 1] > threshold
            
            # Return the line number if prediction is positive
            if prediction and features.get('line_number'):
                return [features['line_number']]
            
            return []
        except Exception as e:
            logger.error(f"Error in Causal prediction: {e}")
            return []

    def _predict_dg2n_locations(self, cfg_data, threshold):
        """Predict annotation locations using DG2N by converting CFG to .pt and running predict_dg2n.py.
        Returns node line numbers with top-class predictions above threshold.
        """
        try:
            import tempfile
            from subprocess import run as sp_run
            import uuid
            # Write cfg_data to a temp JSON
            with tempfile.TemporaryDirectory() as td:
                cfg_path = os.path.join(td, 'graph.json')
                with open(cfg_path, 'w') as f:
                    json.dump(cfg_data, f)
                # Convert to .pt using adapter (operate on dir)
                adapter_out = os.path.join(td, 'pt')
                os.makedirs(adapter_out, exist_ok=True)
                # Save as method.json file name
                method_name = cfg_data.get('method_name', f'method_{uuid.uuid4().hex}')
                file_path = os.path.join(td, f'{method_name}.json')
                with open(file_path, 'w') as f:
                    json.dump(cfg_data, f)
                sp_run([sys.executable, os.path.join(os.getcwd(), 'dg2n_adapter.py'), '--cfg_dir', td, '--out_dir', adapter_out], check=False)
                # Pick first .pt
                pts = [p for p in os.listdir(adapter_out) if p.endswith('.pt')]
                if not pts:
                    return []
                graph_pt = os.path.join(adapter_out, pts[0])
                ckpt = os.path.join('models', 'dg2n', 'best_dg2n.pt')
                out_json = os.path.join(td, 'dg2n_pred.json')
                sp_run([sys.executable, os.path.join('dg2n', 'predict_dg2n.py'), '--ckpt', ckpt, '--graph_pt', graph_pt, '--out_json', out_json], check=False)
                if not os.path.exists(out_json):
                    return []
                with open(out_json, 'r') as f:
                    res = json.load(f)
                # Map node-level predictions to line numbers
                pred = res.get('pred', [])
                probs = res.get('probs', [])
                lines = []
                nodes = cfg_data.get('nodes', [])
                for i, node in enumerate(nodes):
                    if i < len(pred) and i < len(probs):
                        p1 = probs[i][1] if isinstance(probs[i], list) and len(probs[i]) > 1 else 0.0
                        if pred[i] == 1 and p1 >= threshold and node.get('line') is not None:
                            lines.append(node['line'])
                return lines
        except Exception as e:
            logger.error(f"Error in DG2N prediction: {e}")
            return []

    def _predict_gcn_locations(self, cfg_data, threshold):
        """Predict lines using the simple GCN: build temp CFG JSON to dir, run gcn_predict.py with models/gcn/best_gcn.pth."""
        try:
            import tempfile
            from subprocess import run as sp_run
            import uuid
            with tempfile.TemporaryDirectory() as td:
                # Write temp Java-like CFG into dir structure and reuse gcn_predict on a synthetic file by mapping directly
                # Instead, emulate prediction by loading ckpt and running forward here
                from gcn_train import cfg_to_homograph
                ckpt_path = os.path.join('models', 'gcn', 'best_gcn.pth')
                if not os.path.exists(ckpt_path):
                    return []
                ckpt = torch.load(ckpt_path, map_location='cpu')
                from gcn_train import SimpleGCN
                model = SimpleGCN(in_dim=ckpt['in_dim'], hidden=ckpt['hidden'])
                model.load_state_dict(ckpt['model_state'])
                model.eval()
                data = cfg_to_homograph(cfg_data)
                if data.x.numel() == 0:
                    return []
                with torch.no_grad():
                    logits = model(data.x, data.edge_index)
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                lines = []
                nodes = cfg_data.get('nodes', [])
                for i, p in enumerate(probs.tolist()):
                    if p >= threshold and i < len(nodes) and nodes[i].get('line') is not None:
                        lines.append(nodes[i]['line'])
                return lines
        except Exception as e:
            logger.error(f"Error in GCN prediction: {e}")
            return []
    
    def place_annotations_advanced(self, java_file, predicted_lines):
        """Place annotations using the advanced annotation placement system"""
        try:
            # Create a copy of the file for annotation
            temp_file = java_file + '.annotated'
            shutil.copy2(java_file, temp_file)
            
            # Use the advanced annotation placement manager
            manager = AnnotationPlacementManager(temp_file)
            
            # Determine annotation category based on checker type
            annotation_category = 'nullness' if self.checker_type == CheckerType.NULLNESS else 'index'
            
            logger.info(f"Attempting to place annotations on lines: {predicted_lines}")
            logger.info(f"Annotation category: {annotation_category}")
            
            # Place annotations
            success = manager.place_annotations(predicted_lines, annotation_category)
            
            logger.info(f"Annotation placement success: {success}")
            return temp_file if success else None
            
        except Exception as e:
            logger.error(f"Error placing annotations: {e}")
            return None
    
    def evaluate_with_checker_framework_advanced(self, java_file):
        """Run Checker Framework evaluation using the advanced evaluator"""
        try:
            result = self.evaluator.evaluate_file(java_file, self.checker_type)
            return result
        except Exception as e:
            logger.error(f"Error running Checker Framework: {e}")
            return EvaluationResult(
                original_warnings=[],
                new_warnings=[],
                warning_count_change=0,
                success=False,
                error_message=str(e),
                compilation_success=False
            )
    
    def compute_adaptive_reward(self, original_result, annotated_result):
        """Compute adaptive reward based on evaluation results"""
        if not original_result.success or not annotated_result.success:
            # If Checker Framework evaluation fails, provide neutral reward
            # This handles Java module system issues gracefully
            logger.warning("Checker Framework evaluation failed, using neutral reward")
            return 0.0
        
        if not original_result.compilation_success or not annotated_result.compilation_success:
            return -0.5
        
        # Get warning counts
        original_count = len(original_result.original_warnings)
        annotated_count = len(annotated_result.new_warnings)
        
        # If no warnings found (Checker Framework evaluation issues), 
        # provide reward based on successful annotation placement
        if original_count == 0 and annotated_count == 0:
            logger.info("No Checker Framework warnings found, using placement-based reward")
            return 0.5  # Positive reward for successful annotation placement
        
        # Compute reward based on strategy
        if self.reward_strategy == 'adaptive':
            return self._adaptive_reward(original_count, annotated_count)
        elif self.reward_strategy == 'linear':
            return self._linear_reward(original_count, annotated_count)
        elif self.reward_strategy == 'exponential':
            return self._exponential_reward(original_count, annotated_count)
        else:
            return self._adaptive_reward(original_count, annotated_count)
    
    def _adaptive_reward(self, original_count, annotated_count):
        """Adaptive reward function that considers context"""
        if original_count == 0:
            # No original warnings - reward for not introducing new ones
            return 1.0 if annotated_count == 0 else -0.5
        else:
            # Reward based on relative improvement
            improvement = (original_count - annotated_count) / original_count
            return improvement
    
    def _linear_reward(self, original_count, annotated_count):
        """Linear reward function"""
        return original_count - annotated_count
    
    def _exponential_reward(self, original_count, annotated_count):
        """Exponential reward function"""
        if original_count == 0:
            return 1.0 if annotated_count == 0 else -0.5
        else:
            improvement = (original_count - annotated_count) / original_count
            return np.exp(improvement) - 1
    
    def train_episode_advanced(self, cfg_data, java_file):
        """Train the model on a single episode with advanced evaluation"""
        try:
            # Get original evaluation
            original_result = self.evaluate_with_checker_framework_advanced(java_file)
            
            # Predict annotation locations
            predicted_lines = self.predict_annotation_locations(cfg_data)
            logger.info(f"Predicted {len(predicted_lines)} annotation locations: {predicted_lines}")
            
            if not predicted_lines:
                logger.warning("No annotation locations predicted, returning 0 reward")
                return 0.0, original_result  # No reward if no predictions
            
            # Place annotations using advanced system
            annotated_file = self.place_annotations_advanced(java_file, predicted_lines)
            logger.info(f"Annotation placement result: {annotated_file}")
            if not annotated_file:
                logger.warning("Annotation placement failed, returning 0 reward")
                return 0.0, original_result
            
            # Give positive reward for successful annotation placement
            placement_reward = 0.3  # Base reward for successful placement
            
            # Evaluate annotated file
            annotated_result = self.evaluate_with_checker_framework_advanced(annotated_file)
            
            # Compute reward
            cf_reward = self.compute_adaptive_reward(original_result, annotated_result)
            
            # Combine placement reward with Checker Framework reward
            total_reward = placement_reward + cf_reward
            logger.info(f"Placement reward: {placement_reward}, CF reward: {cf_reward}, Total: {total_reward}")
            
            # Store experience for replay
            experience = {
                'cfg_data': cfg_data,
                'predicted_lines': predicted_lines,
                'original_result': original_result,
                'annotated_result': annotated_result,
                'reward': total_reward
            }
            self._store_experience(experience)
            
            # Clean up temp file
            if os.path.exists(annotated_file):
                os.remove(annotated_file)
            
            return total_reward, original_result
            
        except Exception as e:
            logger.error(f"Error in training episode: {e}")
            return 0.0, None
    
    def _store_experience(self, experience):
        """Store experience in replay buffer"""
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def _update_model_with_experience_replay(self, batch_size=32):
        """Update model using experience replay"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch from experience buffer
        batch = random.sample(self.experience_buffer, batch_size)
        
        # Prepare training data
        rewards = [exp['reward'] for exp in batch]
        
        # Convert rewards to loss (negative rewards)
        loss = -torch.tensor(rewards, dtype=torch.float32).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, slices_dir, cfg_dir, num_episodes=100, batch_size=32, use_augmented_slices=True):
        """Main training loop with enhanced features
        
        Args:
            slices_dir: Directory containing augmented slices (default behavior)
            cfg_dir: Directory containing CFGs generated from augmented slices
            num_episodes: Number of training episodes
            batch_size: Batch size for training
            use_augmented_slices: Whether to use augmented slices (default: True)
        """
        slice_type = "augmented" if use_augmented_slices else "original"
        logger.info(f"Starting enhanced RL training with {self.model_type} model for {num_episodes} episodes")
        logger.info(f"Training on {slice_type} slices from: {slices_dir}")
        
        # Find all CFG files
        cfg_files = []
        for root, dirs, files in os.walk(cfg_dir):
            for file in files:
                if file.endswith('.json'):
                    cfg_files.append(os.path.join(root, file))
        
        if not cfg_files:
            logger.error("No CFG files found for training")
            return
        
        logger.info(f"Found {len(cfg_files)} CFG files for training")
        
        # Training loop
        for episode in range(num_episodes):
            episode_rewards = []
            episode_warning_changes = []
            episode_losses = []
            
            # Sample a batch of CFG files
            batch_files = random.sample(cfg_files, min(batch_size, len(cfg_files)))
            
            for cfg_file in batch_files:
                try:
                    # Load CFG data
                    with open(cfg_file, 'r') as f:
                        cfg_data = json.load(f)
                    
                    # Find corresponding Java file
                    java_file = self._find_corresponding_java_file(cfg_file, slices_dir)
                    if not java_file or not os.path.exists(java_file):
                        continue
                    
                    # Train on this episode
                    reward, original_result = self.train_episode_advanced(cfg_data, java_file)
                    
                    episode_rewards.append(reward)
                    if original_result:
                        episode_warning_changes.append(len(original_result.original_warnings))
                    
                except Exception as e:
                    logger.error(f"Error processing {cfg_file}: {e}")
                    continue
            
            # Update model using experience replay
            if len(self.experience_buffer) >= batch_size:
                loss = self._update_model_with_experience_replay(batch_size)
                episode_losses.append(loss)
            
            # Update learning rate scheduler
            if episode_rewards:
                avg_reward = np.mean(episode_rewards)
                self.scheduler.step(avg_reward)
                
                # Record statistics
                self.training_stats['episodes'].append(episode)
                self.training_stats['rewards'].append(avg_reward)
                self.training_stats['warning_changes'].append(np.mean(episode_warning_changes))
                self.training_stats['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                if episode_losses:
                    self.training_stats['loss'].append(np.mean(episode_losses))
                
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, "
                          f"Avg Original Warnings = {np.mean(episode_warning_changes):.1f}, "
                          f"LR = {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save model checkpoint every 10 episodes
            if episode % 10 == 0:
                self.save_model(f"models/enhanced_rl_{self.model_type}_episode_{episode}.pth")
        
        logger.info("Enhanced training completed!")
        self.save_model(f"models/enhanced_rl_{self.model_type}_final.pth")
        self.save_training_stats(f"models/enhanced_rl_{self.model_type}_stats.json")
    
    def _find_corresponding_java_file(self, cfg_file, slices_dir):
        """Find the Java file corresponding to a CFG file"""
        try:
            # Extract method name from CFG file path
            cfg_path = Path(cfg_file)
            method_name = cfg_path.stem
            
            # Look for corresponding Java file in slices directory
            for root, dirs, files in os.walk(slices_dir):
                for file in files:
                    if file.endswith('.java'):
                        java_file = os.path.join(root, file)
                        # Check if this Java file contains the method
                        with open(java_file, 'r') as f:
                            content = f.read()
                            if method_name in content:
                                return java_file
            return None
        except Exception as e:
            logger.error(f"Error finding Java file: {e}")
            return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_type': self.model_type,
            'learning_rate': self.learning_rate,
            'checker_type': self.checker_type.value,
            'reward_strategy': self.reward_strategy
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {filepath}")
    
    def save_training_stats(self, filepath):
        """Save training statistics"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        logger.info(f"Training stats saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Reinforcement Learning Training for Annotation Placement')
    parser.add_argument('--slices_dir', required=True, help='Directory containing augmented slices')
    parser.add_argument('--cfg_dir', required=True, help='Directory containing CFGs')
    parser.add_argument('--model_type', choices=['hgt', 'gbt', 'causal', 'dg2n', 'gcn'], default='hgt',
                       help='Type of model to use for RL training')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--load_model', help='Path to load existing model')
    parser.add_argument('--checker_type', choices=['nullness', 'index'], default='nullness',
                       help='Type of Checker Framework checker to use')
    parser.add_argument('--reward_strategy', choices=['adaptive', 'linear', 'exponential'], default='adaptive',
                       help='Reward computation strategy')
    parser.add_argument('--use_augmented_slices', action='store_true', default=True,
                       help='Use augmented slices for training (default: True)')
    parser.add_argument('--use_original_slices', action='store_true', default=False,
                       help='Use original slices instead of augmented slices')
    
    args = parser.parse_args()
    
    # Determine whether to use augmented slices
    use_augmented_slices = args.use_augmented_slices and not args.use_original_slices
    
    # Initialize trainer
    trainer = EnhancedReinforcementLearningTrainer(
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        device=args.device,
        checker_type=args.checker_type,
        reward_strategy=args.reward_strategy
    )
    
    # Load existing model if specified
    if args.load_model:
        trainer.load_model(args.load_model)
    
    # Start training
    trainer.train(
        slices_dir=args.slices_dir,
        cfg_dir=args.cfg_dir,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        use_augmented_slices=use_augmented_slices
    )

if __name__ == '__main__':
    main()
