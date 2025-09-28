#!/usr/bin/env python3
"""
RL Annotation Type Training System

This module provides reinforcement learning training specifically for annotation type prediction,
integrating the multi-class annotation type prediction system with RL training.
"""

import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from annotation_type_prediction import (
    LowerBoundAnnotationType, AnnotationTypeGBTModel, AnnotationTypeHGTModel
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RLAnnotationTypeEnvironment:
    """Environment for RL annotation type training"""
    cfg_data: Dict[str, Any]
    target_annotations: List[Dict[str, Any]]
    current_node_index: int = 0
    episode_reward: float = 0.0
    done: bool = False

class AnnotationTypeRLAgent(nn.Module):
    """RL agent for annotation type prediction"""
    
    def __init__(self, state_dim: int = 23, action_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        
        # State includes features + current annotation type context
        self.state_dim = state_dim
        self.action_dim = action_dim  # Number of annotation types
        self.hidden_dim = hidden_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
            # Remove Softmax - will apply in forward pass with better numerical stability
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.annotation_types = list(LowerBoundAnnotationType)
        
    def get_state(self, env: RLAnnotationTypeEnvironment, node: Dict) -> torch.Tensor:
        """Extract state representation from environment and node"""
        from annotation_type_prediction import AnnotationTypeClassifier
        
        classifier = AnnotationTypeClassifier()
        features = classifier.extract_features(node, env.cfg_data)
        feature_vector = classifier.features_to_vector(features)
        
        return torch.tensor(feature_vector, dtype=torch.float32)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[int, torch.Tensor]:
        """Select action (annotation type) based on current state"""
        logits = self.policy_net(state)
        # Apply softmax with better numerical stability
        policy_output = F.softmax(logits, dim=-1)
        
        # Add small epsilon to prevent numerical issues
        policy_output = policy_output + 1e-8
        policy_output = policy_output / policy_output.sum()
        
        if training:
            # Sample from policy distribution
            action_dist = torch.distributions.Categorical(policy_output)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        else:
            # Take most likely action
            action = torch.argmax(policy_output)
            log_prob = torch.log(policy_output[action] + 1e-8)
        
        return action.item(), log_prob
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for current state"""
        return self.value_net(state)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both policy and value"""
        logits = self.policy_net(state)
        policy = F.softmax(logits, dim=-1)
        value = self.value_net(state)
        return policy, value

class AnnotationTypeRLTrainer:
    """RL trainer for annotation type prediction"""
    
    def __init__(self, state_dim: int = 23, action_dim: int = 12, learning_rate: float = 0.001):
        self.agent = AnnotationTypeRLAgent(state_dim, action_dim)
        self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        self.training_history = []
        self.gamma = 0.99  # Discount factor
        self.lambda_gae = 0.95  # GAE parameter
        
        # Reward weights
        self.correct_annotation_reward = 10.0
        self.wrong_annotation_penalty = -5.0
        self.no_annotation_reward = 1.0
        self.wrong_category_penalty = -2.0
    
    def create_environment(self, cfg_data: Dict[str, Any]) -> RLAnnotationTypeEnvironment:
        """Create RL environment from CFG data"""
        # Get annotation targets
        from annotation_type_prediction import AnnotationTypeClassifier
        from node_level_models import NodeClassifier
        
        classifier = AnnotationTypeClassifier()
        target_annotations = []
        
        for i, node in enumerate(cfg_data.get('nodes', [])):
            if NodeClassifier.is_annotation_target(node):
                features = classifier.extract_features(node, cfg_data)
                ground_truth_type = classifier.determine_annotation_type(features)
                
                target_annotations.append({
                    'node_index': i,
                    'node': node,
                    'ground_truth': ground_truth_type.value,
                    'features': features
                })
        
        return RLAnnotationTypeEnvironment(
            cfg_data=cfg_data,
            target_annotations=target_annotations
        )
    
    def calculate_reward(self, predicted_annotation: str, ground_truth_annotation: str) -> float:
        """Calculate reward for annotation type prediction"""
        if predicted_annotation == ground_truth_annotation:
            if ground_truth_annotation == LowerBoundAnnotationType.NO_ANNOTATION.value:
                return self.no_annotation_reward
            else:
                return self.correct_annotation_reward
        else:
            # Check if it's a related category error (less penalty)
            if self._is_related_category(predicted_annotation, ground_truth_annotation):
                return self.wrong_category_penalty
            else:
                return self.wrong_annotation_penalty
    
    def _is_related_category(self, pred: str, true: str) -> bool:
        """Check if prediction is in related category to ground truth"""
        positive_types = ["@Positive", "@NonNegative", "@GTENegativeOne"]
        length_types = ["@MinLen", "@ArrayLen", "@LengthOf", "@LTLengthOf", "@GTLengthOf"]
        index_types = ["@IndexFor", "@SearchIndexFor", "@SearchIndexBottom"]
        
        categories = [positive_types, length_types, index_types]
        
        for category in categories:
            if pred in category and true in category:
                return True
        return False
    
    def run_episode(self, env: RLAnnotationTypeEnvironment) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], List[torch.Tensor]]:
        """Run a single episode of RL training"""
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        env.current_node_index = 0
        env.episode_reward = 0.0
        env.done = False
        
        for target_annotation in env.target_annotations:
            node = target_annotation['node']
            ground_truth = target_annotation['ground_truth']
            
            # Get current state
            state = self.agent.get_state(env, node)
            
            # Select action
            action, log_prob = self.agent.select_action(state, training=True)
            
            # Get value estimate
            value = self.agent.get_value(state)
            
            # Convert action to annotation type
            predicted_annotation = self.agent.annotation_types[action].value
            
            # Calculate reward
            reward = self.calculate_reward(predicted_annotation, ground_truth)
            
            # Store experience
            states.append(state)
            actions.append(torch.tensor(action))
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            env.episode_reward += reward
            env.current_node_index += 1
        
        env.done = True
        return states, actions, rewards, log_probs, values
    
    def compute_gae(self, rewards: List[float], values: List[torch.Tensor], next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        values = [v.item() for v in values] + [next_value]
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] - values[step]
            gae = delta + self.gamma * self.lambda_gae * gae
            advantages.insert(0, gae)
        
        returns = []
        for step in range(len(rewards)):
            returns.append(advantages[step] + values[step])
        
        return advantages, returns
    
    def update_policy(self, states: List[torch.Tensor], actions: List[torch.Tensor], 
                     advantages: List[float], returns: List[float], log_probs: List[torch.Tensor]):
        """Update policy using PPO-style optimization"""
        # Convert to tensors
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        old_log_probs = torch.stack(log_probs)
        
        # Normalize advantages only if we have more than 1 sample
        if len(advantages_tensor) > 1:
            std_val = advantages_tensor.std()
            if std_val > 1e-8:
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / std_val
            else:
                advantages_tensor = advantages_tensor - advantages_tensor.mean()
        else:
            advantages_tensor = advantages_tensor - advantages_tensor.mean()
        
        # Get current policy and values
        policy_outputs, value_outputs = self.agent(states_tensor)
        
        # Calculate policy loss
        action_dist = torch.distributions.Categorical(policy_outputs)
        new_log_probs = action_dist.log_prob(actions_tensor)
        
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        policy_loss = -torch.mean(ratio * advantages_tensor)
        
        # Calculate value loss with proper tensor dimensions
        value_outputs_squeezed = value_outputs.squeeze()
        if value_outputs_squeezed.dim() == 0:
            value_outputs_squeezed = value_outputs_squeezed.unsqueeze(0)
        if returns_tensor.dim() == 0:
            returns_tensor = returns_tensor.unsqueeze(0)
        
        value_loss = F.mse_loss(value_outputs_squeezed, returns_tensor)
        
        # Calculate entropy bonus for exploration
        entropy = action_dist.entropy().mean()
        entropy_bonus = 0.01 * entropy
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - entropy_bonus
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, cfg_files: List[Dict], episodes: int = 1000) -> Dict[str, Any]:
        """Train the RL agent for annotation type prediction"""
        logger.info(f"Starting RL annotation type training: {len(cfg_files)} CFGs, {episodes} episodes")
        
        start_time = time.time()
        episode_rewards = []
        training_metrics = []
        
        for episode in range(episodes):
            # Randomly select a CFG for this episode
            cfg_file = np.random.choice(cfg_files)
            cfg_data = cfg_file['data']
            
            # Create environment
            env = self.create_environment(cfg_data)
            
            if not env.target_annotations:
                continue  # Skip if no annotation targets
            
            # Run episode
            states, actions, rewards, log_probs, values = self.run_episode(env)
            
            if not states:
                continue  # Skip if no states
            
            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values)
            
            # Update policy
            update_metrics = self.update_policy(states, actions, advantages, returns, log_probs)
            
            # Track metrics
            episode_rewards.append(env.episode_reward)
            training_metrics.append(update_metrics)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                logger.info(f"Episode {episode}: avg_reward={avg_reward:.2f}, "
                           f"policy_loss={update_metrics['policy_loss']:.4f}")
        
        training_time = time.time() - start_time
        
        # Compute final metrics
        final_avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        final_metrics = {
            'episodes': episodes,
            'training_time': training_time,
            'final_avg_reward': final_avg_reward,
            'total_cfg_files': len(cfg_files),
            'avg_policy_loss': np.mean([m['policy_loss'] for m in training_metrics[-100:]]),
            'avg_value_loss': np.mean([m['value_loss'] for m in training_metrics[-100:]]),
            'avg_entropy': np.mean([m['entropy'] for m in training_metrics[-100:]])
        }
        
        self.training_history.append(final_metrics)
        
        logger.info(f"RL annotation type training completed in {training_time:.2f}s")
        logger.info(f"Final average reward: {final_avg_reward:.2f}")
        
        return final_metrics
    
    def predict_annotation_types(self, cfg_data: Dict[str, Any], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Predict annotation types using trained RL agent"""
        env = self.create_environment(cfg_data)
        predictions = []
        
        self.agent.eval()
        with torch.no_grad():
            for target_annotation in env.target_annotations:
                node = target_annotation['node']
                
                # Get state
                state = self.agent.get_state(env, node)
                
                # Get action probabilities
                policy_output = self.agent.policy_net(state)
                
                # Get predicted action and confidence
                action = torch.argmax(policy_output).item()
                confidence = policy_output[action].item()
                
                if confidence >= threshold:
                    predicted_type = self.agent.annotation_types[action].value
                    
                    if predicted_type != LowerBoundAnnotationType.NO_ANNOTATION.value:
                        prediction = {
                            'node_id': target_annotation['node_index'],
                            'line': node.get('line'),
                            'annotation_type': predicted_type,
                            'confidence': confidence,
                            'model': 'RL_AnnotationType',
                            'context': target_annotation['features']
                        }
                        predictions.append(prediction)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained RL model"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        logger.info(f"RL annotation type model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained RL model"""
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, weights_only=False)
                self.agent.load_state_dict(checkpoint['agent_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_history = checkpoint['training_history']
                logger.info(f"RL annotation type model loaded from {filepath}")
                return True
            except Exception as e:
                logger.warning(f"Error loading model from {filepath}: {e}")
                return False
        else:
            logger.warning(f"Model file not found: {filepath}")
            return False

def main():
    """Test the RL annotation type training system"""
    logger.info("Testing RL Annotation Type Training System")
    
    # Create test data
    test_cfg = {
        'method_name': 'testMethod',
        'nodes': [
            {
                'id': 1,
                'label': 'LocalVariableDeclaration: int index = 0',
                'line': 5,
                'node_type': 'variable'
            },
            {
                'id': 2,
                'label': 'ArrayAccess: array[index]',
                'line': 6,
                'node_type': 'expression'
            },
            {
                'id': 3,
                'label': 'MethodCall: array.length',
                'line': 7,
                'node_type': 'expression'
            }
        ],
        'control_edges': [{'source': 1, 'target': 2}, {'source': 2, 'target': 3}],
        'dataflow_edges': [{'source': 1, 'target': 2}]
    }
    
    cfg_files = [{'data': test_cfg} for _ in range(10)]  # Multiple copies for training
    
    # Initialize and train RL agent
    trainer = AnnotationTypeRLTrainer()
    
    # Train the model
    logger.info("Training RL annotation type model...")
    training_result = trainer.train(cfg_files, episodes=200)
    logger.info(f"Training result: {training_result}")
    
    # Test prediction
    logger.info("Testing RL annotation type prediction...")
    predictions = trainer.predict_annotation_types(test_cfg)
    logger.info(f"RL predictions: {len(predictions)} annotations predicted")
    for pred in predictions:
        logger.info(f"  - Line {pred['line']}: {pred['annotation_type']} (confidence: {pred['confidence']:.3f})")
    
    # Save model
    model_path = "models/rl_annotation_type_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)

if __name__ == '__main__':
    main()
