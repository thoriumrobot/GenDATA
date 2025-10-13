#!/usr/bin/env python3
"""
Random Walk Policy Network

This module implements a neural network that learns to perform intelligent
random walks through the transformation space, optimizing for warning reduction.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import networkx as nx

from recursive_augmentation_engine import TransformationType, TransformationState
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator

logger = logging.getLogger(__name__)

@dataclass
class WalkExperience:
    """Experience from a random walk"""
    walk_sequence: List[TransformationType]
    warning_reduction: float
    overall_score: float
    graph_context: Dict[str, Any]

class GraphAttentionLayer(nn.Module):
    """Graph attention layer for transformation graph"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Linear transformations
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        
        # Edge features (for transformation dependencies)
        self.W_edge = nn.Linear(1, self.head_dim, bias=False)
        
        # Output projection
        self.W_out = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
        """
        batch_size, num_nodes, input_dim = node_features.shape
        
        # Linear transformations
        Q = self.W_q(node_features)  # [batch_size, num_nodes, output_dim]
        K = self.W_k(node_features)  # [batch_size, num_nodes, output_dim]
        V = self.W_v(node_features)  # [batch_size, num_nodes, output_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply edge weights
        if edge_index.size(1) > 0:
            edge_weights_expanded = edge_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            edge_weights_expanded = edge_weights_expanded.expand(batch_size, self.num_heads, -1, 1)
            
            # Apply edge weights to attention scores
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]
                scores[:, :, src, dst] += edge_weights[i]
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.output_dim)
        
        # Final projection
        out = self.W_out(out)
        
        return out

class RandomWalkPolicyNet(nn.Module):
    """Neural network for learned random walk policy"""
    
    def __init__(self, transformation_vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_heads: int = 4, max_walk_length: int = 10):
        super().__init__()
        
        self.transformation_vocab_size = transformation_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_walk_length = max_walk_length
        
        # Transformation embeddings
        self.transformation_embedding = nn.Embedding(transformation_vocab_size, embedding_dim)
        
        # Graph attention layers
        self.graph_attention1 = GraphAttentionLayer(embedding_dim, hidden_dim, num_heads)
        self.graph_attention2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
        
        # Walk history encoder (LSTM)
        self.history_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Current state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # graph + history
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, transformation_vocab_size)
        )
        
        # Value head (for advantage estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, current_transformation: torch.Tensor, 
                walk_history: torch.Tensor, 
                graph_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weights: torch.Tensor,
                valid_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            current_transformation: [batch_size] - current transformation index
            walk_history: [batch_size, max_walk_length] - walk history indices
            graph_features: [batch_size, num_nodes, embedding_dim] - graph node features
            edge_index: [2, num_edges] - graph edge indices
            edge_weights: [num_edges] - edge weights
            valid_actions: [batch_size, vocab_size] - mask for valid actions
            
        Returns:
            policy_logits: [batch_size, vocab_size]
            value: [batch_size, 1]
        """
        batch_size = current_transformation.size(0)
        
        # Encode current transformation
        current_embedding = self.transformation_embedding(current_transformation)  # [batch_size, embedding_dim]
        
        # Encode walk history
        history_embeddings = self.transformation_embedding(walk_history)  # [batch_size, max_walk_length, embedding_dim]
        history_output, _ = self.history_encoder(history_embeddings)  # [batch_size, max_walk_length, hidden_dim]
        history_encoded = history_output[:, -1, :]  # Take last output: [batch_size, hidden_dim]
        
        # Process graph features
        graph_encoded = self.graph_attention1(graph_features, edge_index, edge_weights)
        graph_encoded = self.graph_attention2(graph_encoded, edge_index, edge_weights)
        
        # Global graph representation (mean pooling)
        graph_global = torch.mean(graph_encoded, dim=1)  # [batch_size, hidden_dim]
        
        # Encode current state
        state_encoded = self.state_encoder(current_embedding)  # [batch_size, hidden_dim]
        
        # Combine graph and history information
        combined = torch.cat([graph_global, history_encoded], dim=-1)  # [batch_size, hidden_dim * 2]
        
        # Policy and value
        policy_logits = self.policy_head(combined)  # [batch_size, vocab_size]
        value = self.value_head(combined)  # [batch_size, 1]
        
        # Mask invalid actions
        masked_logits = policy_logits * valid_actions + (1 - valid_actions) * (-1e8)
        
        return masked_logits, value
    
    def select_action(self, current_transformation: int, walk_history: List[int],
                     graph_features: torch.Tensor, edge_index: torch.Tensor,
                     edge_weights: torch.Tensor, valid_actions: List[int]) -> int:
        """Select next action using the policy network"""
        self.eval()
        
        with torch.no_grad():
            # Prepare inputs
            current_tensor = torch.tensor([current_transformation], dtype=torch.long)
            
            # Pad or truncate history
            history_padded = walk_history[-self.max_walk_length:] + [0] * max(0, self.max_walk_length - len(walk_history))
            history_tensor = torch.tensor([history_padded], dtype=torch.long)
            
            # Prepare graph features
            graph_tensor = graph_features.unsqueeze(0)
            edge_index_tensor = edge_index
            edge_weights_tensor = edge_weights
            
            # Prepare valid actions mask
            valid_mask = torch.zeros(self.transformation_vocab_size)
            for action in valid_actions:
                valid_mask[action] = 1.0
            valid_tensor = valid_mask.unsqueeze(0)
            
            # Forward pass
            policy_logits, _ = self.forward(
                current_tensor, history_tensor, graph_tensor,
                edge_index_tensor, edge_weights_tensor, valid_tensor
            )
            
            # Sample action
            action_probs = F.softmax(policy_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
            return action

class RandomWalkPolicyTrainer:
    """Trainer for the random walk policy network"""
    
    def __init__(self, model: RandomWalkPolicyNet, device: str = 'cpu',
                 learning_rate: float = 3e-4, gamma: float = 0.99):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # Experience buffer
        self.experience_buffer = []
        self.buffer_size = 10000
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'total_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'warning_reductions': []
        }
    
    def add_experience(self, experience: WalkExperience):
        """Add experience to buffer"""
        self.experience_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer = self.experience_buffer[-self.buffer_size:]
    
    def train_on_batch(self, batch_size: int = 32, epochs: int = 3):
        """Train the model on a batch of experiences"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        for epoch in range(epochs):
            total_policy_loss = 0
            total_value_loss = 0
            
            for experience in batch:
                # Prepare training data
                # This is a simplified training loop - in practice, you'd need
                # more sophisticated experience replay with proper state-action-reward sequences
                
                # For now, use warning reduction as reward
                reward = experience.warning_reduction
                
                # Compute loss (simplified)
                # In practice, you'd need proper PPO or other RL algorithm implementation
                dummy_policy_loss = torch.tensor(0.0, requires_grad=True)
                dummy_value_loss = torch.tensor(0.0, requires_grad=True)
                
                total_policy_loss += dummy_policy_loss
                total_value_loss += dummy_value_loss
            
            # Update model
            self.optimizer.zero_grad()
            total_loss = total_policy_loss + total_value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Update statistics
            self.training_stats['policy_losses'].append(total_policy_loss.item() / batch_size)
            self.training_stats['value_losses'].append(total_value_loss.item() / batch_size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.training_stats.copy()

class RandomWalkPolicyNetwork:
    """Main class for random walk policy learning"""
    
    def __init__(self, device: str = 'cpu', embedding_dim: int = 128, 
                 hidden_dim: int = 256, learning_rate: float = 3e-4):
        self.device = device
        
        # Initialize model
        transformation_vocab_size = len(TransformationType)
        self.model = RandomWalkPolicyNet(
            transformation_vocab_size=transformation_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        # Initialize trainer
        self.trainer = RandomWalkPolicyTrainer(
            model=self.model,
            device=device,
            learning_rate=learning_rate
        )
        
        # Transformation mappings
        self.transformation_to_idx = {t: i for i, t in enumerate(TransformationType)}
        self.idx_to_transformation = {i: t for t, i in self.transformation_to_idx.items()}
        
        # Statistics
        self.stats = {
            'total_walks': 0,
            'successful_walks': 0,
            'average_warning_reduction': 0.0
        }
    
    def generate_walk_with_policy(self, initial_state: TransformationState, 
                                max_length: int = 10) -> List[TransformationType]:
        """Generate a random walk using the learned policy"""
        walk = []
        current_state = initial_state
        
        for step in range(max_length):
            # Get valid next transformations
            valid_transformations = self._get_valid_transformations(current_state)
            
            if not valid_transformations:
                break
            
            # Select action using policy network
            current_idx = self.transformation_to_idx.get(
                current_state.transformation_history[-1] if current_state.transformation_history else TransformationType.LOOP_CONVERSION,
                0
            )
            
            history_indices = [self.transformation_to_idx[t] for t in current_state.transformation_history]
            valid_indices = [self.transformation_to_idx[t] for t in valid_transformations]
            
            # Create dummy graph features for now
            graph_features = torch.randn(len(TransformationType), 128)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0)
            
            action_idx = self.model.select_action(
                current_idx, history_indices, graph_features,
                edge_index, edge_weights, valid_indices
            )
            
            action = self.idx_to_transformation[action_idx]
            walk.append(action)
            
            # Update state (simplified)
            new_history = current_state.transformation_history + [action]
            current_state = TransformationState(
                code=current_state.code,
                transformation_history=new_history,
                depth=current_state.depth + 1,
                complexity_score=current_state.complexity_score,
                compilation_status=current_state.compilation_status,
                semantic_preservation=current_state.semantic_preservation,
                metadata=current_state.metadata
            )
        
        return walk
    
    def _get_valid_transformations(self, state: TransformationState) -> List[TransformationType]:
        """Get valid next transformations for current state"""
        # Simplified implementation - return all transformations
        # In practice, this would use the recursive augmentation engine
        return list(TransformationType)
    
    def learn_from_walks(self, walks: List[List[TransformationType]], 
                        rewards: List[float]):
        """Learn from successful walks"""
        for walk, reward in zip(walks, rewards):
            experience = WalkExperience(
                walk_sequence=walk,
                warning_reduction=reward,
                overall_score=reward,
                graph_context={}
            )
            self.trainer.add_experience(experience)
        
        # Train on batch
        self.trainer.train_on_batch(batch_size=min(32, len(walks)))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = self.stats.copy()
        stats.update(self.trainer.get_statistics())
        return stats


def main():
    """Test the random walk policy network"""
    logger.info("Testing Random Walk Policy Network...")
    
    # Create test state
    test_state = TransformationState(
        code="public class Test { public int method() { return 0; } }",
        transformation_history=[],
        depth=0,
        complexity_score=2.0,
        compilation_status=True,
        semantic_preservation=True,
        metadata={}
    )
    
    # Initialize policy network
    policy_net = RandomWalkPolicyNetwork(device='cpu')
    
    # Generate walk
    walk = policy_net.generate_walk_with_policy(test_state, max_length=5)
    
    logger.info(f"Generated walk: {[t.value for t in walk]}")
    
    # Get statistics
    stats = policy_net.get_statistics()
    logger.info(f"Statistics: {stats}")


if __name__ == '__main__':
    import random
    random.seed(42)
    main()

