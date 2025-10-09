#!/usr/bin/env python3
"""
Transformation Policy GNN

This module implements a Graph Neural Network with random walk embeddings
for predicting optimal next transformations in the augmentation pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from recursive_augmentation_engine import TransformationType, TransformationState

logger = logging.getLogger(__name__)

@dataclass
class CodeGraph:
    """Represents a code Control Flow Graph"""
    nodes: List[Dict[str, Any]]  # Node features
    edges: List[Tuple[int, int]]  # Edge indices
    node_types: List[str]  # Node type labels
    edge_types: List[str]  # Edge type labels
    adjacency_matrix: torch.Tensor

class RandomWalkEmbedder:
    """Generates random walk embeddings for code graphs"""
    
    def __init__(self, walk_length: int = 10, num_walks: int = 80, 
                 window_size: int = 5, embedding_dim: int = 128):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.embedding_dim = embedding_dim
    
    def generate_random_walks(self, graph: CodeGraph) -> List[List[int]]:
        """Generate random walks on the code graph"""
        walks = []
        num_nodes = len(graph.nodes)
        
        # Convert to NetworkX graph for easier manipulation
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(num_nodes))
        
        for edge in graph.edges:
            nx_graph.add_edge(edge[0], edge[1])
        
        for _ in range(self.num_walks):
            # Start from random node
            start_node = random.randint(0, num_nodes - 1)
            walk = [start_node]
            
            current_node = start_node
            for _ in range(self.walk_length - 1):
                neighbors = list(nx_graph.neighbors(current_node))
                if neighbors:
                    current_node = random.choice(neighbors)
                    walk.append(current_node)
                else:
                    break
            
            walks.append(walk)
        
        return walks
    
    def compute_node_embeddings(self, walks: List[List[int]], 
                               vocab_size: int) -> torch.Tensor:
        """Compute node embeddings using Skip-gram"""
        # Initialize embeddings
        embeddings = torch.randn(vocab_size, self.embedding_dim) * 0.1
        
        # Simple Skip-gram training
        learning_rate = 0.01
        for walk in walks:
            for i, center_node in enumerate(walk):
                # Get context nodes within window
                start = max(0, i - self.window_size)
                end = min(len(walk), i + self.window_size + 1)
                context_nodes = walk[start:i] + walk[i+1:end]
                
                # Update embeddings (simplified gradient descent)
                for context_node in context_nodes:
                    # Positive sample
                    center_emb = embeddings[center_node]
                    context_emb = embeddings[context_node]
                    
                    # Simple similarity update
                    similarity = torch.dot(center_emb, context_emb)
                    gradient = 1.0 - torch.sigmoid(similarity)
                    
                    embeddings[center_node] += learning_rate * gradient * context_emb
                    embeddings[context_node] += learning_rate * gradient * center_emb
        
        return embeddings

class CodeGraphEncoder(nn.Module):
    """Encodes code graphs using GNN with random walk positional encodings"""
    
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32,
                 hidden_dim: int = 128, num_layers: int = 3, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Node feature projection
        self.node_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # Edge feature projection
        self.edge_projection = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Random walk embedder
        self.random_walk_embedder = RandomWalkEmbedder(embedding_dim=hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
    def forward(self, graph: CodeGraph, random_walk_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode code graph"""
        num_nodes = len(graph.nodes)
        
        # Project node features
        node_features = torch.stack([torch.tensor(node['features'], dtype=torch.float32) 
                                   for node in graph.nodes])
        node_features = self.node_projection(node_features)
        
        # Add random walk embeddings
        node_features = node_features + random_walk_embeddings
        
        # Create edge features
        edge_features = torch.zeros(len(graph.edges), self.hidden_dim)
        for i, edge in enumerate(graph.edges):
            edge_type = graph.edge_types[i] if i < len(graph.edge_types) else 'default'
            edge_features[i] = self._encode_edge_type(edge_type)
        
        # Apply GNN layers
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            residual = node_features
            node_features = gnn_layer(node_features, graph.edges, edge_features)
            node_features = layer_norm(node_features + residual)
            node_features = self.dropout(node_features)
        
        # Global attention pooling
        node_features = node_features.unsqueeze(0)  # Add batch dimension
        pooled_features, _ = self.global_attention(node_features, node_features, node_features)
        pooled_features = pooled_features.squeeze(0).mean(dim=0)  # Global average
        
        return pooled_features
    
    def _encode_edge_type(self, edge_type: str) -> torch.Tensor:
        """Encode edge type as feature vector"""
        # Simple one-hot encoding for different edge types
        edge_types = ['control_flow', 'data_flow', 'call', 'return', 'default']
        edge_encoding = torch.zeros(len(edge_types))
        
        if edge_type in edge_types:
            edge_encoding[edge_types.index(edge_type)] = 1.0
        else:
            edge_encoding[-1] = 1.0  # Default
        
        # Project to hidden dimension
        return F.linear(edge_encoding, torch.randn(self.hidden_dim, len(edge_types)))

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer with edge features"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention weights
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge attention
        self.W_e = nn.Linear(hidden_dim, num_heads)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, node_features: torch.Tensor, edges: List[Tuple[int, int]], 
                edge_features: torch.Tensor) -> torch.Tensor:
        """Apply graph attention"""
        num_nodes = node_features.size(0)
        
        # Project features
        Q = self.W_q(node_features).view(num_nodes, self.num_heads, self.head_dim)
        K = self.W_k(node_features).view(num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(node_features).view(num_nodes, self.num_heads, self.head_dim)
        
        # Initialize attention matrix
        attention_scores = torch.zeros(num_nodes, num_nodes, self.num_heads)
        
        # Compute attention for connected nodes
        for i, (src, dst) in enumerate(edges):
            # Node-to-node attention
            node_attention = torch.sum(Q[src] * K[dst], dim=-1) * self.scale
            attention_scores[src, dst] = node_attention
            
            # Edge attention
            edge_attention = self.W_e(edge_features[i]).view(self.num_heads)
            attention_scores[src, dst] += edge_attention
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.zeros_like(node_features)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if attention_weights[i, j].sum() > 0:
                    weighted_values = (attention_weights[i, j].unsqueeze(-1) * V[j]).sum(dim=0)
                    output[i] += weighted_values
        
        # Output projection
        output = self.W_o(output)
        
        return output

class TransformationHistoryEncoder(nn.Module):
    """Encodes transformation history using LSTM/Transformer"""
    
    def __init__(self, input_dim: int = len(TransformationType), 
                 hidden_dim: int = 128, num_layers: int = 2, 
                 use_transformer: bool = False):
        super().__init__()
        
        self.use_transformer = use_transformer
        
        if use_transformer:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            # LSTM encoder
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0
            )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, history_sequence: torch.Tensor) -> torch.Tensor:
        """Encode transformation history"""
        if self.use_transformer:
            # Project input
            x = self.input_projection(history_sequence)
            
            # Apply transformer
            encoded = self.encoder(x)
            
            # Return last output
            return encoded[:, -1, :]
        else:
            # Apply LSTM
            encoded, (hidden, cell) = self.encoder(history_sequence)
            
            # Return last hidden state
            return hidden[-1]

class TransformationPolicyGNN(nn.Module):
    """Main GNN model for transformation policy prediction"""
    
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32,
                 hidden_dim: int = 256, num_gnn_layers: int = 3,
                 num_heads: int = 8, dropout: float = 0.1,
                 use_transformer_history: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Code graph encoder
        self.graph_encoder = CodeGraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Transformation history encoder
        self.history_encoder = TransformationHistoryEncoder(
            input_dim=len(TransformationType),
            hidden_dim=hidden_dim,
            use_transformer=use_transformer_history
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Policy head (predicts next transformation)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(TransformationType))
        )
        
        # Value head (estimates augmentation sequence quality)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Random walk embedder
        self.random_walk_embedder = RandomWalkEmbedder(embedding_dim=hidden_dim)
    
    def forward(self, code_graph: CodeGraph, history_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Generate random walk embeddings
        walks = self.random_walk_embedder.generate_random_walks(code_graph)
        vocab_size = len(code_graph.nodes)
        random_walk_embeddings = self.random_walk_embedder.compute_node_embeddings(walks, vocab_size)
        
        # Encode code graph
        graph_embedding = self.graph_encoder(code_graph, random_walk_embeddings)
        
        # Encode transformation history
        history_embedding = self.history_encoder(history_sequence)
        
        # Fuse embeddings
        combined = torch.cat([graph_embedding, history_embedding], dim=-1)
        fused = self.fusion(combined)
        
        # Predict policy and value
        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)
        
        return policy_logits, value
    
    def select_action(self, code_graph: CodeGraph, history_sequence: torch.Tensor,
                     valid_actions: List[TransformationType], 
                     temperature: float = 1.0) -> TransformationType:
        """Select next transformation"""
        with torch.no_grad():
            policy_logits, _ = self.forward(code_graph, history_sequence)
            
            # Apply temperature
            policy_logits = policy_logits / temperature
            
            # Mask invalid actions
            action_mask = torch.zeros(len(TransformationType))
            for action in valid_actions:
                action_idx = list(TransformationType).index(action)
                action_mask[action_idx] = 1.0
            
            # Apply mask
            masked_logits = policy_logits + torch.log(action_mask + 1e-8)
            
            # Sample action
            action_probs = F.softmax(masked_logits, dim=-1)
            action_idx = torch.multinomial(action_probs, 1).item()
            
            return list(TransformationType)[action_idx]

class RandomWalkAgent:
    """Agent that uses random walk exploration for transformation selection"""
    
    def __init__(self, policy_gnn: TransformationPolicyGNN, 
                 exploration_rate: float = 0.1, temperature: float = 1.0):
        self.policy_gnn = policy_gnn
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        
        # Statistics
        self.stats = {
            'exploration_actions': 0,
            'exploitation_actions': 0,
            'random_walk_steps': 0
        }
    
    def select_action(self, state: TransformationState, 
                     valid_actions: List[TransformationType]) -> TransformationType:
        """Select action using random walk exploration"""
        # Decide between exploration and exploitation
        if random.random() < self.exploration_rate:
            # Random walk exploration
            action = self._random_walk_exploration(valid_actions)
            self.stats['exploration_actions'] += 1
        else:
            # Policy exploitation
            action = self._policy_exploitation(state, valid_actions)
            self.stats['exploitation_actions'] += 1
        
        return action
    
    def _random_walk_exploration(self, valid_actions: List[TransformationType]) -> TransformationType:
        """Random walk exploration strategy"""
        # Perform random walk through action space
        walk_length = random.randint(1, 3)
        current_actions = valid_actions.copy()
        
        for _ in range(walk_length):
            if current_actions:
                # Randomly select from current actions
                selected = random.choice(current_actions)
                
                # Simulate transition to related actions
                # This is a simplified model - in practice, you'd have action relationships
                related_actions = self._get_related_actions(selected)
                current_actions = [a for a in related_actions if a in valid_actions]
                
                self.stats['random_walk_steps'] += 1
        
        # Return final action
        return random.choice(valid_actions) if valid_actions else TransformationType.VARIABLE_OPERATION
    
    def _policy_exploitation(self, state: TransformationState, 
                           valid_actions: List[TransformationType]) -> TransformationType:
        """Policy exploitation using trained GNN"""
        try:
            # Convert state to code graph (simplified)
            code_graph = self._state_to_code_graph(state)
            
            # Convert history to tensor
            history_tensor = self._history_to_tensor(state.transformation_history)
            
            # Use policy GNN to select action
            return self.policy_gnn.select_action(
                code_graph, history_tensor, valid_actions, self.temperature
            )
        except Exception as e:
            logger.warning(f"Error in policy exploitation: {e}")
            return random.choice(valid_actions) if valid_actions else TransformationType.VARIABLE_OPERATION
    
    def _get_related_actions(self, action: TransformationType) -> List[TransformationType]:
        """Get actions related to the given action"""
        # Define action relationships (simplified)
        relationships = {
            TransformationType.VARIABLE_OPERATION: [
                TransformationType.METHOD_EXTRACTION,
                TransformationType.MATHEMATICAL_EXPRESSION
            ],
            TransformationType.METHOD_EXTRACTION: [
                TransformationType.VARIABLE_OPERATION,
                TransformationType.LOOP_CONVERSION
            ],
            TransformationType.LOOP_CONVERSION: [
                TransformationType.GUARD_REVERSAL,
                TransformationType.VARIABLE_OPERATION
            ],
            TransformationType.GUARD_REVERSAL: [
                TransformationType.CONDITIONAL_EXPRESSION,
                TransformationType.TERNARY_OPERATOR
            ]
        }
        
        return relationships.get(action, list(TransformationType))
    
    def _state_to_code_graph(self, state: TransformationState) -> CodeGraph:
        """Convert transformation state to code graph (simplified)"""
        # This is a simplified conversion - in practice, you'd parse the actual code
        code = state.code
        
        # Create simple graph structure
        nodes = []
        edges = []
        node_types = []
        edge_types = []
        
        # Parse code into nodes (simplified)
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        for i, line in enumerate(lines):
            # Determine node type based on line content
            if 'for ' in line or 'while ' in line:
                node_type = 'loop'
            elif 'if ' in line:
                node_type = 'condition'
            elif 'return ' in line:
                node_type = 'return'
            elif '=' in line:
                node_type = 'assignment'
            else:
                node_type = 'statement'
            
            # Create node features
            features = [
                1.0 if 'for ' in line else 0.0,
                1.0 if 'while ' in line else 0.0,
                1.0 if 'if ' in line else 0.0,
                1.0 if 'return ' in line else 0.0,
                1.0 if '=' in line else 0.0,
                len(line) / 100.0,  # Normalized line length
                state.complexity_score / 10.0,  # Normalized complexity
                1.0 if state.compilation_status else 0.0
            ]
            
            nodes.append({'features': features})
            node_types.append(node_type)
            
            # Create edges (simplified - sequential connections)
            if i > 0:
                edges.append((i-1, i))
                edge_types.append('control_flow')
        
        # Create adjacency matrix
        num_nodes = len(nodes)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes)
        for src, dst in edges:
            adjacency_matrix[src, dst] = 1.0
        
        return CodeGraph(
            nodes=nodes,
            edges=edges,
            node_types=node_types,
            edge_types=edge_types,
            adjacency_matrix=adjacency_matrix
        )
    
    def _history_to_tensor(self, history: List[TransformationType]) -> torch.Tensor:
        """Convert transformation history to tensor"""
        # One-hot encoding of transformation history
        max_history = 10
        history_tensor = torch.zeros(1, max_history, len(TransformationType))
        
        for i, transformation in enumerate(history[-max_history:]):
            action_idx = list(TransformationType).index(transformation)
            history_tensor[0, i, action_idx] = 1.0
        
        return history_tensor


def main():
    """Test the transformation policy GNN"""
    logger.info("Testing Transformation Policy GNN...")
    
    # Create model
    policy_gnn = TransformationPolicyGNN(
        node_feature_dim=8,
        edge_feature_dim=5,
        hidden_dim=128,
        num_gnn_layers=2,
        num_heads=4
    )
    
    logger.info("Policy GNN created successfully")
    
    # Create random walk agent
    agent = RandomWalkAgent(policy_gnn, exploration_rate=0.2)
    
    logger.info("Random walk agent created successfully")
    
    # Test with dummy data
    dummy_graph = CodeGraph(
        nodes=[{'features': [0, 0, 1, 0, 1, 0.5, 0.3, 1]}],
        edges=[],
        node_types=['condition'],
        edge_types=[],
        adjacency_matrix=torch.eye(1)
    )
    
    dummy_history = torch.zeros(1, 5, len(TransformationType))
    
    try:
        policy_logits, value = policy_gnn(dummy_graph, dummy_history)
        logger.info(f"Forward pass successful: policy shape={policy_logits.shape}, value shape={value.shape}")
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")


if __name__ == '__main__':
    main()
