#!/usr/bin/env python3
"""
Checker Value Attention Module

Learnable attention mechanism that automatically learns which values to emphasize
for each Checker Framework checker during training. Replaces manual feature scaling
with learned emphasis weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from checker_config import CheckerType, get_checker_config

class CheckerValueAttention(nn.Module):
    """
    Learnable attention for checker-specific value emphasis.
    
    This module takes raw value pattern features and learns attention weights
    to automatically emphasize relevant values for each checker type.
    """
    
    def __init__(
        self,
        checker_type: CheckerType,
        pattern_dim: int,
        hidden_dim: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize checker-specific value attention module
        
        Args:
            checker_type: Type of checker (INDEX, NULLNESS, SIGNATURE, etc.)
            pattern_dim: Dimension of input pattern features
            hidden_dim: Hidden dimension for attention layers
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.checker_type = checker_type
        self.pattern_dim = pattern_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_attention_heads
        
        config = get_checker_config(checker_type)
        self.checker_name = config.get('name', 'Unknown')
        
        # Project pattern features to hidden dimension
        self.pattern_projection = nn.Linear(pattern_dim, hidden_dim)
        
        # Multi-head self-attention for learning value importance
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable emphasis scaling (replaces manual 1.5x-3.0x scaling)
        # This learns how much to emphasize each pattern
        self.emphasis_weights = nn.Parameter(torch.ones(pattern_dim))
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network for attention refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection back to pattern dimension
        self.output_projection = nn.Linear(hidden_dim, pattern_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize emphasis weights based on checker type
        self._init_emphasis_weights()
    
    def _init_emphasis_weights(self):
        """Initialize emphasis weights based on checker type"""
        # Initialize with small positive values to encourage learning
        # Higher initial values for patterns that are likely important
        with torch.no_grad():
            if self.checker_type == CheckerType.INDEX:
                # For Lower Bound Checker, emphasize zero and negative_one patterns
                # Initialize with slightly higher values for these
                self.emphasis_weights.fill_(1.0)
            elif self.checker_type == CheckerType.NULLNESS:
                # For Null Checker, emphasize null-related patterns
                self.emphasis_weights.fill_(1.0)
            elif self.checker_type == CheckerType.SIGNATURE:
                # For Signature String Checker, emphasize string patterns
                self.emphasis_weights.fill_(1.0)
            else:
                # Default initialization
                self.emphasis_weights.fill_(1.0)
    
    def forward(
        self,
        pattern_features: torch.Tensor,
        return_attention_weights: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply checker-specific value attention
        
        Args:
            pattern_features: Raw pattern features [batch_size, pattern_dim] or [pattern_dim]
            return_attention_weights: Whether to return attention weights for interpretability
            
        Returns:
            Emphasized features with same shape as input
            Optionally returns attention weights if return_attention_weights=True
        """
        # Handle single sample (add batch dimension)
        if pattern_features.dim() == 1:
            pattern_features = pattern_features.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = pattern_features.size(0)
        
        # Apply learnable emphasis weights (element-wise scaling)
        emphasized = pattern_features * self.emphasis_weights.unsqueeze(0)
        
        # Project to hidden dimension
        x = self.pattern_projection(emphasized)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention (treat each pattern as a sequence element)
        # Reshape to [batch_size, seq_len=1, hidden_dim] for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply multi-head self-attention
        # Query, key, value are all the same (self-attention)
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Residual connection and layer norm
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # Project back to pattern dimension
        x = self.output_projection(x)  # [batch_size, 1, pattern_dim]
        x = x.squeeze(1)  # [batch_size, pattern_dim]
        
        # Add residual connection from emphasized input
        output = emphasized + x
        
        # Remove batch dimension if single sample
        if single_sample:
            output = output.squeeze(0)
            if return_attention_weights:
                attn_weights = attn_weights.squeeze(0)
        
        if return_attention_weights:
            return output, attn_weights
        return output
    
    def get_emphasis_weights(self) -> torch.Tensor:
        """Get current emphasis weights for interpretability"""
        return self.emphasis_weights.detach()
    
    def get_attention_summary(self, pattern_features: torch.Tensor) -> Dict[str, float]:
        """
        Get summary of which patterns are being emphasized
        
        Args:
            pattern_features: Pattern features to analyze
            
        Returns:
            Dictionary mapping pattern names to emphasis scores
        """
        config = get_checker_config(self.checker_type)
        pattern_names = config.get('value_patterns', [])
        
        # Get emphasis weights
        weights = self.get_emphasis_weights().cpu().numpy()
        
        # Create summary
        summary = {}
        for i, pattern_name in enumerate(pattern_names):
            if i < len(weights):
                summary[pattern_name] = float(weights[i])
        
        return summary


class CheckerValueAttentionPool(nn.Module):
    """
    Pooled attention across multiple nodes for graph-based models.
    Applies value attention to each node and pools the results.
    """
    
    def __init__(
        self,
        checker_type: CheckerType,
        pattern_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        pooling: str = 'mean'  # 'mean', 'max', 'sum', 'attention'
    ):
        super().__init__()
        
        self.checker_type = checker_type
        self.pattern_dim = pattern_dim
        self.pooling = pooling
        
        # Individual attention module
        self.attention = CheckerValueAttention(
            checker_type=checker_type,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_heads
        )
        
        # Global attention pooling if needed
        if pooling == 'attention':
            self.global_attention = nn.MultiheadAttention(
                embed_dim=pattern_dim,
                num_heads=num_heads,
                batch_first=True
            )
    
    def forward(self, pattern_features: torch.Tensor) -> torch.Tensor:
        """
        Apply attention and pool across nodes
        
        Args:
            pattern_features: [num_nodes, pattern_dim] or [batch_size, num_nodes, pattern_dim]
            
        Returns:
            Pooled emphasized features
        """
        # Handle batch dimension
        if pattern_features.dim() == 2:
            # Single graph: [num_nodes, pattern_dim]
            pattern_features = pattern_features.unsqueeze(0)
            single_graph = True
        else:
            single_graph = False
        
        batch_size, num_nodes, pattern_dim = pattern_features.shape
        
        # Apply attention to each node
        # Reshape to [batch_size * num_nodes, pattern_dim]
        flat_features = pattern_features.view(-1, pattern_dim)
        attended = self.attention(flat_features)
        attended = attended.view(batch_size, num_nodes, pattern_dim)
        
        # Pool across nodes
        if self.pooling == 'mean':
            pooled = attended.mean(dim=1)
        elif self.pooling == 'max':
            pooled = attended.max(dim=1)[0]
        elif self.pooling == 'sum':
            pooled = attended.sum(dim=1)
        elif self.pooling == 'attention':
            # Use attention-based pooling
            # Query is learnable, keys/values are node features
            query = self.global_attention.query.weight.mean(dim=0).unsqueeze(0).unsqueeze(0)
            query = query.expand(batch_size, 1, pattern_dim)
            pooled, _ = self.global_attention(query, attended, attended)
            pooled = pooled.squeeze(1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        if single_graph:
            pooled = pooled.squeeze(0)
        
        return pooled

