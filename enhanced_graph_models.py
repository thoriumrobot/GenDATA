#!/usr/bin/env python3
"""
Enhanced Graph-Based Models with Large Input Support

This module provides enhanced versions of the graph-based annotation type models
that can handle variable-sized CFG inputs with proper batching and dynamic sizing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import PNAConv, GATv2Conv
from torch_geometric.data import Data, Batch
import logging
from typing import Dict, Any, Optional, Tuple

# Import our CFG dataloader
from cfg_dataloader import CFGSizeConfig, CFGBatchProcessor

logger = logging.getLogger(__name__)

class EnhancedGraphBasedModel(nn.Module):
    """Enhanced base class for graph-based annotation type models with large input support"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256,  # Increased for better capacity
                 out_dim: int = 2, 
                 num_layers: int = 4,    # Increased layers
                 dropout: float = 0.1,
                 max_nodes: int = None,
                 use_attention_pooling: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_nodes = max_nodes or CFGSizeConfig.MAX_NODES
        self.use_attention_pooling = use_attention_pooling
        
        # Input projection with layer normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced classification head with residual connections
        if use_attention_pooling:
            self.pooling_dim = hidden_dim * 4  # Multiple pooling strategies + attention
        else:
            self.pooling_dim = hidden_dim * 3  # Multiple pooling strategies
        
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
        # Attention pooling for better graph-level representation
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(hidden_dim)
        else:
            self.attention_pool = None
    
    def forward(self, data):
        """Forward pass with enhanced graph processing"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply graph convolutions (implemented in subclasses)
        x = self.graph_forward(x, edge_index)
        
        # Enhanced global pooling
        if self.use_attention_pooling and hasattr(self, 'attention_pool'):
            # Use attention pooling + multiple strategies
            x_att = self.attention_pool(x, batch)
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            
            # Concatenate all pooling results
            x_global = torch.cat([x_att, x_mean, x_max, x_sum], dim=1)
        else:
            # Use multiple pooling strategies
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            
            # Concatenate pooling results
            x_global = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Classification
        return self.classifier(x_global)
    
    def graph_forward(self, x, edge_index):
        """Graph convolution forward pass - implemented by subclasses"""
        raise NotImplementedError


class AttentionPooling(nn.Module):
    """Attention-based global pooling for better graph representation"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, batch):
        """Apply attention pooling"""
        # Compute attention weights
        att_weights = self.attention(x).squeeze(-1)
        
        # Apply softmax within each batch
        att_weights = torch.softmax(att_weights, dim=0)
        
        # Weighted sum
        x_weighted = x * att_weights.unsqueeze(-1)
        
        # Global sum pooling
        return global_add_pool(x_weighted, batch)


class EnhancedGCNModel(EnhancedGraphBasedModel):
    """Enhanced GCN model with residual connections and better capacity for GPU acceleration"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 2, 
                 num_layers: int = 6, dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs)
        
        # GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def graph_forward(self, x, edge_index):
        """Enhanced GCN forward pass with residual connections"""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_res
                
        return x


class EnhancedGATModel(EnhancedGraphBasedModel):
    """Enhanced GAT model with multi-head attention and residual connections - GPU optimized"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 2, 
                 num_layers: int = 6, dropout: float = 0.1, heads: int = 16, **kwargs):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs)
        self.heads = heads
        
        # GAT layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def graph_forward(self, x, edge_index):
        """Enhanced GAT forward pass with residual connections"""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_res
                
        return x


class EnhancedTransformerModel(EnhancedGraphBasedModel):
    """Enhanced Graph Transformer with edge encodings and residual connections - GPU optimized"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 2, 
                 num_layers: int = 6, dropout: float = 0.1, heads: int = 16, **kwargs):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs)
        self.heads = heads
        
        # Transformer layers with edge encodings
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        edge_dim = CFGSizeConfig.EDGE_FEATURE_DIM
        
        for i in range(num_layers):
            self.convs.append(TransformerConv(
                hidden_dim, hidden_dim // heads, 
                heads=heads, 
                edge_dim=edge_dim,
                dropout=dropout,
                beta=True  # Enable beta parameter for better attention
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def graph_forward(self, x, edge_index):
        """Enhanced Transformer forward pass with residual connections"""
        # Get edge attributes if available
        edge_attr = getattr(self, 'edge_attr', None)
        if edge_attr is None:
            # Create default edge attributes
            edge_attr = torch.zeros(edge_index.size(1), CFGSizeConfig.EDGE_FEATURE_DIM, device=x.device)
            # Assume first half are control flow, second half are data flow
            mid = edge_index.size(1) // 2
            edge_attr[mid:, 1] = 1  # Data flow edges
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            x = x + x_res
            
        return x


class EnhancedHybridModel(nn.Module):
    """Enhanced hybrid model combining multiple architectures with better fusion - GPU optimized"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 2, 
                 num_layers: int = 6, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Multiple enhanced graph models
        self.gcn_model = EnhancedGCNModel(input_dim, hidden_dim, hidden_dim, num_layers, dropout, **kwargs)
        self.gat_model = EnhancedGATModel(input_dim, hidden_dim, hidden_dim, num_layers, dropout, **kwargs)
        self.transformer_model = EnhancedTransformerModel(input_dim, hidden_dim, hidden_dim, num_layers, dropout, **kwargs)
        
        # Enhanced fusion with attention
        fusion_dim = hidden_dim * 4 * 3  # 3 models * 4 pooling strategies each
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 4,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, data):
        """Enhanced hybrid forward pass with attention fusion"""
        # Get embeddings from different models
        gcn_emb = self.gcn_model.graph_forward(
            self.gcn_model.input_proj(data.x), 
            data.edge_index
        )
        gat_emb = self.gat_model.graph_forward(
            self.gat_model.input_proj(data.x), 
            data.edge_index
        )
        trans_emb = self.transformer_model.graph_forward(
            self.transformer_model.input_proj(data.x), 
            data.edge_index
        )
        
        # Global pooling for each
        gcn_pool = self._global_pooling(gcn_emb, data.batch)
        gat_pool = self._global_pooling(gat_emb, data.batch)
        trans_pool = self._global_pooling(trans_emb, data.batch)
        
        # Stack for attention fusion
        pooled_features = torch.stack([gcn_pool, gat_pool, trans_pool], dim=1)  # [batch, 3, features]
        
        # Apply attention fusion
        attended_features, _ = self.fusion_attention(pooled_features, pooled_features, pooled_features)
        
        # Flatten and fuse
        combined = attended_features.reshape(attended_features.size(0), -1)
        return self.fusion(combined)
    
    def _global_pooling(self, x, batch):
        """Apply multiple global pooling strategies"""
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_att = self.gcn_model.attention_pool(x, batch)
        
        return torch.cat([x_att, x_mean, x_max, x_sum], dim=1)


# ============================================================================
# EMBEDDING-BASED MODELS (for GBT, Causal, Enhanced Causal)
# These models take sophisticated graph embeddings as input, not direct graphs
# ============================================================================

class EnhancedEmbeddingBaseModel(nn.Module):
    """Base class for enhanced embedding-based models"""
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 512, out_dim: int = 2, 
                 dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_attention = use_attention
        
        # Graph encoder for creating embeddings from CFG graphs
        from graph_encoder import build_graph_encoder
        self.graph_encoder = build_graph_encoder(
            in_dim=15,  # CFG node features
            edge_dim=2,  # CFG edge features
            out_dim=embedding_dim,
            variant='transformer'
        )
        
        # Embedding processor
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for embedding processing
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, out_dim)
        )
    
    def forward(self, data):
        # First, encode the graph into embeddings
        embeddings = self.graph_encoder(data)  # [batch_size, embedding_dim]
        
        # Process embeddings
        processed = self.embedding_processor(embeddings)  # [batch_size, hidden_dim]
        
        # Apply attention if enabled (treat as sequence of length 1)
        if self.use_attention:
            processed_seq = processed.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            attended, _ = self.attention(processed_seq, processed_seq, processed_seq)
            processed = attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Final classification
        return self.classifier(processed)


class EnhancedEmbeddingGBTModel(EnhancedEmbeddingBaseModel):
    """Enhanced GBT model that operates on graph embeddings"""
    
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256, out_dim: int = 2, **kwargs):
        super().__init__(embedding_dim, hidden_dim, out_dim, use_attention=False, **kwargs)
        
        # GBT-specific processing (gradient boosting inspired)
        self.boost_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)  # 3 boosting stages
        ])
        
        # Weighted combination of boosting stages
        self.boost_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Override classifier to handle correct input dimensions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Input is hidden_dim // 2 from boosting
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, out_dim)
        )
    
    def forward(self, data):
        # Get base embedding
        embeddings = self.graph_encoder(data)
        processed = self.embedding_processor(embeddings)
        
        # Apply boosting layers
        boost_outputs = []
        for layer in self.boost_layers:
            boost_outputs.append(layer(processed))
        
        # Weighted combination (like gradient boosting)
        weighted_output = sum(w * output for w, output in zip(self.boost_weights, boost_outputs))
        
        # Final classification - ensure dimensions match
        return self.classifier(weighted_output)


class EnhancedEmbeddingCausalModel(EnhancedEmbeddingBaseModel):
    """Enhanced Causal model that operates on graph embeddings with causal attention"""
    
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256, out_dim: int = 2, **kwargs):
        super().__init__(embedding_dim, hidden_dim, out_dim, use_attention=True, **kwargs)
        
        # Causal-specific components
        self.causal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.causal_norm = nn.LayerNorm(hidden_dim)
        
        # Causal relationship processor
        self.causal_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Enhanced classifier with causal features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, data):
        # Get base embedding
        embeddings = self.graph_encoder(data)
        processed = self.embedding_processor(embeddings)
        
        # Apply causal attention
        processed_seq = processed.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        causal_attended, _ = self.causal_attention(processed_seq, processed_seq, processed_seq)
        causal_features = self.causal_norm(causal_attended.squeeze(1))
        
        # Process causal relationships
        causal_processed = self.causal_processor(causal_features)
        
        # Combine causal and base features
        combined = torch.cat([causal_features, causal_processed], dim=1)
        
        # Final classification
        return self.classifier(combined)


class EnhancedEmbeddingHybridModel(EnhancedEmbeddingBaseModel):
    """Enhanced Hybrid model for embeddings (Enhanced Causal variant)"""
    
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256, out_dim: int = 2, **kwargs):
        super().__init__(embedding_dim, hidden_dim, out_dim, use_attention=True, **kwargs)
        
        # Multiple embedding processors (like hybrid graph model)
        self.gbt_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.causal_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.transformer_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion attention
        self.fusion_attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, batch_first=True)
        
        # Final fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, data):
        # Get base embedding
        embeddings = self.graph_encoder(data)
        processed = self.embedding_processor(embeddings)
        
        # Process with different architectures
        gbt_features = self.gbt_processor(processed)
        causal_features = self.causal_processor(processed)
        transformer_features = self.transformer_processor(processed)
        
        # Fusion with attention
        combined = torch.stack([gbt_features, causal_features, transformer_features], dim=1)
        fused, _ = self.fusion_attention(combined, combined, combined)
        fused = fused.mean(dim=1)  # [batch_size, hidden_dim // 2]
        
        # Concatenate all features
        all_features = torch.cat([gbt_features, causal_features, transformer_features], dim=1)
        
        # Final classification
        return self.fusion(all_features)


# ============================================================================
# NATIVE GRAPH CAUSAL MODELS (Direct CFG Processing with Causal Mechanisms)
# These models process CFG graphs directly with causal attention mechanisms
# ============================================================================

class EnhancedGraphCausalModel(EnhancedGraphBasedModel):
    """
    Enhanced Graph Causal model with native graph input support.
    
    This model processes CFG graphs directly using graph convolutions
    and causal attention mechanisms, without requiring embedding conversion.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, out_dim: int = 2, 
                 num_layers: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs)
        
        # Causal graph convolution layers
        self.causal_convs = nn.ModuleList()
        self.causal_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.causal_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.causal_norms.append(nn.LayerNorm(hidden_dim))
        
        # Causal attention mechanism for modeling causal relationships
        self.causal_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Causal relationship encoder
        self.causal_relationship_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Enhanced pooling dimension (includes causal features)
        self.pooling_dim = hidden_dim * 4 + hidden_dim // 4  # 4 pooling strategies + causal features
        
        # Enhanced classifier with causal features
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def graph_forward(self, x, edge_index):
        """
        Apply causal graph convolutions with residual connections.
        """
        for i, (conv, norm) in enumerate(zip(self.causal_convs, self.causal_norms)):
            x_res = x
            
            # Apply graph convolution
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_res
        
        return x
    
    def apply_causal_attention(self, x, batch):
        """
        Apply causal attention mechanism to model causal relationships.
        """
        batch_size = batch.max().item() + 1
        
        # Apply attention within each graph
        attended_features = []
        for graph_id in range(batch_size):
            # Get nodes for this graph
            mask = batch == graph_id
            graph_nodes = x[mask]  # [num_nodes_in_graph, hidden_dim]
            
            if graph_nodes.size(0) == 0:
                continue
            
            # Apply self-attention for causal modeling
            graph_nodes_seq = graph_nodes.unsqueeze(0)  # [1, num_nodes, hidden_dim]
            attended_nodes, _ = self.causal_attention(
                graph_nodes_seq, graph_nodes_seq, graph_nodes_seq
            )
            attended_nodes = attended_nodes.squeeze(0)  # [num_nodes, hidden_dim]
            
            # Apply layer norm with residual connection
            attended_nodes = self.attention_norm(attended_nodes + graph_nodes)
            attended_features.append(attended_nodes)
        
        # Concatenate back
        if attended_features:
            x = torch.cat(attended_features, dim=0)
        
        return x
    
    def forward(self, data):
        """
        Enhanced forward pass with causal relationship modeling.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply causal graph convolutions
        x = self.graph_forward(x, edge_index)
        
        # Apply causal attention
        x = self.apply_causal_attention(x, batch)
        
        # Enhanced global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        
        # Apply attention pooling if available
        if self.use_attention_pooling and hasattr(self, 'attention_pool'):
            x_att = self.attention_pool(x, batch)
            x_pooled = torch.cat([x_att, x_mean, x_max, x_sum], dim=1)
        else:
            x_pooled = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # Model causal relationships
        x_causal = self.causal_relationship_encoder(x)
        x_causal_pooled = global_mean_pool(x_causal, batch)
        
        # Combine all features
        x_combined = torch.cat([x_pooled, x_causal_pooled], dim=1)
        
        # Final classification
        return self.classifier(x_combined)


class GraphITECausalModel(EnhancedGraphBasedModel):
    """
    GraphITE-inspired model for CFG-based causal inference.
    
    Based on "GraphITE: Individual Treatment Effect Estimation for Graph Data"
    (Harada & Kashima, 2021). This model treats CFG structures as graph treatments
    and estimates causal effects of different CFG configurations on annotation placement.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, out_dim: int = 2, 
                 num_layers: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs)
        
        # GNN encoder for graph treatments (CFG structures)
        self.treatment_encoder = nn.ModuleList()
        self.treatment_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.treatment_encoder.append(GCNConv(hidden_dim, hidden_dim))
            self.treatment_norms.append(nn.LayerNorm(hidden_dim))
        
        # Treatment effect estimator (causal effect of CFG structure)
        self.treatment_effect_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Counterfactual predictor (what-if scenarios)
        self.counterfactual_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
        # Treatment effect pooling
        self.treatment_pooling_dim = hidden_dim + hidden_dim // 4  # treatment + effect features
        
        # Final classifier combining treatment effects and counterfactuals
        self.classifier = nn.Sequential(
            nn.Linear(self.treatment_pooling_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def encode_treatment(self, x, edge_index):
        """
        Encode CFG structure as a treatment representation.
        """
        for i, (conv, norm) in enumerate(zip(self.treatment_encoder, self.treatment_norms)):
            x_res = x
            
            # Apply treatment encoding
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_res
        
        return x
    
    def estimate_treatment_effect(self, treatment_embedding):
        """
        Estimate the causal effect of CFG structure (treatment).
        """
        return self.treatment_effect_estimator(treatment_embedding)
    
    def predict_counterfactual(self, treatment_embedding):
        """
        Predict counterfactual outcomes for different CFG structures.
        """
        return self.counterfactual_predictor(treatment_embedding)
    
    def forward(self, data):
        """
        GraphITE forward pass: treatment effect estimation + counterfactual prediction.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Encode CFG structure as treatment
        treatment_embedding = self.encode_treatment(x, edge_index)
        
        # Global pooling for treatment representation
        treatment_global = global_mean_pool(treatment_embedding, batch)
        
        # Estimate treatment effect (causal effect of CFG structure)
        treatment_effect = self.estimate_treatment_effect(treatment_global)
        
        # Predict counterfactual outcomes
        counterfactual = self.predict_counterfactual(treatment_global)
        
        # Combine treatment and effect features
        combined_features = torch.cat([treatment_global, treatment_effect], dim=1)
        
        # Final classification
        return self.classifier(combined_features)


def create_enhanced_model(model_type: str, input_dim: int, hidden_dim: int = 256, 
                         out_dim: int = 2, **kwargs) -> nn.Module:
    """Factory function to create enhanced models with proper input handling"""
    
    # Models that take direct graph inputs (CFG graphs)
    graph_input_models = {
        'gcn': EnhancedGCNModel,
        'gat': EnhancedGATModel,
        'transformer': EnhancedTransformerModel,
        'hybrid': EnhancedHybridModel,
        'hgt': EnhancedGATModel,  # HGT uses GAT architecture (distinct from transformer)
        'gcsn': EnhancedTransformerModel,  # GCSN uses Transformer architecture (distinct from GCN)
        'dg2n': EnhancedGATModel,  # DG2N uses GAT architecture (distinct from GCN)
        # Enhanced variants that take graph inputs
        'enhanced_gcn': EnhancedGCNModel,
        'enhanced_gat': EnhancedGATModel,
        'enhanced_transformer': EnhancedTransformerModel,
        'enhanced_hybrid': EnhancedHybridModel,
        # Native graph causal models
        'graph_causal': EnhancedGraphCausalModel,  # NEW: Native graph causal
        'enhanced_graph_causal': EnhancedHybridModel,  # Enhanced variant uses Hybrid (distinct from GraphCausal)
        # GraphITE-inspired causal models
        'graphite': GraphITECausalModel,  # NEW: GraphITE treatment effect estimation
    }
    
    # Models that take sophisticated graph embeddings (not direct graphs)
    embedding_input_models = {
        'gbt': EnhancedEmbeddingGBTModel,  # GBT takes embeddings
        'causal': EnhancedEmbeddingCausalModel,  # Causal takes embeddings
        'enhanced_causal': EnhancedEmbeddingHybridModel,  # Enhanced Causal takes embeddings
    }
    
    # Check which category the model belongs to
    if model_type in graph_input_models:
        return graph_input_models[model_type](input_dim, hidden_dim, out_dim, **kwargs)
    elif model_type in embedding_input_models:
        return embedding_input_models[model_type](input_dim, hidden_dim, out_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported enhanced model type: {model_type}. Available graph inputs: {list(graph_input_models.keys())}, Available embedding inputs: {list(embedding_input_models.keys())}")


# Legacy compatibility classes
class AnnotationTypeEnhancedGCNModel(EnhancedGCNModel):
    """Legacy compatibility wrapper for enhanced GCN model"""
    pass

class AnnotationTypeEnhancedGATModel(EnhancedGATModel):
    """Legacy compatibility wrapper for enhanced GAT model"""
    pass

class AnnotationTypeEnhancedTransformerModel(EnhancedTransformerModel):
    """Legacy compatibility wrapper for enhanced Transformer model"""
    pass

class AnnotationTypeEnhancedHybridModel(EnhancedHybridModel):
    """Legacy compatibility wrapper for enhanced hybrid model"""
    pass
