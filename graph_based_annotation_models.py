#!/usr/bin/env python3
"""
Graph-based annotation type models that use CFG graphs directly as input.

This module provides sophisticated graph neural network models for predicting
Lower Bound Checker annotation types (@Positive, @NonNegative, @GTENegativeOne)
using control flow graphs as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import PNAConv, GATv2Conv
from torch_geometric.data import Data, Batch
import logging

logger = logging.getLogger(__name__)

class GraphBasedAnnotationModel(nn.Module):
    """Base class for graph-based annotation type models"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 pooling strategies
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, data):
        """Forward pass for graph-based models"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply graph convolutions (implemented in subclasses)
        x = self.graph_forward(x, edge_index)
        
        # Global pooling - use multiple strategies for robustness
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


class GraphBasedGCNModel(GraphBasedAnnotationModel):
    """Graph Convolutional Network for annotation type prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, num_layers: int = 3, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
    
    def graph_forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphBasedGATModel(GraphBasedAnnotationModel):
    """Graph Attention Network for annotation type prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, num_layers: int = 3, dropout: float = 0.1, heads: int = 4):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout)
        self.heads = heads
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
    
    def graph_forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphBasedTransformerModel(GraphBasedAnnotationModel):
    """Graph Transformer for annotation type prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, num_layers: int = 3, dropout: float = 0.1, heads: int = 4):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout)
        self.heads = heads
        
        # Transformer layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        edge_dim = 2  # Control flow and data flow edges
        
        for i in range(num_layers):
            self.convs.append(TransformerConv(
                hidden_dim, hidden_dim // heads, 
                heads=heads, 
                edge_dim=edge_dim,
                dropout=dropout,
                beta=True
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def graph_forward(self, x, edge_index):
        # Get edge attributes if available
        edge_attr = getattr(self, 'edge_attr', None)
        if edge_attr is None:
            # Create default edge attributes (control flow = 0, data flow = 1)
            edge_attr = torch.zeros(edge_index.size(1), 2, device=x.device)
            # Assume first half are control flow, second half are data flow
            mid = edge_index.size(1) // 2
            edge_attr[mid:, 1] = 1  # Data flow edges
        
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
        return x


class GraphBasedPNATModel(GraphBasedAnnotationModel):
    """Principal Neighbourhood Aggregation for annotation type prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, num_layers: int = 3, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, out_dim, num_layers, dropout)
        
        # PNA layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        aggregators = ['mean', 'max', 'min', 'std', 'sum']
        scalers = ['identity', 'amplification', 'attenuation']
        
        for i in range(num_layers):
            self.convs.append(PNAConv(
                hidden_dim, hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                edge_dim=2  # Control flow and data flow edges
            ))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
    
    def graph_forward(self, x, edge_index):
        # Get edge attributes if available
        edge_attr = getattr(self, 'edge_attr', None)
        if edge_attr is None:
            # Create default edge attributes
            edge_attr = torch.zeros(edge_index.size(1), 2, device=x.device)
            mid = edge_index.size(1) // 2
            edge_attr[mid:, 1] = 1
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphBasedHybridModel(nn.Module):
    """Hybrid model combining multiple graph neural network architectures"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Multiple graph models
        self.gcn_model = GraphBasedGCNModel(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.gat_model = GraphBasedGATModel(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.transformer_model = GraphBasedTransformerModel(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Project input
        x_proj = self.gcn_model.input_proj(x)
        
        # Get embeddings from different models
        gcn_emb = self.gcn_model.graph_forward(x_proj, edge_index)
        gat_emb = self.gat_model.graph_forward(x_proj, edge_index)
        trans_emb = self.transformer_model.graph_forward(x_proj, edge_index)
        
        # Global pooling for each
        gcn_pool = global_mean_pool(gcn_emb, batch)
        gat_pool = global_mean_pool(gat_emb, batch)
        trans_pool = global_mean_pool(trans_emb, batch)
        
        # Concatenate and fuse
        combined = torch.cat([gcn_pool, gat_pool, trans_pool], dim=1)
        return self.fusion(combined)


def create_graph_based_model(model_type: str, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, **kwargs) -> nn.Module:
    """Factory function to create graph-based annotation type models"""
    
    models = {
        'gcn': GraphBasedGCNModel,
        'gat': GraphBasedGATModel,
        'transformer': GraphBasedTransformerModel,
        'pna': GraphBasedPNATModel,
        'hybrid': GraphBasedHybridModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](input_dim, hidden_dim, out_dim, **kwargs)


class GraphBasedGBTModel:
    """Graph-based Gradient Boosting Tree model that uses graph embeddings"""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.model = None
        self.graph_encoder = None
        self.is_trained = False
    
    def set_graph_encoder(self, encoder):
        """Set the graph encoder for producing embeddings"""
        self.graph_encoder = encoder
    
    def fit(self, graph_data_list, targets):
        """Train the GBT model on graph embeddings"""
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Extract embeddings from graphs
        embeddings = []
        for graph_data in graph_data_list:
            if self.graph_encoder is not None:
                with torch.no_grad():
                    emb = self.graph_encoder(graph_data)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    embeddings.append(emb)
            else:
                # Fallback to zero embedding
                embeddings.append(np.zeros(self.embedding_dim))
        
        # Train GBT model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.model.fit(embeddings, targets)
        self.is_trained = True
    
    def predict_proba(self, graph_data_list):
        """Predict probabilities for graph data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Extract embeddings
        embeddings = []
        for graph_data in graph_data_list:
            if self.graph_encoder is not None:
                with torch.no_grad():
                    emb = self.graph_encoder(graph_data)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.embedding_dim))
        
        return self.model.predict_proba(embeddings)
    
    def predict(self, graph_data_list):
        """Predict classes for graph data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(self.predict_proba(graph_data_list))


# Legacy compatibility classes that wrap the new graph-based models
class AnnotationTypeGCNModel(GraphBasedGCNModel):
    """Legacy compatibility wrapper for GCN model"""
    pass

class AnnotationTypeGATModel(GraphBasedGATModel):
    """Legacy compatibility wrapper for GAT model"""
    pass

class AnnotationTypeTransformerModel(GraphBasedTransformerModel):
    """Legacy compatibility wrapper for Transformer model"""
    pass

class AnnotationTypeHGTModel(GraphBasedTransformerModel):
    """Legacy compatibility wrapper - HGT uses Transformer architecture"""
    pass

class AnnotationTypeGCSNModel(GraphBasedPNATModel):
    """Legacy compatibility wrapper - GCSN uses PNA architecture"""
    pass

class AnnotationTypeDG2NModel(GraphBasedHybridModel):
    """Legacy compatibility wrapper - DG2N uses hybrid architecture"""
    pass

class AnnotationTypeCausalModel(GraphBasedTransformerModel):
    """Legacy compatibility wrapper - Causal uses Transformer architecture"""
    pass

class AnnotationTypeEnhancedCausalModel(GraphBasedHybridModel):
    """Legacy compatibility wrapper - Enhanced Causal uses hybrid architecture"""
    pass
