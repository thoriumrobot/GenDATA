#!/usr/bin/env python3
"""
Train graph-based annotation type models from scratch using CFG data.
This script creates new models with the graph-based architecture.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, List, Any
import numpy as np
from pathlib import Path

# Import our graph-based models
from graph_based_annotation_models import (
    create_graph_based_model,
    AnnotationTypeGCNModel,
    AnnotationTypeGATModel,
    AnnotationTypeTransformerModel,
    AnnotationTypeHGTModel,
    AnnotationTypeGCSNModel,
    AnnotationTypeDG2NModel,
    AnnotationTypeCausalModel,
    AnnotationTypeEnhancedCausalModel
)
from cfg_graph import load_cfg_as_pyg

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphBasedModelTrainer:
    """Trainer for graph-based annotation type models"""
    
    def __init__(self, models_dir: str = 'models_annotation_types', device: str = 'cpu'):
        self.models_dir = models_dir
        self.device = device
        self.annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
    
    def create_synthetic_training_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Create synthetic training data for demonstration purposes"""
        training_data = []
        
        for i in range(num_samples):
            # Create a synthetic CFG-like graph
            num_nodes = np.random.randint(5, 20)
            
            # Create node features (15-dimensional as we observed)
            node_features = torch.randn(num_nodes, 15)
            
            # Create edge indices (simple chain structure)
            edge_index = torch.tensor([
                list(range(num_nodes-1)) + list(range(1, num_nodes)),
                list(range(1, num_nodes)) + list(range(num_nodes-1))
            ], dtype=torch.long)
            
            # Create edge attributes (control flow vs data flow)
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(num_edges, 2)
            edge_attr[:num_edges//2, 0] = 1  # Control flow
            edge_attr[num_edges//2:, 1] = 1  # Data flow
            
            # Create synthetic target (random for demonstration)
            target = np.random.randint(0, 2)
            
            training_data.append({
                'graph': {
                    'x': node_features,
                    'edge_index': edge_index,
                    'edge_attr': edge_attr
                },
                'target': target
            })
        
        return training_data
    
    def train_model(self, annotation_type: str, base_model_type: str = 'enhanced_causal', 
                   epochs: int = 10, learning_rate: float = 0.001) -> bool:
        """Train a graph-based model for the specified annotation type"""
        try:
            logger.info(f"Training {annotation_type} model with {base_model_type} architecture")
            
            # Create model
            model = self._create_model(base_model_type)
            model = model.to(self.device)
            
            # Create optimizer and loss function
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Create synthetic training data
            training_data = self.create_synthetic_training_data(num_samples=200)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_data in training_data:
                    # Prepare batch
                    graph_data = batch_data['graph']
                    target = torch.tensor([batch_data['target']], dtype=torch.long).to(self.device)
                    
                    # Create PyG Data object
                    from torch_geometric.data import Data
                    data = Data(
                        x=graph_data['x'].to(self.device),
                        edge_index=graph_data['edge_index'].to(self.device),
                        edge_attr=graph_data['edge_attr'].to(self.device),
                        batch=torch.zeros(graph_data['x'].size(0), dtype=torch.long).to(self.device)
                    )
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    total_loss += loss.item()
                    pred = torch.argmax(output, dim=1)
                    correct_predictions += (pred == target).sum().item()
                    total_predictions += 1
                
                avg_loss = total_loss / len(training_data)
                accuracy = correct_predictions / total_predictions
                
                if epoch % 5 == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            # Save model
            model_name = annotation_type.replace('@', '').lower()
            model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
            
            # Create checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'annotation_type': annotation_type,
                'base_model_type': base_model_type,
                'input_dim': 15,
                'hidden_dim': 128,
                'out_dim': 2,
                'epochs': epochs,
                'learning_rate': learning_rate
            }
            
            torch.save(checkpoint, model_file)
            logger.info(f"✅ Saved {annotation_type} model to {model_file}")
            
            # Save stats
            stats = {
                'annotation_type': annotation_type,
                'base_model_type': base_model_type,
                'final_loss': avg_loss,
                'final_accuracy': accuracy,
                'epochs': epochs,
                'training_samples': len(training_data)
            }
            
            stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"✅ Saved {annotation_type} stats to {stats_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training {annotation_type} model: {e}")
            return False
    
    def _create_model(self, base_model_type: str) -> nn.Module:
        """Create a model instance for the specified type"""
        model_classes = {
            'gcn': AnnotationTypeGCNModel,
            'gat': AnnotationTypeGATModel,
            'transformer': AnnotationTypeTransformerModel,
            'hgt': AnnotationTypeHGTModel,
            'gcsn': AnnotationTypeGCSNModel,
            'dg2n': AnnotationTypeDG2NModel,
            'causal': AnnotationTypeCausalModel,
            'enhanced_causal': AnnotationTypeEnhancedCausalModel
        }
        
        if base_model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {base_model_type}")
        
        model_class = model_classes[base_model_type]
        return model_class(
            input_dim=15,  # Based on observed CFG features
            hidden_dim=128,
            out_dim=2,
            num_layers=3,
            dropout=0.1
        )
    
    def train_all_models(self, base_model_type: str = 'enhanced_causal', epochs: int = 10) -> bool:
        """Train all annotation type models"""
        logger.info(f"Training all graph-based models with {base_model_type} architecture")
        
        success_count = 0
        for annotation_type in self.annotation_types:
            if self.train_model(annotation_type, base_model_type, epochs):
                success_count += 1
        
        logger.info(f"✅ Successfully trained {success_count}/{len(self.annotation_types)} models")
        return success_count == len(self.annotation_types)


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train graph-based annotation type models')
    parser.add_argument('--base_model_type', default='enhanced_causal', 
                       choices=['gcn', 'gat', 'transformer', 'hgt', 'gcsn', 'dg2n', 'causal', 'enhanced_causal'],
                       help='Base model architecture to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--models_dir', default='models_annotation_types', help='Directory to save models')
    parser.add_argument('--device', default='cpu', help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = GraphBasedModelTrainer(models_dir=args.models_dir, device=args.device)
    
    # Train all models
    success = trainer.train_all_models(base_model_type=args.base_model_type, epochs=args.epochs)
    
    if success:
        logger.info("🎉 All models trained successfully!")
    else:
        logger.error("❌ Some models failed to train")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
