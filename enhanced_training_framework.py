#!/usr/bin/env python3
"""
Enhanced Training Framework for Graph-Based Annotation Type Models

This module provides a comprehensive training framework that supports large CFG inputs,
proper batching, and enhanced model architectures for all annotation type models.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
from collections import defaultdict
import random

# Import our enhanced components
from cfg_dataloader import CFGDataLoader, CFGSizeConfig, find_cfg_files, CFGBatchProcessor
from enhanced_graph_models import (
    create_enhanced_model,
    EnhancedGCNModel,
    EnhancedGATModel, 
    EnhancedTransformerModel,
    EnhancedHybridModel
)

logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Enhanced trainer for graph-based annotation type models with large input support"""
    
    def __init__(self, 
                 models_dir: str = 'models_annotation_types',
                 device: str = 'cpu',
                 max_nodes: int = 1000,
                 max_edges: int = 2000,
                 max_batch_size: int = 16):
        """
        Initialize enhanced trainer
        
        Args:
            models_dir: Directory to save models
            device: Device to use for training
            max_nodes: Maximum nodes per graph
            max_edges: Maximum edges per graph  
            max_batch_size: Maximum batch size
        """
        self.models_dir = models_dir
        self.device = device
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_batch_size = max_batch_size
        
        # Update global config
        CFGSizeConfig.update_limits(max_nodes, max_edges, max_batch_size)
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Training history
        self.training_history = defaultdict(list)
        
        # TensorBoard writer
        self.writer = None
        
    def create_training_data(self, 
                           cfg_files: List[str],
                           targets: Optional[List[int]] = None,
                           synthetic_ratio: float = 0.3) -> Tuple[CFGDataLoader, CFGDataLoader]:
        """
        Create training and validation dataloaders
        
        Args:
            cfg_files: List of CFG file paths
            targets: Optional target labels
            synthetic_ratio: Ratio of synthetic data to add
        """
        # Create synthetic targets if not provided
        if targets is None:
            targets = [random.randint(0, 1) for _ in cfg_files]
        
        # Add synthetic data for better training
        if synthetic_ratio > 0:
            synthetic_files, synthetic_targets = self._generate_synthetic_data(
                len(cfg_files), synthetic_ratio
            )
            cfg_files.extend(synthetic_files)
            targets.extend(synthetic_targets)
        
        # Split data for training and validation
        split_idx = int(0.8 * len(cfg_files))
        train_files = cfg_files[:split_idx]
        train_targets = targets[:split_idx]
        val_files = cfg_files[split_idx:]
        val_targets = targets[split_idx:]
        
        # Create dataloaders
        train_loader = CFGDataLoader(
            cfg_files=train_files,
            targets=train_targets,
            batch_size=self.max_batch_size,
            shuffle=True,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges
        )
        
        val_loader = CFGDataLoader(
            cfg_files=val_files,
            targets=val_targets,
            batch_size=self.max_batch_size,
            shuffle=False,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges
        )
        
        logger.info(f"Created training dataloader: {len(train_loader)} batches")
        logger.info(f"Created validation dataloader: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _generate_synthetic_data(self, base_size: int, ratio: float) -> Tuple[List[str], List[int]]:
        """Generate synthetic CFG data for training augmentation"""
        synthetic_count = int(base_size * ratio)
        
        # For now, return empty lists - in production, this would generate synthetic CFGs
        # This is a placeholder for future implementation
        logger.info(f"Would generate {synthetic_count} synthetic CFG files")
        return [], []
    
    def train_model(self, 
                   annotation_type: str,
                   base_model_type: str = 'enhanced_hybrid',
                   epochs: int = 50,
                   learning_rate: float = 0.001,
                   weight_decay: float = 1e-4,
                   cfg_files: Optional[List[str]] = None,
                   targets: Optional[List[int]] = None,
                   use_tensorboard: bool = True) -> Dict[str, Any]:
        """
        Train an enhanced model for the specified annotation type
        
        Args:
            annotation_type: Type of annotation (@Positive, @NonNegative, @GTENegativeOne)
            base_model_type: Type of model architecture
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            cfg_files: Optional CFG files for training
            targets: Optional target labels
            use_tensorboard: Whether to use TensorBoard logging
        """
        logger.info(f"Training {annotation_type} model with {base_model_type} architecture")
        
        # Setup TensorBoard
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = os.path.join(self.models_dir, 'tensorboard', annotation_type)
            self.writer = SummaryWriter(log_dir)
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available, disabling logging")
            use_tensorboard = False
        
        # Create model
        model = self._create_enhanced_model(base_model_type, annotation_type)
        model = model.to(self.device)
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Create training data
        if cfg_files is None:
            # Use synthetic training data
            train_loader, val_loader = self._create_synthetic_dataloaders()
        else:
            train_loader, val_loader = self.create_training_data(cfg_files, targets)
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        training_stats = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'learning_rates': []
        }
        
        model.train()
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            training_stats['train_losses'].append(train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['train_accuracies'].append(train_acc)
            training_stats['val_accuracies'].append(val_acc)
            training_stats['learning_rates'].append(current_lr)
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")
        
        # Save best model
        model_name = annotation_type.replace('@', '').lower()
        model_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_model.pth")
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': best_model_state,
            'annotation_type': annotation_type,
            'base_model_type': base_model_type,
            'input_dim': CFGSizeConfig.NODE_FEATURE_DIM,
            'hidden_dim': model.hidden_dim,
            'out_dim': model.out_dim,
            'max_nodes': self.max_nodes,
            'max_edges': self.max_edges,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'best_val_loss': best_val_loss,
            'training_stats': training_stats
        }
        
        torch.save(checkpoint, model_file)
        logger.info(f"✅ Saved {annotation_type} model to {model_file}")
        
        # Save training statistics
        stats_file = os.path.join(self.models_dir, f"{model_name}_{base_model_type}_stats.json")
        stats_data = {
            'annotation_type': annotation_type,
            'base_model_type': base_model_type,
            'final_train_loss': training_stats['train_losses'][-1],
            'final_val_loss': training_stats['val_losses'][-1],
            'final_train_accuracy': training_stats['train_accuracies'][-1],
            'final_val_accuracy': training_stats['val_accuracies'][-1],
            'best_val_loss': best_val_loss,
            'epochs': epochs,
            'max_nodes': self.max_nodes,
            'max_edges': self.max_edges,
            'training_samples': len(train_loader.dataset),
            'validation_samples': len(val_loader.dataset)
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"✅ Saved {annotation_type} stats to {stats_file}")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return {
            'success': True,
            'model_file': model_file,
            'stats_file': stats_file,
            'best_val_loss': best_val_loss,
            'final_val_accuracy': training_stats['val_accuracies'][-1]
        }
    
    def _create_enhanced_model(self, base_model_type: str, annotation_type: str) -> nn.Module:
        """Create an enhanced model for the specified type"""
        model_type = base_model_type.replace('enhanced_', '')
        
        return create_enhanced_model(
            model_type=model_type,
            input_dim=CFGSizeConfig.NODE_FEATURE_DIM,
            hidden_dim=256,
            out_dim=2,
            num_layers=4,
            dropout=0.1,
            max_nodes=self.max_nodes
        )
    
    def _create_synthetic_dataloaders(self) -> Tuple[CFGDataLoader, CFGDataLoader]:
        """Create synthetic dataloaders for demonstration"""
        # Find available CFG files for structure reference
        cfg_dir = "/home/ubuntu/GenDATA/test_case_study_cfg_output"
        available_files = find_cfg_files(cfg_dir)
        
        if available_files:
            # Use available files for both training and validation
            targets = [random.randint(0, 1) for _ in available_files]
            
            # Split for train/val
            split_idx = len(available_files) // 2
            train_files = available_files[:split_idx]
            train_targets = targets[:split_idx]
            val_files = available_files[split_idx:]
            val_targets = targets[split_idx:]
            
            train_loader = CFGDataLoader(train_files, train_targets, batch_size=2, max_nodes=self.max_nodes)
            val_loader = CFGDataLoader(val_files, val_targets, batch_size=2, max_nodes=self.max_nodes)
            
            return train_loader, val_loader
        else:
            # Create minimal synthetic dataloaders
            logger.warning("No CFG files found, creating minimal synthetic dataloaders")
            return self._create_minimal_dataloaders()
    
    def _create_minimal_dataloaders(self) -> Tuple[CFGDataLoader, CFGDataLoader]:
        """Create minimal dataloaders when no CFG files are available"""
        # This is a fallback - in production, you'd want to generate proper synthetic CFGs
        from cfg_dataloader import CFGDataset
        
        # Create empty datasets
        train_dataset = CFGDataset([], [])
        val_dataset = CFGDataset([], [])
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        return train_loader, val_loader
    
    def _train_epoch(self, model, dataloader, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = CFGBatchProcessor.move_to_device(batch, self.device)
            targets = CFGBatchProcessor.extract_targets(batch)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct_predictions += (pred == targets).sum().item()
            total_predictions += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model, dataloader, criterion) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = CFGBatchProcessor.move_to_device(batch, self.device)
                targets = CFGBatchProcessor.extract_targets(batch)
                
                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, targets)
                
                # Statistics
                total_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                correct_predictions += (pred == targets).sum().item()
                total_predictions += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train_all_models(self, 
                        base_model_type: str = 'enhanced_hybrid',
                        epochs: int = 50,
                        cfg_files: Optional[List[str]] = None) -> Dict[str, bool]:
        """Train all annotation type models"""
        annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        results = {}
        
        logger.info(f"Training all enhanced models with {base_model_type} architecture")
        
        for annotation_type in annotation_types:
            try:
                result = self.train_model(
                    annotation_type=annotation_type,
                    base_model_type=base_model_type,
                    epochs=epochs,
                    cfg_files=cfg_files
                )
                results[annotation_type] = result['success']
                
            except Exception as e:
                logger.error(f"Failed to train {annotation_type} model: {e}")
                results[annotation_type] = False
        
        success_count = sum(results.values())
        logger.info(f"✅ Successfully trained {success_count}/{len(annotation_types)} enhanced models")
        
        return results


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train enhanced graph-based annotation type models')
    parser.add_argument('--base_model_type', default='enhanced_hybrid',
                       choices=['enhanced_gcn', 'enhanced_gat', 'enhanced_transformer', 'enhanced_hybrid'],
                       help='Enhanced model architecture to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--max_nodes', type=int, default=1000, help='Maximum nodes per graph')
    parser.add_argument('--max_edges', type=int, default=2000, help='Maximum edges per graph')
    parser.add_argument('--max_batch_size', type=int, default=16, help='Maximum batch size')
    parser.add_argument('--models_dir', default='models_annotation_types', help='Directory to save models')
    parser.add_argument('--device', default='cpu', help='Device to use for training')
    parser.add_argument('--cfg_dir', help='Directory containing CFG files for training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = EnhancedModelTrainer(
        models_dir=args.models_dir,
        device=args.device,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_batch_size=args.max_batch_size
    )
    
    # Find CFG files if directory provided
    cfg_files = None
    if args.cfg_dir:
        cfg_files = find_cfg_files(args.cfg_dir)
        logger.info(f"Found {len(cfg_files)} CFG files for training")
    
    # Train all models
    results = trainer.train_all_models(
        base_model_type=args.base_model_type,
        epochs=args.epochs,
        cfg_files=cfg_files
    )
    
    success = all(results.values())
    if success:
        logger.info("🎉 All enhanced models trained successfully!")
    else:
        logger.error("❌ Some enhanced models failed to train")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
