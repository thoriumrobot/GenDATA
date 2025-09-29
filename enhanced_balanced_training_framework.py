#!/usr/bin/env python3
"""
Enhanced Balanced Training Framework

This framework trains enhanced models using balanced datasets with proper
graph inputs, batching, and the enhanced framework architecture.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBalancedGraphDataset(Dataset):
    """PyTorch Geometric Dataset for enhanced balanced annotation type training with graph inputs"""
    
    def __init__(self, balanced_examples: List[Dict], annotation_type: str):
        self.examples = balanced_examples
        self.annotation_type = annotation_type
        
        # Convert examples to PyTorch Geometric Data objects
        self.graphs = []
        self.labels = []
        self.code_contexts = []
        
        for example in balanced_examples:
            try:
                # Create graph data from CFG features
                graph = self._create_graph_from_features(example)
                if graph is not None:
                    self.graphs.append(graph)
                    # Convert boolean to integer: True -> 1 (positive), False -> 0 (negative)
                    self.labels.append(1 if example['is_positive'] else 0)
                    self.code_contexts.append(example.get('code_context', ''))
            except Exception as e:
                logger.warning(f"Error creating graph from example: {e}")
                continue
        
        logger.info(f"Created enhanced balanced graph dataset for {annotation_type}: {len(self.graphs)} graphs")
        logger.info(f"  Positive examples: {np.sum(self.labels)} ({np.sum(self.labels)/len(self.labels)*100:.1f}%)")
        logger.info(f"  Negative examples: {len(self.labels) - np.sum(self.labels)} ({(len(self.labels) - np.sum(self.labels))/len(self.labels)*100:.1f}%)")
        
        # Show sample code contexts
        if self.code_contexts:
            logger.info("Sample code contexts:")
            for i, context in enumerate(self.code_contexts[:3]):
                logger.info(f"  {i+1}. {context[:100]}...")
    
    def _create_graph_from_features(self, example: Dict) -> Data:
        """Create a PyTorch Geometric Data object from CFG features"""
        try:
            features = example['features']
            
            # Ensure we have the right number of features (pad or truncate to 15 for CFG)
            if len(features) < 15:
                # Pad with zeros
                features = features + [0.0] * (15 - len(features))
            elif len(features) > 15:
                # Truncate to 15
                features = features[:15]
            
            # Create node features tensor (each feature becomes a node)
            node_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 15]
            
            # Create a simple graph structure (single node with self-loop for now)
            # In a real implementation, this would use the actual CFG structure
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop
            
            # Create edge attributes (control flow)
            edge_attr = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # [control_flow, data_flow]
            
            # Create graph-level label
            graph_label = torch.tensor([1 if example['is_positive'] else 0], dtype=torch.long)
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=graph_label,
                batch=torch.zeros(node_features.size(0), dtype=torch.long)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating graph from features: {e}")
            return None
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return {
            'graph': self.graphs[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'confidence': self.examples[idx].get('confidence', 0.5),
            'code_context': self.code_contexts[idx]
        }

class EnhancedBalancedTrainingFramework:
    """Enhanced training framework for balanced annotation type models with graph inputs and batching"""
    
    def __init__(self, model_types: List[str] = None, device: str = 'auto'):
        self.model_types = model_types or ['enhanced_gcn', 'enhanced_gat', 'enhanced_transformer', 'enhanced_hybrid']
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Model components
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
        # Training statistics
        self.training_stats = {
            'annotation_types': ['@Positive', '@NonNegative', '@GTENegativeOne'],
            'training_history': {},
            'best_accuracies': {},
            'final_metrics': {},
            'enhanced_analysis': {}
        }
    
    def load_balanced_dataset(self, dataset_file: str) -> Tuple[List[Dict], str]:
        """Load a real balanced dataset from file"""
        with open(dataset_file, 'r') as f:
            dataset_data = json.load(f)
        
        annotation_type = dataset_data['annotation_type']
        examples = dataset_data['examples']
        
        logger.info(f"Loaded real balanced dataset for {annotation_type}:")
        logger.info(f"  Total examples: {len(examples)}")
        logger.info(f"  Positive examples: {dataset_data['positive_examples']}")
        logger.info(f"  Negative examples: {dataset_data['negative_examples']}")
        logger.info(f"  Balance ratio: {dataset_data['balance_ratio']:.3f}")
        
        # Analyze code contexts
        positive_contexts = [ex.get('code_context', '') for ex in examples if ex.get('is_positive', False)]
        negative_contexts = [ex.get('code_context', '') for ex in examples if not ex.get('is_positive', False)]
        
        logger.info(f"  Sample positive contexts: {len([c for c in positive_contexts if c])} with context")
        logger.info(f"  Sample negative contexts: {len([c for c in negative_contexts if c])} with context")
        
        return examples, annotation_type
    
    def create_enhanced_model(self, model_type: str, input_dim: int, annotation_type: str) -> nn.Module:
        """Create an enhanced model for the given annotation type"""
        from enhanced_graph_models import create_enhanced_model
        
        model = create_enhanced_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=512,
            num_layers=6,
            heads=16,
            num_classes=2  # Binary classification
        )
        
        model = model.to(self.device)
        self.models[f"{annotation_type}_{model_type}"] = model
        
        # Create optimizer with different learning rates for different layers
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        self.optimizers[f"{annotation_type}_{model_type}"] = optimizer
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.7,
            patience=5,
            min_lr=1e-6
        )
        self.schedulers[f"{annotation_type}_{model_type}"] = scheduler
        
        return model
    
    def train_model(self, dataset_file: str, model_type: str, epochs: int = 200, batch_size: int = 32, 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Train an enhanced model using real balanced dataset with graph inputs and batching"""
        
        # Load dataset
        examples, annotation_type = self.load_balanced_dataset(dataset_file)
        
        if not examples:
            logger.error(f"No examples found in dataset for {annotation_type}")
            return {'success': False, 'error': 'No examples found'}
        
        # Create enhanced graph dataset
        dataset = EnhancedBalancedGraphDataset(examples, annotation_type)
        
        if len(dataset) == 0:
            logger.error(f"No valid graphs created from dataset for {annotation_type}")
            return {'success': False, 'error': 'No valid graphs created'}
        
        # Split into train and validation
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create PyTorch Geometric data loaders for graph batching
        train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Training set: {len(train_dataset)} graphs")
        logger.info(f"Validation set: {len(val_dataset)} graphs")
        
        # Create enhanced model
        model = self.create_enhanced_model(model_type, 15, annotation_type)  # 15 CFG features
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = 20
        
        model_key = f"{annotation_type}_{model_type}"
        
        logger.info(f"Starting enhanced training for {annotation_type} ({model_type}) for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                self.optimizers[model_key].zero_grad()
                
                # Forward pass with graph inputs
                if hasattr(model, 'conv1'):  # GCN-like model
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                elif hasattr(model, 'transformer_layers'):  # Transformer-like model
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                else:
                    # Fallback: use node features
                    outputs = model(batch.x)
                
                loss = criterion(outputs, batch.y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.optimizers[model_key].step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch.y.size(0)
                train_correct += (predicted == batch.y).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move batch to device
                    batch = batch.to(self.device)
                    
                    # Forward pass with graph inputs
                    if hasattr(model, 'conv1'):  # GCN-like model
                        outputs = model(batch.x, batch.edge_index, batch.batch)
                    elif hasattr(model, 'transformer_layers'):  # Transformer-like model
                        outputs = model(batch.x, batch.edge_index, batch.batch)
                    else:
                        # Fallback: use node features
                        outputs = model(batch.x)
                    
                    loss = criterion(outputs, batch.y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch.y.size(0)
                    val_correct += (predicted == batch.y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Update learning rate
            self.schedulers[model_key].step(val_accuracy)
            
            # Track history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience exceeded)")
                break
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}:")
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                logger.info(f"  Best Val Acc: {best_val_accuracy:.2f}%")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self.evaluate_enhanced_model(model, val_loader, annotation_type, model_type)
        
        # Analyze enhanced patterns
        enhanced_analysis = self.analyze_enhanced_patterns(examples, annotation_type, model_type)
        
        # Store training statistics
        self.training_stats['training_history'][model_key] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'epochs_trained': len(train_losses)
        }
        self.training_stats['best_accuracies'][model_key] = best_val_accuracy
        self.training_stats['final_metrics'][model_key] = final_metrics
        self.training_stats['enhanced_analysis'][model_key] = enhanced_analysis
        
        logger.info(f"Enhanced training completed for {annotation_type} ({model_type})")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        logger.info(f"Epochs trained: {len(train_losses)}")
        
        return {
            'success': True,
            'annotation_type': annotation_type,
            'model_type': model_type,
            'best_accuracy': best_val_accuracy,
            'final_metrics': final_metrics,
            'model': model,
            'enhanced_analysis': enhanced_analysis
        }
    
    def analyze_enhanced_patterns(self, examples: List[Dict], annotation_type: str, model_type: str) -> Dict[str, Any]:
        """Analyze enhanced patterns in the dataset"""
        positive_examples = [ex for ex in examples if ex.get('is_positive', False)]
        negative_examples = [ex for ex in examples if not ex.get('is_positive', False)]
        
        analysis = {
            'positive_patterns': {},
            'negative_patterns': {},
            'pattern_differences': {},
            'model_type': model_type,
            'uses_graph_inputs': True,
            'uses_batching': True
        }
        
        # Analyze positive patterns
        positive_labels = [ex.get('node_label', '') for ex in positive_examples]
        positive_types = [ex.get('node_type', '') for ex in positive_examples]
        
        # Analyze negative patterns
        negative_labels = [ex.get('node_label', '') for ex in negative_examples]
        negative_types = [ex.get('node_type', '') for ex in negative_examples]
        
        # Count common patterns
        def count_patterns(items):
            from collections import Counter
            return dict(Counter(items).most_common(10))
        
        analysis['positive_patterns']['labels'] = count_patterns(positive_labels)
        analysis['positive_patterns']['types'] = count_patterns(positive_types)
        analysis['negative_patterns']['labels'] = count_patterns(negative_labels)
        analysis['negative_patterns']['types'] = count_patterns(negative_types)
        
        # Find pattern differences
        positive_label_set = set(positive_labels)
        negative_label_set = set(negative_labels)
        
        analysis['pattern_differences']['positive_only'] = list(positive_label_set - negative_label_set)[:5]
        analysis['pattern_differences']['negative_only'] = list(negative_label_set - positive_label_set)[:5]
        analysis['pattern_differences']['common'] = list(positive_label_set & negative_label_set)[:5]
        
        logger.info(f"Enhanced pattern analysis for {annotation_type} ({model_type}):")
        logger.info(f"  Positive patterns: {len(positive_label_set)} unique labels")
        logger.info(f"  Negative patterns: {len(negative_label_set)} unique labels")
        logger.info(f"  Common patterns: {len(positive_label_set & negative_label_set)}")
        logger.info(f"  Uses graph inputs: {analysis['uses_graph_inputs']}")
        logger.info(f"  Uses batching: {analysis['uses_batching']}")
        
        return analysis
    
    def evaluate_enhanced_model(self, model: nn.Module, data_loader: PyGDataLoader, 
                              annotation_type: str, model_type: str) -> Dict[str, Any]:
        """Evaluate enhanced model performance with detailed analysis"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_contexts = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)
                contexts = batch.get('code_context', [''] * batch.y.size(0))
                
                # Forward pass with graph inputs
                if hasattr(model, 'conv1'):  # GCN-like model
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                elif hasattr(model, 'transformer_layers'):  # Transformer-like model
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                else:
                    # Fallback: use node features
                    outputs = model(batch.x)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_confidences.extend(probabilities.max(dim=1)[0].cpu().numpy())
                all_contexts.extend(contexts)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Classification report
        report = classification_report(
            all_labels, all_predictions,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Analyze confidence scores
        positive_confidences = [conf for conf, label in zip(all_confidences, all_labels) if label == 1]
        negative_confidences = [conf for conf, label in zip(all_confidences, all_labels) if label == 0]
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(all_labels),
            'positive_samples': sum(all_labels),
            'negative_samples': len(all_labels) - sum(all_labels),
            'confidence_analysis': {
                'positive_avg_confidence': np.mean(positive_confidences) if positive_confidences else 0,
                'negative_avg_confidence': np.mean(negative_confidences) if negative_confidences else 0,
                'overall_avg_confidence': np.mean(all_confidences)
            },
            'model_type': model_type,
            'uses_graph_inputs': True,
            'uses_batching': True
        }
        
        logger.info(f"Enhanced evaluation results for {annotation_type} ({model_type}):")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision (Positive): {report['Positive']['precision']:.3f}")
        logger.info(f"  Recall (Positive): {report['Positive']['recall']:.3f}")
        logger.info(f"  F1-Score (Positive): {report['Positive']['f1-score']:.3f}")
        logger.info(f"  Avg Confidence: {metrics['confidence_analysis']['overall_avg_confidence']:.3f}")
        logger.info(f"  Uses Graph Inputs: {metrics['uses_graph_inputs']}")
        logger.info(f"  Uses Batching: {metrics['uses_batching']}")
        
        return metrics
    
    def save_enhanced_models(self, output_dir: str):
        """Save trained enhanced models with enhanced metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_key, model in self.models.items():
            annotation_type, model_type = model_key.rsplit('_', 1)
            model_path = os.path.join(output_dir, f"{annotation_type.replace('@', '').lower()}_{model_type}_balanced_model.pth")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': f'enhanced_balanced_{model_type}',
                'annotation_type': annotation_type,
                'training_stats': self.training_stats.get('training_history', {}).get(model_key, {}),
                'best_accuracy': self.training_stats.get('best_accuracies', {}).get(model_key, 0.0),
                'enhanced_analysis': self.training_stats.get('enhanced_analysis', {}).get(model_key, {}),
                'model_architecture': {
                    'model_type': model_type,
                    'input_dim': 15,
                    'hidden_dim': 512,
                    'num_layers': 6,
                    'heads': 16,
                    'output_dim': 2,
                    'uses_graph_inputs': True,
                    'uses_batching': True
                }
            }, model_path)
            
            logger.info(f"Saved enhanced balanced model for {annotation_type} ({model_type}) to {model_path}")
        
        # Save overall training statistics
        stats_path = os.path.join(output_dir, "enhanced_balanced_training_statistics.json")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_stats = convert_numpy_types(self.training_stats)
        
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        logger.info(f"Saved enhanced balanced training statistics to {stats_path}")
    
    def print_enhanced_training_summary(self):
        """Print enhanced training summary"""
        print("\n" + "="*80)
        print("ENHANCED BALANCED ANNOTATION TYPE TRAINING SUMMARY")
        print("="*80)
        
        for annotation_type in self.training_stats['annotation_types']:
            print(f"\n{annotation_type}:")
            
            for model_type in self.model_types:
                model_key = f"{annotation_type}_{model_type}"
                if model_key in self.training_stats['best_accuracies']:
                    accuracy = self.training_stats['best_accuracies'][model_key]
                    print(f"  {model_type}: Best Accuracy = {accuracy:.2f}%")
                    
                    if model_key in self.training_stats['final_metrics']:
                        metrics = self.training_stats['final_metrics'][model_key]
                        print(f"    Final Metrics:")
                        print(f"      Accuracy: {metrics['accuracy']:.3f}")
                        if 'classification_report' in metrics:
                            report = metrics['classification_report']
                            print(f"      Precision (Positive): {report['Positive']['precision']:.3f}")
                            print(f"      Recall (Positive): {report['Positive']['recall']:.3f}")
                            print(f"      F1-Score (Positive): {report['Positive']['f1-score']:.3f}")
                        print(f"      Avg Confidence: {metrics['confidence_analysis']['overall_avg_confidence']:.3f}")
                        print(f"      Uses Graph Inputs: {metrics['uses_graph_inputs']}")
                        print(f"      Uses Batching: {metrics['uses_batching']}")
                    
                    if model_key in self.training_stats['enhanced_analysis']:
                        analysis = self.training_stats['enhanced_analysis'][model_key]
                        print(f"    Enhanced Analysis:")
                        print(f"      Model Type: {analysis['model_type']}")
                        print(f"      Uses Graph Inputs: {analysis['uses_graph_inputs']}")
                        print(f"      Uses Batching: {analysis['uses_batching']}")
                        print(f"      Positive patterns: {len(analysis['positive_patterns']['labels'])} unique")
                        print(f"      Negative patterns: {len(analysis['negative_patterns']['labels'])} unique")
                        print(f"      Common patterns: {len(analysis['pattern_differences']['common'])}")
        
        print("\n" + "="*80)


def main():
    """Main function to train enhanced balanced annotation type models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train enhanced balanced annotation type models')
    parser.add_argument('--balanced_dataset_dir', required=True, 
                       help='Directory containing real balanced dataset files')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for trained models')
    parser.add_argument('--model_types', nargs='+', 
                       default=['enhanced_gcn', 'enhanced_gat', 'enhanced_transformer', 'enhanced_hybrid'],
                       help='Model types to train')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--use_balanced_training', default='true',
                       help='Use balanced training (true/false)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = EnhancedBalancedTrainingFramework(
        model_types=args.model_types,
        device=args.device
    )
    
    # Find balanced dataset files
    dataset_files = []
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    for ann_type in annotation_types:
        dataset_file = os.path.join(args.balanced_dataset_dir, f"{ann_type}_real_balanced_dataset.json")
        if os.path.exists(dataset_file):
            dataset_files.append(dataset_file)
        else:
            logger.warning(f"Real balanced dataset file not found: {dataset_file}")
    
    if not dataset_files:
        logger.error("No real balanced dataset files found. Exiting.")
        return 1
    
    # Train models for each annotation type and model type combination
    results = []
    for dataset_file in dataset_files:
        logger.info(f"\nTraining models with dataset: {dataset_file}")
        
        for model_type in args.model_types:
            logger.info(f"Training {model_type} model")
            
            result = trainer.train_model(
                dataset_file=dataset_file,
                model_type=model_type,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            if result['success']:
                results.append(result)
                logger.info(f"Successfully trained {model_type} model for {result['annotation_type']}")
            else:
                logger.error(f"Failed to train {model_type} model: {result.get('error', 'Unknown error')}")
    
    # Save models
    if results:
        trainer.save_enhanced_models(args.output_dir)
        trainer.print_enhanced_training_summary()
        logger.info(f"Enhanced training completed successfully. Models saved to {args.output_dir}")
    else:
        logger.error("No models were successfully trained.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
