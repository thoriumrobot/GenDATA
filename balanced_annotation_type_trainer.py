#!/usr/bin/env python3
"""
Balanced Annotation Type Trainer

This module trains annotation type models using balanced datasets where each annotation type
appears in approximately 50% of examples and is absent in the other 50%.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BalancedAnnotationDataset(Dataset):
    """PyTorch Dataset for balanced annotation type training"""
    
    def __init__(self, balanced_examples: List[Dict], annotation_type: str):
        self.examples = balanced_examples
        self.annotation_type = annotation_type
        
        # Extract features and labels
        self.features = []
        self.labels = []
        
        for example in balanced_examples:
            self.features.append(example['features'])
            # Convert boolean to integer: True -> 1 (positive), False -> 0 (negative)
            self.labels.append(1 if example['is_positive'] else 0)
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        logger.info(f"Created dataset for {annotation_type}: {len(self.features)} examples")
        logger.info(f"  Positive examples: {np.sum(self.labels)} ({np.sum(self.labels)/len(self.labels)*100:.1f}%)")
        logger.info(f"  Negative examples: {len(self.labels) - np.sum(self.labels)} ({(len(self.labels) - np.sum(self.labels))/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'confidence': self.examples[idx].get('confidence', 0.5)
        }

class BalancedAnnotationTypeModel(nn.Module):
    """Neural network model for balanced annotation type prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (binary classification: positive/negative)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class BalancedAnnotationTypeTrainer:
    """Trainer for balanced annotation type models"""
    
    def __init__(self, model_type: str = 'enhanced_causal', device: str = 'auto'):
        self.model_type = model_type
        
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
            'final_metrics': {}
        }
    
    def load_balanced_dataset(self, dataset_file: str) -> Tuple[List[Dict], str]:
        """Load a balanced dataset from file"""
        with open(dataset_file, 'r') as f:
            dataset_data = json.load(f)
        
        annotation_type = dataset_data['annotation_type']
        examples = dataset_data['examples']
        
        logger.info(f"Loaded balanced dataset for {annotation_type}:")
        logger.info(f"  Total examples: {len(examples)}")
        logger.info(f"  Positive examples: {dataset_data['positive_examples']}")
        logger.info(f"  Negative examples: {dataset_data['negative_examples']}")
        logger.info(f"  Balance ratio: {dataset_data['balance_ratio']:.3f}")
        
        return examples, annotation_type
    
    def create_model(self, input_dim: int, annotation_type: str) -> nn.Module:
        """Create a model for the given annotation type"""
        model = BalancedAnnotationTypeModel(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=0.3
        )
        
        model = model.to(self.device)
        self.models[annotation_type] = model
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        self.optimizers[annotation_type] = optimizer
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )
        self.schedulers[annotation_type] = scheduler
        
        return model
    
    def train_model(self, dataset_file: str, epochs: int = 100, batch_size: int = 32, 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """Train a model using balanced dataset"""
        
        # Load dataset
        examples, annotation_type = self.load_balanced_dataset(dataset_file)
        
        if not examples:
            logger.error(f"No examples found in dataset for {annotation_type}")
            return {'success': False, 'error': 'No examples found'}
        
        # Create dataset
        dataset = BalancedAnnotationDataset(examples, annotation_type)
        
        # Split into train and validation
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Training set: {len(train_dataset)} examples")
        logger.info(f"Validation set: {len(val_dataset)} examples")
        
        # Create model
        input_dim = len(examples[0]['features'])
        model = self.create_model(input_dim, annotation_type)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_accuracy = 0.0
        best_model_state = None
        
        logger.info(f"Starting training for {annotation_type} for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizers[annotation_type].zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizers[annotation_type].step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Update learning rate
            self.schedulers[annotation_type].step(val_accuracy)
            
            # Track history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}:")
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self.evaluate_model(model, val_loader, annotation_type)
        
        # Store training statistics
        self.training_stats['training_history'][annotation_type] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
        self.training_stats['best_accuracies'][annotation_type] = best_val_accuracy
        self.training_stats['final_metrics'][annotation_type] = final_metrics
        
        logger.info(f"Training completed for {annotation_type}")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        
        return {
            'success': True,
            'annotation_type': annotation_type,
            'best_accuracy': best_val_accuracy,
            'final_metrics': final_metrics,
            'model': model
        }
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader, 
                      annotation_type: str) -> Dict[str, Any]:
        """Evaluate model performance"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
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
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(all_labels),
            'positive_samples': sum(all_labels),
            'negative_samples': len(all_labels) - sum(all_labels)
        }
        
        logger.info(f"Evaluation results for {annotation_type}:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision (Positive): {report['Positive']['precision']:.3f}")
        logger.info(f"  Recall (Positive): {report['Positive']['recall']:.3f}")
        logger.info(f"  F1-Score (Positive): {report['Positive']['f1-score']:.3f}")
        
        return metrics
    
    def save_models(self, output_dir: str):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for annotation_type, model in self.models.items():
            model_path = os.path.join(output_dir, f"{annotation_type.replace('@', '').lower()}_balanced_model.pth")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': self.model_type,
                'annotation_type': annotation_type,
                'training_stats': self.training_stats.get('training_history', {}).get(annotation_type, {}),
                'best_accuracy': self.training_stats.get('best_accuracies', {}).get(annotation_type, 0.0)
            }, model_path)
            
            logger.info(f"Saved model for {annotation_type} to {model_path}")
        
        # Save overall training statistics
        stats_path = os.path.join(output_dir, "balanced_training_statistics.json")
        
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
        
        logger.info(f"Saved training statistics to {stats_path}")
    
    def print_training_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("BALANCED ANNOTATION TYPE TRAINING SUMMARY")
        print("="*60)
        
        for annotation_type in self.training_stats['annotation_types']:
            if annotation_type in self.training_stats['best_accuracies']:
                accuracy = self.training_stats['best_accuracies'][annotation_type]
                print(f"{annotation_type}: Best Accuracy = {accuracy:.2f}%")
                
                if annotation_type in self.training_stats['final_metrics']:
                    metrics = self.training_stats['final_metrics'][annotation_type]
                    print(f"  Final Metrics:")
                    print(f"    Accuracy: {metrics['accuracy']:.3f}")
                    if 'classification_report' in metrics:
                        report = metrics['classification_report']
                        print(f"    Precision (Positive): {report['Positive']['precision']:.3f}")
                        print(f"    Recall (Positive): {report['Positive']['recall']:.3f}")
                        print(f"    F1-Score (Positive): {report['Positive']['f1-score']:.3f}")
                    print()
        
        print("="*60)


def main():
    """Main function to train balanced annotation type models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train balanced annotation type models')
    parser.add_argument('--balanced_dataset_dir', required=True, 
                       help='Directory containing balanced dataset files')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for trained models')
    parser.add_argument('--model_type', default='enhanced_causal',
                       help='Model type identifier')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BalancedAnnotationTypeTrainer(
        model_type=args.model_type,
        device=args.device
    )
    
    # Find balanced dataset files
    dataset_files = []
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    for ann_type in annotation_types:
        dataset_file = os.path.join(args.balanced_dataset_dir, f"{ann_type}_balanced_dataset.json")
        if os.path.exists(dataset_file):
            dataset_files.append(dataset_file)
        else:
            logger.warning(f"Balanced dataset file not found: {dataset_file}")
    
    if not dataset_files:
        logger.error("No balanced dataset files found. Exiting.")
        return 1
    
    # Train models for each annotation type
    results = []
    for dataset_file in dataset_files:
        logger.info(f"\nTraining model with dataset: {dataset_file}")
        
        result = trainer.train_model(
            dataset_file=dataset_file,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        if result['success']:
            results.append(result)
            logger.info(f"Successfully trained model for {result['annotation_type']}")
        else:
            logger.error(f"Failed to train model: {result.get('error', 'Unknown error')}")
    
    # Save models
    if results:
        trainer.save_models(args.output_dir)
        trainer.print_training_summary()
        logger.info(f"Training completed successfully. Models saved to {args.output_dir}")
    else:
        logger.error("No models were successfully trained.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
