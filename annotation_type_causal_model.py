#!/usr/bin/env python3
"""
Annotation Type Causal Model - A specialized causal model for multi-class annotation type prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from sklearn.preprocessing import LabelEncoder
import random

from annotation_type_prediction import AnnotationTypeClassifier, LowerBoundAnnotationType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnnotationTypeCausalModel(nn.Module):
    """
    Causal model for multi-class annotation type prediction.
    Uses causal relationships between nodes to predict annotation types.
    """
    
    def __init__(self, input_dim: int = 23, hidden_dim: int = 128, num_classes: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Causal-aware neural network
        self.causal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Causal relationship processor
        self.causal_processor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim // 2 + hidden_dim // 4, num_classes)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([at.value for at in LowerBoundAnnotationType])
        self.training_history = []
        
        logger.info(f"Initialized AnnotationTypeCausalModel with input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}")
    
    def extract_causal_features(self, node: Dict[str, Any], cfg_data: Dict[str, Any]) -> List[float]:
        """Extract causal-specific features for annotation type prediction."""
        features = []
        
        # Basic node features
        node_type = node.get('type', '')
        node_label = node.get('label', '').lower()
        node_id = node.get('id', 0)
        
        # Control flow causal features
        control_edges = cfg_data.get('control_edges', [])
        dataflow_edges = cfg_data.get('dataflow_edges', [])
        
        # Causal influence (how many nodes this affects)
        causal_influence = sum(1 for edge in control_edges if edge.get('source') == node_id)
        causal_dependence = sum(1 for edge in control_edges if edge.get('target') == node_id)
        
        # Dataflow causal relationships
        dataflow_vars = set()
        for edge in dataflow_edges:
            if edge.get('source') == node_id or edge.get('target') == node_id:
                if 'variable' in edge:
                    dataflow_vars.add(edge['variable'])
        
        # Semantic features for causal reasoning
        semantic_features = [
            1 if 'return' in node_label else 0,
            1 if 'throw' in node_label else 0,
            1 if 'new' in node_label else 0,
            1 if 'call' in node_label else 0,
            1 if 'assign' in node_label else 0,
            1 if 'if' in node_label or 'else' in node_label else 0,
            1 if 'for' in node_label or 'while' in node_label else 0,
            1 if 'array' in node_label else 0,
            1 if 'index' in node_label else 0,
            1 if 'length' in node_label or 'size' in node_label else 0,
            1 if 'param' in node_type else 0,
            1 if 'variable' in node_type else 0,
            1 if 'field' in node_type else 0,
            1 if 'method' in node_type else 0,
        ]
        
        # Combine all features
        features.extend([
            len(node_label),  # label length
            node.get('line_number', 0),  # line number
            causal_influence,  # causal influence
            causal_dependence,  # causal dependence
            len(dataflow_vars),  # dataflow variables
            len(dataflow_edges),  # dataflow edges
        ])
        features.extend(semantic_features)
        
        # Pad to expected size
        while len(features) < self.input_dim:
            features.append(0.0)
        
        return features[:self.input_dim]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the causal model."""
        # Extract causal features
        causal_features = self.causal_encoder(x)
        
        # Process causal relationships
        causal_relationships = self.causal_processor(causal_features)
        
        # Combine features for final classification
        combined_features = torch.cat([causal_features, causal_relationships], dim=-1)
        
        # Final classification
        output = self.classifier(combined_features)
        return output
    
    def train_model(self, cfg_files: List[Dict[str, Any]], classifier: AnnotationTypeClassifier, epochs: int = 100, learning_rate: float = 0.001):
        """Train the causal model for annotation type prediction."""
        logger.info(f"Starting annotation type Causal training: {len(cfg_files)} CFG files, {epochs} epochs")
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        all_features = []
        all_labels = []
        parameter_free = getattr(self, 'parameter_free', False)
        
        for cfg_file in cfg_files:
            cfg_data = cfg_file['data']
            nodes = cfg_data.get('nodes', [])
            
            for node in nodes:
                from node_level_models import NodeClassifier
                if NodeClassifier.is_annotation_target(node):
                    features = self.extract_causal_features(node, cfg_data)
                    node_features = classifier.extract_features(node, cfg_data)
                    annotation_type = classifier.determine_annotation_type(node_features)
                    label_value = annotation_type.value
                    if parameter_free:
                        from annotation_type_prediction import ParameterFreeConfig
                        if not ParameterFreeConfig.is_parameter_free(label_value):
                            continue
                    
                    all_features.append(features)
                    all_labels.append(label_value)
        
        if len(all_features) < 2:
            logger.warning("Insufficient features for causal training. Skipping.")
            return
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels)
        
        # Fit encoder to appropriate label space
        if parameter_free:
            from annotation_type_prediction import ParameterFreeConfig
            self.label_encoder.fit(sorted(list(ParameterFreeConfig.PARAMETER_FREE_TYPES)))
        else:
            self.label_encoder.fit([at.value for at in LowerBoundAnnotationType])
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Ensure class diversity
        unique_classes = np.unique(y_encoded)
        if len(unique_classes) < 2:
            logger.warning(f"Training with {len(unique_classes)} annotation types: {unique_classes}. Adding class diversity.")
            classes_all = list(range(len(self.label_encoder.classes_)))
            missing = [c for c in classes_all if c not in unique_classes]
            if missing:
                num_synth = max(5, len(X)//10)
                synth = (np.random.randn(num_synth, X.shape[1]) * 0.05).astype(np.float32)
                X = np.vstack([X, synth])
                y_encoded = np.hstack([y_encoded, np.random.choice(missing, size=num_synth)])
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        
        # Simple train/val split
        dataset_size = len(X_tensor)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        if train_size == 0 or val_size == 0:
            train_features, train_labels = X_tensor, y_tensor
            val_features, val_labels = X_tensor, y_tensor
        else:
            train_features, val_features = torch.split(X_tensor, [train_size, val_size])
            train_labels, val_labels = torch.split(y_tensor, [train_size, val_size])
        
        logger.info(f"Causal training data prepared: {len(train_features)} samples, {train_features.shape[1]} features, {len(np.unique(train_labels))} classes")
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(val_features)
                    _, predicted = torch.max(val_outputs.data, 1)
                    val_accuracy = (predicted == val_labels).float().mean().item() if len(val_labels)>0 else 0.0
                logger.debug(f"Causal Epoch {epoch+1}/{epochs} Val Acc: {val_accuracy:.3f}")
        
        self.eval()
        with torch.no_grad():
            train_outputs = self.forward(train_features)
            _, train_predicted = torch.max(train_outputs.data, 1)
            train_total = train_labels.size(0)
            train_correct = (train_predicted == train_labels).sum().item()
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0
            
            val_outputs = self.forward(val_features)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total = val_labels.size(0)
            val_correct = (val_predicted == val_labels).sum().item()
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        logger.info(f"Causal training completed - Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}")
    
    def predict_annotation_type(self, node: Dict[str, Any], cfg_data: Dict[str, Any], classifier: AnnotationTypeClassifier) -> Tuple[LowerBoundAnnotationType, float]:
        """Predict the specific annotation type for a single node."""
        self.eval()
        features = self.extract_causal_features(node, cfg_data)
        X_pred = torch.tensor([features], dtype=torch.float32)
        
        from node_level_models import NodeClassifier
        if not NodeClassifier.is_annotation_target(node):
            return LowerBoundAnnotationType.NO_ANNOTATION, 1.0
        
        with torch.no_grad():
            outputs = self.forward(X_pred)
            probabilities = F.softmax(outputs, dim=-1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            confidence = probabilities[predicted_class_idx].item()
            return LowerBoundAnnotationType(predicted_label), confidence

def main():
    """Test the causal annotation type model"""
    logger.info("Testing Annotation Type Causal Model")
    
    # Simple test
    classifier = AnnotationTypeClassifier()
    model = AnnotationTypeCausalModel()
    
    # Create dummy data
    test_cfg = {
        'nodes': [{'id': 1, 'label': 'int index = 0', 'type': 'variable', 'line_number': 5}],
        'control_edges': [],
        'dataflow_edges': []
    }
    
    test_node = test_cfg['nodes'][0]
    prediction, confidence = model.predict_annotation_type(test_node, test_cfg, classifier)
    logger.info(f"Test prediction: {prediction.value} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
