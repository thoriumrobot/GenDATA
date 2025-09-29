#!/usr/bin/env python3
"""
Enhanced Causal Model Implementation for CFWR
Integrates with existing annotation type scripts as an enhanced causal option
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class EnhancedCausalFeatureExtractor:
    """Enhanced causal feature extractor with 32-dimensional features"""
    
    def __init__(self):
        self.feature_cache = {}
    
    def extract_features(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract 32-dimensional causal features"""
        features = []
        
        # 1. Structural Causal Features (8 features)
        features.extend(self._extract_structural_causal(node, cfg_data))
        
        # 2. Dataflow Causal Features (8 features)
        features.extend(self._extract_dataflow_causal(node, cfg_data))
        
        # 3. Semantic Causal Features (8 features)
        features.extend(self._extract_semantic_causal(node, cfg_data))
        
        # 4. Temporal Causal Features (8 features)
        features.extend(self._extract_temporal_causal(node, cfg_data))
        
        # Ensure exactly 32 features
        if len(features) > 32:
            features = features[:32]
        elif len(features) < 32:
            # Pad with zeros if we have fewer than 32 features
            features.extend([0.0] * (32 - len(features)))
        
        return features
    
    def _extract_structural_causal(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract structural causal relationships"""
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        
        # Control flow causal influence
        control_in_degree = self._get_control_in_degree(node, cfg_data)
        control_out_degree = self._get_control_out_degree(node, cfg_data)
        
        # Data dependency causal chains
        dataflow_in_degree = self._get_dataflow_in_degree(node, cfg_data)
        dataflow_out_degree = self._get_dataflow_out_degree(node, cfg_data)
        
        # Method call causal propagation
        method_call_causal = float('(' in label and ')' in label and '.' not in label)
        
        # Variable scope causal boundaries
        scope_causal = float('private' in label or 'public' in label or 'protected' in label)
        
        # Loop causal complexity
        loop_causal = float('for' in label or 'while' in label or 'do' in label)
        
        # Exception handling causal flows
        exception_causal = float('try' in label or 'catch' in label or 'throw' in label)
        
        return [
            control_in_degree,
            control_out_degree,
            dataflow_in_degree,
            dataflow_out_degree,
            method_call_causal,
            scope_causal,
            loop_causal,
            exception_causal
        ]
    
    def _extract_dataflow_causal(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract dataflow causal relationships"""
        label = node.get('label', '')
        
        # Variable definition-use causal chains
        var_definition = float('=' in label and '==' not in label)
        var_usage = float(any(char.isalpha() for char in label) and '=' not in label)
        
        # Parameter passing causal effects
        parameter_causal = float('parameter' in label.lower() or 'param' in label.lower())
        
        # Return value causal propagation
        return_causal = float('return' in label.lower())
        
        # Array access causal patterns
        array_access = float('[' in label and ']' in label)
        
        # Method invocation causal chains
        method_invocation = float('.' in label and '(' in label and ')' in label)
        
        # Assignment causal propagation
        assignment_causal = float('=' in label and '==' not in label and '!=' not in label)
        
        # Field access causal patterns
        field_access = float('this.' in label or '.' in label)
        
        # Type casting causal effects
        type_casting = float('(' in label and ')' in label and any(t in label for t in ['int', 'String', 'Object', 'List']))
        
        return [
            var_definition,
            var_usage,
            parameter_causal,
            return_causal,
            array_access,
            method_invocation,
            assignment_causal,
            field_access,
            type_casting
        ][:8]  # Ensure exactly 8 features
    
    def _extract_semantic_causal(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract semantic causal relationships"""
        label = node.get('label', '')
        node_type = node.get('node_type', '')
        
        # Type relationships causal patterns
        primitive_type = float(any(t in label for t in ['int', 'long', 'double', 'float', 'boolean', 'char']))
        object_type = float(any(t in label for t in ['String', 'Object', 'List', 'Map', 'Set', 'Collection']))
        
        # Method signature causal patterns
        method_signature = float('method' in node_type.lower() or 'constructor' in node_type.lower())
        
        # Field declaration causal patterns
        field_declaration = float('field' in node_type.lower() or 'variable' in node_type.lower())
        
        # Annotation causal patterns
        annotation_present = float('@' in label)
        
        # Generic type causal patterns
        generic_type = float('<' in label and '>' in label)
        
        # Static causal patterns
        static_causal = float('static' in label)
        
        # Final causal patterns
        final_causal = float('final' in label)
        
        # Synchronized causal patterns
        synchronized_causal = float('synchronized' in label)
        
        return [
            primitive_type,
            object_type,
            method_signature,
            field_declaration,
            annotation_present,
            generic_type,
            static_causal,
            final_causal
        ]
    
    def _extract_temporal_causal(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract temporal causal relationships"""
        label = node.get('label', '')
        line = node.get('line', 0)
        nodes = cfg_data.get('nodes', [])
        
        # Execution order causal patterns
        line_position = float(line) / max(1, len(nodes))  # Normalized line position
        
        # Sequential execution causal patterns
        sequential_causal = float(';' in label and '{' not in label)
        
        # Conditional execution causal patterns
        conditional_causal = float('if' in label or 'else' in label or 'switch' in label or 'case' in label)
        
        # Iterative execution causal patterns
        iterative_causal = float('for' in label or 'while' in label or 'do' in label)
        
        # Exception handling temporal patterns
        exception_temporal = float('try' in label or 'catch' in label or 'finally' in label)
        
        # Method call temporal patterns
        method_call_temporal = float('(' in label and ')' in label)
        
        # Variable lifecycle temporal patterns
        variable_lifecycle = float('=' in label and ';' in label)
        
        # Control flow temporal patterns
        control_flow_temporal = float('return' in label or 'break' in label or 'continue' in label)
        
        return [
            line_position,
            sequential_causal,
            conditional_causal,
            iterative_causal,
            exception_temporal,
            method_call_temporal,
            variable_lifecycle,
            control_flow_temporal
        ]
    
    def _get_control_in_degree(self, node: Dict, cfg_data: Dict) -> float:
        """Get control flow in-degree for causal analysis"""
        node_id = node.get('id', -1)
        control_edges = cfg_data.get('control_edges', [])
        
        in_degree = sum(1 for edge in control_edges if edge.get('target') == node_id)
        return float(in_degree)
    
    def _get_control_out_degree(self, node: Dict, cfg_data: Dict) -> float:
        """Get control flow out-degree for causal analysis"""
        node_id = node.get('id', -1)
        control_edges = cfg_data.get('control_edges', [])
        
        out_degree = sum(1 for edge in control_edges if edge.get('source') == node_id)
        return float(out_degree)
    
    def _get_dataflow_in_degree(self, node: Dict, cfg_data: Dict) -> float:
        """Get dataflow in-degree for causal analysis"""
        node_id = node.get('id', -1)
        dataflow_edges = cfg_data.get('dataflow_edges', [])
        
        in_degree = sum(1 for edge in dataflow_edges if edge.get('target') == node_id)
        return float(in_degree)
    
    def _get_dataflow_out_degree(self, node: Dict, cfg_data: Dict) -> float:
        """Get dataflow out-degree for causal analysis"""
        node_id = node.get('id', -1)
        dataflow_edges = cfg_data.get('dataflow_edges', [])
        
        out_degree = sum(1 for edge in dataflow_edges if edge.get('source') == node_id)
        return float(out_degree)

class EnhancedCausalModel(nn.Module):
    """Enhanced causal model with sophisticated architecture"""
    
    def __init__(self, input_dim=32, hidden_dim=256, out_dim=2, annotation_type='@Positive'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.annotation_type = annotation_type
        
        # Shared causal feature extractor
        self.causal_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Causal attention mechanism
        self.causal_attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=8, batch_first=True)
        
        # Annotation type specific causal reasoning layers
        if annotation_type == '@Positive':
            self.causal_layers = PositiveCausalLayers(hidden_dim // 2)
        elif annotation_type == '@NonNegative':
            self.causal_layers = NonNegativeCausalLayers(hidden_dim // 2)
        elif annotation_type == '@GTENegativeOne':
            self.causal_layers = GTENegativeOneCausalLayers(hidden_dim // 2)
        else:
            # Generic causal layers
            self.causal_layers = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, out_dim)
        )
        
        # Causal intervention mechanism
        self.intervention_module = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
    
    def forward(self, x):
        # Extract causal features
        causal_features = self.causal_extractor(x)
        
        # Apply causal attention if input is 3D
        if len(causal_features.shape) == 3:
            attended_features, _ = self.causal_attention(causal_features, causal_features, causal_features)
            causal_features = attended_features.mean(dim=1)  # Global average pooling
        
        # Apply causal intervention
        intervened_features = self.intervention_module(causal_features)
        combined_features = causal_features + intervened_features
        
        # Apply annotation type specific causal reasoning
        processed_features = self.causal_layers(combined_features)
        
        # Final classification
        logits = self.classifier(processed_features)
        return logits

class PositiveCausalLayers(nn.Module):
    """Causal reasoning for @Positive annotations"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Focus on count, size, length relationships
        # Use smaller output dimensions to avoid concatenation issues
        output_dim = hidden_dim // 4  # Make sure it divides evenly
        self.count_causal = nn.Linear(hidden_dim, output_dim)
        self.size_causal = nn.Linear(hidden_dim, output_dim)
        self.length_causal = nn.Linear(hidden_dim, output_dim)
        # Concatenated features will be output_dim * 3
        self.merge = nn.Linear(output_dim * 3, hidden_dim)
        
    def forward(self, x):
        count_features = torch.relu(self.count_causal(x))
        size_features = torch.relu(self.size_causal(x))
        length_features = torch.relu(self.length_causal(x))
        
        combined = torch.cat([count_features, size_features, length_features], dim=-1)
        return torch.relu(self.merge(combined))

class NonNegativeCausalLayers(nn.Module):
    """Causal reasoning for @NonNegative annotations"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Focus on index, offset, position relationships
        output_dim = hidden_dim // 4  # Make sure it divides evenly
        self.index_causal = nn.Linear(hidden_dim, output_dim)
        self.offset_causal = nn.Linear(hidden_dim, output_dim)
        self.position_causal = nn.Linear(hidden_dim, output_dim)
        self.merge = nn.Linear(output_dim * 3, hidden_dim)
        
    def forward(self, x):
        index_features = torch.relu(self.index_causal(x))
        offset_features = torch.relu(self.offset_causal(x))
        position_features = torch.relu(self.position_causal(x))
        
        combined = torch.cat([index_features, offset_features, position_features], dim=-1)
        return torch.relu(self.merge(combined))

class GTENegativeOneCausalLayers(nn.Module):
    """Causal reasoning for @GTENegativeOne annotations"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Focus on capacity, limit, bound relationships
        output_dim = hidden_dim // 4  # Make sure it divides evenly
        self.capacity_causal = nn.Linear(hidden_dim, output_dim)
        self.limit_causal = nn.Linear(hidden_dim, output_dim)
        self.bound_causal = nn.Linear(hidden_dim, output_dim)
        self.merge = nn.Linear(output_dim * 3, hidden_dim)
        
    def forward(self, x):
        capacity_features = torch.relu(self.capacity_causal(x))
        limit_features = torch.relu(self.limit_causal(x))
        bound_features = torch.relu(self.bound_causal(x))
        
        combined = torch.cat([capacity_features, limit_features, bound_features], dim=-1)
        return torch.relu(self.merge(combined))

# Global feature extractor instance
enhanced_feature_extractor = EnhancedCausalFeatureExtractor()

def extract_enhanced_causal_features(node: Dict, cfg_data: Dict) -> List[float]:
    """Extract enhanced causal features for a node"""
    return enhanced_feature_extractor.extract_features(node, cfg_data)
