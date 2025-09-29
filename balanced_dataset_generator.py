#!/usr/bin/env python3
"""
Balanced Dataset Generator for Annotation Type Models

This module creates a balanced training dataset where each annotation type
appears in approximately 50% of examples and is absent in the other 50%.
This helps with model convergence by providing balanced positive and negative examples.
"""

import os
import json
import random
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BalancedExample:
    """Represents a balanced training example"""
    node_id: int
    file_path: str
    method_name: str
    line_number: int
    node_type: str
    node_label: str
    features: List[float]
    annotation_type: str  # The target annotation type
    is_positive: bool     # True if this annotation should be present, False if absent
    confidence: float     # Synthetic confidence score

class BalancedDatasetGenerator:
    """Generates balanced training datasets for annotation type models"""
    
    def __init__(self, target_balance: float = 0.5, random_seed: int = 42):
        """
        Initialize the balanced dataset generator
        
        Args:
            target_balance: Target ratio of positive examples (0.5 = 50% positive, 50% negative)
            random_seed: Random seed for reproducible results
        """
        self.target_balance = target_balance
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Annotation types to balance
        self.annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        
        # Statistics tracking
        self.generation_stats = {
            'total_examples': 0,
            'positive_examples': 0,
            'negative_examples': 0,
            'annotation_type_counts': {ann_type: {'positive': 0, 'negative': 0} 
                                     for ann_type in self.annotation_types}
        }
    
    def load_cfg_files(self, cfg_directory: str) -> List[Dict[str, Any]]:
        """Load all CFG files from a directory"""
        cfg_files = []
        
        if not os.path.exists(cfg_directory):
            logger.warning(f"CFG directory does not exist: {cfg_directory}")
            return cfg_files
        
        for root, dirs, files in os.walk(cfg_directory):
            for file in files:
                if file.endswith('.json'):
                    cfg_path = os.path.join(root, file)
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg_data = json.load(f)
                        
                        # Extract method name from path
                        method_name = os.path.splitext(file)[0]
                        
                        cfg_files.append({
                            'file': cfg_path,
                            'method': method_name,
                            'data': cfg_data
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load CFG file {cfg_path}: {e}")
        
        logger.info(f"Loaded {len(cfg_files)} CFG files from {cfg_directory}")
        return cfg_files
    
    def extract_node_features(self, node: Dict, cfg_data: Dict) -> List[float]:
        """Extract features from a CFG node (simplified version)"""
        label = node.get('label', '')
        node_type = node.get('type', '')
        line = node.get('line', 0)
        
        # Basic features for annotation type prediction
        features = [
            float(len(label)),  # label_length
            float(line if line is not None else 0),  # line_number
            float('method' in node_type.lower()),  # is_method
            float('field' in node_type.lower()),  # is_field
            float('parameter' in node_type.lower()),  # is_parameter
            float('variable' in node_type.lower()),  # is_variable
            float('positive' in label.lower()),  # contains_positive
            float('negative' in label.lower()),  # contains_negative
            float('int' in label.lower()),  # contains_int
            float('array' in label.lower()),  # contains_array
            float('length' in label.lower()),  # contains_length
            float('index' in label.lower()),  # contains_index
            float('size' in label.lower()),  # contains_size
            float('count' in label.lower()),  # contains_count
            float('bound' in label.lower()),  # contains_bound
        ]
        
        return features
    
    def determine_annotation_type(self, node: Dict, cfg_data: Dict) -> str:
        """Determine the most appropriate annotation type for a node"""
        label = node.get('label', '').lower()
        node_type = node.get('type', '').lower()
        
        # Rule-based annotation type determination
        if 'array' in label or 'index' in label:
            if 'length' in label:
                return '@NonNegative'
            else:
                return '@Positive'
        elif 'size' in label or 'count' in label:
            return '@Positive'
        elif 'bound' in label or 'limit' in label:
            return '@GTENegativeOne'
        elif 'parameter' in node_type:
            return '@NonNegative'
        elif 'variable' in node_type:
            return '@GTENegativeOne'
        else:
            # Default based on context
            if 'positive' in label:
                return '@Positive'
            elif 'negative' in label:
                return '@GTENegativeOne'
            else:
                return '@NonNegative'
    
    def generate_balanced_examples(self, cfg_files: List[Dict[str, Any]], 
                                 examples_per_annotation: int = 1000) -> Dict[str, List[BalancedExample]]:
        """
        Generate balanced examples for each annotation type
        
        Args:
            cfg_files: List of CFG file data
            examples_per_annotation: Target number of examples per annotation type
            
        Returns:
            Dictionary mapping annotation types to lists of balanced examples
        """
        balanced_datasets = {ann_type: [] for ann_type in self.annotation_types}
        
        logger.info(f"Generating balanced datasets with {examples_per_annotation} examples per annotation type")
        logger.info(f"Target balance: {self.target_balance*100:.1f}% positive, {(1-self.target_balance)*100:.1f}% negative")
        
        for ann_type in self.annotation_types:
            logger.info(f"\nGenerating examples for {ann_type}...")
            
            # Collect all potential nodes
            all_nodes = []
            for cfg_file in cfg_files:
                cfg_data = cfg_file['data']
                for node in cfg_data.get('nodes', []):
                    all_nodes.append({
                        'node': node,
                        'cfg_data': cfg_data,
                        'file_path': cfg_file['file'],
                        'method_name': cfg_file['method']
                    })
            
            # Filter nodes that could potentially have this annotation type
            relevant_nodes = []
            for node_info in all_nodes:
                node = node_info['node']
                predicted_type = self.determine_annotation_type(node, node_info['cfg_data'])
                
                # Include nodes that match the target annotation type or could be relevant
                if predicted_type == ann_type or self._is_relevant_for_annotation(node, ann_type):
                    relevant_nodes.append(node_info)
            
            logger.info(f"Found {len(relevant_nodes)} relevant nodes for {ann_type}")
            
            if len(relevant_nodes) == 0:
                logger.warning(f"No relevant nodes found for {ann_type}")
                continue
            
            # Generate positive examples (annotation should be present)
            num_positive = int(examples_per_annotation * self.target_balance)
            positive_examples = self._generate_positive_examples(
                relevant_nodes, ann_type, num_positive
            )
            
            # Generate negative examples (annotation should be absent)
            num_negative = examples_per_annotation - num_positive
            negative_examples = self._generate_negative_examples(
                relevant_nodes, ann_type, num_negative
            )
            
            # Combine and shuffle
            all_examples = positive_examples + negative_examples
            random.shuffle(all_examples)
            
            balanced_datasets[ann_type] = all_examples
            
            # Update statistics
            self.generation_stats['annotation_type_counts'][ann_type]['positive'] = len(positive_examples)
            self.generation_stats['annotation_type_counts'][ann_type]['negative'] = len(negative_examples)
            
            logger.info(f"Generated {len(positive_examples)} positive and {len(negative_examples)} negative examples for {ann_type}")
        
        # Update overall statistics
        self.generation_stats['total_examples'] = sum(len(examples) for examples in balanced_datasets.values())
        self.generation_stats['positive_examples'] = sum(
            stats['positive'] for stats in self.generation_stats['annotation_type_counts'].values()
        )
        self.generation_stats['negative_examples'] = sum(
            stats['negative'] for stats in self.generation_stats['annotation_type_counts'].values()
        )
        
        return balanced_datasets
    
    def _is_relevant_for_annotation(self, node: Dict, annotation_type: str) -> bool:
        """Check if a node could be relevant for a specific annotation type"""
        label = node.get('label', '').lower()
        node_type = node.get('type', '').lower()
        
        # Define relevance patterns for each annotation type
        relevance_patterns = {
            '@Positive': ['int', 'long', 'count', 'size', 'length', 'capacity', 'number'],
            '@NonNegative': ['index', 'offset', 'position', 'parameter', 'int', 'array'],
            '@GTENegativeOne': ['bound', 'limit', 'capacity', 'size', 'count', 'variable']
        }
        
        patterns = relevance_patterns.get(annotation_type, [])
        return any(pattern in label or pattern in node_type for pattern in patterns)
    
    def _generate_positive_examples(self, relevant_nodes: List[Dict], 
                                  annotation_type: str, num_examples: int) -> List[BalancedExample]:
        """Generate positive examples where the annotation should be present"""
        examples = []
        
        # Sample nodes with replacement if needed
        for i in range(num_examples):
            node_info = relevant_nodes[i % len(relevant_nodes)]
            node = node_info['node']
            cfg_data = node_info['cfg_data']
            
            features = self.extract_node_features(node, cfg_data)
            
            # Add some variation to features for diversity
            features = self._add_feature_variation(features)
            
            example = BalancedExample(
                node_id=node.get('id', i),
                file_path=node_info['file_path'],
                method_name=node_info['method_name'],
                line_number=node.get('line', 0),
                node_type=node.get('type', ''),
                node_label=node.get('label', ''),
                features=features,
                annotation_type=annotation_type,
                is_positive=True,
                confidence=random.uniform(0.6, 1.0)  # High confidence for positive examples
            )
            
            examples.append(example)
        
        return examples
    
    def _generate_negative_examples(self, relevant_nodes: List[Dict], 
                                  annotation_type: str, num_examples: int) -> List[BalancedExample]:
        """Generate negative examples where the annotation should be absent"""
        examples = []
        
        # Sample nodes with replacement if needed
        for i in range(num_examples):
            node_info = relevant_nodes[i % len(relevant_nodes)]
            node = node_info['node']
            cfg_data = node_info['cfg_data']
            
            features = self.extract_node_features(node, cfg_data)
            
            # Add variation and potentially modify features to make them less suitable for the annotation
            features = self._add_feature_variation(features)
            features = self._modify_features_for_negative(features, annotation_type)
            
            example = BalancedExample(
                node_id=node.get('id', i),
                file_path=node_info['file_path'],
                method_name=node_info['method_name'],
                line_number=node.get('line', 0),
                node_type=node.get('type', ''),
                node_label=node.get('label', ''),
                features=features,
                annotation_type=annotation_type,
                is_positive=False,
                confidence=random.uniform(0.1, 0.5)  # Low confidence for negative examples
            )
            
            examples.append(example)
        
        return examples
    
    def _add_feature_variation(self, features: List[float]) -> List[float]:
        """Add small random variations to features for diversity"""
        varied_features = []
        for feature in features:
            # Add small random noise (5% variation)
            variation = random.uniform(-0.05, 0.05) * abs(feature) if feature != 0 else random.uniform(-0.1, 0.1)
            varied_features.append(feature + variation)
        return varied_features
    
    def _modify_features_for_negative(self, features: List[float], annotation_type: str) -> List[float]:
        """Modify features to make them less suitable for the target annotation type"""
        modified_features = features.copy()
        
        # Reduce features that would suggest the annotation is needed
        if annotation_type == '@Positive':
            # Reduce size/count related features
            if len(modified_features) > 10:
                modified_features[10] *= 0.3  # contains_length
                modified_features[12] *= 0.3  # contains_size
                modified_features[13] *= 0.3  # contains_count
        elif annotation_type == '@NonNegative':
            # Reduce index/array related features
            if len(modified_features) > 9:
                modified_features[9] *= 0.3   # contains_array
                modified_features[11] *= 0.3  # contains_index
        elif annotation_type == '@GTENegativeOne':
            # Reduce bound/limit related features
            if len(modified_features) > 14:
                modified_features[14] *= 0.3  # contains_bound
        
        return modified_features
    
    def save_balanced_dataset(self, balanced_datasets: Dict[str, List[BalancedExample]], 
                            output_dir: str):
        """Save the balanced datasets to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for ann_type, examples in balanced_datasets.items():
            if not examples:
                continue
            
            # Save as JSON
            output_file = os.path.join(output_dir, f"{ann_type.replace('@', '').lower()}_balanced_dataset.json")
            
            dataset_data = {
                'annotation_type': ann_type,
                'total_examples': len(examples),
                'positive_examples': sum(1 for ex in examples if ex.is_positive),
                'negative_examples': sum(1 for ex in examples if not ex.is_positive),
                'balance_ratio': sum(1 for ex in examples if ex.is_positive) / len(examples),
                'examples': [
                    {
                        'node_id': ex.node_id,
                        'file_path': ex.file_path,
                        'method_name': ex.method_name,
                        'line_number': ex.line_number,
                        'node_type': ex.node_type,
                        'node_label': ex.node_label,
                        'features': ex.features,
                        'is_positive': ex.is_positive,
                        'confidence': ex.confidence
                    }
                    for ex in examples
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(dataset_data, f, indent=2)
            
            logger.info(f"Saved {len(examples)} examples for {ann_type} to {output_file}")
        
        # Save overall statistics
        stats_file = os.path.join(output_dir, "generation_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(self.generation_stats, f, indent=2)
        
        logger.info(f"Saved generation statistics to {stats_file}")
    
    def print_statistics(self):
        """Print generation statistics"""
        print("\n" + "="*60)
        print("BALANCED DATASET GENERATION STATISTICS")
        print("="*60)
        
        print(f"Total examples generated: {self.generation_stats['total_examples']}")
        print(f"Overall balance: {self.generation_stats['positive_examples']} positive, {self.generation_stats['negative_examples']} negative")
        
        if self.generation_stats['total_examples'] > 0:
            overall_balance = self.generation_stats['positive_examples'] / self.generation_stats['total_examples']
            print(f"Overall balance ratio: {overall_balance:.3f} (target: {self.target_balance:.3f})")
        
        print("\nPer-annotation-type statistics:")
        for ann_type, stats in self.generation_stats['annotation_type_counts'].items():
            total = stats['positive'] + stats['negative']
            if total > 0:
                balance = stats['positive'] / total
                print(f"  {ann_type}: {stats['positive']} positive, {stats['negative']} negative (balance: {balance:.3f})")
        
        print("="*60)


def main():
    """Main function to generate balanced datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate balanced training datasets for annotation type models')
    parser.add_argument('--cfg_dir', required=True, help='Directory containing CFG files')
    parser.add_argument('--output_dir', required=True, help='Output directory for balanced datasets')
    parser.add_argument('--examples_per_annotation', type=int, default=1000, 
                       help='Number of examples to generate per annotation type')
    parser.add_argument('--target_balance', type=float, default=0.5,
                       help='Target balance ratio for positive examples (0.5 = 50 percent positive)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Create generator
    generator = BalancedDatasetGenerator(
        target_balance=args.target_balance,
        random_seed=args.random_seed
    )
    
    # Load CFG files
    cfg_files = generator.load_cfg_files(args.cfg_dir)
    
    if not cfg_files:
        logger.error("No CFG files found. Exiting.")
        return 1
    
    # Generate balanced datasets
    balanced_datasets = generator.generate_balanced_examples(
        cfg_files, 
        examples_per_annotation=args.examples_per_annotation
    )
    
    # Save datasets
    generator.save_balanced_dataset(balanced_datasets, args.output_dir)
    
    # Print statistics
    generator.print_statistics()
    
    return 0


if __name__ == '__main__':
    exit(main())
