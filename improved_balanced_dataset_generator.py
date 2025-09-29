#!/usr/bin/env python3
"""
Improved Balanced Dataset Generator for Annotation Type Models

This module creates a balanced training dataset using REAL code examples where:
- Positive examples: Code nodes that actually need the specific annotation type
- Negative examples: Code nodes that don't need the specific annotation type (real code, not artificial)

This ensures models learn from meaningful code patterns rather than artificial feature modifications.
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
class RealBalancedExample:
    """Represents a balanced training example using real code"""
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
    code_context: str     # The actual code context for this node

class ImprovedBalancedDatasetGenerator:
    """Generates balanced training datasets using real code examples"""
    
    def __init__(self, target_balance: float = 0.5, random_seed: int = 42):
        """
        Initialize the improved balanced dataset generator
        
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
        """Extract features from a CFG node (enhanced version)"""
        label = node.get('label', '')
        node_type = node.get('type', '')
        line = node.get('line', 0)
        
        # Enhanced features for annotation type prediction
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
            float('string' in label.lower()),  # contains_string
            float('collection' in label.lower()),  # contains_collection
            float('loop' in label.lower()),  # contains_loop
            float('condition' in label.lower()),  # contains_condition
            float('return' in label.lower()),  # contains_return
            float('call' in label.lower()),  # contains_call
        ]
        
        return features
    
    def get_code_context(self, node: Dict, cfg_data: Dict) -> str:
        """Extract the actual code context for a node"""
        label = node.get('label', '')
        node_type = node.get('type', '')
        line = node.get('line', 0)
        
        # Create a meaningful code context string
        context_parts = []
        
        if label:
            context_parts.append(f"Label: {label}")
        if node_type:
            context_parts.append(f"Type: {node_type}")
        if line and line > 0:
            context_parts.append(f"Line: {line}")
        
        # Add context from surrounding nodes if available
        nodes = cfg_data.get('nodes', [])
        if nodes:
            current_idx = next((i for i, n in enumerate(nodes) if n.get('id') == node.get('id')), -1)
            if current_idx >= 0:
                # Add context from previous and next nodes
                if current_idx > 0:
                    prev_node = nodes[current_idx - 1]
                    if prev_node.get('label'):
                        context_parts.append(f"Prev: {prev_node['label'][:30]}...")
                if current_idx < len(nodes) - 1:
                    next_node = nodes[current_idx + 1]
                    if next_node.get('label'):
                        context_parts.append(f"Next: {next_node['label'][:30]}...")
        
        return " | ".join(context_parts)
    
    def determine_annotation_type(self, node: Dict, cfg_data: Dict) -> str:
        """Determine the most appropriate annotation type for a node using enhanced rules"""
        label = node.get('label', '').lower()
        node_type = node.get('type', '').lower()
        
        # Enhanced rule-based annotation type determination
        # Rule 1: Array and index-related annotations
        if any(keyword in label for keyword in ['array', 'index', 'subscript']):
            if 'length' in label or 'size' in label:
                return '@NonNegative'
            elif 'bound' in label or 'limit' in label:
                return '@GTENegativeOne'
            else:
                return '@Positive'
        
        # Rule 2: Loop and iteration variables
        if any(keyword in label for keyword in ['loop', 'iter', 'i', 'j', 'k']):
            if 'array' in label or 'list' in label:
                return '@NonNegative'
            else:
                return '@GTENegativeOne'
        
        # Rule 3: Size and length related
        if any(keyword in label for keyword in ['length', 'size', 'count', 'capacity']):
            if 'parameter' in node_type:
                return '@Positive'
            else:
                return '@NonNegative'
        
        # Rule 4: Numeric types and parameters
        if 'parameter' in node_type:
            if any(keyword in label for keyword in ['int', 'long', 'double', 'float']):
                return '@NonNegative'
            else:
                return '@Positive'
        
        # Rule 5: String and collection types
        if any(keyword in label for keyword in ['string', 'list', 'map', 'set']):
            return '@Positive'
        
        # Rule 6: Method calls and complex patterns
        if 'method' in node_type or 'call' in label:
            return '@Positive'
        
        # Rule 7: Variable declarations
        if 'variable' in node_type:
            if any(keyword in label for keyword in ['temp', 'result', 'value']):
                return '@GTENegativeOne'
            else:
                return '@NonNegative'
        
        # Default based on context
        if 'positive' in label:
            return '@Positive'
        elif 'negative' in label:
            return '@GTENegativeOne'
        else:
            return '@NonNegative'
    
    def classify_node_for_annotation_type(self, node: Dict, cfg_data: Dict, target_annotation: str) -> Tuple[bool, float]:
        """
        Classify whether a node should have the target annotation type
        
        Returns:
            (is_positive, confidence): Whether the node needs the annotation and confidence score
        """
        predicted_annotation = self.determine_annotation_type(node, cfg_data)
        
        # Check if the node actually needs this annotation type
        is_positive = (predicted_annotation == target_annotation)
        
        # Calculate confidence based on how well the node matches the target annotation
        label = node.get('label', '').lower()
        node_type = node.get('type', '').lower()
        
        confidence = 0.5  # Base confidence
        
        if target_annotation == '@Positive':
            # Features that suggest @Positive annotation
            if any(keyword in label for keyword in ['size', 'length', 'count', 'capacity']):
                confidence += 0.3
            if 'parameter' in node_type:
                confidence += 0.2
            if any(keyword in label for keyword in ['int', 'long', 'double']):
                confidence += 0.1
        
        elif target_annotation == '@NonNegative':
            # Features that suggest @NonNegative annotation
            if any(keyword in label for keyword in ['index', 'offset', 'position']):
                confidence += 0.3
            if any(keyword in label for keyword in ['array', 'list']):
                confidence += 0.2
            if 'parameter' in node_type:
                confidence += 0.2
        
        elif target_annotation == '@GTENegativeOne':
            # Features that suggest @GTENegativeOne annotation
            if any(keyword in label for keyword in ['bound', 'limit', 'capacity']):
                confidence += 0.3
            if any(keyword in label for keyword in ['variable', 'temp', 'result']):
                confidence += 0.2
            if any(keyword in label for keyword in ['count', 'size']):
                confidence += 0.1
        
        # Ensure confidence is in [0.1, 1.0] range
        confidence = max(0.1, min(1.0, confidence))
        
        return is_positive, confidence
    
    def generate_balanced_examples(self, cfg_files: List[Dict[str, Any]], 
                                 examples_per_annotation: int = 1000) -> Dict[str, List[RealBalancedExample]]:
        """
        Generate balanced examples using REAL code examples
        
        Args:
            cfg_files: List of CFG file data
            examples_per_annotation: Target number of examples per annotation type
            
        Returns:
            Dictionary mapping annotation types to lists of balanced examples
        """
        balanced_datasets = {ann_type: [] for ann_type in self.annotation_types}
        
        logger.info(f"Generating balanced datasets with {examples_per_annotation} examples per annotation type")
        logger.info(f"Target balance: {self.target_balance*100:.1f} percent positive, {(1-self.target_balance)*100:.1f} percent negative")
        
        for ann_type in self.annotation_types:
            logger.info(f"\nGenerating examples for {ann_type}...")
            
            # Collect all nodes and classify them for this annotation type
            positive_nodes = []
            negative_nodes = []
            
            for cfg_file in cfg_files:
                cfg_data = cfg_file['data']
                for node in cfg_data.get('nodes', []):
                    is_positive, confidence = self.classify_node_for_annotation_type(node, cfg_data, ann_type)
                    
                    node_info = {
                        'node': node,
                        'cfg_data': cfg_data,
                        'file_path': cfg_file['file'],
                        'method_name': cfg_file['method'],
                        'confidence': confidence
                    }
                    
                    if is_positive:
                        positive_nodes.append(node_info)
                    else:
                        negative_nodes.append(node_info)
            
            logger.info(f"Found {len(positive_nodes)} positive and {len(negative_nodes)} negative nodes for {ann_type}")
            
            if len(positive_nodes) == 0 or len(negative_nodes) == 0:
                logger.warning(f"Insufficient nodes for {ann_type}: {len(positive_nodes)} positive, {len(negative_nodes)} negative")
                continue
            
            # Generate positive examples (real nodes that need this annotation)
            num_positive = int(examples_per_annotation * self.target_balance)
            positive_examples = self._generate_real_positive_examples(
                positive_nodes, ann_type, num_positive
            )
            
            # Generate negative examples (real nodes that don't need this annotation)
            num_negative = examples_per_annotation - num_positive
            negative_examples = self._generate_real_negative_examples(
                negative_nodes, ann_type, num_negative
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
    
    def _generate_real_positive_examples(self, positive_nodes: List[Dict], 
                                       annotation_type: str, num_examples: int) -> List[RealBalancedExample]:
        """Generate positive examples using real nodes that need the annotation"""
        examples = []
        
        # Sample nodes with replacement if needed, prioritizing higher confidence
        positive_nodes_sorted = sorted(positive_nodes, key=lambda x: x['confidence'], reverse=True)
        
        for i in range(num_examples):
            node_info = positive_nodes_sorted[i % len(positive_nodes_sorted)]
            node = node_info['node']
            cfg_data = node_info['cfg_data']
            
            features = self.extract_node_features(node, cfg_data)
            code_context = self.get_code_context(node, cfg_data)
            
            example = RealBalancedExample(
                node_id=node.get('id', i),
                file_path=node_info['file_path'],
                method_name=node_info['method_name'],
                line_number=node.get('line', 0),
                node_type=node.get('type', ''),
                node_label=node.get('label', ''),
                features=features,
                annotation_type=annotation_type,
                is_positive=True,
                confidence=node_info['confidence'],
                code_context=code_context
            )
            
            examples.append(example)
        
        return examples
    
    def _generate_real_negative_examples(self, negative_nodes: List[Dict], 
                                       annotation_type: str, num_examples: int) -> List[RealBalancedExample]:
        """Generate negative examples using real nodes that don't need the annotation"""
        examples = []
        
        # Sample nodes with replacement if needed, prioritizing lower confidence (more certain negatives)
        negative_nodes_sorted = sorted(negative_nodes, key=lambda x: x['confidence'])
        
        for i in range(num_examples):
            node_info = negative_nodes_sorted[i % len(negative_nodes_sorted)]
            node = node_info['node']
            cfg_data = node_info['cfg_data']
            
            features = self.extract_node_features(node, cfg_data)
            code_context = self.get_code_context(node, cfg_data)
            
            example = RealBalancedExample(
                node_id=node.get('id', i),
                file_path=node_info['file_path'],
                method_name=node_info['method_name'],
                line_number=node.get('line', 0),
                node_type=node.get('type', ''),
                node_label=node.get('label', ''),
                features=features,
                annotation_type=annotation_type,
                is_positive=False,
                confidence=node_info['confidence'],
                code_context=code_context
            )
            
            examples.append(example)
        
        return examples
    
    def save_balanced_dataset(self, balanced_datasets: Dict[str, List[RealBalancedExample]], 
                            output_dir: str):
        """Save the balanced datasets to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for ann_type, examples in balanced_datasets.items():
            if not examples:
                continue
            
            # Save as JSON
            output_file = os.path.join(output_dir, f"{ann_type.replace('@', '').lower()}_real_balanced_dataset.json")
            
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
                        'confidence': ex.confidence,
                        'code_context': ex.code_context
                    }
                    for ex in examples
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(dataset_data, f, indent=2)
            
            logger.info(f"Saved {len(examples)} real examples for {ann_type} to {output_file}")
        
        # Save overall statistics
        stats_file = os.path.join(output_dir, "real_generation_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(self.generation_stats, f, indent=2)
        
        logger.info(f"Saved generation statistics to {stats_file}")
    
    def print_statistics(self):
        """Print generation statistics"""
        print("\n" + "="*60)
        print("REAL BALANCED DATASET GENERATION STATISTICS")
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
    """Main function to generate balanced datasets using real code examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate balanced training datasets using real code examples')
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
    generator = ImprovedBalancedDatasetGenerator(
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
