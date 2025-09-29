#!/usr/bin/env python3
"""
CFG DataLoader and Batching Framework for Large Input Support

This module provides a comprehensive framework for handling CFG graphs of varying sizes
with proper batching, padding, and dynamic sizing support for all annotation type models.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from collections import defaultdict
import random

# Import CFG processing utilities
from cfg_graph import load_cfg_as_pyg

logger = logging.getLogger(__name__)

class CFGSizeConfig:
    """Configuration for CFG size limits and batching parameters"""
    
    # Maximum sizes (can be increased based on available memory)
    MAX_NODES = 1000      # Increased from default ~20
    MAX_EDGES = 2000      # Increased from default ~40
    MAX_BATCH_SIZE = 32   # Batch size for training
    
    # Padding and truncation settings
    PAD_TO_MAX = True     # Pad smaller graphs to max size
    TRUNCATE_LARGE = True # Truncate graphs that exceed max size
    
    # Feature dimensions
    NODE_FEATURE_DIM = 15
    EDGE_FEATURE_DIM = 2
    
    @classmethod
    def update_limits(cls, max_nodes: int = None, max_edges: int = None, max_batch_size: int = None):
        """Update size limits dynamically"""
        if max_nodes is not None:
            cls.MAX_NODES = max_nodes
        if max_edges is not None:
            cls.MAX_EDGES = max_edges
        if max_batch_size is not None:
            cls.MAX_BATCH_SIZE = max_batch_size


class CFGDataset(Dataset):
    """Dataset class for CFG graphs with dynamic sizing support"""
    
    def __init__(self, 
                 cfg_files: List[str],
                 targets: Optional[List[int]] = None,
                 max_nodes: int = None,
                 max_edges: int = None,
                 pad_to_max: bool = True,
                 truncate_large: bool = True):
        """
        Initialize CFG dataset
        
        Args:
            cfg_files: List of paths to CFG JSON files
            targets: Optional list of target labels
            max_nodes: Maximum number of nodes per graph
            max_edges: Maximum number of edges per graph
            pad_to_max: Whether to pad smaller graphs to max size
            truncate_large: Whether to truncate graphs exceeding max size
        """
        self.cfg_files = cfg_files
        self.targets = targets if targets is not None else [0] * len(cfg_files)
        
        # Use provided limits or defaults from config
        self.max_nodes = max_nodes or CFGSizeConfig.MAX_NODES
        self.max_edges = max_edges or CFGSizeConfig.MAX_EDGES
        self.pad_to_max = pad_to_max
        self.truncate_large = truncate_large
        
        # Validate inputs
        if len(self.cfg_files) != len(self.targets):
            raise ValueError("Number of CFG files must match number of targets")
        
        # Analyze dataset statistics
        self._analyze_dataset()
        
    def _analyze_dataset(self):
        """Analyze the dataset to understand size distribution"""
        node_counts = []
        edge_counts = []
        
        for cfg_file in self.cfg_files[:min(100, len(self.cfg_files))]:  # Sample first 100
            try:
                with open(cfg_file, 'r') as f:
                    cfg_data = json.load(f)
                
                num_nodes = len(cfg_data.get('nodes', []))
                num_edges = len(cfg_data.get('control_edges', [])) + len(cfg_data.get('dataflow_edges', []))
                
                node_counts.append(num_nodes)
                edge_counts.append(num_edges)
                
            except Exception as e:
                logger.warning(f"Error analyzing {cfg_file}: {e}")
        
        if node_counts:
            self.stats = {
                'avg_nodes': np.mean(node_counts),
                'max_nodes': np.max(node_counts),
                'avg_edges': np.mean(edge_counts),
                'max_edges': np.max(edge_counts),
                'samples_analyzed': len(node_counts)
            }
            
            logger.info(f"Dataset analysis: {self.stats}")
            
            # Adjust limits if dataset has larger graphs
            if self.stats['max_nodes'] > self.max_nodes:
                logger.warning(f"Dataset has graphs with up to {self.stats['max_nodes']} nodes, "
                             f"but limit is {self.max_nodes}. Consider increasing MAX_NODES.")
            
            if self.stats['max_edges'] > self.max_edges:
                logger.warning(f"Dataset has graphs with up to {self.stats['max_edges']} edges, "
                             f"but limit is {self.max_edges}. Consider increasing MAX_EDGES.")
        else:
            self.stats = None
    
    def __len__(self):
        return len(self.cfg_files)
    
    def __getitem__(self, idx):
        """Get a single CFG graph with proper sizing"""
        cfg_file = self.cfg_files[idx]
        target = self.targets[idx]
        
        try:
            # Load CFG as PyG graph
            graph_data = load_cfg_as_pyg(cfg_file)
            
            if graph_data.x is None or graph_data.x.numel() == 0:
                # Create empty graph with proper structure
                graph_data = self._create_empty_graph()
            
            # Ensure batch tensor exists
            if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            else:
                # Ensure batch tensor matches number of nodes
                if graph_data.batch.size(0) != graph_data.x.size(0):
                    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            
            # Apply size constraints
            graph_data = self._apply_size_constraints(graph_data)
            
            # Add target as a property
            graph_data.target = torch.tensor([target], dtype=torch.long)
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error loading CFG {cfg_file}: {e}")
            # Return empty graph on error
            return self._create_empty_graph()
    
    def _create_empty_graph(self):
        """Create an empty graph with proper structure"""
        return Data(
            x=torch.zeros(1, CFGSizeConfig.NODE_FEATURE_DIM),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, CFGSizeConfig.EDGE_FEATURE_DIM),
            batch=torch.zeros(1, dtype=torch.long),
            target=torch.tensor([0], dtype=torch.long)
        )
    
    def _apply_size_constraints(self, graph_data: Data) -> Data:
        """Apply size constraints to a graph"""
        num_nodes = graph_data.x.size(0)
        num_edges = graph_data.edge_index.size(1)
        
        # Handle node constraints
        if num_nodes > self.max_nodes and self.truncate_large:
            logger.debug(f"Truncating graph from {num_nodes} to {self.max_nodes} nodes")
            # Keep first max_nodes nodes
            node_mask = torch.arange(self.max_nodes, device=graph_data.x.device)
            graph_data.x = graph_data.x[node_mask]
            
            # Update edge indices and remove edges involving truncated nodes
            edge_mask = (graph_data.edge_index[0] < self.max_nodes) & (graph_data.edge_index[1] < self.max_nodes)
            graph_data.edge_index = graph_data.edge_index[:, edge_mask]
            if graph_data.edge_attr is not None:
                graph_data.edge_attr = graph_data.edge_attr[edge_mask]
            
            # Update batch tensor
            graph_data.batch = graph_data.batch[:self.max_nodes]
            
        elif num_nodes < self.max_nodes and self.pad_to_max:
            # Pad with zero nodes
            padding_size = self.max_nodes - num_nodes
            padding_nodes = torch.zeros(padding_size, graph_data.x.size(1), device=graph_data.x.device)
            graph_data.x = torch.cat([graph_data.x, padding_nodes], dim=0)
            
            # Update batch tensor - all nodes belong to batch 0
            padding_batch = torch.zeros(padding_size, dtype=torch.long, device=graph_data.batch.device)
            graph_data.batch = torch.cat([graph_data.batch, padding_batch], dim=0)
        
        # Handle edge constraints
        if num_edges > self.max_edges and self.truncate_large:
            logger.debug(f"Truncating graph from {num_edges} to {self.max_edges} edges")
            # Keep first max_edges edges
            edge_mask = torch.arange(self.max_edges, device=graph_data.edge_index.device)
            graph_data.edge_index = graph_data.edge_index[:, edge_mask]
            if graph_data.edge_attr is not None:
                graph_data.edge_attr = graph_data.edge_attr[edge_mask]
        
        elif num_edges < self.max_edges and self.pad_to_max:
            # Pad with self-loops (edges from node 0 to node 0)
            padding_size = self.max_edges - num_edges
            padding_edges = torch.zeros(2, padding_size, dtype=torch.long, device=graph_data.edge_index.device)
            graph_data.edge_index = torch.cat([graph_data.edge_index, padding_edges], dim=1)
            
            if graph_data.edge_attr is not None:
                padding_attrs = torch.zeros(padding_size, graph_data.edge_attr.size(1), device=graph_data.edge_attr.device)
                graph_data.edge_attr = torch.cat([graph_data.edge_attr, padding_attrs], dim=0)
        
        return graph_data


class CFGBatchCollator:
    """Custom collator for batching CFG graphs with proper padding"""
    
    def __init__(self, max_nodes: int = None, max_edges: int = None):
        self.max_nodes = max_nodes or CFGSizeConfig.MAX_NODES
        self.max_edges = max_edges or CFGSizeConfig.MAX_EDGES
    
    def __call__(self, batch: List[Data]) -> Batch:
        """Collate a batch of graphs with proper padding"""
        # Use PyTorch Geometric's batching
        batch_data = Batch.from_data_list(batch)
        
        # Fix batch tensor to match the actual number of nodes
        # PyTorch Geometric's batching may not handle our padding correctly
        batch_data.batch = torch.zeros(batch_data.x.size(0), dtype=torch.long)
        
        # Assign batch indices to each graph
        start_idx = 0
        for i, graph in enumerate(batch):
            num_nodes = graph.x.size(0)
            batch_data.batch[start_idx:start_idx + num_nodes] = i
            start_idx += num_nodes
        
        # Ensure batch size constraints
        total_nodes = batch_data.x.size(0)
        total_edges = batch_data.edge_index.size(1)
        
        if total_nodes > self.max_nodes * len(batch):
            logger.warning(f"Batch has {total_nodes} total nodes, exceeding limit of {self.max_nodes * len(batch)}")
        
        if total_edges > self.max_edges * len(batch):
            logger.warning(f"Batch has {total_edges} total edges, exceeding limit of {self.max_edges * len(batch)}")
        
        return batch_data


class CFGDataLoader:
    """Main dataloader class for CFG graphs with comprehensive batching support"""
    
    def __init__(self,
                 cfg_files: List[str],
                 targets: Optional[List[int]] = None,
                 batch_size: int = None,
                 shuffle: bool = True,
                 max_nodes: int = None,
                 max_edges: int = None,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        """
        Initialize CFG dataloader
        
        Args:
            cfg_files: List of CFG file paths
            targets: Optional target labels
            batch_size: Batch size (defaults to CFGSizeConfig.MAX_BATCH_SIZE)
            shuffle: Whether to shuffle data
            max_nodes: Maximum nodes per graph
            max_edges: Maximum edges per graph
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
        """
        self.batch_size = batch_size or CFGSizeConfig.MAX_BATCH_SIZE
        self.max_nodes = max_nodes or CFGSizeConfig.MAX_NODES
        self.max_edges = max_edges or CFGSizeConfig.MAX_EDGES
        
        # Create dataset
        self.dataset = CFGDataset(
            cfg_files=cfg_files,
            targets=targets,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges
        )
        
        # Create custom collator
        self.collator = CFGBatchCollator(max_nodes=self.max_nodes, max_edges=self.max_edges)
        
        # Create dataloader
        # Note: PyG DataLoader may not use custom collators properly, so we'll handle batching manually
        self.dataloader = PyGDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def __iter__(self):
        """Custom iterator that applies our collator to each batch"""
        for batch in self.dataloader:
            # The batch from PyG DataLoader is already a Batch object, but with incorrect batch tensor
            # We need to fix the batch tensor manually
            batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long)
            
            # Assign batch indices to each graph
            start_idx = 0
            batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') and batch.batch.numel() > 0 else 1
            
            # If we can't determine batch size from the original batch tensor, estimate it
            if batch_size == 1 and batch.x.size(0) > 0:
                # Estimate batch size based on node count and max_nodes
                estimated_batch_size = max(1, batch.x.size(0) // self.max_nodes)
                batch_size = estimated_batch_size
            
            # Distribute nodes evenly across batch
            nodes_per_graph = batch.x.size(0) // batch_size
            for i in range(batch_size):
                end_idx = start_idx + nodes_per_graph
                if i == batch_size - 1:  # Last batch gets remaining nodes
                    end_idx = batch.x.size(0)
                batch.batch[start_idx:end_idx] = i
                start_idx = end_idx
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_stats(self):
        """Get dataset statistics"""
        return self.dataset.stats


class CFGBatchProcessor:
    """Utility class for processing batches of CFG graphs"""
    
    @staticmethod
    def get_batch_info(batch: Batch) -> Dict[str, Any]:
        """Get information about a batch"""
        return {
            'batch_size': batch.batch.max().item() + 1,
            'total_nodes': batch.x.size(0),
            'total_edges': batch.edge_index.size(1),
            'avg_nodes_per_graph': batch.x.size(0) / (batch.batch.max().item() + 1),
            'avg_edges_per_graph': batch.edge_index.size(1) / (batch.batch.max().item() + 1),
            'node_feature_dim': batch.x.size(1),
            'edge_feature_dim': batch.edge_attr.size(1) if batch.edge_attr is not None else 0
        }
    
    @staticmethod
    def move_to_device(batch: Batch, device: torch.device) -> Batch:
        """Move batch to specified device"""
        return batch.to(device)
    
    @staticmethod
    def extract_targets(batch: Batch) -> torch.Tensor:
        """Extract targets from batch"""
        if hasattr(batch, 'target'):
            return batch.target
        else:
            # If targets are not in batch, return zeros
            return torch.zeros(batch.batch.max().item() + 1, dtype=torch.long, device=batch.x.device)


def create_cfg_dataloader(cfg_files: List[str],
                         targets: Optional[List[int]] = None,
                         **kwargs) -> CFGDataLoader:
    """Convenience function to create a CFG dataloader"""
    return CFGDataLoader(cfg_files, targets, **kwargs)


def find_cfg_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all CFG JSON files in a directory"""
    cfg_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    cfg_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.endswith('.json'):
                cfg_files.append(os.path.join(directory, file))
    
    return sorted(cfg_files)


# Example usage and testing
if __name__ == "__main__":
    # Test the dataloader
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Find CFG files
    cfg_dir = "/home/ubuntu/GenDATA/test_case_study_cfg_output"
    cfg_files = find_cfg_files(cfg_dir)
    
    if cfg_files:
        print(f"Found {len(cfg_files)} CFG files")
        
        # Create synthetic targets
        targets = [random.randint(0, 1) for _ in cfg_files]
        
        # Create dataloader
        dataloader = create_cfg_dataloader(
            cfg_files=cfg_files,
            targets=targets,
            batch_size=4,
            max_nodes=50,  # Small limit for testing
            max_edges=100
        )
        
        print(f"Created dataloader with {len(dataloader)} batches")
        
        # Test batching
        for i, batch in enumerate(dataloader):
            batch_info = CFGBatchProcessor.get_batch_info(batch)
            print(f"Batch {i}: {batch_info}")
            
            if i >= 2:  # Test first 3 batches
                break
    else:
        print("No CFG files found for testing")
