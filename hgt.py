#!/usr/bin/env python3
"""
HGT Training Script with Best Practices Defaults

This script trains Heterogeneous Graph Transformer models using:
- Dataflow-augmented CFGs by default
- Augmented slices as default training data
- Enhanced graph features including dataflow information

Best Practices:
- Uses dataflow edges for better graph representation
- Prefers augmented slices for improved model generalization
- Integrates seamlessly with prediction pipeline
- Maintains consistency across training and inference
"""

import os
import json
import random
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cfg import generate_control_flow_graphs, save_cfgs

def ensure_cfg(java_file):
    base = os.path.splitext(os.path.basename(java_file))[0]
    out_dir = os.path.join(cfg_output_dir, base)
    if not os.path.exists(out_dir) or not any(name.endswith('.json') for name in os.listdir(out_dir)):
        cfgs = generate_control_flow_graphs(java_file, cfg_output_dir)
        save_cfgs(cfgs, out_dir)

# Directory paths
cfg_output_dir = os.environ.get("CFG_OUTPUT_DIR", "cfg_output")
slices_dir = os.environ.get("SLICES_DIR", "slices")

# Default behavior: Use augmented slices if available, otherwise fall back to regular slices
def find_best_slices_directory():
    """Find the best available slices directory, preferring augmented slices."""
    # First, check if SLICES_DIR is explicitly set and contains Java files
    if os.path.exists(slices_dir) and any(f.endswith('.java') for f in os.listdir(slices_dir) if os.path.isfile(os.path.join(slices_dir, f))):
        return slices_dir
    
    # Look for augmented slices directories (preferred)
    base_dir = os.path.dirname(slices_dir) if os.path.dirname(slices_dir) else "."
    for slicer in ['specimin', 'wala']:  # Prefer specimin since it's working
        aug_dir = os.path.join(base_dir, f"slices_aug_{slicer}")
        if os.path.exists(aug_dir) and any(f.endswith('.java') for f in os.listdir(aug_dir) if os.path.isfile(os.path.join(aug_dir, f))):
            print(f"Using augmented slices from: {aug_dir}")
            return aug_dir
    
    # Look for general augmented slices directory
    aug_dir = os.path.join(base_dir, "slices_aug")
    if os.path.exists(aug_dir) and any(f.endswith('.java') for f in os.listdir(aug_dir) if os.path.isfile(os.path.join(aug_dir, f))):
        print(f"Using augmented slices from: {aug_dir}")
        return aug_dir
    
    # Fall back to regular slices
    if os.path.exists(slices_dir):
        print(f"Using regular slices from: {slices_dir}")
        return slices_dir
    
    # Last resort: look for any slices directory
    for potential_dir in ["slices", "slices_specimin", "slices_wala"]:
        if os.path.exists(potential_dir) and any(f.endswith('.java') for f in os.listdir(potential_dir) if os.path.isfile(os.path.join(potential_dir, f))):
            print(f"Using slices from: {potential_dir}")
            return potential_dir
    
    raise FileNotFoundError("No slices directory found with Java files")

slices_dir = find_best_slices_directory()
index_checker_cp = os.environ.get("CHECKERFRAMEWORK_CP", "")
models_dir = os.environ.get("MODELS_DIR", "models")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Function to load CFGs and convert them into HeteroData objects
def iter_java_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".java"):
                yield os.path.join(root, f)

def load_data():
    graphs = []
    labels = []

    for java_file_path in iter_java_files(slices_dir):
        # Run Index Checker to get warnings
        warnings = run_index_checker(java_file_path)
        annotations = parse_warnings(warnings)
        # Load CFGs
        ensure_cfg(java_file_path)
        method_cfgs = load_cfgs(java_file_path)
        for cfg_entry in method_cfgs:
            cfg_data = cfg_entry.get('data', cfg_entry)
            # Create HeteroData object
            data = create_heterodata(cfg_data)
            if data is None:
                continue
            # Label nodes based on warnings
            label_nodes(data, cfg_data, annotations)
            graphs.append(data)
    return graphs

def load_cfgs(java_file, cfg_output_dir=None):
    """
    Load the saved CFGs for a given Java file with proper structure for node-level models.
    """
    if cfg_output_dir is None:
        cfg_output_dir = os.environ.get("CFG_OUTPUT_DIR", "cfg_output")
    
    method_cfgs = []
    java_file_name = os.path.splitext(os.path.basename(java_file))[0]
    cfg_dir = os.path.join(cfg_output_dir, java_file_name)
    if os.path.exists(cfg_dir):
        for cfg_file in os.listdir(cfg_dir):
            if cfg_file.endswith('.json'):
                cfg_file_path = os.path.join(cfg_dir, cfg_file)
                with open(cfg_file_path, 'r') as f:
                    cfg_data = json.load(f)
                    # Add method name to cfg_data for identification
                    cfg_data['method_name'] = os.path.splitext(cfg_file)[0]
                    cfg_data['java_file'] = java_file
                    # Wrap in structure expected by node-level models
                    method_cfgs.append({
                        'file': cfg_file_path,
                        'method': cfg_data.get('method_name', 'unknown'),
                        'data': cfg_data
                    })
    else:
        print(f"CFG directory {cfg_dir} does not exist for Java file {java_file}")
    return method_cfgs

def run_index_checker(java_file):
    """
    Run the Checker Framework's Index Checker on the given Java file and capture warnings.
    """
    # Construct the command to run the Index Checker
    cp = index_checker_cp
    command = ['javac']
    if cp:
        command += ['-cp', cp]
    command += ['-processor', 'org.checkerframework.checker.index.IndexChecker', java_file]
    result = subprocess.run(command, capture_output=True, text=True)
    warnings = result.stderr  # Warnings are typically output to stderr
    return warnings

def parse_warnings(warnings):
    """
    Parse the warnings generated by Index Checker to identify nodes for annotations.
    """
    import re
    pattern = re.compile(r'^(.*\.java):(\d+):\s*(error|warning):\s*(.*)$')
    annotations = []  # List of dictionaries with file, line, and message

    for line in warnings.split('\n'):
        match = pattern.match(line)
        if match:
            file_path = match.group(1).strip()
            line_number = int(match.group(2).strip())
            message_type = match.group(3).strip()
            message = match.group(4).strip()
            annotations.append({
                'file': file_path,
                'line': line_number,
                'message_type': message_type,
                'message': message
            })
    return annotations

def create_heterodata(cfg_data):
    """
    Convert CFG data into a HeteroData object for PyTorch Geometric.
    Now handles both control flow and dataflow edges as different edge types.
    """
    data = HeteroData()
    nodes = cfg_data['nodes']
    
    # Get both control and dataflow edges
    control_edges = cfg_data.get('control_edges', [])
    dataflow_edges = cfg_data.get('dataflow_edges', [])
    
    # Fallback to general edges if specific edge types not available
    if not control_edges and not dataflow_edges:
        edges = cfg_data.get('edges', [])
        control_edges = edges  # Treat all as control edges for backward compatibility

    # Create node features (e.g., label encoding, node type)
    node_features = []
    node_labels = []
    node_indices = {}
    for node in nodes:
        node_id = node['id']
        label = node['label']
        node_type = node.get('node_type', 'control')
        
        # Enhanced features: label length, node type encoding
        feature = [len(label)]
        # Add node type encoding (0 for control, 1 for dataflow if we had such nodes)
        feature.append(0 if node_type == 'control' else 1)
        
        node_features.append(feature)
        node_indices[node_id] = len(node_indices)
        # Initialize labels to 0 (no annotation)
        node_labels.append(0)

    if not node_features:
        return None  # Skip if no nodes

    data['node'].x = torch.tensor(node_features, dtype=torch.float)
    data['node'].y = torch.tensor(node_labels, dtype=torch.long)

    # Create control flow edge index
    control_edge_index = [[], []]
    for edge in control_edges:
        source = node_indices.get(edge.get('source', edge.get('from')))
        target = node_indices.get(edge.get('target', edge.get('to')))
        if source is not None and target is not None:
            control_edge_index[0].append(source)
            control_edge_index[1].append(target)

    # Create dataflow edge index
    dataflow_edge_index = [[], []]
    for edge in dataflow_edges:
        source = node_indices.get(edge.get('source', edge.get('from')))
        target = node_indices.get(edge.get('target', edge.get('to')))
        if source is not None and target is not None:
            dataflow_edge_index[0].append(source)
            dataflow_edge_index[1].append(target)

    # Combine all edges into a single edge type for HGT compatibility
    # This treats both control flow and dataflow edges as the same type
    all_edges = control_edges + dataflow_edges
    combined_edge_index = [[], []]
    for edge in all_edges:
        source = node_indices.get(edge.get('source', edge.get('from')))
        target = node_indices.get(edge.get('target', edge.get('to')))
        if source is not None and target is not None:
            combined_edge_index[0].append(source)
            combined_edge_index[1].append(target)
    
    # Use the standard edge type that HGT expects
    if combined_edge_index[0]:
        data['node', 'to', 'node'].edge_index = torch.tensor(combined_edge_index, dtype=torch.long)

    # Check if we have any edges at all
    if not combined_edge_index[0]:
        return None  # Skip if no edges

    return data

def label_nodes(data, cfg_data, annotations):
    """
    Label nodes in the data object based on warnings.
    """
    node_labels = data['node'].y.numpy()
    nodes = cfg_data['nodes']
    node_line_map = {}
    for idx, node in enumerate(nodes):
        line_number = node.get('line', None)
        if line_number is not None:
            node_line_map[line_number] = idx

    # Build a set of annotation line numbers
    annotation_lines = set()
    for annotation in annotations:
        if os.path.abspath(annotation['file']) == os.path.abspath(cfg_data['java_file']):
            annotation_lines.add(annotation['line'])

    # Label nodes
    for line_number in annotation_lines:
        node_idx = node_line_map.get(line_number)
        if node_idx is not None:
            node_labels[node_idx] = 1  # Mark node for annotation

    data['node'].y = torch.tensor(node_labels, dtype=torch.long)

# Define the HGT model
class HGTModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super(HGTModel, self).__init__()
        self.convs = nn.ModuleList()
        current_in = in_channels
        for layer_idx in range(num_layers):
            conv = HGTConv(
                in_channels=current_in,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads
            )
            self.convs.append(conv)
            current_in = hidden_channels
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # Apply activation function
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        # Output layer
        out = self.fc(x_dict['node'])
        return out

# Main training loop
def main():
    # Load data
    graphs = load_data()
    if not graphs:
        print("No data available for training.")
        return

    # Split data into training and validation sets
    random.shuffle(graphs)
    train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)

    # Get metadata from a sample graph
    metadata = graphs[0].metadata()
    in_channels = graphs[0]['node'].x.size(-1)
    out_channels = 2  # Binary classification (annotate or not)

    # Define model, loss function, and optimizer
    model = HGTModel(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels,
        num_heads=2,
        num_layers=2,
        metadata=metadata
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_loss = float('inf')
    num_epochs = 40

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            loss = criterion(out, data['node'].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data['node'].num_nodes

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(models_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with validation loss {val_loss:.4f}")

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_nodes = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x_dict, data.edge_index_dict)
            loss = criterion(out, data['node'].y)
            total_loss += loss.item() * data['node'].num_nodes
            pred = out.argmax(dim=1)
            correct += (pred == data['node'].y).sum().item()
            total_nodes += data['node'].num_nodes
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total_nodes
    return avg_loss

if __name__ == '__main__':
    main()
