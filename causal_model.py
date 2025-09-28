#!/usr/bin/env python3
"""
Causal Model Training Script with Best Practices Defaults

This script trains Causal models using:
- Dataflow-augmented CFGs by default
- Augmented slices as default training data
- Enhanced features including dataflow information

Best Practices:
- Uses dataflow features for better model performance
- Prefers augmented slices for improved generalization
- Integrates seamlessly with prediction pipeline
- Maintains consistency across training and inference
"""

# causal_model.py

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import subprocess
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
try:
    from dowhy import CausalModel  # type: ignore
    DOWHY_AVAILABLE = True
except Exception:
    CausalModel = None
    DOWHY_AVAILABLE = False

from cfg import generate_control_flow_graphs, save_cfgs

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

def iter_java_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.java'):
                yield os.path.join(root, f)

def ensure_cfg(java_file):
    base = os.path.splitext(os.path.basename(java_file))[0]
    cfg_dir = os.path.join(cfg_output_dir, base)
    if not os.path.exists(cfg_dir) or not any(name.endswith('.json') for name in os.listdir(cfg_dir)):
        cfgs = generate_control_flow_graphs(java_file, cfg_output_dir)
        save_cfgs(cfgs, cfg_dir)

def load_data():
    """
    Load CFGs and prepare data for causal modeling.
    """
    data_records = []
    for java_file_path in iter_java_files(slices_dir):
        ensure_cfg(java_file_path)
        method_cfgs = load_cfgs(java_file_path)
        for cfg_entry in method_cfgs:
            cfg_data = cfg_entry.get('data', cfg_entry)
            records = extract_features_and_labels_synthetic(cfg_data)
            data_records.extend(records)
    return pd.DataFrame(data_records)

def run_index_checker(java_file):
    """
    Run the Checker Framework's Index Checker on the given Java file and capture warnings.
    """
    # Construct the command to run the Index Checker
    command = ['javac']
    if index_checker_cp:
        command += ['-cp', index_checker_cp]
    command += ['-processor', 'org.checkerframework.checker.index.IndexChecker', java_file]
    result = subprocess.run(command, capture_output=True, text=True)
    warnings_output = result.stderr  # Warnings are typically output to stderr
    return warnings_output

def parse_warnings(warnings_output):
    """
    Parse the warnings generated by Index Checker to identify nodes for annotations.
    """
    pattern = re.compile(r'^(.*\.java):(\d+):\s*(error|warning):\s*(.*)$', re.MULTILINE)
    annotations = []  # List of dictionaries with file, line, and message

    for match in pattern.finditer(warnings_output):
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

def extract_features_and_labels(cfg_data, annotations):
    """
    Extract features and labels from CFG data and annotations for causal modeling.
    """
    records = []
    nodes = cfg_data['nodes']
    edges = cfg_data['edges']
    method_name = cfg_data['method_name']
    java_file = cfg_data['java_file']
    # Build a set of annotation line numbers
    annotation_lines = set()
    for annotation in annotations:
        if os.path.abspath(annotation['file']) == os.path.abspath(java_file):
            annotation_lines.add(annotation['line'])
    # Create a mapping from node IDs to line numbers (if available)
    node_line_map = {}
    for node in nodes:
        node_id = node['id']
        line_number = node.get('line', None)
        if line_number is not None:
            node_line_map[node_id] = line_number
    for node in nodes:
        node_id = node['id']
        label = node['label']
        # Extract features from the node
        features = {
            'node_id': node_id,
            'method_name': method_name,
            'java_file': java_file,
            'label_length': len(label),
            'label': label,
        }
        # Degree features
        in_degree = 0
        out_degree = 0
        for edge in edges:
            if edge['target'] == node_id:
                in_degree += 1
            if edge['source'] == node_id:
                out_degree += 1
        features['in_degree'] = in_degree
        features['out_degree'] = out_degree
        # Line number (if available)
        line_number = node_line_map.get(node_id, None)
        features['line_number'] = line_number
        # Target variable: whether annotation is needed
        if line_number in annotation_lines:
            features['needs_annotation'] = 1
        else:
            features['needs_annotation'] = 0
        records.append(features)
    return records

def extract_features_and_labels_synthetic(cfg_data):
    """
    Extract features and synthetic labels from a CFG for causal modeling.
    Now includes dataflow information.
    """
    records = []
    nodes = cfg_data['nodes']
    edges = cfg_data.get('edges', [])
    control_edges = cfg_data.get('control_edges', [])
    dataflow_edges = cfg_data.get('dataflow_edges', [])
    method_name = cfg_data['method_name']
    java_file = cfg_data['java_file']
    
    # Calculate CFG complexity for synthetic labeling
    node_labels = [node.get('label', '') for node in nodes]
    complexity_score = (
        len([label for label in node_labels if 'if' in label.lower()]) +
        len([label for label in node_labels if 'for' in label.lower()]) +
        len([label for label in node_labels if 'while' in label.lower()]) +
        len([label for label in node_labels if 'switch' in label.lower()]) +
        len([label for label in node_labels if 'try' in label.lower()])
    )
    
    # Calculate dataflow complexity
    dataflow_complexity = len(dataflow_edges)
    control_complexity = len(control_edges)
    
    for node in nodes:
        node_id = node['id']
        label = node['label']
        # Extract features from the node
        features = {
            'node_id': node_id,
            'method_name': method_name,
            'java_file': java_file,
            'label_length': len(label),
            'label': label,
        }
        
        # Control flow degree features
        control_in_degree = 0
        control_out_degree = 0
        for edge in control_edges:
            if edge['target'] == node_id:
                control_in_degree += 1
            if edge['source'] == node_id:
                control_out_degree += 1
        
        # Dataflow degree features
        dataflow_in_degree = 0
        dataflow_out_degree = 0
        variables_used = set()
        for edge in dataflow_edges:
            if edge['target'] == node_id:
                dataflow_in_degree += 1
                if 'variable' in edge:
                    variables_used.add(edge['variable'])
            if edge['source'] == node_id:
                dataflow_out_degree += 1
                if 'variable' in edge:
                    variables_used.add(edge['variable'])
        
        # Legacy degree features (for backward compatibility)
        in_degree = control_in_degree + dataflow_in_degree
        out_degree = control_out_degree + dataflow_out_degree
        
        features['in_degree'] = in_degree
        features['out_degree'] = out_degree
        features['control_in_degree'] = control_in_degree
        features['control_out_degree'] = control_out_degree
        features['dataflow_in_degree'] = dataflow_in_degree
        features['dataflow_out_degree'] = dataflow_out_degree
        features['variables_used'] = len(variables_used)
        
        # Line number (if available)
        line_number = node.get('line', 0)
        features['line_number'] = line_number
        
        # Synthetic label: nodes in complex CFGs with dataflow are more likely to need annotations
        node_complexity = 0
        label_lower = label.lower()
        if 'if' in label_lower or 'for' in label_lower or 'while' in label_lower:
            node_complexity += 2
        if 'return' in label_lower:
            node_complexity += 1
        if 'assignment' in label_lower or '=' in label_lower:
            node_complexity += 1
        
        # Dataflow complexity bonus
        if dataflow_in_degree > 0 or dataflow_out_degree > 0:
            node_complexity += 1
        
        # Combine CFG complexity, dataflow complexity, and node complexity
        total_complexity = complexity_score + dataflow_complexity + node_complexity
        features['needs_annotation'] = 1 if total_complexity > 4 else 0
        
        records.append(features)
    return records

def main():
    # Load data
    data = load_data()
    if data.empty:
        print("No data available for training.")
        return
    
    print(f"Loaded {len(data)} data points for causal model training")
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Skip complex DoWhy causal inference and focus on predictive modeling
    print("Using simplified causal model approach (predictive classifier only)")
    
    # Use the causal model to predict where annotations should be placed
    data['predicted_annotation'], clf = predict_annotations(data)
    
    # Evaluate the model
    accuracy = accuracy_score(data['needs_annotation'], data['predicted_annotation'])
    f1 = f1_score(data['needs_annotation'], data['predicted_annotation'])
    print(f"Causal model accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    
    # Save predictive classifier for inference
    clf_path = os.path.join(models_dir, 'causal_clf.joblib')
    joblib.dump(clf, clf_path)
    print(f"Predictive classifier saved at {clf_path}")
    
    # Print feature importance if available
    if hasattr(clf, 'feature_importances_'):
        feature_names = ['label_length', 'in_degree', 'out_degree', 'control_in_degree', 'control_out_degree', 
                        'dataflow_in_degree', 'dataflow_out_degree', 'variables_used', 'label_encoded', 'line_number']
        importances = clf.feature_importances_
        print("\nFeature importance:")
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.4f}")

def preprocess_data(data):
    """
    Preprocess data for causal modeling.
    """
    # Encode categorical variables
    data['label_encoded'] = data['label'].astype('category').cat.codes
    # Fill missing values
    data = data.fillna(0)
    return data

def define_causal_model(data):
    """
    Define the causal model using DoWhy.
    """
    # Specify the causal graph
    # Assuming the following relationships:
    # - Features affect whether an annotation is needed
    model = CausalModel(
        data=data,
        treatment=['label_length', 'in_degree', 'out_degree', 'label_encoded', 'line_number'],
        outcome='needs_annotation',
        graph="digraph{"
              "label_length -> needs_annotation;"
              "in_degree -> needs_annotation;"
              "out_degree -> needs_annotation;"
              "label_encoded -> needs_annotation;"
              "line_number -> needs_annotation;"
              "}",
    )
    return model

def predict_annotations(data):
    """
    Use the causal model to predict where annotations should be placed.
    Now includes dataflow features.
    """
    # Features used for prediction (including dataflow features)
    features = ['label_length', 'in_degree', 'out_degree', 'control_in_degree', 'control_out_degree', 
                'dataflow_in_degree', 'dataflow_out_degree', 'variables_used', 'label_encoded', 'line_number']
    
    # Filter features that exist in the data
    available_features = [f for f in features if f in data.columns]
    X = data[available_features]
    y = data['needs_annotation']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a classifier
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Predict on the full dataset
    y_pred = clf.predict(X)
    return y_pred, clf

if __name__ == '__main__':
    main()
