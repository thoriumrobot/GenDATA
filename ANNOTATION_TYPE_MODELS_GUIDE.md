# Annotation Type Models Guide (Graph-Based Inputs)

This guide explains how to use the new annotation-specific models that predict specific Checker Framework annotation types: `@Positive`, `@NonNegative`, and `@GTENegativeOne`, with CFG graph inputs. Graph models consume PyTorch Geometric graphs directly; non-graph models use a Graph Transformer encoder to obtain a fixed-length embedding.

## Overview

The annotation type models build upon the binary RL models to provide more precise annotation placement. Instead of just predicting whether an annotation should be placed, these models predict the specific type of annotation needed.

### Supported Annotation Types

1. **`@Positive`** - For values that must be greater than zero (e.g., count, size, length)
2. **`@NonNegative`** - For values that must be greater than or equal to zero (e.g., index, offset, position)
3. **`@GTENegativeOne`** - For values that must be greater than or equal to -1 (e.g., capacity, limit, bound)

## Architecture

The annotation type models use a two-stage approach:

1. **Binary Stage**: First, binary RL models predict whether an annotation should be placed
2. **Type Stage**: Then, annotation type models predict the specific annotation type

This approach ensures that only valid annotation targets are considered for type prediction.

## Graph-Based Inputs

- CFGs are generated via Checker Framework’s CFG Builder, then converted to **PyTorch Geometric** graphs using `cfg_graph.py`.
- Node features include: node-type one-hots, degree, normalized line number, Laplacian positional encodings (k eigenvectors), random-walk structural encodings (RWSE), and edge-type indicators (control vs dataflow).
- For non-graph models (GBT, causal, enhanced_causal), a **Graph Transformer encoder** (`graph_encoder.py`, with edge encodings and global attention pooling) produces a fixed-length embedding from the CFG graph. Trainers (`annotation_type_rl_*.py`) append this embedding to their feature vectors using `annotation_graph_input.py`.

## Model-Based Prediction System

The pipeline uses **trained machine learning models** for prediction by default. The system includes:

- **Enhanced Causal Model (Default)**
  - Graph-augmented features via embeddings
  - Dynamic confidence scores based on model certainty
- **Model-Based Predictions**
  - Model attribution in outputs
  - Confidence scores from model inference

## Scripts

### Individual Model Training Scripts

- `annotation_type_rl_positive.py` - Trains model for `@Positive` annotations (now appends CFG graph embeddings)
- `annotation_type_rl_nonnegative.py` - Trains model for `@NonNegative` annotations (with embeddings)
- `annotation_type_rl_gtenegativeone.py` - Trains model for `@GTENegativeOne` annotations (with embeddings)

### Pipeline Scripts

- `simple_annotation_type_pipeline.py` - Simplified pipeline for training and prediction (uses trained models by default)
- `annotation_type_pipeline.py` - Full pipeline with Specimin, augmentation, and CFG integration
- `model_based_predictor.py` - Loads trained models; feeds CFG graphs to graph models and embeddings to non-graph models

## Usage

### Training Annotation Type Models (Graph-Based)

#### Individual Model Training

```bash
# Train @Positive model (Enhanced Causal recommended), using real CFG data/embeddings
python annotation_type_rl_positive.py --episodes 50 --base_model enhanced_causal \
  --project_root /home/ubuntu/checker-framework/checker/tests/index --use_real_cfg_data

# Train @NonNegative
python annotation_type_rl_nonnegative.py --episodes 50 --base_model enhanced_causal \
  --project_root /home/ubuntu/checker-framework/checker/tests/index --use_real_cfg_data

# Train @GTENegativeOne
python annotation_type_rl_gtenegativeone.py --episodes 50 --base_model enhanced_causal \
  --project_root /home/ubuntu/checker-framework/checker/tests/index --use_real_cfg_data
```

#### Using the Pipeline

```bash
# Train all annotation type models (Enhanced Causal default)
python simple_annotation_type_pipeline.py --mode train --episodes 50 \
  --project_root /home/ubuntu/checker-framework/checker/tests/index

# Train and predict in one command
python simple_annotation_type_pipeline.py --mode both --episodes 50 \
  --project_root /home/ubuntu/checker-framework/checker/tests/index
```

### Prediction (Graph-Based)

#### Using the Pipeline

```bash
# Predict annotations on specific file (uses trained models; graph inputs under the hood)
python simple_annotation_type_pipeline.py --mode predict --target_file /path/to/MyClass.java
```

## Files

- `annotation_type_rl_positive.py`
- `annotation_type_rl_nonnegative.py`
- `annotation_type_rl_gtenegativeone.py`
- `cfg_graph.py` (CFG → PyG graph)
- `graph_encoder.py` (Graph Transformer encoder)
- `annotation_graph_input.py` (Embeddings for trainers)
