# Annotation Type Models Guide (Graph-Based Inputs)

This guide explains how to use the annotation-specific models that predict Checker Framework annotation types for multiple checkers (Lower Bound, SQL Quotes, Signature String) with CFG graph inputs. Graph models consume PyTorch Geometric graphs directly; non-graph models use a Graph Transformer encoder to obtain a fixed-length embedding.

## Overview

The annotation type models build upon the binary RL models to provide more precise annotation placement. Instead of just predicting whether an annotation should be placed, these models predict the specific type of annotation needed. The system uses **confidence-based selection** to place only the highest-confidence annotation at each location.

### Multi-Checker Support

GenDATA supports multiple Checker Framework checkers, each with its own set of annotation types:

#### Lower Bound Checker
1. **`@Positive`** - For values that must be greater than zero (e.g., count, size, length)
2. **`@NonNegative`** - For values that must be greater than or equal to zero (e.g., index, offset, position)
3. **`@GTENegativeOne`** - For values that must be greater than or equal to -1 (e.g., capacity, limit, bound)

#### SQL Quotes Checker
1. **`@SqlEvenQuotes`** - For SQL strings with even number of quotes (balanced quote pairs)
2. **`@SqlOddQuotes`** - For SQL strings with odd number of quotes (unbalanced quotes)

#### Signature String Checker
1. **`@FullyQualifiedName`** - For fully qualified class names (e.g., `java.lang.String`)
2. **`@BinaryName`** - For binary class names (e.g., `java/lang/String`)
3. **`@FieldDescriptor`** - For field descriptors (e.g., `Ljava/lang/String;`)

### Confidence-Based Selection

The system uses `MultiCheckerPredictor` for unified prediction across all checkers:

- **Single Annotation Per Location**: For each code location, only one annotation is placed (the highest confidence one)
- **Multi-Model Evaluation**: All annotation type models for a checker are evaluated at each location
- **Highest Confidence Selection**: If multiple models predict annotations, only the annotation with the highest confidence is placed
- **Automatic Checker Detection**: Checker is automatically detected from warnings file path or can be specified explicitly

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

The pipeline uses **MultiCheckerPredictor** for unified prediction across all checkers. The system includes:

- **MultiCheckerPredictor**: Unified predictor that handles all checkers with confidence-based selection
- **Checker-Specific Model Loading**: Models are loaded from checker-specific directories
- **Confidence-Based Selection**: For each location, selects the annotation with highest confidence
- **Enhanced Causal Model (Default)**
  - Graph-augmented features via embeddings
  - Dynamic confidence scores based on model certainty
- **Model-Based Predictions**
  - Model attribution in outputs
  - Confidence scores from model inference

## Scripts

### Individual Model Training Scripts

#### Lower Bound Checker
- `annotation_type_rl_positive.py` - Trains model for `@Positive` annotations
- `annotation_type_rl_nonnegative.py` - Trains model for `@NonNegative` annotations
- `annotation_type_rl_gtenegativeone.py` - Trains model for `@GTENegativeOne` annotations

#### SQL Quotes Checker
- `train_balanced_sql_quotes_models.py` - Trains models for `@SqlEvenQuotes` and `@SqlOddQuotes` annotations

#### Signature String Checker
- `annotation_type_rl_signature_string_fullyqualified.py` - Trains model for `@FullyQualifiedName` annotations
- `annotation_type_rl_signature_string_binary.py` - Trains model for `@BinaryName` annotations
- `annotation_type_rl_signature_string_fielddescriptor.py` - Trains model for `@FieldDescriptor` annotations

### Pipeline Scripts

- `simple_annotation_type_pipeline.py` - Simplified pipeline for training and prediction (uses MultiCheckerPredictor by default, supports all checkers)
- `annotation_type_pipeline.py` - Full pipeline with Specimin, augmentation, and CFG integration
- `multi_checker_predictor.py` - Unified predictor for all checkers with confidence-based selection
- `model_based_predictor.py` - Legacy predictor (superseded by MultiCheckerPredictor)

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
