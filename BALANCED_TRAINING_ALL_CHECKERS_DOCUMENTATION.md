# Balanced Training for All Checkers - Complete Documentation

## Overview

This document describes how GenDATA trains models for **all checkers** (Lower Bound, SQL Quotes, and Signature String) using balanced datasets. Balanced training ensures that each annotation type model is trained on a dataset with approximately 50% positive examples (nodes that need the annotation) and 50% negative examples (nodes that don't need the annotation), improving model convergence and reducing prediction bias.

## Annotation Placement with Confidence-Based Selection

After training, models are used for prediction through the **MultiCheckerPredictor** system, which implements **confidence-based annotation selection**:

- **Single Annotation Per Location**: For each code location, only one annotation is placed (the highest confidence one)
- **Multi-Model Evaluation**: All annotation type models for a checker are evaluated at each location
- **Highest Confidence Selection**: If multiple models predict annotations, only the annotation with the highest confidence is placed
- **Unified Predictor**: `MultiCheckerPredictor` handles all checkers with consistent confidence-based selection
- **Automatic Checker Detection**: Checker is automatically detected from warnings file path or can be specified explicitly

See `place_annotations.py` and `multi_checker_predictor.py` for implementation details.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Balanced Dataset Generation](#balanced-dataset-generation)
4. [Training Process](#training-process)
5. [Checker-Specific Implementation](#checker-specific-implementation)
6. [Usage Guide](#usage-guide)
7. [Results and Metrics](#results-and-metrics)

## Introduction

### Problem Statement

Traditional annotation type model training suffers from data imbalance:
- **Imbalanced Data**: Most training examples are positive (need annotations)
- **Poor Convergence**: Models learn to always predict positive
- **Biased Predictions**: Low confidence in negative predictions
- **Poor Generalization**: Models don't learn to distinguish when annotations are NOT needed

### Solution: Balanced Training

Balanced training creates datasets where:
- **50% positive examples**: Nodes that need the specific annotation type
- **50% negative examples**: Nodes that don't need the specific annotation type
- **Per annotation type**: Separate balanced datasets for each annotation type
- **Real code examples**: Both positive and negative examples are real code patterns, not artificial modifications

## System Architecture

### Core Components

1. **`improved_balanced_dataset_generator.py`**: Generates balanced datasets from CFG files
2. **`improved_balanced_annotation_type_trainer.py`**: Trains models on balanced datasets
3. **`train_balanced_sql_quotes_models.py`**: Orchestrates balanced training for SQL Quotes Checker
4. **`train_balanced_signature_string_models.py`**: Orchestrates balanced training for Signature String Checker
5. **`generate_balanced_training_metrics_report.py`**: Generates comprehensive metrics reports

### Directory Structure

```
/home/ubuntu/GenDATA/
├── balanced_datasets/                          # Lower Bound Checker balanced datasets
│   ├── positive_real_balanced_dataset.json
│   ├── nonnegative_real_balanced_dataset.json
│   └── gtenegativeone_real_balanced_dataset.json
├── balanced_datasets_sql_quotes/               # SQL Quotes Checker balanced datasets
│   ├── sqlevenquotes_real_balanced_dataset.json
│   ├── sqloddquotes_real_balanced_dataset.json
│   └── real_generation_statistics.json
├── balanced_datasets_signature_string/         # Signature String Checker balanced datasets
│   ├── fullyqualifiedname_real_balanced_dataset.json
│   ├── binaryname_real_balanced_dataset.json
│   ├── fielddescriptor_real_balanced_dataset.json
│   └── real_generation_statistics.json
├── models_annotation_types/                    # Lower Bound Checker models
├── models_annotation_types_sql_quotes/         # SQL Quotes Checker balanced models
│   └── *_balanced_model.pth
└── models_annotation_types_signature_string/   # Signature String Checker balanced models
    └── *_balanced_model.pth
```

## Balanced Dataset Generation

### Process Overview

The balanced dataset generation process works as follows:

1. **Load CFG Files**: Load all CFG JSON files from the checker-specific CFG directory
2. **Classify Nodes**: For each annotation type, classify each CFG node as positive (needs annotation) or negative (doesn't need annotation)
3. **Balance Examples**: Select equal numbers of positive and negative examples
4. **Extract Features**: Extract checker-specific features from each node
5. **Save Datasets**: Save balanced datasets as JSON files

### Classification Rules

#### SQL Quotes Checker

**`@SqlEvenQuotes`**:
- **Positive**: String literals with even number of single quotes, SQL strings in prepared statements
- **Negative**: String literals with odd number of quotes, non-SQL strings

**`@SqlOddQuotes`**:
- **Positive**: String literals with odd number of single quotes, unsafe SQL concatenation patterns
- **Negative**: String literals with even number of quotes, safe SQL strings

**Implementation**: `classify_node_for_sql_quotes_annotation()` in `improved_balanced_dataset_generator.py`

#### Signature String Checker

**`@FullyQualifiedName`**:
- **Positive**: Dotted format (`java.lang.String`), class.getName() results
- **Negative**: Non-dotted formats, binary names, field descriptors

**`@BinaryName`**:
- **Positive**: Slashed format (`java/lang/String`), Class.forName() patterns
- **Negative**: Dotted formats, field descriptors

**`@FieldDescriptor`**:
- **Positive**: JVM format (`Ljava/lang/String;`), reflection API usage
- **Negative**: Dotted or slashed formats

**Implementation**: `classify_node_for_signature_string_annotation()` in `improved_balanced_dataset_generator.py`

### Feature Extraction

Each checker uses checker-specific feature extraction:

- **SQL Quotes**: Quote counts, SQL method patterns, string concatenation, prepared statement indicators
- **Signature String**: Format detection (dots, slashes, field descriptor patterns), structural features, pattern features, context features
- **Lower Bound**: Comparison patterns, array access patterns, loop variables, numeric operations

See `_extract_sql_quotes_features()` and `_extract_signature_string_features()` in `improved_balanced_dataset_generator.py`.

## Training Process

### Training Pipeline

For each checker, the training process follows these steps:

1. **Generate Balanced Datasets** (if not already generated):
   ```bash
   python3 train_balanced_sql_quotes_models.py
   # or
   python3 train_balanced_signature_string_models.py
   ```

2. **Train All Models**: For each annotation type and base model combination:
   - Load balanced dataset
   - Create model architecture
   - Train with 80/20 train/validation split
   - Save model with `_balanced` suffix

### Training Configuration

- **Epochs**: 100-200 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 20%
- **Early Stopping**: Patience of 20 epochs
- **Optimizer**: AdamW with multi-layer learning rates
- **Loss Function**: CrossEntropyLoss with class weights

### Model Architecture

All balanced models use the `ImprovedBalancedAnnotationTypeModel` architecture:
- **Input Layer**: Variable dimension (depends on checker features)
- **Hidden Layers**: [512, 256, 128, 64] with BatchNorm and Dropout
- **Output Layer**: 2 classes (positive/negative)
- **Regularization**: Dropout rate 0.4, BatchNorm, Gradient Clipping

## Checker-Specific Implementation

### SQL Quotes Checker

**Script**: `train_balanced_sql_quotes_models.py`

**Annotation Types**: `@SqlEvenQuotes`, `@SqlOddQuotes`

**Total Models**: 14 (7 base models × 2 annotation types)

**Base Models**: gcn, hgt, gbt, causal, enhanced_causal, gcsn, dg2n

**Usage**:
```bash
# Generate datasets and train all models
python3 train_balanced_sql_quotes_models.py

# Use existing datasets
python3 train_balanced_sql_quotes_models.py --skip_dataset_generation

# Custom configuration
python3 train_balanced_sql_quotes_models.py \
    --examples_per_annotation 1000 \
    --epochs 200 \
    --batch_size 32
```

**Dataset Location**: `cfg_output_adaptive_specimin_sql_quotes/`

**Output Models**: `models_annotation_types_sql_quotes/*_balanced_model.pth`

### Signature String Checker

**Script**: `train_balanced_signature_string_models.py`

**Annotation Types**: `@FullyQualifiedName`, `@BinaryName`, `@FieldDescriptor`

**Total Models**: 21 (7 base models × 3 annotation types)

**Base Models**: gcn, hgt, gbt, causal, enhanced_causal, gcsn, dg2n

**Usage**:
```bash
# Generate datasets and train all models
python3 train_balanced_signature_string_models.py

# Use existing datasets
python3 train_balanced_signature_string_models.py --skip_dataset_generation

# Custom configuration
python3 train_balanced_signature_string_models.py \
    --examples_per_annotation 1000 \
    --epochs 200 \
    --batch_size 32
```

**Dataset Location**: `cfg_output_adaptive_specimin_signature_string/`

**Output Models**: `models_annotation_types_signature_string/*_balanced_model.pth`

### Lower Bound Checker

**Status**: Already uses balanced training (reference implementation)

**Annotation Types**: `@Positive`, `@NonNegative`, `@GTENegativeOne`

**Total Models**: 21 (7 base models × 3 annotation types)

**Dataset Location**: `balanced_datasets/`

**Output Models**: `models_annotation_types/`

## Usage Guide

### Prerequisites

1. **CFG Files**: Ensure CFG files are generated for the target checker:
   ```bash
   python3 create_training_datasets.py --checker sql_quotes
   python3 create_training_datasets.py --checker signature_string
   ```

2. **Warning Files** (optional): Warning files can help identify positive examples, but balanced dataset generation works from CFG files alone.

### Step-by-Step Workflow

#### For SQL Quotes Checker

1. **Generate Training Datasets** (if not done):
   ```bash
   python3 create_training_datasets.py --checker sql_quotes
   ```

2. **Train Balanced Models**:
   ```bash
   python3 train_balanced_sql_quotes_models.py \
       --examples_per_annotation 1000 \
       --epochs 200 \
       --batch_size 32
   ```

3. **Verify Training**:
   ```bash
   ls -la models_annotation_types_sql_quotes/*_balanced_model.pth
   ```

#### For Signature String Checker

1. **Generate Training Datasets** (if not done):
   ```bash
   python3 create_training_datasets.py --checker signature_string
   ```

2. **Train Balanced Models**:
   ```bash
   python3 train_balanced_signature_string_models.py \
       --examples_per_annotation 1000 \
       --epochs 200 \
       --batch_size 32
   ```

3. **Verify Training**:
   ```bash
   ls -la models_annotation_types_signature_string/*_balanced_model.pth
   ```

### Command-Line Options

Both training scripts support the following options:

- `--cfg_dir`: Directory containing CFG files (default: checker-specific directory)
- `--balanced_dataset_dir`: Directory to save balanced datasets (default: `balanced_datasets_{checker_name}/`)
- `--models_dir`: Directory to save trained models (default: `models_annotation_types_{checker_name}/`)
- `--examples_per_annotation`: Number of examples per annotation type (default: 1000)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--device`: Device to use ('auto', 'cuda', or 'cpu', default: 'auto')
- `--skip_dataset_generation`: Skip dataset generation and use existing datasets

## Results and Metrics

### Dataset Statistics (Verified)

#### SQL Quotes Checker

**Location**: `balanced_datasets_sql_quotes/real_generation_statistics.json`

- **Total Examples**: 2,000
- **Positive Examples**: 1,000 (50.0%)
- **Negative Examples**: 1,000 (50.0%)
- **Balance Ratio**: 0.500 ✅ (Perfect balance)
- **Per-Annotation-Type**:
  - `@SqlEvenQuotes`: 500 positive, 500 negative (balance: 0.500) ✅
  - `@SqlOddQuotes`: 500 positive, 500 negative (balance: 0.500) ✅

#### Signature String Checker

**Location**: `balanced_datasets_signature_string/real_generation_statistics.json`

- **Total Examples**: 3,000
- **Positive Examples**: 1,500 (50.0%)
- **Negative Examples**: 1,500 (50.0%)
- **Balance Ratio**: 0.500 ✅ (Perfect balance)
- **Per-Annotation-Type**:
  - `@FullyQualifiedName`: 500 positive, 500 negative (balance: 0.500) ✅
  - `@BinaryName`: 500 positive, 500 negative (balance: 0.500) ✅
  - `@FieldDescriptor`: 500 positive, 500 negative (balance: 0.500) ✅

### Training Status (Verified as of 2025-12-08)

**Important**: Always verify actual model files and their training statistics. Script outputs may be optimistic.

#### SQL Quotes Checker

- **Expected Models**: 14 (7 base models × 2 annotation types)
- **Model Files Found**: 11
- **Models with Valid Training Stats**: 9
- **Completion**: 64.3% (9/14 with valid stats)

**Verified Performance** (9 models with valid stats):
- Average Best Accuracy: 100.00%
- All trained models achieved perfect accuracy

**Missing Models**: 3 models not yet trained
**Incomplete Models**: 2 model files exist but lack valid training statistics

#### Signature String Checker

- **Expected Models**: 21 (7 base models × 3 annotation types)
- **Model Files Found**: 14
- **Models with Valid Training Stats**: 11
- **Completion**: 52.4% (11/21 with valid stats)

**Verified Performance** (11 models with valid stats):
- Average Best Accuracy: 95.18%
- Range: 75.50% - 100.00%
- Note: `binaryname_gcsn_balanced` achieved only 75.50% (may need retraining)

**Missing Models**: 7 models not yet trained (including all FieldDescriptor models)
**Incomplete Models**: 3 model files exist but lack valid training statistics

### Generating Metrics Reports

Training metrics are saved with each model file. To generate a report:

```bash
python3 generate_balanced_training_metrics_report.py
```

**Note**: The metrics report only includes models with valid training statistics. Models without stats will show 0% accuracy (not actual accuracy). Always verify by checking model files directly.

### Model Files

Balanced models are saved with the following naming convention:
- **SQL Quotes**: `{annotation}_{model}_balanced_model.pth`
  - Example: `sqlevenquotes_gcn_balanced_model.pth`
- **Signature String**: `{annotation}_{model}_balanced_model.pth`
  - Example: `fullyqualifiedname_gcn_balanced_model.pth`

Each model file contains:
- Model state dictionary
- Model type
- Annotation type
- Input dimension
- Training statistics (accuracy, loss, etc.)

## Key Implementation Details

### Classification Logic

The balanced dataset generator uses checker-specific classification methods:

1. **SQL Quotes**: Analyzes quote parity in string literals, SQL method calls, and prepared statement patterns
2. **Signature String**: Detects format patterns (dotted, slashed, field descriptor) using `signature_string_feature_extractor.py`
3. **Lower Bound**: Uses comparison patterns, array access patterns, and numeric operations

### Feature Extraction

- **SQL Quotes**: 13 features (quote counts, SQL patterns, concatenation, sanitization)
- **Signature String**: 30 features (format detection, structural features, pattern features, context features)
- **Lower Bound**: 20+ features (comparison patterns, array access, loop variables, etc.)

### Training Improvements

Balanced training provides several benefits:

1. **Better Convergence**: Models learn proper decision boundaries
2. **Reduced Bias**: Balanced learning prevents always-positive predictions
3. **Improved Accuracy**: Better performance on both positive and negative examples
4. **Reliable Confidence**: More accurate confidence scores
5. **Better Generalization**: Models work better on new, unseen code

## Verification

### Check Dataset Balance

```bash
# SQL Quotes
cat balanced_datasets_sql_quotes/real_generation_statistics.json

# Signature String
cat balanced_datasets_signature_string/real_generation_statistics.json
```

### Check Trained Models

```bash
# SQL Quotes (should have 14 models)
ls -1 models_annotation_types_sql_quotes/*_balanced_model.pth | wc -l

# Signature String (should have 21 models)
ls -1 models_annotation_types_signature_string/*_balanced_model.pth | wc -l
```

### Generate Metrics Report

```bash
python3 generate_balanced_training_metrics_report.py
cat BALANCED_TRAINING_METRICS_REPORT.md
```

## Summary

All GenDATA checkers (Lower Bound, SQL Quotes, and Signature String) now use balanced training infrastructure:

- **Lower Bound Checker**: ✅ Reference implementation (21 models)
- **SQL Quotes Checker**: ⚠️ Training in progress (9/14 models with valid stats, 64.3% complete)
- **Signature String Checker**: ⚠️ Training in progress (11/21 models with valid stats, 52.4% complete)

**Verified Achievements**:
- ✅ Perfect 50/50 dataset balance for all checkers
- ✅ Real code examples (not artificial modifications)
- ✅ Training infrastructure fully functional
- ✅ Excellent performance for trained models (95-100% accuracy)

**Current Status**:
- ⚠️ Training incomplete - some models still need to be trained
- ⚠️ Some model files exist but lack complete training statistics
- ⚠️ One model (`binaryname_gcsn_balanced`) shows lower performance (75.50%)

**Important**: Always verify actual model files and training statistics rather than relying solely on script outputs. Use the verification commands in this document to check the real state of training.

All models are saved with `_balanced` suffix to distinguish them from non-balanced models.

