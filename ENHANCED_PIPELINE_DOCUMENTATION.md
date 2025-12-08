# Enhanced Pipeline Documentation

## Overview

The GenDATA pipeline has been enhanced with comprehensive "could be zero" detection features to improve the distinction between `@Positive` (must be > 0) and `@NonNegative` (can be 0) annotation types.

## Pipeline Architecture

### 1. Feature Extraction

#### Graph-Based Models (GCN, HGT, GCSN)
- **File**: `cfg_graph.py`
- **Features**: 22-dimensional node features including:
  - Node type one-hot encoding
  - Degree features (1)
  - Normalized line numbers (1)
  - Laplacian positional encodings (8, k=8)
  - Random-walk structural encodings (4, steps=4)
  - **Semantic "could be zero" features (7, NEW)**:
    - Array index usage detection
    - Loop variable detection
    - Subtraction result detection
    - Offset/position variable detection
    - Nonnegative check detection
    - Length comparison detection
    - Aggregated "could be zero" score (2.0x scaling)

#### Feature-Based Models (GBT, Causal, Enhanced Causal)
- **File**: `improved_balanced_dataset_generator.py`
- **Features**: Enhanced feature set with "could be zero" detection patterns
- **Training**: Uses balanced datasets with 50/50 positive/negative examples

#### Prediction-Time Extractors
- **Files**: `annotation_type_rl_positive.py`, `annotation_type_rl_nonnegative.py`, `annotation_type_rl_gtenegativeone.py`
- **Features**: Match training-time features for consistency

### 2. Model Training

#### Graph-Based Models
- **Training Script**: `retrain_graph_models.py`
- **Models**: GCN, HGT, GCSN for all 3 annotation types (9 models total)
- **Location**: `models_annotation_types/`
- **Naming**: `{annotation_type}_{base_model}_model.pth`

#### Feature-Based Models
- **Training Script**: `improved_balanced_annotation_type_trainer.py`
- **Models**: GBT, Causal, Enhanced Causal, DG2N, DGCRF
- **Features**: Cost-sensitive loss to penalize `@Positive` ↔ `@NonNegative` confusion

### 3. "Could Be Zero" Detection Patterns

The enhanced features detect 8 patterns that indicate a value could be zero:

1. **Array index usage**: Variables used as array indices (can start at 0)
2. **Loop iteration variables**: Loop counters (often start at 0)
3. **Subtraction results**: Expressions like `length - 1` (could be 0)
4. **Parameter in array context**: Parameters used near array access
5. **Comparison with length/size**: Comparisons suggesting range [0, length)
6. **Initialization to 0**: Variables explicitly initialized to 0
7. **Nonnegative checks**: Explicit `>= 0` or `>= -1` checks
8. **Offset/position variables**: Variables named offset, position, etc.

These patterns are aggregated into a `could_be_zero_score` feature (scaled 3.0x for emphasis).

## "Could Be Zero" Features

For detailed documentation on the "could be zero" features, see **`COULD_BE_ZERO_FEATURES_DOCUMENTATION.md`**.

### Quick Summary

The "could be zero" features detect 8 semantic patterns that indicate a value might be zero:
1. **Array index usage**: Variables used as array indices (can start at 0)
2. **Loop iteration variables**: Loop counters (often start at 0)
3. **Subtraction results**: Expressions like `length - 1` (could be 0)
4. **Parameter in array context**: Parameters used near array access
5. **Comparison with length/size**: Comparisons suggesting range [0, length)
6. **Initialization to 0**: Variables explicitly initialized to 0
7. **Nonnegative checks**: Explicit `>= 0` or `>= -1` checks
8. **Offset/position variables**: Variables named offset, position, etc.

These patterns are aggregated into a `could_be_zero_score` feature (scaled 3.0x for emphasis).

**Impact**: These features help models distinguish between `@Positive` (must be > 0) and `@NonNegative` (can be >= 0), reducing label confusion.

## Usage

### Training Graph Models

```bash
# Retrain all graph-based models
python retrain_graph_models.py --episodes 100 --device cpu

# Retrain specific model
python retrain_graph_models.py --model gcn --annotation_type @Positive --episodes 100
```

### Training Feature-Based Models

```bash
# First, generate balanced datasets
python improved_balanced_dataset_generator.py \
  --cfg_dir cfg_output_specimin \
  --output_dir real_balanced_datasets \
  --examples_per_annotation 2000 \
  --target_balance 0.5

# Then train models
python improved_balanced_annotation_type_trainer.py \
  --balanced_dataset_dir real_balanced_datasets \
  --output_dir models_annotation_types \
  --epochs 200 \
  --batch_size 32
```

### Prediction

Models are automatically loaded by `ModelBasedPredictor`:

```python
from model_based_predictor import ModelBasedPredictor

predictor = ModelBasedPredictor(models_dir='models_annotation_types')
predictions = predictor.predict_annotations_for_file_with_cfg(
    java_file='path/to/file.java',
    cfg_dir='path/to/cfg/dir',
    threshold=0.3
)
```

## Ablation Study Framework

### Augmentation vs. No Augmentation

To run an ablation study comparing augmentation vs. no augmentation:

1. **With Augmentation** (Current Pipeline):
   - Uses enhanced semantic augmentation
   - Generates multiple variants per slice
   - Trains on augmented CFG data

2. **Without Augmentation** (Requires Pipeline Modification):
   - Skip augmentation step
   - Use only original (non-augmented) slices
   - Train on original CFG data only

### Running Ablation Study

```bash
# Track training/validation accuracy for all models
python run_training_accuracy_ablation.py \
  --output_dir ablation_training_accuracy \
  --balanced_dataset_dir real_balanced_datasets \
  --episodes 50 \
  --device cpu
```

**Note**: For a proper no-augmentation comparison, you need to:
1. Generate a separate dataset without augmentation
2. Train models on that dataset
3. Compare results

## Expected Improvements

1. **Reduced Label Confusion**: Better distinction between `@Positive` and `@NonNegative`
2. **Better @NonNegative Detection**: Explicit signals for nonnegative semantics
3. **Improved Accuracy**: Should reduce confusion observed in case studies

## Model Files

All trained models are saved in `models_annotation_types/`:

```
models_annotation_types/
├── positive_gcn_model.pth
├── positive_hgt_model.pth
├── positive_gcsn_model.pth
├── nonnegative_gcn_model.pth
├── nonnegative_hgt_model.pth
├── nonnegative_gcsn_model.pth
├── gtenegativeone_gcn_model.pth
├── gtenegativeone_hgt_model.pth
├── gtenegativeone_gcsn_model.pth
└── ... (feature-based models)
```

## Validation

After training, evaluate on case studies:

```bash
# Run predictions
python studies/run_annotation_type_predictions.py

# Compute metrics
python studies/compute_case_study_metrics.py

# Collect results
python studies/case_study_metrics_collector.py
```

Check for improvements in:
- `near_match_pos_vs_nn` count (should decrease)
- `accuracy_exact` and `accuracy_partial` (should increase)
- `near_match_same_label` count (should increase)

