# Graph Models Retraining Summary

## Overview
All graph-based models (GCN, HGT, GCSN) have been successfully retrained with enhanced "could be zero" features to improve the distinction between `@Positive` (must be > 0) and `@NonNegative` (can be 0).

## Changes Implemented

### 1. Enhanced Feature Representation
- **Graph-based models** (`cfg_graph.py`): Added 7 semantic "could be zero" features to node representations
  - Array index usage detection
  - Loop variable detection
  - Subtraction result detection
  - Offset/position variable detection
  - Nonnegative check detection
  - Length comparison detection
  - Aggregated "could be zero" score (emphasized with 2.0x scaling)
- **Feature dimension**: Increased from ~15 to 22 features per node

### 2. Models Retrained
Successfully retrained **9 graph-based models**:
- **GCN models**: `positive_gcn_model.pth`, `nonnegative_gcn_model.pth`, `gtenegativeone_gcn_model.pth`
- **HGT models**: `positive_hgt_model.pth`, `nonnegative_hgt_model.pth`, `gtenegativeone_hgt_model.pth`
- **GCSN models**: `positive_gcsn_model.pth`, `nonnegative_gcsn_model.pth`, `gtenegativeone_gcsn_model.pth`

### 3. Training Configuration
- **Episodes**: 50 per model
- **Device**: CPU
- **CFG Directory**: `cfg_output_specimin`
- **Training Script**: `retrain_graph_models.py`

## Feature Breakdown

The graph node features now include:
1. **Node type one-hot encoding** (variable size based on node types)
2. **Degree features** (1 feature)
3. **Normalized line numbers** (1 feature)
4. **Laplacian positional encodings** (8 features, k=8)
5. **Random-walk structural encodings** (4 features, steps=4)
6. **Semantic "could be zero" features** (7 features, NEW):
   - `is_array_index` (scaled 1.0x)
   - `is_loop_var` (scaled 1.0x)
   - `is_subtraction` (scaled 1.0x)
   - `is_offset` (scaled 1.0x)
   - `has_nonneg_check` (scaled 1.0x)
   - `compared_with_len` (scaled 1.0x)
   - `could_be_zero_score` (scaled 2.0x, emphasized)

**Total**: 22 features per node

## Expected Impact

1. **Reduced Label Confusion**: Models should better distinguish `@Positive` from `@NonNegative` by explicitly detecting when a value could be 0
2. **Better @NonNegative Detection**: The "could be zero" features provide explicit signals for nonnegative semantics
3. **Improved Accuracy**: Should reduce the confusion between `@Positive` and `@NonNegative` observed in case studies

## Verification

- ✅ All 9 models retrained successfully
- ✅ Feature dimension confirmed: 22 features per node
- ✅ Models saved to `models_annotation_types/`
- ✅ Training completed without errors

## Next Steps

1. **Evaluate on case studies**: Run case study evaluation to measure improvement
2. **Compare metrics**: Check if `near_match_pos_vs_nn` count decreased
3. **Monitor accuracy**: Verify improvements in `accuracy_exact` and `accuracy_partial`

## Usage

The retrained models are automatically used by `ModelBasedPredictor` when loading models. No changes needed to prediction code - the enhanced features are automatically included when loading CFG graphs via `cfg_graph.py`.

To retrain again:
```bash
python retrain_graph_models.py --episodes 100 --device cpu
```

To retrain a specific model:
```bash
python retrain_graph_models.py --model gcn --annotation_type @Positive --episodes 100
```

