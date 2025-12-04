# Model Retraining with Improvements

## Date: 2025-12-04

## Summary

Retrained annotation type models with improved label semantics, enhanced features, and cost-sensitive loss to address the accuracy issues identified in the case study evaluation.

## Improvements Implemented

### 1. Label Semantics Fixes (`improved_balanced_dataset_generator.py`)

- **Fixed Rule 3**: Changed length/size parameters from `@Positive` to `@NonNegative` to better match Index Checker semantics
- **Enhanced pattern detection**: Added explicit checks for comparison patterns (`> 0`, `>= 0`, `>= -1`) in node labels
- **Better default**: Parameters now default to `@NonNegative` (can be 0) rather than `@Positive` (must be > 0)

### 2. Feature Enhancements (`improved_balanced_dataset_generator.py`)

Added 6 new semantic pattern features:
- `has_strict_positive_comparison`: Detects `> 0` patterns
- `has_nonnegative_comparison`: Detects `>= 0` patterns  
- `has_gtenegativeone_comparison`: Detects `>= -1` patterns
- `has_array_length_minus_one`: Detects `length - 1` patterns
- `has_strict_positive_context`: Checks surrounding nodes for `> 0`
- `has_nonnegative_context`: Checks surrounding nodes for `>= 0`

These features help models distinguish between `@Positive` (> 0) and `@NonNegative` (>= 0) semantics.

### 3. Cost-Sensitive Loss (`improved_balanced_annotation_type_trainer.py`)

- Added class weights to penalize misclassification more heavily
- For `@Positive` and `@NonNegative` models, increased weight for positive class (1.5x) to reduce confusion
- Helps models learn the distinction between strictly positive and nonnegative values

## Retraining Process

### Datasets Regenerated

- **Total examples**: 6000 (2000 per annotation type)
- **Balance**: 50% positive, 50% negative for each annotation type
- **Features**: 27 features per example (21 original + 6 new semantic features)

### Models Being Retrained

The balanced training framework (`improved_balanced_annotation_type_trainer.py`) trains:
- **Feature-based models**: GBT, Causal (these use the balanced datasets)

**Note**: Graph-based models (GCN, HGT, GCSN, DG2N, DGCRF) use different training pipelines and may need separate retraining. They currently use their own training scripts that may not directly benefit from the balanced dataset improvements.

### Training Configuration

- **Epochs**: 200
- **Batch size**: 32
- **Device**: Auto (CUDA if available, else CPU)
- **Early stopping**: Patience of 20 epochs
- **Learning rate**: Adaptive with ReduceLROnPlateau scheduler

## Expected Impact

Based on the analysis in `case_studies/evaluation_results/label_semantics_analysis.json`:

1. **Reduced @Positive ↔ @NonNegative confusion**: The improved labeling rules and cost-sensitive loss should reduce the 8.1% confusion rate observed in case studies
2. **Better @GTENegativeOne detection**: Enhanced features for `>= -1` patterns should improve recall (currently 0%)
3. **Improved localization**: While not directly addressed in this retraining, the enhanced features may help models better understand semantic patterns

## Next Steps

1. **Wait for training completion**: Monitor `retraining.log` for progress
2. **Evaluate on case studies**: Re-run predictions and metrics to measure improvement
3. **Consider graph-based model retraining**: If needed, modify graph-based training pipelines to use improved datasets or apply similar fixes
4. **Compare metrics**: Compare new metrics with baseline from `case_studies/evaluation_results/baseline_metrics_summary.json`

## Files Modified

- `improved_balanced_dataset_generator.py`: Label semantics fixes and feature enhancements
- `improved_balanced_annotation_type_trainer.py`: Cost-sensitive loss
- `retrain_with_improvements.py`: New retraining script

## Backup

Original models backed up to: `models_annotation_types_backup_YYYYMMDD_HHMMSS/`

