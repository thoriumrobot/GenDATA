# Localization Fix Implementation Summary

## Date: 2025-12-04

## Fixes Implemented

### 1. CFG Line Number Adjustment (`model_based_predictor.py`)

**Problem**: CFG nodes are typically 1-2 lines after GT annotation lines because CFG captures statement boundaries while annotations are on parameter declarations.

**Solution**: Added logic to adjust CFG node line numbers backwards when detecting parameter/variable declaration patterns:
- Checks for parameter/variable declaration nodes
- Adjusts line numbers backwards by 1 line for statement nodes that follow declarations
- Helps align predictions closer to GT annotation lines

**Code Location**: `model_based_predictor.py` lines 458-473

### 2. Evaluation Alignment Improvement (`studies/compute_case_study_metrics.py`)

**Problem**: Even when predictions are within ±1 line with correct labels, they weren't counted as matches.

**Solution**: Enhanced `align_labels()` function to check ±1 line for same-label matches before falling back to the ±3 window:
- First checks exact line match
- Then checks ±1 line for same label (accounts for CFG offset)
- Finally falls back to ±3 window search

**Code Location**: `studies/compute_case_study_metrics.py` lines 78-117

### 3. Model Naming Fix

**Problem**: Retrained models (`positive_real_balanced_model.pth`) weren't being loaded because predictor expected different names (`positive_enhanced_causal_model.pth`).

**Solution**: Created copies of retrained models with expected naming convention.

## Results

### Metrics After Fixes (plume-lib)

| Model | Exact Accuracy | Partial Accuracy | Coverage | Near Same Label | Pos vs NN |
|-------|---------------|-----------------|----------|-----------------|-----------|
| GCN   | 0.0000        | 0.1667          | 0.0000   | 0               | 5         |
| HGT   | 0.0000        | 0.1667          | 0.0000   | 0               | 5         |
| Causal| 0.0000        | 0.1667          | 0.0000   | 0               | 5         |
| GCSN  | 0.0000        | 0.1667          | 0.0000   | 0               | 5         |
| GBT   | 0.0000        | 0.0000          | 0.0000   | 0               | 0         |
| DG2N  | 0.0000        | 0.0000          | 0.0000   | 0               | 0         |
| DGCRF | 0.0000        | 0.0000          | 0.0000   | 0               | 0         |

### Key Observations

1. **Metrics unchanged**: The localization fixes didn't improve metrics, indicating the primary issue is still label confusion, not just line offsets.

2. **Label confusion persists**: All graph-based models show 5 cases of `@Positive` vs `@NonNegative` confusion, which accounts for the partial accuracy (0.1667 = 5/15 * 0.5 partial credit).

3. **Missing predictions**: 10 out of 15 GT points have no predictions within ±3 lines, indicating coverage issues.

4. **Graph vs feature models**: Graph-based models (GCN, HGT, Causal, GCSN) produce predictions but with wrong labels. Feature-based models (GBT, DG2N, DGCRF) produce very few predictions.

## Root Cause Analysis

The localization fixes address the symptom (line offsets) but not the root cause:

1. **Label semantics mismatch**: Models predict `@Positive` where GT is `@NonNegative` because:
   - Training data labeling rules may still not perfectly match Index Checker semantics
   - Graph-based models weren't retrained with improved datasets
   - Models learned patterns that don't align with actual annotation requirements

2. **CFG coverage gaps**: Many GT lines have no CFG nodes nearby because:
   - CFG generation may miss parameter declarations
   - Some annotations are on field initializations or method signatures not captured in CFG

3. **Model architecture limitations**: Feature-based models (GBT, DG2N, DGCRF) may not capture the semantic patterns needed for accurate predictions.

## Next Steps

1. **Retrain graph-based models**: The retrained models only cover feature-based architectures. Graph-based models (GCN, HGT, GCSN) need retraining with improved datasets.

2. **Improve CFG generation**: Ensure CFG captures parameter declarations, field initializations, and method signatures where annotations typically appear.

3. **Post-processing label correction**: Consider adding a post-processing step that adjusts `@Positive` predictions to `@NonNegative` when context suggests nonnegative semantics (e.g., array indices, lengths).

4. **Expand ground truth**: Current evaluation has only 15 GT points in plume-lib, making metrics brittle. More GT data would provide better signal.

## Files Modified

- `model_based_predictor.py`: Added CFG line number adjustment logic
- `studies/compute_case_study_metrics.py`: Enhanced evaluation alignment to check ±1 line for same-label matches
- `models_annotation_types/`: Renamed retrained models to match predictor expectations



