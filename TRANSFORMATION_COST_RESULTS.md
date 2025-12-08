# Transformation Cost Ablation Results

## Executive Summary

This document presents the results of the second ablation study measuring the **accuracy cost** of dropping each semantic transformation. The study shows that all tested transformations have a significant impact on model performance, with costs ranging from 4.58% to 6.34%.

## Study Configuration

- **Baseline**: All transformations enabled (92.35% average validation accuracy)
- **Transformations Tested**: 5 sample transformations
- **Models Tested**: 12 models with validation accuracy
- **Training Episodes**: 5 per model

## Results

### Overall Cost Summary

| Rank | Transformation | Accuracy Cost | % Cost | Ablated Accuracy |
|------|---------------|---------------|--------|------------------|
| 1 | **simple_array_access** | **0.0585** | **6.34%** | 86.50% |
| 2 | **simple_constructor_call** | **0.0519** | **5.62%** | 87.17% |
| 3 | **loop_conversion** | **0.0490** | **5.30%** | 87.46% |
| 4 | **logical_expression** | **0.0458** | **4.96%** | 87.77% |
| 5 | **conditional_expression** | **0.0423** | **4.58%** | 88.12% |

**Baseline Average**: 92.35%
**Average Cost**: 0.0495 (4.95%)
**Range**: 0.0423 - 0.0585

### Key Findings

1. **All Transformations Are Critical**: Every tested transformation shows a meaningful cost (4-6% drop)
2. **Simple Transformations Matter Most**: `simple_array_access` and `simple_constructor_call` have the highest costs
3. **Loop Conversion Is Important**: 5.30% cost demonstrates the value of loop transformations
4. **Consistent Impact**: All transformations affect model performance significantly

### Detailed Breakdown: Top 3 Most Critical

#### 1. simple_array_access (Cost: 6.34%)

**Most Affected Models**:
- @GTENegativeOne_causal: **11.67% cost** (94.25% → 83.25%)
- @Positive_gbt: **8.88% cost** (98.50% → 89.75%)
- @Positive_dgcrf: **8.38% cost** (98.50% → 90.25%)
- @Positive_causal: **7.87% cost** (98.50% → 90.75%)

**Insight**: Array access patterns are critical for annotation type prediction, especially for @Positive and @GTENegativeOne models.

#### 2. simple_constructor_call (Cost: 5.62%)

**Most Affected Models**:
- @GTENegativeOne_causal: **10.88% cost** (94.25% → 84.00%)
- @GTENegativeOne_dgcrf: **9.62% cost** (91.00% → 82.25%)
- @Positive_dg2n: **7.29% cost** (99.50% → 92.25%)

**Insight**: Constructor call patterns are important, especially for @GTENegativeOne models.

#### 3. loop_conversion (Cost: 5.30%)

**Most Affected Models**:
- @GTENegativeOne_causal: **10.34% cost** (94.25% → 84.50%)
- @GTENegativeOne_dgcrf: **9.34% cost** (91.00% → 82.50%)
- @Positive_dgcrf: **6.60% cost** (98.50% → 92.00%)

**Insight**: Loop conversion transformations provide important diversity for training, especially for @GTENegativeOne models.

### Per-Model Cost Analysis

**Models Most Affected by Transformation Drops**:
1. @GTENegativeOne_causal: Highest cost across all transformations (10-12%)
2. @GTENegativeOne_dgcrf: High cost (8-10%)
3. @Positive models: Moderate cost (6-9%)

**Models Least Affected**:
1. @NonNegative_dg2n: Lowest cost (2-3%)
2. @NonNegative_dgcrf: Low cost (2-4%)

### Per-Annotation Type Impact

**@GTENegativeOne**: Most sensitive to transformation drops
- Average cost: ~10% across tested transformations
- Most affected by: simple_array_access, simple_constructor_call, loop_conversion

**@Positive**: Moderate sensitivity
- Average cost: ~7% across tested transformations
- Most affected by: simple_array_access, simple_constructor_call

**@NonNegative**: Least sensitive
- Average cost: ~3% across tested transformations
- More robust to transformation drops

## Interpretation

### Why These Costs Matter

1. **Data Diversity**: Each transformation adds unique code patterns to the training data
2. **Model Robustness**: Dropping transformations reduces the variety of patterns models see
3. **Annotation-Specific Impact**: Different annotation types rely on different transformation patterns

### Recommendations

1. **Keep All Transformations**: All tested transformations show significant value
2. **Prioritize Simple Transformations**: `simple_array_access` and `simple_constructor_call` are most critical
3. **Focus on Loop Patterns**: `loop_conversion` is important for @GTENegativeOne models
4. **Consider Annotation-Specific Needs**: @GTENegativeOne models benefit most from transformations

## Methodology Notes

### Current Study Limitations

1. **Sample Size**: Only 5 transformations tested (out of 27)
2. **Dataset Reuse**: Uses existing datasets rather than regenerating with disabled transformations
3. **Training Episodes**: 5 episodes (shorter than full training)

### True Ablation Requirements

For a complete transformation cost study:
1. Generate slices from original code
2. Augment with each transformation disabled (one at a time)
3. Generate CFGs from augmented slices
4. Generate balanced datasets from those CFGs
5. Train all models on each dataset
6. Compare against baseline

This would require 27 × full pipeline runs, which is computationally expensive.

## Next Steps

1. ✅ **Sample Study**: Completed (5 transformations)
2. ⏳ **Full Study**: Test all 27 transformations
3. ⏳ **True Ablation**: Regenerate datasets for each transformation
4. ⏳ **Extended Training**: Use full training episodes (50-100)

## Files

- **Results**: `ablation_transformation_costs_sample/transformation_cost_results.json`
- **Analysis Script**: `analyze_transformation_costs.py`
- **Study Script**: `run_transformation_cost_ablation.py`

## Usage

```bash
# Analyze results
python analyze_transformation_costs.py \
  --results_file ablation_transformation_costs_sample/transformation_cost_results.json

# Run full study (all transformations)
python run_transformation_cost_ablation.py \
  --baseline_file ablation_baseline_final/ablation_results.json \
  --episodes 10 \
  --output_dir ablation_transformation_costs_full
```

