# Transformation Cost Ablation Study Guide

## Overview

The transformation cost ablation study measures the **accuracy cost** (performance drop) when each semantic transformation is disabled. This helps identify which transformations are most critical for model performance.

## Study Design

### Baseline
- Train all models with **all transformations enabled**
- Measure validation accuracy
- This serves as the reference point

### Ablation
For each of the 27 semantic transformations:
1. Train all models with that specific transformation **disabled**
2. Measure validation accuracy
3. Calculate cost: `baseline_accuracy - ablated_accuracy`
4. Rank transformations by cost

### Output
- **Cost**: Accuracy drop (positive value = performance loss)
- **Percent Cost**: Relative cost as percentage of baseline
- **Ranking**: Most critical to least critical transformations
- **Per-Model Breakdown**: Cost for each model configuration

## Running the Study

### Quick Test (Sample)
```bash
python run_transformation_cost_ablation.py \
  --baseline_file ablation_baseline_final/ablation_results.json \
  --episodes 5 \
  --sample 5 \
  --output_dir ablation_transformation_costs_sample
```

### Full Study (All Transformations)
```bash
python run_transformation_cost_ablation.py \
  --baseline_file ablation_baseline_final/ablation_results.json \
  --episodes 10 \
  --output_dir ablation_transformation_costs
```

### Analyze Results
```bash
python analyze_transformation_costs.py \
  --results_file ablation_transformation_costs/transformation_cost_results.json
```

## Results Interpretation

### Cost Values
- **Positive cost**: Dropping this transformation hurts performance
- **Higher cost**: More critical transformation
- **Zero/negative cost**: Transformation may not be essential (or dataset limitation)

### Ranking
Transformations are ranked by cost:
1. **Most Critical**: Highest cost (biggest performance drop when disabled)
2. **Least Critical**: Lowest cost (smallest impact when disabled)

### Example Output
```
Rank  Transformation                  Cost        % Cost    Ablated Avg
1     loop_conversion                0.0234      2.54%     0.9001
2     guard_reversal                 0.0198      2.15%     0.9037
3     mathematical_expression        0.0156      1.69%     0.9079
...
```

## Methodology Notes

### Current Implementation
The current study uses existing balanced datasets. For a true transformation ablation:
1. Generate slices from original code
2. Augment slices with specific transformation **disabled**
3. Generate CFGs from augmented slices
4. Generate balanced datasets from those CFGs
5. Train models on the new dataset
6. Compare against baseline

### Limitations
- Using existing datasets means transformations were already applied during augmentation
- True ablation requires regenerating the full pipeline for each transformation
- This is computationally expensive (27 transformations × full pipeline)

### Future Work
- Implement full pipeline regeneration for each transformation
- Generate separate datasets for each ablation case
- Run complete study for all 27 transformations

## Transformations Tested

### Enhanced Transformations (17)
- loop_conversion
- guard_reversal
- mathematical_expression
- logical_expression
- ternary_operator
- switch_statement
- variable_operation
- method_extraction
- conditional_expression
- array_access_pattern
- string_concatenation
- numeric_literal
- exception_handling
- lambda_expression
- stream_api
- builder_pattern
- functional_conversion

### Simple Transformations (10)
- simple_method_call
- simple_assignment
- simple_conditional
- simple_array_access
- simple_return_statement
- simple_variable_declaration
- simple_constructor_call
- simple_field_access
- simple_string_operation
- simple_numeric_operation

## Expected Findings

Based on the study design:
1. **Most transformations will have some cost**: Dropping any transformation reduces data diversity
2. **Some transformations more critical**: Certain transformations may be more important for specific annotation types
3. **Model-specific costs**: Different models may be affected differently by dropping transformations
4. **Annotation-type specific**: @Positive, @NonNegative, @GTENegativeOne may have different critical transformations

## Files and Scripts

### Study Scripts
- `run_transformation_cost_ablation.py`: Main study script
- `analyze_transformation_costs.py`: Results analysis and presentation

### Results Files
- `ablation_transformation_costs/transformation_cost_results.json`: Full results
- `ablation_transformation_costs_sample/transformation_cost_results.json`: Sample results

### Documentation
- This guide
- `ABLATION_STUDY_RESULTS.md`: Main results document
- `ABLATION_STUDY_AUGMENTATION.md`: Ablation study overview

