# Training and Prediction Status

## Current Situation

### Models Available
- **Lower Bound Checker**: 15 models available (GCN, HGT, GCSN, Enhanced Causal for @Positive, @NonNegative, @GTENegativeOne)
- **SQL Quotes Checker**: 0 models (test suite not found)
- **Signature String Checker**: 0 models (training scripts are placeholders)

### Project Warnings Status
All evaluation projects currently have **0 warnings** for all checkers:
- guava: 0 warnings
- jfreechart: 0 warnings  
- plume-lib: 0 warnings
- agrona: 0 warnings
- hipparchus: 0 warnings
- eclipse-collections: 0 warnings

### Test Suite Warnings Status
- **Lower Bound test suite**: 0 warnings (files are already well-annotated)
- **Signature String test suite**: 0 warnings (files are already well-annotated)
- **SQL Quotes test suite**: Not found in current Checker Framework installation

## Issue

**Cannot generate predictions without warnings**: The prediction pipeline requires:
1. Checker warnings to identify code locations
2. Slices generated from warnings
3. CFGs generated from slices
4. Models to make predictions on CFGs

Since all projects have 0 warnings, the pipeline cannot proceed past step 1.

## Training Requirements

Training models requires:
1. **Warnings file** with actual checker warnings
2. **Slices** generated from warnings using Soot slicer
3. **CFGs** generated from slices using Checker Framework CFG Builder
4. **Training data** (CFGs with ground truth annotations)

## Solutions

### Option 1: Use Projects with Warnings
Find or create projects that actually trigger checker warnings. This is the normal workflow.

### Option 2: Use Pre-existing Training Data
If training data (slices, CFGs) exists from previous runs, models can be trained on that data.

### Option 3: Create Synthetic Warnings
Manually create warnings files for testing purposes, though this won't reflect real-world performance.

## Next Steps

1. **For Lower Bound Checker**: Models exist but need projects with warnings to generate predictions
2. **For Signature String Checker**: 
   - Implement training scripts (currently placeholders)
   - Find projects with Signature String warnings OR use test suite with warnings
   - Train models
3. **For SQL Quotes Checker**:
   - Obtain test suite or update Checker Framework installation
   - Implement training scripts
   - Train models

## Recommendation

To proceed with training and predictions:
1. Identify projects that trigger warnings for each checker
2. Run checker on those projects to generate warnings
3. Use warnings to generate slices and CFGs
4. Train models on CFGs
5. Generate predictions on evaluation projects

The current "0 warnings" situation suggests the evaluation projects are well-annotated, which is actually a positive outcome but prevents demonstration of the prediction pipeline.

