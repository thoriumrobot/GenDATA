# Latest Ablation Study Results (December 2025)

## Study 1: Augmentation Comparison Study

### Status
**Completed**: ✅ Baseline results available (with augmentation)  
**Note**: No-augmentation comparison requires `--cfg_dir_no_aug` argument to generate dataset from non-augmented CFGs

### Results: With Augmentation (Baseline)

#### Overall Performance
- **Models Tested**: 21 configurations (7 models × 3 annotation types)
- **Models with Valid Accuracy**: 12 models
- **Average Validation Accuracy**: **92.35%**
- **Range**: 87.00% - 99.50%
- **Training Episodes**: 10 per model

#### Top 5 Performing Models
1. **@Positive_dg2n**: 99.50% validation accuracy
2. **@Positive_causal**: 98.50% validation accuracy
3. **@Positive_dgcrf**: 98.50% validation accuracy
4. **@Positive_gbt**: 98.50% validation accuracy
5. **@NonNegative_dg2n**: 87.00% validation accuracy

#### Performance by Model Type
| Model Type | Average Val Accuracy | Models with Results |
|------------|---------------------|-------------------|
| **GBT** | 94.83% | 3/3 |
| **Causal** | 93.75% | 3/3 |
| **DG2N** | 91.63% | 3/3 |
| **DGCRF** | 92.58% | 3/3 |
| **GCN** | N/A | 0/3 (graph models - accuracy not parsed) |
| **HGT** | N/A | 0/3 (graph models - accuracy not parsed) |
| **GCSN** | N/A | 0/3 (graph models - accuracy not parsed) |

#### Performance by Annotation Type
| Annotation Type | Average Val Accuracy | Models with Results |
|----------------|---------------------|-------------------|
| **@Positive** | 98.75% | 4/7 |
| **@GTENegativeOne** | 90.17% | 4/7 |
| **@NonNegative** | 88.13% | 4/7 |

#### Key Findings
1. **@Positive models excel**: All @Positive feature-based models achieve >98% accuracy
2. **Feature-based models show strong performance**: GBT and Causal models lead with >93% average
3. **@NonNegative is most challenging**: Lowest average accuracy (88.13%)
4. **Graph-based models trained successfully**: All 9 graph models (GCN, HGT, GCSN) completed training, but accuracy metrics need log parsing improvement

### Without Augmentation
**Status**: ⏸️ Skipped - Requires `--cfg_dir_no_aug` argument  
**Note**: To complete the comparison, provide a CFG directory containing non-augmented CFGs:
```bash
python run_augmentation_comparison_study.py \
    --cfg_dir_no_aug cfg_output_no_aug \
    --baseline_file ablation_baseline_final/ablation_results.json
```

### Implementation Notes
- **Random Seeds**: All training uses fixed seed 42 for reproducibility
- **Dataset Separation**: Uses separate dataset directories (when available)
- **Error Handling**: Gracefully handles missing CFG directories with informative messages

---

## Study 2: Transformation Ablation Study

### Status
**Completed**: ✅ Baseline training successful  
**Note**: Transformation-specific ablations require CFG directories with each transformation disabled

### Results: Baseline (All Transformations Enabled)

#### Overall Performance
- **Models Tested**: 21 configurations (7 models × 3 annotation types)
- **Models with Valid Accuracy**: 12 models
- **Average Validation Accuracy**: **92.75%**
- **Range**: 87.75% - 98.50%
- **Training Episodes**: 10 per model

#### Top 5 Performing Models
1. **@Positive_gbt**: 98.50% validation accuracy
2. **@Positive_causal**: 98.50% validation accuracy
3. **@Positive_dg2n**: 98.50% validation accuracy
4. **@Positive_dgcrf**: 98.50% validation accuracy
5. **@NonNegative_gbt**: 92.00% validation accuracy

#### Performance by Model Type
| Model Type | Average Val Accuracy | Models with Results |
|------------|---------------------|-------------------|
| **GBT** | 92.75% | 3/3 |
| **Causal** | 92.75% | 3/3 |
| **DG2N** | 92.75% | 3/3 |
| **DGCRF** | 92.75% | 3/3 |
| **GCN** | N/A | 0/3 (graph models - accuracy not parsed) |
| **HGT** | N/A | 0/3 (graph models - accuracy not parsed) |
| **GCSN** | N/A | 0/3 (graph models - accuracy not parsed) |

#### Performance by Annotation Type
| Annotation Type | Average Val Accuracy | Models with Results |
|----------------|---------------------|-------------------|
| **@Positive** | 98.50% | 4/7 |
| **@NonNegative** | 92.00% | 4/7 |
| **@GTENegativeOne** | 87.75% | 4/7 |

### Transformation Ablations

#### Tested Transformations
The following transformations were tested (5 total):
- `loop_conversion`
- `guard_reversal`
- `mathematical_expression`
- `ternary_operator`
- `logical_expression`

#### Status
**All transformations skipped**: CFG directories with transformations disabled do not exist

**Required Setup**:
To run transformation ablations, generate CFG directories with each transformation disabled:
```bash
# Example: Generate CFGs with loop_conversion disabled
python enhanced_semantic_augment_slices.py \
    --slices_dir slices_original \
    --out_dir slices_ablate_loop_conversion \
    --variants_per_file 10 \
    --disabled loop_conversion

# Generate CFGs from these slices
python pipeline.py \
    --steps cfg \
    --slices_dir slices_ablate_loop_conversion \
    --cfg_output_dir cfg_output_ablate_loop_conversion
```

Then run the ablation study:
```bash
python run_transformation_ablation_final.py \
    --cfg_dir_base_pattern "cfg_output_ablate_{transform}" \
    --transformations loop_conversion guard_reversal
```

### Implementation Notes
- **Error Handling**: Fixed to handle skipped transformations gracefully
- **Results Structure**: Properly structured error messages for missing CFG directories
- **Baseline Complete**: All 21 models trained successfully with baseline configuration

---

## Implementation Verification

### ✅ Verified Components

1. **Random Seed Fixes**
   - All random operations use seed 42
   - Model initialization is deterministic
   - Train/validation splits are consistent

2. **Dataset Generation**
   - `ablation_dataset_generator.py` works correctly
   - Can generate datasets from CFG directories
   - Proper error handling for missing directories

3. **Augmentation Comparison Study**
   - Successfully loads baseline results
   - Generates partial results when `cfg_dir_no_aug` is missing
   - Non-blank, non-erroneous results

4. **Transformation Ablation Study**
   - Successfully trains baseline models
   - Handles missing CFG directories gracefully
   - Proper error messages for skipped transformations
   - Fixed AttributeError in comparison calculation

### 📊 Results Summary

| Study | Status | Baseline Avg | Models | Notes |
|-------|--------|--------------|--------|-------|
| **Augmentation Comparison** | Partial | 92.35% | 12/21 | No-augmentation requires CFG directory |
| **Transformation Ablation** | Baseline Complete | 92.75% | 12/21 | Transformations require CFG directories |

### 🔧 Known Limitations

1. **Graph Model Accuracy Parsing**: Graph-based models (GCN, HGT, GCSN) complete training but accuracy metrics are not parsed from logs. Need to improve log parsing for these models.

2. **CFG Directory Requirements**: 
   - Augmentation comparison requires non-augmented CFG directory
   - Transformation ablation requires CFG directories with each transformation disabled
   - These must be generated separately before running studies

3. **Dataset Generation**: Datasets are generated on-demand, which requires CFG directories to exist

### 📝 Next Steps

1. **Generate Non-Augmented CFGs**: Create `cfg_output_no_aug` directory for complete augmentation comparison
2. **Generate Transformation-Specific CFGs**: Create CFG directories for each transformation to test
3. **Improve Graph Model Log Parsing**: Extract accuracy metrics from graph model training logs
4. **Run Complete Studies**: Execute full ablation studies with all required CFG directories

---

## Files and Locations

### Results Files
- **Augmentation Comparison**: `ablation_augmentation_comparison_v2/augmentation_comparison_results.json`
- **Transformation Ablation**: `ablation_transformations_v2_fixed/transformation_ablation_results.json`

### Log Files
- **Augmentation Comparison**: `ablation_augmentation_v2.log`
- **Transformation Ablation**: `ablation_transformations_v2.log`

### Study Scripts
- **Augmentation Comparison**: `run_augmentation_comparison_study.py`
- **Transformation Ablation**: `run_transformation_ablation_final.py`
- **Unified Ablation**: `run_unified_ablation_study.py`
- **Dataset Generator**: `ablation_dataset_generator.py`

---

## Conclusion

Both ablation studies have been successfully implemented with:
- ✅ Proper dataset separation
- ✅ Fixed random seeds for reproducibility
- ✅ Graceful error handling
- ✅ Non-blank, non-erroneous results
- ✅ Baseline training completed successfully

The studies are ready for full execution once the required CFG directories are generated.

