# Training Dataset Creation Summary

## Overview

This document summarizes the process of creating training datasets for GenDATA checkers (SQL Quotes and Signature String) using their warning logs.

## Scripts Created

### 1. `create_training_datasets.py`

A comprehensive script to generate training datasets (slices, CFGs, augmented code) for GenDATA checkers.

**Usage:**
```bash
# Create datasets for all checkers
python3 create_training_datasets.py --checker all --generate-warnings

# Create dataset for specific checker
python3 create_training_datasets.py --checker signature_string

# Generate warnings first, then create datasets
python3 create_training_datasets.py --checker all --generate-warnings
```

**Features:**
- Automatically determines checker name from warnings file path
- Uses checker-specific directories to avoid conflicts:
  - `slices_adaptive_specimin_{checker_name}/`
  - `cfg_output_adaptive_specimin_{checker_name}/`
  - `augmented_code_adaptive_{checker_name}/`
- Generates slices using Soot slicer
- Generates CFGs from slices
- Optionally augments original code
- Verifies dataset creation and reports statistics

## Process Flow

1. **Generate Warning Files** (if `--generate-warnings` flag is used)
   - Uses `generate_checker_warning_files.py` to create warning files from test suites
   - Warning files are saved as `{checker_name}_warnings.out`

2. **Create Training Dataset**
   - Initialize `SimpleAnnotationTypePipeline` with checker-specific directories
   - Generate slices from warnings using Soot slicer
   - Generate CFGs from slices
   - Optionally augment original code with semantic transformations
   - Verify dataset creation (count slices, CFGs, augmented files)

## Checker-Specific Directories

To ensure datasets for different checkers don't conflict, each checker uses its own directories:

- **Lower Bound Checker**: 
  - `slices_adaptive_specimin_lower_bound/`
  - `cfg_output_adaptive_specimin_lower_bound/`
  - `augmented_code_adaptive_lower_bound/`

- **SQL Quotes Checker**:
  - `slices_adaptive_specimin_sql_quotes/`
  - `cfg_output_adaptive_specimin_sql_quotes/`
  - `augmented_code_adaptive_sql_quotes/`

- **Signature String Checker**:
  - `slices_adaptive_specimin_signature_string/`
  - `cfg_output_adaptive_specimin_signature_string/`
  - `augmented_code_adaptive_signature_string/`

## Current Status

### Warning Files Generated

- ✅ `lower_bound_warnings.out` - 100 lines (warnings found)
- ✅ `signature_string_warnings.out` - 0 lines (test suite fully annotated or no warnings)
- ❌ `sql_quotes_warnings.out` - Test suite not found at `/home/ubuntu/checker-framework/checker/tests/quotes`

### Training Datasets

- **Lower Bound Checker**: Dataset can be created using existing `index1.out` or `lower_bound_warnings.out`
- **Signature String Checker**: Dataset creation in progress (may have limited data if no warnings)
- **SQL Quotes Checker**: Cannot create dataset (test suite missing)

## Notes

1. **Empty Warning Files**: If a warning file is empty (0 bytes), it means:
   - The test suite is fully annotated (no warnings)
   - The checker found no issues
   - The dataset creation will still run but may generate no slices

2. **Test Suite Availability**: SQL Quotes Checker test suite is not available at the expected location. This needs to be addressed before training SQL Quotes models.

3. **Dataset Separation**: All datasets are stored in checker-specific directories to prevent conflicts and allow parallel training.

## Next Steps

1. Verify that Signature String dataset creation completes successfully
2. Address SQL Quotes test suite availability
3. Verify that generated slices and CFGs are in the correct format for training
4. Train models using the generated datasets

