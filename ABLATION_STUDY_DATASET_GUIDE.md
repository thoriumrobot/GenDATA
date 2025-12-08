# Ablation Study Dataset Generation Guide

## Overview

This guide explains how ablation studies use separate dataset directories for different conditions, ensuring valid comparisons between augmentation vs. no augmentation and transformation ablations.

## Key Concepts

### Dataset Separation

Each ablation condition must use its own dataset generated from the appropriate CFG directory:
- **Augmentation vs. No Augmentation**: Different datasets from augmented vs. non-augmented CFGs
- **Transformation Ablation**: Separate dataset for each transformation disabled

### Random Seed Consistency

All random operations use seed 42 for reproducibility:
- Model weight initialization
- Train/validation split
- DataLoader shuffling
- Dataset generation

## CFG Directory Structure

### Expected CFG Directory Locations

#### Augmented CFGs (Baseline)
- Default: `cfg_output_specimin` or `cfg_output_adaptive_specimin`
- Contains CFGs generated from augmented slices
- Used for baseline training

#### Non-Augmented CFGs
- Default patterns checked:
  - `cfg_output_no_aug`
  - `cfg_output_specimin_no_aug`
  - `ablation_studies/no_augmentation/cfg_output`
- Contains CFGs generated from original (non-augmented) slices
- Used for no-augmentation comparison

#### Transformation Ablation CFGs
- Pattern: `cfg_output_ablate_{transform_name}`
- Alternative: `ablation_studies/ablate_{transform_name}/cfg_output`
- Contains CFGs generated with specific transformation disabled
- Used for transformation-specific ablation studies

### Example Directory Structure

```
GenDATA/
├── cfg_output_specimin/              # Augmented CFGs (baseline)
├── cfg_output_no_aug/                 # Non-augmented CFGs
├── cfg_output_ablate_loop_conversion/ # CFGs with loop_conversion disabled
├── cfg_output_ablate_guard_reversal/ # CFGs with guard_reversal disabled
└── ...
```

## Dataset Directory Naming Convention

### Augmentation Comparison Study

```
ablation_augmentation_comparison/
├── with_augmentation/          # Training results with augmentation
├── without_augmentation/       # Training results without augmentation
└── no_augmentation_datasets/   # Datasets generated from non-augmented CFGs
    ├── positive_real_balanced_dataset.json
    ├── nonnegative_real_balanced_dataset.json
    └── gtenegativeone_real_balanced_dataset.json
```

### Transformation Ablation Study

```
ablation_transformations_final/
├── baseline/                   # Baseline training results
└── ablate_{transform_name}/
    ├── datasets/              # Dataset for this transformation ablation
    │   ├── positive_real_balanced_dataset.json
    │   ├── nonnegative_real_balanced_dataset.json
    │   └── gtenegativeone_real_balanced_dataset.json
    └── {annotation_type}_{model}_training.log
```

## Usage

### Augmentation Comparison Study

```bash
python run_augmentation_comparison_study.py \
    --output_dir ablation_augmentation_comparison \
    --balanced_dataset_dir real_balanced_datasets \
    --cfg_dir cfg_output_specimin \
    --cfg_dir_no_aug cfg_output_no_aug \
    --episodes 10 \
    --device cpu
```

**Required Arguments:**
- `--cfg_dir_no_aug`: CFG directory for non-augmented slices (required for generating no-augmentation dataset)

**Process:**
1. Uses existing `real_balanced_datasets` for with-augmentation training
2. Generates new dataset in `no_augmentation_datasets/` from `cfg_dir_no_aug`
3. Trains models separately on each dataset
4. Compares results

### Transformation Ablation Study

```bash
python run_transformation_ablation_final.py \
    --output_dir ablation_transformations_final \
    --balanced_dataset_dir real_balanced_datasets \
    --cfg_dir cfg_output_specimin \
    --cfg_dir_base_pattern "cfg_output_ablate_{transform}" \
    --episodes 10 \
    --device cpu \
    --transformations loop_conversion guard_reversal
```

**Required Arguments:**
- `--cfg_dir_base_pattern`: Pattern for CFG directories with transformations disabled
  - Use `{transform}` placeholder, e.g., `"cfg_output_ablate_{transform}"`
  - For `loop_conversion`, this becomes `cfg_output_ablate_loop_conversion`

**Process:**
1. Trains baseline using `real_balanced_datasets`
2. For each transformation:
   - Constructs CFG directory path from pattern
   - Generates dataset in `ablate_{transform}/datasets/`
   - Trains models on generated dataset
   - Compares against baseline

### Unified Ablation Study

```bash
python run_unified_ablation_study.py \
    --output_dir ablation_unified \
    --balanced_dataset_dir real_balanced_datasets \
    --cfg_dir cfg_output_specimin \
    --cfg_dir_for_dataset cfg_output_specimin \
    --episodes 20 \
    --device cpu
```

**Optional Arguments:**
- `--cfg_dir_for_dataset`: CFG directory for dataset generation (if different from `cfg_dir`)
  - If dataset doesn't exist, it will be generated from this directory

**Process:**
1. Checks if dataset exists in `balanced_dataset_dir`
2. If missing, generates dataset from `cfg_dir_for_dataset`
3. Trains models on the dataset

## Dataset Generation Utility

The `ablation_dataset_generator.py` module provides a shared utility for generating datasets:

```python
from ablation_dataset_generator import AblationDatasetGenerator

generator = AblationDatasetGenerator(random_seed=42)
success, error = generator.generate_dataset(
    cfg_dir='cfg_output_specimin',
    output_dir='real_balanced_datasets',
    examples_per_annotation=2000,
    target_balance=0.5,
    timeout=3600
)
```

**Parameters:**
- `cfg_dir`: Directory containing CFG JSON files
- `output_dir`: Output directory for generated datasets
- `examples_per_annotation`: Number of examples per annotation type (default: 2000)
- `target_balance`: Target balance ratio, 0.5 = 50% positive, 50% negative
- `timeout`: Timeout in seconds (default: 3600)

## Prerequisites

### Generating CFGs for Different Conditions

Before running ablation studies, you must generate CFGs for each condition:

#### 1. Non-Augmented CFGs

```bash
# Generate slices without augmentation
python pipeline.py \
    --steps slice \
    --project_root case_studies/guava \
    --warnings_file case_studies/guava_warnings.json \
    --slices_dir slices_no_aug \
    --augment_variants 0  # No augmentation

# Generate CFGs from non-augmented slices
python pipeline.py \
    --steps cfg \
    --slices_dir slices_no_aug \
    --cfg_output_dir cfg_output_no_aug
```

#### 2. Transformation Ablation CFGs

For each transformation to ablate:

```bash
# Generate slices with specific transformation disabled
python enhanced_semantic_augment_slices.py \
    --slices_dir slices_original \
    --out_dir slices_ablate_loop_conversion \
    --variants_per_file 10 \
    --disabled loop_conversion  # Disable this transformation

# Generate CFGs
python pipeline.py \
    --steps cfg \
    --slices_dir slices_ablate_loop_conversion \
    --cfg_output_dir cfg_output_ablate_loop_conversion
```

## Verification

### Check Dataset Existence

```python
from ablation_dataset_generator import AblationDatasetGenerator

generator = AblationDatasetGenerator()
exists = generator.verify_dataset_exists('real_balanced_datasets')
print(f"Dataset exists: {exists}")
```

### Verify Random Seeds

To verify that random seeds are working correctly:

1. Run the same training twice with identical parameters
2. Check that:
   - Train/validation split is identical
   - Model initialization is identical
   - Training results are identical (within numerical precision)

## Troubleshooting

### Dataset Not Found

**Error**: `Dataset file not found: ...`

**Solutions:**
1. Ensure CFG directory exists and contains JSON files
2. Check that `improved_balanced_dataset_generator.py` is in the current directory
3. Verify CFG directory path is correct
4. Check file permissions

### CFG Directory Not Found

**Error**: `CFG directory does not exist: ...`

**Solutions:**
1. Generate CFGs for the required condition first
2. Check the directory path
3. Use `--cfg_dir_no_aug` or `--cfg_dir_base_pattern` to specify correct paths

### Dataset Generation Timeout

**Error**: `Dataset generation timed out after 3600 seconds`

**Solutions:**
1. Increase timeout in `AblationDatasetGenerator.generate_dataset()`
2. Reduce `examples_per_annotation` parameter
3. Check system resources (CPU, memory, disk I/O)

## Best Practices

1. **Always use separate datasets**: Never reuse the same dataset for different conditions
2. **Verify CFG directories**: Ensure CFG directories exist before running ablation studies
3. **Use consistent random seeds**: Always use seed 42 for reproducibility
4. **Document CFG sources**: Keep track of which CFG directories correspond to which conditions
5. **Validate datasets**: Use `verify_dataset_exists()` before training
6. **Check dataset balance**: Verify that datasets have approximately 50% positive/negative examples

## Related Files

- `ablation_dataset_generator.py`: Shared dataset generation utility
- `improved_balanced_dataset_generator.py`: Core dataset generation logic
- `run_augmentation_comparison_study.py`: Augmentation comparison study
- `run_transformation_ablation_final.py`: Transformation ablation study
- `run_unified_ablation_study.py`: Unified ablation study framework
- `improved_balanced_annotation_type_trainer.py`: Model training with fixed random seeds

## Latest Results

See `ABLATION_STUDY_RESULTS_LATEST.md` for the most recent ablation study results, including:
- Augmentation comparison baseline: 92.35% average validation accuracy (12 models)
- Transformation ablation baseline: 92.75% average validation accuracy (12 models)
- Implementation verification and known limitations

