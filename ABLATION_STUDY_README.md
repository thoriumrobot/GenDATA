# Ablation Study Pipeline

This document describes the ablation study pipeline for evaluating the impact of different augmentation techniques on model performance in the GenDATA project.

## Overview

The ablation study pipeline evaluates three key aspects of the GenDATA system:

1. **No Augmentation Impact**: Performance loss when no data augmentation is used
2. **Individual Transformation Impact**: Performance loss when each of the 27 semantic transformations is removed individually
3. **Random Walk Optimization Impact**: Performance loss when augmentation is used but random walk optimization is disabled

## Architecture

### Core Components

- **`ablation_study_pipeline.py`**: Main pipeline class that orchestrates ablation studies
- **`ablation_study_evaluator.py`**: Evaluates and compares performance metrics across studies
- **`ablation_study_report_generator.py`**: Generates comprehensive reports and visualizations
- **`run_ablation_studies.py`**: Main execution script with command-line interface

### Directory Structure

Each ablation study creates isolated directories to prevent data contamination:

```
ablation_studies/
├── baseline/                      # Full pipeline (reference)
├── no_augmentation/              # No augmentation
├── ablate_transform_X/           # Each transformation removed (27 dirs)
└── no_random_walk/               # Augmentation without random walk
    ├── slices/
    ├── cfg_output/
    └── models/
```

## Usage

### Basic Usage

```bash
# Run all ablation studies
python run_ablation_studies.py --mode all

# Run specific ablation studies
python run_ablation_studies.py --mode baseline
python run_ablation_studies.py --mode no_aug
python run_ablation_studies.py --mode no_rw
python run_ablation_studies.py --mode transformations

# Run transformation ablation for specific transformations
python run_ablation_studies.py --mode transformations --transform_names loop_conversion guard_reversal
```

### Advanced Usage

```bash
# Custom parameters
python run_ablation_studies.py --mode all \
    --project_root /path/to/project \
    --warnings_file /path/to/warnings.out \
    --output_dir custom_ablation_studies \
    --device cuda

# Evaluate existing results
python run_ablation_studies.py --mode evaluate --results_dir ablation_studies

# Generate reports only
python run_ablation_studies.py --mode report --results_dir ablation_studies
```

### Command Line Options

- `--mode`: Ablation study mode (`all`, `baseline`, `no_aug`, `no_rw`, `transformations`, `evaluate`, `report`)
- `--project_root`: Root directory of the Java project
- `--warnings_file`: Path to warnings file
- `--output_dir`: Output directory for ablation studies
- `--device`: Device to use (`cpu`, `cuda`, `auto`)
- `--transform_names`: Specific transformation names for transformation mode
- `--results_dir`: Results directory for evaluation/reporting modes

## Ablation Studies

### 1. No Augmentation Study

**Purpose**: Measure the impact of data augmentation on model performance.

**Method**: 
- Skip augmentation step entirely
- Use original slices directly for training
- Train all 7 models × 3 annotation types
- Compare performance against baseline

**Expected Outcome**: Significant performance loss demonstrating the value of data augmentation.

### 2. Individual Transformation Ablation Studies

**Purpose**: Identify which semantic transformations contribute most to model performance.

**Method**:
- For each of 27 transformations:
  - Disable that specific transformation
  - Run full pipeline with remaining 26 transformations
  - Train all models
  - Compare performance against baseline

**Transformations Evaluated**:

**Enhanced Transformations (17)**:
- `loop_conversion`, `guard_reversal`, `mathematical_expression`, `logical_expression`
- `ternary_operator`, `switch_statement`, `variable_operation`, `method_extraction`
- `conditional_expression`, `array_access_pattern`, `string_concatenation`
- `numeric_literal`, `exception_handling`, `lambda_expression`, `stream_api`
- `builder_pattern`, `functional_conversion`

**Simple Transformations (10)**:
- `simple_method_call`, `simple_assignment`, `simple_conditional`
- `simple_array_access`, `simple_return_statement`, `simple_variable_declaration`
- `simple_constructor_call`, `simple_field_access`, `simple_string_operation`
- `simple_numeric_operation`

**Expected Outcome**: Identification of critical transformations that should not be removed.

### 3. No Random Walk Optimization Study

**Purpose**: Measure the impact of random walk-based optimization on augmentation quality.

**Method**:
- Use all 27 semantic transformations
- Disable random walk optimizer:
  - No RL policy learning
  - No MCTS search
  - No evolutionary algorithms
  - No graph-based walks
- Use deterministic transformation selection instead

**Expected Outcome**: Moderate performance loss showing the value of intelligent transformation selection.

## Performance Metrics

### Primary Metrics

- **Warning Reduction Percentage**: Primary measure of model effectiveness
- **Training Time**: Efficiency measure
- **Data Generation**: Number of slices and CFGs generated
- **Model Training**: Number of models successfully trained

### Derived Metrics

- **Performance Loss**: Percentage reduction in performance compared to baseline
- **Statistical Significance**: Confidence in performance differences
- **Transformation Impact Ranking**: Ordered list of transformation importance

## Output Files

### Results Files

- `ablation_results_summary.json`: Comprehensive results from all studies
- `ablation_analysis_report.json`: Detailed analysis with statistical measures
- `ablation_results_report.md`: Human-readable markdown report

### Visualizations

- `performance_loss_bar.png`: Bar chart showing performance loss per ablation case
- `transformation_impact_heatmap.png`: Heatmap of transformation impact
- `ablation_comparison.png`: Overall ablation comparison charts

### Individual Study Results

Each ablation case generates:
- `results.json`: Individual study metrics
- `slices/`: Generated code slices
- `cfg_output/`: Generated CFGs
- `models/`: Trained models

## Implementation Details

### Modified Components

The ablation pipeline modifies existing components to support ablation studies:

1. **Enhanced Semantic Transformer**: Added `disabled_transformations` parameter
2. **Simple Code Semantic Transformer**: Added `disabled_transformations` parameter  
3. **Random Walk Optimizer**: Added `enable_random_walk` flag

### Directory Isolation

Each ablation study uses completely separate directories to prevent:
- Data contamination between studies
- Model interference
- CFG cross-contamination
- Slicing result mixing

### Performance Tracking

The pipeline tracks:
- Training time for each ablation case
- Number of files generated at each stage
- Model training success/failure rates
- Memory and computational resource usage

## Example Output

### Markdown Report Structure

```markdown
# Ablation Study Results Report

## Study Overview
- Total Ablation Cases: 30
- Baseline Case: baseline
- Study Timestamp: 2025-10-13T16:29:19

## Key Findings
- Average performance loss across all ablations: 15.2%
- Maximum performance loss observed: 45.3%
- Data augmentation has significant impact on model performance

## Performance Analysis
### No Augmentation Impact
- Baseline Performance: 1250.5s
- No Augmentation Performance: 890.2s
- Performance Loss: 28.8%

## Individual Transformation Impact
### Enhanced Transformations (17 methods)
| Transformation | Performance Loss (%) |
|----------------|---------------------|
| loop_conversion | 12.5 |
| guard_reversal | 8.3 |
| ...

### Simple Transformations (10 methods)
| Transformation | Performance Loss (%) |
|----------------|---------------------|
| simple_method_call | 5.2 |
| simple_assignment | 3.8 |
| ...
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce episodes or use CPU instead of GPU
2. **File Permissions**: Ensure write access to output directories
3. **Missing Dependencies**: Install required Python packages
4. **Path Issues**: Use absolute paths for project_root and warnings_file

### Validation

Run the test script to validate the pipeline:

```bash
python test_ablation_pipeline.py
```

This will test:
- Pipeline initialization
- Directory creation
- Component integration
- Transformation mapping
- Random walk optimization

## Performance Expectations

### Estimated Runtime

- **Baseline**: ~1 hour (train 21 models)
- **No augmentation**: ~30 minutes (simpler data)
- **Each transformation ablation**: ~45 minutes × 27 = ~20 hours
- **No random walk**: ~1 hour
- **Total estimated time**: ~23 hours (can parallelize transformation ablations)

### Resource Requirements

- **Memory**: 8GB+ RAM recommended
- **Storage**: 50GB+ for all ablation results
- **CPU**: Multi-core recommended for parallel processing
- **GPU**: Optional but recommended for faster training

## Integration with GenDATA

The ablation pipeline integrates seamlessly with the existing GenDATA architecture:

- Uses existing `SimpleAnnotationTypePipeline` as base
- Leverages current augmentation systems
- Compatible with all 7 RL models
- Works with existing evaluation metrics
- Maintains compatibility with case study projects

## Troubleshooting

### Common Issues and Solutions

#### 1. "No .java slices produced" Error

**Problem**: `[SLICE] No .java slices produced by 'soot'. Falling back to 'cf' slicer...`

**Root Cause**: Soot slicer fails to produce slices, then CF slicer can't find the JAR.

**Solution**: The pipeline now uses Specimin slicer by default, which is more reliable for Checker Framework code.

#### 2. "Unknown transformation" Warnings

**Problem**: `WARNING - Unknown transformation: TransformationType.RANDOM_METHOD_INSERTION`

**Root Cause**: Random walk optimizer tries to use random transformation types that don't exist in semantic transformers.

**Solution**: Random walk optimizer is disabled in ablation studies to prevent these warnings.

#### 3. "CheckerFrameworkSlicer JAR not found" Error

**Problem**: `Error: CheckerFrameworkSlicer JAR not found at <path>/build/libs/GenDATA-all.jar`

**Root Cause**: Pipeline looks for JAR in wrong directory.

**Solution**: Pipeline now uses main GenDATA directory (`/home/ubuntu/GenDATA`) as `cfwr_root` to access the JAR.

#### 4. CUDA/GPU Issues

**Problem**: `RuntimeError: No CUDA GPUs are available`

**Root Cause**: Environment has no CUDA GPUs.

**Solution**: Use `--device cpu` flag for CPU-only environments.

#### 5. Blank Output Directories

**Problem**: Slices, CFGs, or models directories are empty.

**Root Cause**: Pipeline failure during slicing, CFG generation, or model training.

**Solution**: Check logs for specific errors and ensure all dependencies are properly installed.

### Slicer Comparison

| Slicer | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Specimin** | ✅ More reliable for CF code<br>✅ No pre-compilation needed<br>✅ Better error handling | ⚠️ Slower than Soot | **Default choice** |
| **Soot** | ✅ Faster bytecode analysis<br>✅ More precise slicing | ❌ Requires compilation<br>❌ Less reliable with CF | Legacy compatibility |
| **CF** | ✅ Direct CF integration<br>✅ Fastest execution | ❌ Requires JAR<br>❌ Limited functionality | Fallback only |

### Device Configuration

- **GPU (CUDA)**: Default for better performance when available
- **CPU**: Use when no GPU available or for testing
- **Auto**: Automatically detects best available device

### File Path Issues

The pipeline expects:
- Warnings file: `index1.out` (will copy from `index1.small.out` if needed)
- JAR location: `/home/ubuntu/GenDATA/build/libs/GenDATA-all.jar`
- Project root: `/home/ubuntu/checker-framework/checker/tests/index`

### Testing Commands

```bash
# Test no augmentation mode (recommended for initial testing)
python run_ablation_studies.py \
  --mode no_aug \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/GenDATA/index1.small.out \
  --output_dir ablation_studies_test \
  --device cpu

# Verify outputs
ls -la ablation_studies_test/no_augmentation/cfg_output/
ls -la ablation_studies_test/no_augmentation/models/
cat ablation_studies_test/no_augmentation/results.json
```

## Future Enhancements

Potential improvements to the ablation pipeline:

1. **Parallel Execution**: Run transformation ablations in parallel
2. **Incremental Studies**: Resume interrupted studies
3. **Custom Metrics**: Add domain-specific performance measures
4. **Interactive Reports**: Web-based visualization interface
5. **Automated Analysis**: Machine learning-based pattern detection
6. **Cross-Validation**: Multiple runs with different random seeds

## Conclusion

The ablation study pipeline provides comprehensive evaluation of the GenDATA augmentation system, enabling data-driven decisions about which components are most critical for model performance. The isolated directory structure ensures reliable results, while the detailed reporting and visualization capabilities facilitate interpretation and communication of findings.
