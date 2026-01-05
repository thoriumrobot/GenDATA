# GenDATA Usage Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Training Models](#training-models)
6. [Running Predictions](#running-predictions)
7. [Evaluating Results](#evaluating-results)
8. [Configuration](#configuration)
9. [Case Studies](#case-studies)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)

---

## Overview

GenDATA (Generation of Data for Annotation Type prediction) is a Reinforcement Learning-based system for automatically placing Checker Framework annotations in Java code. The system:

- **Trains RL models** on Checker Framework test suites to learn annotation placement patterns
- **Predicts annotations** on new code using trained models
- **Evaluates results** by measuring warning reduction after annotation placement
- **Supports multiple checkers**: Lower Bound, SQL Quotes, Signature String

### Key Features

- **Multi-Checker Support**: Unified system for multiple Checker Framework checkers
- **Confidence-Based Selection**: Automatically selects highest-confidence annotation per location
- **Semantic Augmentation**: 20 transformation methods (10 enhanced + 10 simple) for robust training
- **Balanced Training**: 50/50 positive/negative examples for improved model convergence
- **GPU Acceleration**: Optimized for GPU training when available
- **AST-Based Placement**: Accurate annotation placement using Eclipse JDT

---

## Prerequisites

### Required Software

1. **Python 3.8+**
   ```bash
   python3 --version  # Should be 3.8 or higher
   ```

2. **Java Development Kit (JDK) 8+**
   ```bash
   java -version  # Should be Java 8 or higher
   ```

3. **Checker Framework 3.42.0+**
   - Default location: `/home/ubuntu/checker-framework-3.42.0`
   - Or set `CHECKERFRAMEWORK_HOME` environment variable

4. **Checker Framework Warning Resolver (CFWR)**
   - Required for training
   - Default location: `/home/ubuntu/CFWR`
   - Or set `CFWR_ROOT` environment variable

5. **Soot** (included in system) or **Specimin**
   - Used for code slicing
   - Specimin is the default slicer

### Optional

- **CUDA-enabled GPU** for faster training
- **Maven/Gradle** if working with Maven/Gradle projects

---

## Installation

### 1. Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install additional ML libraries
pip install torch torch-geometric javalang scikit-learn joblib numpy
```

### 2. Verify Checker Framework Installation

```bash
# Check if Checker Framework is accessible
ls $CHECKERFRAMEWORK_HOME  # or /home/ubuntu/checker-framework-3.42.0

# Verify javac with Checker Framework
javac -processor org.checkerframework.checker.index.IndexChecker -version
```

### 3. Set Environment Variables (Optional)

```bash
# Add to ~/.bashrc or ~/.zshrc
export CHECKERFRAMEWORK_HOME=/path/to/checker-framework-3.42.0
export CHECKERFRAMEWORK_CP=$CHECKERFRAMEWORK_HOME/checker/dist/checker.jar
export CFWR_ROOT=/path/to/CFWR
```

---

## Quick Start

### 1. Generate Warning Files

Before training, generate warning files from Checker Framework test suites:

```bash
# Generate warnings for all supported checkers
python3 generate_checker_warning_files.py

# Generate for specific checker
python3 generate_checker_warning_files.py --checker lower_bound
python3 generate_checker_warning_files.py --checker sql_quotes
python3 generate_checker_warning_files.py --checker signature_string

# Skip if files already exist
python3 generate_checker_warning_files.py --skip-existing
```

**Generated Warning Files:**
- `lower_bound_warnings.out` - Lower Bound Checker warnings
- `sql_quotes_warnings.out` - SQL Quotes Checker warnings
- `signature_string_warnings.out` - Signature String Checker warnings

**Note**: For backward compatibility, `index1.out` is also recognized as a Lower Bound warnings file.

### 2. Train Models

```bash
# Train all models for all checkers
python3 train_all_checkers.py --generate-warnings

# Train without generating warnings (assumes warning files exist)
python3 train_all_checkers.py

# Train specific checker models
python3 train_all_21_models.py          # Lower Bound Checker (21 models)
python3 train_sql_quotes_models.py      # SQL Quotes Checker (14 models)
python3 train_signature_string_models.py # Signature String Checker (21 models)
```

**Models Trained:**
- **Lower Bound**: 21 models (7 base models × 3 annotation types)
- **SQL Quotes**: 14 models (7 base models × 2 annotation types)
- **Signature String**: 21 models (7 base models × 3 annotation types)

**Training Output:**
- Models saved in `models_annotation_types/` (or checker-specific directories)
- Training logs in console and log files
- Datasets in `slices_adaptive_specimin_{checker}/` and `cfg_output_adaptive_specimin_{checker}/`

### 3. Run Predictions

```bash
# Predict on all case studies (automatically runs checker)
python predict_with_enhanced_pipeline.py

# Predict on specific file
python predict_with_enhanced_pipeline.py --target_file /path/to/MyClass.java

# Predict on specific project directory
python predict_with_enhanced_pipeline.py --case_studies_dir /path/to/project

# Disable automatic checker execution (use provided warnings file)
python predict_with_enhanced_pipeline.py --target_file /path/to/MyClass.java --no_run_checker
```

**Prediction Output:**
- Predictions: `predictions_annotation_types/{checker}/`
- Annotated files: `annotation_evaluation/temp_repos/{project}/`
- Backups: `annotation_evaluation/backups/{project}/`

---

## Training Models

### Training Options

#### Simple Training Pipeline

```bash
python simple_annotation_type_pipeline.py \
    --mode train \
    --project_root /home/ubuntu/checker-framework/checker/tests/index \
    --warnings_file /home/ubuntu/GenDATA/lower_bound_warnings.out \
    --cfwr_root /home/ubuntu/CFWR \
    --episodes 100 \
    --base_model gcn
```

**Parameters:**
- `--mode`: `train` or `predict`
- `--project_root`: Root directory of Java project with warnings
- `--warnings_file`: Path to Checker Framework warnings file
- `--cfwr_root`: Root directory of CFWR project
- `--episodes`: Number of training episodes (default: 50)
- `--base_model`: Base model type (`gcn`, `gbt`, `causal`, `hgt`, `gcsn`, `dg2n`, `enhanced_causal`)
- `--augmentation_factor`: Variants per slice (default: 100)
- `--device`: `auto`, `cpu`, or `cuda` (default: `auto`)
- `--disable_random_walk`: Disable random walk optimization

#### Advanced Training Pipeline

```bash
python main_optimized_pipeline.py --train-all

# Train specific annotation type
python main_optimized_pipeline.py --train nonnegative --model gcn

# Train with custom configuration
python main_optimized_pipeline.py --train positive --config custom_config.json
```

### Training Process

1. **Warning Resolution**: `CheckerFrameworkWarningResolver` identifies target code elements
2. **Code Augmentation**: 20 semantic transformations create variants
   - Enhanced: Loop conversions, guard reversals, mathematical properties, etc.
   - Simple: Method call variations, assignment transformations, etc.
3. **Slicing**: Soot/Specimin generates minimal slices
4. **CFG Generation**: Checker Framework CFG Builder creates control flow graphs
5. **Model Training**: RL models learn from balanced datasets
6. **Model Saving**: Trained models saved to `models_annotation_types/`

### Training Output

**Directory Structure:**
```
models_annotation_types/
├── positive_gcn_model.pth
├── positive_gbt_model.pth
├── positive_causal_model.pth
├── nonnegative_gcn_model.pth
├── nonnegative_gbt_model.pth
├── ...
└── gtenegativeone_dg2n_model.pth
```

**Checkpoint Files:**
- Models saved as `.pth` files (PyTorch) or `.joblib` files (sklearn)
- Naming: `{annotation_type}_{base_model}_model.pth`
- Example: `positive_gcn_model.pth`, `nonnegative_hgt_model.pth`

### Training Tips

- **GPU Acceleration**: Use `--device cuda` if GPU available for faster training
- **Balanced Datasets**: System automatically creates 50/50 positive/negative examples
- **Episode Count**: 50-100 episodes typically sufficient for convergence
- **Data Reuse**: Datasets are not regenerated if they already exist (saves time)

---

## Running Predictions

### Basic Prediction

```bash
# Enhanced pipeline (recommended)
python predict_with_enhanced_pipeline.py \
    --target_file /path/to/File.java \
    --models_dir models_annotation_types \
    --output_dir predictions_annotation_types
```

### Prediction Options

```bash
python predict_with_enhanced_pipeline.py \
    --target_file /path/to/File.java \
    --case_studies_dir /path/to/project \
    --models_dir models_annotation_types \
    --output_dir predictions_annotation_types \
    --no_run_checker  # Use existing warnings file
```

**Parameters:**
- `--target_file`: Specific Java file to predict on
- `--case_studies_dir`: Directory containing case study projects
- `--models_dir`: Directory with trained models (default: `models_annotation_types`)
- `--output_dir`: Output directory for predictions (default: `predictions_annotation_types`)
- `--no_run_checker`: Disable automatic checker execution

### Prediction Process

1. **Checker Execution**: Run Checker Framework checker on target (unless `--no_run_checker`)
2. **Slicing**: Generate slices from warning locations
3. **CFG Generation**: Convert slices to control flow graphs
4. **Model Prediction**: Run all trained models on CFG nodes
5. **Confidence Selection**: Select highest-confidence annotation per location
6. **Annotation Placement**: Place annotations in source code using AST-based placement

### Prediction Output Format

**Predictions JSON:**
```json
[
  {
    "annotation_type": "@NonNegative",
    "confidence": 0.95,
    "model_type": "gcn",
    "line_number": 105,
    "file_path": "/path/to/File.java",
    "reason": "@NonNegative predicted by GCN model (confidence: 0.950)"
  },
  {
    "annotation_type": "@Positive",
    "confidence": 0.87,
    "model_type": "hgt",
    "line_number": 120,
    "file_path": "/path/to/File.java",
    "reason": "@Positive predicted by HGT model (confidence: 0.870)"
  }
]
```

**Annotated Files:**
- Located in: `annotation_evaluation/temp_repos/{project}/`
- Original backups in: `annotation_evaluation/backups/{project}/`
- Annotations placed directly in Java source files

### Viewing Annotations

```bash
# Search for annotations in annotated files
grep -r "@NonNegative\|@Positive\|@GTENegativeOne" annotation_evaluation/temp_repos/{project}/

# Compare original vs annotated
diff annotation_evaluation/backups/{project}/File.java \
      annotation_evaluation/temp_repos/{project}/File.java
```

---

## Evaluating Results

### Evaluation Pipeline

```bash
# Run evaluation on annotated projects
python studies/compute_warning_reduction.py \
    --project_dir annotation_evaluation/temp_repos/{project} \
    --backup_dir annotation_evaluation/backups/{project} \
    --predictions_file predictions_annotation_types/{project}/{model}_predictions.json
```

### Evaluation Metrics

**Warning Reduction:**
- **Baseline Warnings**: Warnings before annotation placement
- **Warnings After**: Warnings after annotation placement
- **Reduction Percentage**: `(baseline - after) / baseline * 100`

**Placement Success:**
- **Compilation Success**: Whether annotated code compiles
- **Annotation Count**: Number of annotations placed
- **Placement Accuracy**: Percentage of annotations placed correctly

### Evaluation Reports

**Location**: `annotation_evaluation/evaluation_report.json`

```json
{
  "metadata": {
    "timestamp": "2025-12-10T18:12:51",
    "projects_evaluated": 3,
    "base_models_tested": 7
  },
  "results": [
    {
      "project_name": "sortpom",
      "baseline_warnings": 96,
      "model_results": [
        {
          "base_model": "gcn",
          "annotations_placed": 780,
          "warnings_after": 0,
          "warning_reduction": 96,
          "reduction_percentage": 100.0,
          "placement_success": true,
          "compilation_success": true
        }
      ]
    }
  ]
}
```

### Annotation Impact Analysis

```bash
# Analyze how annotations reduce warnings
python analyze_annotation_impact.py --project sortpom --base_model gcn

# Generate comprehensive impact report
python comprehensive_annotation_impact_analysis.py
```

**Reports Generated:**
- `ANNOTATION_IMPACT_ANALYSIS_REPORT.md` - Detailed impact analysis
- `DATA_VERIFICATION_REPORT.md` - Data verification results

---

## Configuration

### Configuration File

**Location**: `annotation_type_config.json`

```json
{
  "annotation_types": [
    "NO_ANNOTATION",
    "@Positive",
    "@NonNegative",
    "@GTENegativeOne"
  ],
  "default_threshold": 0.3,
  "training_episodes": 500,
  "evaluation_enabled": true
}
```

**Parameters:**
- `annotation_types`: List of annotation types to predict
- `default_threshold`: Confidence threshold for predictions (0.0-1.0)
- `training_episodes`: Default number of training episodes
- `evaluation_enabled`: Enable evaluation after training

### Environment Variables

```bash
# Checker Framework
export CHECKERFRAMEWORK_HOME=/path/to/checker-framework-3.42.0
export CHECKERFRAMEWORK_CP=$CHECKERFRAMEWORK_HOME/checker/dist/checker.jar

# CFWR
export CFWR_ROOT=/path/to/CFWR

# Device
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

### Directory Configuration

**Default Directories:**
- **Slices**: `slices_adaptive_specimin_{checker}/`
- **CFGs**: `cfg_output_adaptive_specimin_{checker}/`
- **Models**: `models_annotation_types_{checker}/`
- **Predictions**: `predictions_annotation_types_{checker}/`
- **Augmented Code**: `augmented_code_unified_{checker}/`

**Checker-Specific Directories:**
- Lower Bound: Uses `models_annotation_types/` (backward compatibility)
- SQL Quotes: Uses `models_annotation_types_sql_quotes/`
- Signature String: Uses `models_annotation_types_signature_string/`

---

## Case Studies

### Running Case Studies

```bash
# Run all case studies
python predict_with_enhanced_pipeline.py

# Run specific case study
python predict_with_enhanced_pipeline.py \
    --case_studies_dir case_studies/sortpom
```

### Available Case Studies

**Lower Bound Checker:**
- `sortpom` - Maven POM file sorter
- `eclipse-external-annotations-m2e-plugin` - Eclipse plugin
- `pom-tuner` - POM configuration tool

**Location**: `case_studies/` directory

### Case Study Results

**Location**: `annotation_evaluation/evaluation_report.json`

See [Evaluating Results](#evaluating-results) section for details on interpreting results.

### Running Custom Case Studies

1. **Prepare Project**:
   ```bash
   # Copy project to case_studies/
   cp -r /path/to/your/project case_studies/my_project
   ```

2. **Run Prediction**:
   ```bash
   python predict_with_enhanced_pipeline.py \
       --case_studies_dir case_studies/my_project
   ```

3. **Check Results**:
   ```bash
   # View predictions
   cat predictions_annotation_types/my_project/*_predictions.json
   
   # View annotated files
   ls annotation_evaluation/temp_repos/my_project/
   ```

---

## Troubleshooting

### Common Issues

#### 1. Models Not Found

**Error**: `Models directory not found` or `No trained models found`

**Solution**:
```bash
# Train models first
python3 train_all_checkers.py
```

#### 2. Checker Framework Not Found

**Error**: `Checker Framework not found` or compilation errors

**Solution**:
```bash
# Set environment variable
export CHECKERFRAMEWORK_HOME=/path/to/checker-framework-3.42.0

# Verify installation
javac -processor org.checkerframework.checker.index.IndexChecker -version
```

#### 3. Out of Memory

**Error**: CUDA out of memory or Java heap space errors

**Solution**:
```bash
# Use CPU instead of GPU
python simple_annotation_type_pipeline.py --device cpu

# Reduce batch size in configuration
# Or reduce augmentation_factor
python simple_annotation_type_pipeline.py --augmentation_factor 50
```

#### 4. Compilation Failures

**Error**: Annotated code fails to compile

**Solution**:
- Check annotation placement in `annotation_evaluation/temp_repos/`
- Verify Checker Framework annotations are imported
- Check for syntax errors introduced during placement

#### 5. No Warnings Generated

**Error**: Checker produces no warnings

**Solution**:
```bash
# Manually run checker to verify
javac -processor org.checkerframework.checker.index.IndexChecker \
    -cp $CHECKERFRAMEWORK_CP \
    /path/to/File.java

# Check if project compiles without checker
javac /path/to/File.java
```

#### 6. Wrong Checker Detected

**Error**: System uses wrong checker or models

**Solution**:
```bash
# Explicitly specify checker in warnings file name
# lower_bound_warnings.out -> lower_bound
# sql_quotes_warnings.out -> sql_quotes
# signature_string_warnings.out -> signature_string

# Or use --checker_name parameter (if supported)
```

### Getting Help

1. **Check Logs**: Review console output and log files
2. **Verify Configuration**: Check `annotation_type_config.json` and environment variables
3. **Test Components**: Run individual components separately
4. **Review Documentation**: See `README.md` and other documentation files

---

## Advanced Usage

### Custom Model Training

```bash
# Train with custom hyperparameters
python binary_rl_gcn_standalone.py \
    --episodes 100 \
    --learning_rate 0.001 \
    --hidden_dim 128 \
    --dropout_rate 0.5 \
    --device cuda
```

### Ablation Studies

```bash
# Augmentation comparison study
python run_augmentation_comparison_study.py \
    --cfg_dir cfg_output_specimin \
    --cfg_dir_no_aug ablation_studies/no_augmentation/cfg_output \
    --episodes 10

# Transformation ablation study
python run_transformation_ablation_final.py \
    --cfg_dir cfg_output_specimin \
    --cfg_dir_base_pattern "ablation_studies/ablate_{transform}/cfg_output" \
    --episodes 10
```

### Batch Processing

```bash
# Process multiple projects
for project in case_studies/*/; do
    python predict_with_enhanced_pipeline.py \
        --case_studies_dir "$project"
done
```

### Custom Annotation Placement

```bash
# Use custom placement script
python place_annotations.py \
    --predictions_file predictions.json \
    --target_dir /path/to/project \
    --backup_dir /path/to/backup
```

### Multi-Checker Evaluation

```bash
# Evaluate all checkers on same project
python evaluate_multi_checker.py \
    --project_dir /path/to/project \
    --checkers lower_bound sql_quotes signature_string
```

### Data Analysis

```bash
# Analyze annotation patterns
python analyze_annotation_impact.py --project sortpom

# Verify data integrity
python verify_data_accuracy.py

# Generate comprehensive reports
python final_annotation_impact_report.py
```

---

## Additional Resources

### Documentation Files

- `README.md` - Project overview and features
- `LOWER_BOUND_CHECKER_EVALUATION_PIPELINE_DOCUMENTATION.md` - Detailed pipeline documentation
- `ANNOTATION_IMPACT_ANALYSIS_REPORT.md` - How annotations reduce warnings
- `BALANCED_TRAINING_GUIDE.md` - Balanced training documentation
- `ENHANCED_PIPELINE_DOCUMENTATION.md` - Enhanced pipeline features

### Key Scripts

- `predict_with_enhanced_pipeline.py` - Main prediction script
- `simple_annotation_type_pipeline.py` - Training and prediction pipeline
- `multi_checker_predictor.py` - Unified predictor for all checkers
- `place_annotations.py` - Annotation placement engine
- `checker_framework_runner.py` - Checker Framework execution

### Directory Structure

```
GenDATA/
├── models_annotation_types/       # Trained models
├── predictions_annotation_types/  # Prediction results
├── slices_adaptive_specimin/      # Code slices
├── cfg_output_adaptive_specimin/  # Control flow graphs
├── case_studies/                  # Case study projects
├── annotation_evaluation/         # Evaluation results
│   ├── temp_repos/               # Annotated files
│   └── backups/                  # Original file backups
└── studies/                       # Evaluation scripts
```

---

## Quick Reference

### Essential Commands

```bash
# Generate warnings
python3 generate_checker_warning_files.py

# Train models
python3 train_all_checkers.py

# Run predictions
python predict_with_enhanced_pipeline.py

# Evaluate results
python studies/compute_warning_reduction.py
```

### File Locations

- **Models**: `models_annotation_types/`
- **Predictions**: `predictions_annotation_types/`
- **Annotated Files**: `annotation_evaluation/temp_repos/`
- **Evaluation Report**: `annotation_evaluation/evaluation_report.json`

### Key Parameters

- **Confidence Threshold**: 0.3 (default)
- **Training Episodes**: 50-100 (typical)
- **Augmentation Factor**: 100 (default)
- **Device**: `auto` (detects GPU automatically)

---

**For more information, see the main README.md and other documentation files in the project directory.**



