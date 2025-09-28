# CFWR Annotation Type Models - GenDATA

This directory contains the essential files for understanding and running the CFWR (Checker Framework Warning Resolver) annotation type models. These models predict specific Checker Framework annotation types (@Positive, @NonNegative, @GTENegativeOne) using a two-stage approach: binary RL models identify annotation targets, then annotation type models determine the specific annotation type.

## Core Components

### Annotation Type Models
- `annotation_type_rl_positive.py` - Trains model for @Positive annotations
- `annotation_type_rl_nonnegative.py` - Trains model for @NonNegative annotations  
- `annotation_type_rl_gtenegativeone.py` - Trains model for @GTENegativeOne annotations
- `simple_annotation_type_pipeline.py` - Simplified pipeline for training and prediction
- `annotation_type_pipeline.py` - Full pipeline with Specimin, augmentation, and Soot integration

### Binary RL Models (Dependencies)
These models predict whether ANY annotation should be placed (binary classification):
- `binary_rl_gcn_standalone.py` - Graph Convolutional Network model
- `binary_rl_gbt_standalone.py` - Gradient Boosting Trees model
- `binary_rl_causal_standalone.py` - Causal inference model
- `binary_rl_hgt_standalone.py` - Heterogeneous Graph Transformer model
- `binary_rl_gcsn_standalone.py` - Gated Causal Subgraph Network model
- `binary_rl_dg2n_standalone.py` - Deterministic Gates Neural Network model

### Core Model Implementations
- `hgt.py` - HGT model implementation
- `gbt.py` - GBT model implementation
- `causal_model.py` - Causal model implementation

### Supporting Infrastructure
- `checker_framework_integration.py` - Checker Framework integration utilities
- `place_annotations.py` - Annotation placement engine
- `predict_on_project.py` - Project-wide prediction capabilities
- `prediction_saver.py` - Prediction saving and reporting utilities

### Evaluation and Testing
- `run_case_studies.py` - Binary RL model case studies
- `annotation_type_case_studies.py` - Annotation type model case studies
- `comprehensive_annotation_type_evaluation.py` - Comprehensive evaluation framework
- `annotation_type_evaluation.py` - Annotation type evaluation utilities
- `annotation_type_prediction.py` - Annotation type prediction utilities

### Training and Hyperparameter Optimization
- `enhanced_rl_training.py` - Enhanced RL training framework
- `rl_annotation_type_training.py` - RL training for annotation types
- `rl_pipeline.py` - RL training pipeline
- `hyperparameter_search_annotation_types.py` - Hyperparameter search for annotation types
- `simple_hyperparameter_search_annotation_types.py` - Simplified hyperparameter search

### Configuration and Data
- `annotation_type_config.json` - Configuration for annotation type models
- `requirements.txt` - Python dependencies
- `index1.out` - Sample Checker Framework warnings file
- `index1.small.out` - Smaller sample warnings file
- `hyperparameter_search_annotation_types_results_20250927_224114.json` - Hyperparameter search results
- `simple_hyperparameter_search_annotation_types_results_20250927_224445.json` - Simplified search results

### Documentation
- `ANNOTATION_TYPE_MODELS_GUIDE.md` - Comprehensive guide for annotation type models
- `COMPREHENSIVE_CASE_STUDY_RESULTS.md` - Results from comprehensive case studies

### Directories
- `models_annotation_types/` - Trained annotation type models
- `predictions_annotation_types/` - Prediction results and reports

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt includes essential dependencies, but you may need additional packages:
- `torch` - PyTorch for neural network models
- `torch-geometric` - PyTorch Geometric for graph neural networks
- `javalang` - Java language parser
- `sklearn` - Scikit-learn for machine learning
- `joblib` - For model serialization
- `numpy` - Numerical computing
- `pathlib` - Path utilities

Install with:
```bash
pip install torch torch-geometric javalang scikit-learn joblib numpy
```

### 2. Train Annotation Type Models
```bash
# Train @Positive model
python annotation_type_rl_positive.py --episodes 50 --base_model gcn \
  --project_root /path/to/java/project

# Train @NonNegative model  
python annotation_type_rl_nonnegative.py --episodes 50 --base_model gcn \
  --project_root /path/to/java/project

# Train @GTENegativeOne model
python annotation_type_rl_gtenegativeone.py --episodes 50 --base_model gcn \
  --project_root /path/to/java/project
```

### 3. Use Simplified Pipeline
```bash
# Train all annotation type models
python simple_annotation_type_pipeline.py --mode train --episodes 50 --base_model gcn \
  --project_root /path/to/java/project

# Predict annotations
python simple_annotation_type_pipeline.py --mode predict --target_file /path/to/MyClass.java
```

### 4. Run Case Studies
```bash
# Run binary RL case studies
python run_case_studies.py

# Run annotation type case studies
python annotation_type_case_studies.py
```

## Architecture Overview

The annotation type models use a two-stage approach:

1. **Binary Stage**: Binary RL models predict whether an annotation should be placed
2. **Type Stage**: Annotation type models predict the specific annotation type (@Positive, @NonNegative, @GTENegativeOne)

This ensures that only valid annotation targets are considered for type prediction.

## Supported Annotation Types

- **@Positive**: For values that must be greater than zero (e.g., count, size, length)
- **@NonNegative**: For values that must be greater than or equal to zero (e.g., index, offset, position)
- **@GTENegativeOne**: For values that must be greater than or equal to -1 (e.g., capacity, limit, bound)

## Model Performance

Based on comprehensive testing:
- **Binary RL Models**: 6/6 models successfully trained (100% success rate)
- **Annotation Type Models**: 16/18 models successfully trained (89% success rate)
- **Model Consensus**: 100% agreement across all models on annotation placement
- **F1 Scores**: 1.000 for HGT, GBT, and Causal models

## Key Features

- **Node-Level Processing**: All models work at individual node level with semantic filtering
- **Two-Stage Prediction**: Binary classification followed by type-specific prediction
- **Comprehensive Evaluation**: Extensive testing on real-world projects (Guava, JFreeChart, Plume-lib)
- **Production Ready**: Robust error handling and comprehensive logging
- **Manual Inspection**: JSON and human-readable reports for validation

## Environment Variables

Configure the system using these environment variables:

```bash
# Core directories
export SLICES_DIR="/path/to/slices"
export CFG_OUTPUT_DIR="/path/to/cfg_output"  
export MODELS_DIR="/path/to/models"
export AUGMENTED_SLICES_DIR="/path/to/slices_aug"

# Checker Framework
export CHECKERFRAMEWORK_HOME="/path/to/checker-framework-3.42.0"
export CHECKERFRAMEWORK_CP="/path/to/checker-qual.jar:/path/to/checker.jar"
```

## Troubleshooting

1. **Model Not Found Error**: Ensure models are trained before prediction
2. **Dimension Mismatch Error**: Check that all models use consistent feature dimensions
3. **No Predictions Generated**: Verify that binary RL models are working and Java files contain relevant keywords

For detailed information, see `ANNOTATION_TYPE_MODELS_GUIDE.md` and `COMPREHENSIVE_CASE_STUDY_RESULTS.md`.
