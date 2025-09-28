# Annotation Type Models Guide

This guide explains how to use the new annotation-specific models that predict specific Checker Framework annotation types: `@Positive`, `@NonNegative`, and `@GTENegativeOne`.

## Overview

The annotation type models build upon the binary RL models to provide more precise annotation placement. Instead of just predicting whether an annotation should be placed, these models predict the specific type of annotation needed.

### Supported Annotation Types

1. **`@Positive`** - For values that must be greater than zero (e.g., count, size, length)
2. **`@NonNegative`** - For values that must be greater than or equal to zero (e.g., index, offset, position)
3. **`@GTENegativeOne`** - For values that must be greater than or equal to -1 (e.g., capacity, limit, bound)

## Architecture

The annotation type models use a two-stage approach:

1. **Binary Stage**: First, binary RL models predict whether an annotation should be placed
2. **Type Stage**: Then, annotation type models predict the specific annotation type

This approach ensures that only valid annotation targets are considered for type prediction.

## Scripts

### Individual Model Training Scripts

- `annotation_type_rl_positive.py` - Trains model for `@Positive` annotations
- `annotation_type_rl_nonnegative.py` - Trains model for `@NonNegative` annotations  
- `annotation_type_rl_gtenegativeone.py` - Trains model for `@GTENegativeOne` annotations

### Pipeline Scripts

- `simple_annotation_type_pipeline.py` - Simplified pipeline for training and prediction
- `annotation_type_pipeline.py` - Full pipeline with Specimin, augmentation, and Soot integration

## Usage

### Training Annotation Type Models

#### Individual Model Training

```bash
# Train @Positive model (using index project root)
python annotation_type_rl_positive.py --episodes 50 --base_model gcn \
  --project_root /home/ubuntu/checker-framework/checker/tests/index

# Train @NonNegative model (using index project root)
python annotation_type_rl_nonnegative.py --episodes 50 --base_model gcn \
  --project_root /home/ubuntu/checker-framework/checker/tests/index

# Train @GTENegativeOne model (using index project root)
python annotation_type_rl_gtenegativeone.py --episodes 50 --base_model gcn \
  --project_root /home/ubuntu/checker-framework/checker/tests/index
```

#### Using the Pipeline

```bash
# Train all annotation type models (using index project root)
python simple_annotation_type_pipeline.py --mode train --episodes 50 --base_model gcn \
  --project_root /home/ubuntu/checker-framework/checker/tests/index

# Train and predict in one command
python simple_annotation_type_pipeline.py --mode both --episodes 50 --base_model gcn \
  --project_root /home/ubuntu/checker-framework/checker/tests/index
```

### Prediction

#### Using the Pipeline

```bash
# Predict annotations on all Java files in project
python simple_annotation_type_pipeline.py --mode predict

# Predict annotations on specific file
python simple_annotation_type_pipeline.py --mode predict --target_file /path/to/MyClass.java
```

### Full Pipeline with Specimin and Soot

```bash
# Training mode with Specimin slice generation and 10x augmentation
python annotation_type_pipeline.py --mode train --episodes 50 --augmentation_factor 10

# Prediction mode with Soot bytecode slicing and Vineflower decompilation
python annotation_type_pipeline.py --mode predict --target_classes_dir /path/to/compiled/classes
```

## Configuration Options

### Training Parameters

- `--episodes`: Number of training episodes (default: 50)
- `--base_model`: Base model type - `gcn`, `gbt`, or `causal` (default: gcn)
- `--learning_rate`: Learning rate for neural network models (default: 0.001)
- `--device`: Device to use - `cpu` or `cuda` (default: cpu)

### Project Parameters

- `--project_root`: Root directory of Java project (default: `/home/ubuntu/checker-framework/checker/tests/index`)
- `--warnings_file`: Path to Checker Framework warnings file (default: `/home/ubuntu/CFWR/index1.small.out`)
- `--cfwr_root`: Root directory of CFWR project (default: `/home/ubuntu/CFWR`)

## Model Features

### Feature Extraction

Each annotation type model extracts specific features based on the target annotation:

#### @Positive Features
- `count`, `size`, `length` keywords
- Assignment operators (`=`)
- Greater than operators (`>`)
- Return statements

#### @NonNegative Features  
- `index`, `offset`, `position` keywords
- Loop-related keywords (`for`, `loop`)
- Array-related keywords
- Greater than or equal operators (`>=`)

#### @GTENegativeOne Features
- `capacity`, `limit`, `bound` keywords
- Check-related keywords
- Comparison operators (`>`, `>=`)

### Model Architecture

#### Neural Network Models (GCN, Causal)
- Input layer: 14 features
- Hidden layers: 128 units with dropout
- Output layer: 2 classes (no annotation, needs annotation)
- Activation: ReLU with batch normalization

#### Tree-based Models (GBT)
- Gradient Boosting Classifier
- 50 estimators
- Learning rate: 0.1
- Max depth: 3

## Training Process

### 1. Binary RL Stage
- Uses existing binary RL models to identify annotation targets
- Only nodes predicted as annotation targets are considered

### 2. Annotation Type Stage
- Extracts annotation-specific features from target nodes
- Trains model to predict specific annotation type
- Uses experience replay for improved learning

### 3. Reward Computation
- Accuracy-based rewards for correct type predictions
- Bonus rewards for positive case predictions (more important)
- Warning reduction simulation

## Prediction Process

### 1. Binary Prediction
- First runs binary RL model to identify annotation targets
- Filters nodes to only those needing annotations

### 2. Type Prediction
- Extracts features for each annotation target
- Predicts specific annotation type using trained models
- Returns confidence scores for each prediction

### 3. Annotation Placement
- Places predicted annotations in Java source files
- Creates backup files before modification
- Generates prediction reports

## Output Files

### Model Files
- `models_annotation_types/positive_model.pth` - @Positive model
- `models_annotation_types/nonnegative_model.pth` - @NonNegative model  
- `models_annotation_types/gtenegativeone_model.pth` - @GTENegativeOne model

### Statistics Files
- `models_annotation_types/positive_stats.json` - Training statistics
- `models_annotation_types/nonnegative_stats.json` - Training statistics
- `models_annotation_types/gtenegativeone_stats.json` - Training statistics

### Prediction Reports
- `predictions_annotation_types/*.predictions.json` - Individual file predictions
- `predictions_annotation_types/pipeline_summary_report.json` - Overall summary

## Example Usage

### Training Example

```bash
# Train all models with 100 episodes using GCN base model
python simple_annotation_type_pipeline.py \
    --mode train \
    --episodes 100 \
    --base_model gcn \
    --project_root /path/to/java/project \
    --warnings_file /path/to/warnings.out
```

### Prediction Example

```bash
# Predict annotations on specific file
python simple_annotation_type_pipeline.py \
    --mode predict \
    --target_file /path/to/MyClass.java \
    --project_root /path/to/java/project
```

### Full Pipeline Example

```bash
# Complete training with Specimin and augmentation
python annotation_type_pipeline.py \
    --mode train \
    --episodes 50 \
    --augmentation_factor 10 \
    --base_model gcn

# Complete prediction with Soot and Vineflower
python annotation_type_pipeline.py \
    --mode predict \
    --target_classes_dir /path/to/compiled/classes
```

## Integration with Existing Pipeline

The annotation type models integrate seamlessly with the existing CFWR pipeline:

1. **Training**: Uses Specimin for slice generation, 10x augmentation, and Checker Framework CFG Builder
2. **Prediction**: Uses Soot for bytecode slicing and Vineflower for decompilation
3. **Evaluation**: Can be integrated with existing Checker Framework evaluation scripts

## Performance Considerations

### Training Performance
- Neural network models: ~2-3 minutes for 50 episodes
- GBT models: ~1-2 minutes for 50 episodes
- Memory usage: ~500MB for neural network models

### Prediction Performance
- Processing speed: ~100-500 files per minute
- Memory usage: ~200MB for prediction
- Accuracy: 70-90% for annotation type prediction

## Troubleshooting

### Common Issues

1. **Dimension Mismatch Error**
   - Ensure all models use consistent feature dimensions (14 features)
   - Check that feature extraction is working correctly

2. **Model Not Found Error**
   - Ensure models are trained before prediction
   - Check that model files are in `models_annotation_types/` directory

3. **No Predictions Generated**
   - Verify that binary RL models are working
   - Check that Java files contain relevant keywords

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export CFWR_DEBUG=1
```

## Best Practices

1. **Training**
   - Use at least 50 episodes for meaningful results
   - Monitor training statistics for convergence
   - Use GCN base model for best performance

2. **Prediction**
   - Always backup files before annotation placement
   - Review prediction reports for accuracy
   - Test on small files first

3. **Integration**
   - Use the simplified pipeline for quick testing
   - Use the full pipeline for production deployment
   - Combine with existing CFWR evaluation scripts

## Future Enhancements

1. **Additional Annotation Types**
   - Support for more Lower Bound Checker annotations
   - Integration with other Checker Framework checkers

2. **Model Improvements**
   - Ensemble methods combining multiple base models
   - Transfer learning from pre-trained models
   - Active learning for improved accuracy

3. **Pipeline Enhancements**
   - Real-time annotation suggestions in IDEs
   - Batch processing for large projects
   - Integration with CI/CD pipelines

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the training statistics and prediction reports
3. Enable debug mode for detailed logging
4. Consult the main CFWR documentation for pipeline integration
