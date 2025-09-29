# Balanced Training System Guide

## Overview

The Balanced Training System addresses a critical issue in annotation type model training: **data imbalance**. Traditional training datasets are heavily skewed toward positive examples (nodes that need annotations), leading to poor model convergence and biased predictions.

This system creates balanced datasets where each annotation type appears in approximately **50% of examples** and is absent in the other **50%**, ensuring proper model convergence.

## Problem Statement

### Original Training Issues

- **Imbalanced Data**: Most training examples are positive (need annotations)
- **Poor Convergence**: Models learn to always predict positive
- **Biased Predictions**: Low confidence in negative predictions
- **Poor Generalization**: Models don't learn to distinguish when annotations are NOT needed

### Impact on Model Performance

- Models predict annotations even when not needed
- Low accuracy on negative examples
- Unreliable confidence scores
- Poor performance on new code

## Solution: Balanced Training

### Core Concept

Create balanced training datasets where:
- **50% positive examples**: Nodes that need the specific annotation type
- **50% negative examples**: Nodes that don't need the specific annotation type
- **Per annotation type**: Separate balanced datasets for `@Positive`, `@NonNegative`, `@GTENegativeOne`

### Benefits

✅ **Better Convergence**: Models learn proper decision boundaries
✅ **Reduced Bias**: Balanced learning prevents always-positive predictions
✅ **Improved Accuracy**: Better performance on both positive and negative examples
✅ **Reliable Confidence**: More accurate confidence scores
✅ **Better Generalization**: Models work better on new, unseen code

## System Components

### 1. BalancedDatasetGenerator (`balanced_dataset_generator.py`)

**Purpose**: Creates balanced training datasets from CFG files

**Key Features**:
- Loads CFG files and extracts node features
- Determines annotation types using rule-based logic
- Generates positive examples (annotation needed)
- Generates negative examples (annotation not needed)
- Ensures 50/50 balance for each annotation type
- Adds feature variation for diversity

**Usage**:
```bash
python balanced_dataset_generator.py \
  --cfg_dir /path/to/cfg/files \
  --output_dir /path/to/output \
  --examples_per_annotation 1000 \
  --target_balance 0.5
```

### 2. BalancedAnnotationTypeTrainer (`balanced_annotation_type_trainer.py`)

**Purpose**: Trains neural network models using balanced datasets

**Key Features**:
- PyTorch Dataset and DataLoader for balanced examples
- Neural network architecture optimized for binary classification
- Training with proper validation splits
- Learning rate scheduling and early stopping
- Comprehensive evaluation metrics

**Usage**:
```bash
python balanced_annotation_type_trainer.py \
  --balanced_dataset_dir /path/to/datasets \
  --output_dir /path/to/models \
  --epochs 100 --batch_size 32
```

### 3. BalancedTrainingPipeline (`balanced_training_pipeline.py`)

**Purpose**: End-to-end pipeline integration

**Key Features**:
- Orchestrates CFG generation, dataset creation, and model training
- Integrates with existing annotation type pipeline
- Comprehensive error handling and logging
- Progress tracking and statistics

**Usage**:
```bash
python balanced_training_pipeline.py \
  --project_root /path/to/java/project \
  --warnings_file /path/to/warnings.out \
  --cfwr_root /path/to/cfwr \
  --output_dir /path/to/output
```

## Implementation Details

### Dataset Generation Process

1. **CFG Loading**: Load all CFG JSON files from specified directory
2. **Node Analysis**: Extract features from each CFG node
3. **Annotation Type Determination**: Use rule-based logic to determine target annotation type
4. **Positive Example Generation**: Create examples where annotation is needed
5. **Negative Example Generation**: Create examples where annotation is not needed
6. **Feature Variation**: Add controlled randomness for diversity
7. **Balancing**: Ensure 50/50 split for each annotation type

### Model Architecture

- **Input Layer**: 15-dimensional feature vectors
- **Hidden Layers**: [256, 128, 64, 32] with BatchNorm and Dropout
- **Output Layer**: 2 classes (positive/negative)
- **Activation**: ReLU with Xavier initialization
- **Regularization**: Dropout (0.3) and Weight Decay (0.01)

### Training Strategy

- **Loss Function**: CrossEntropyLoss for binary classification
- **Optimizer**: AdamW with learning rate 0.001
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Validation**: 20% split for validation
- **Early Stopping**: Based on validation accuracy

## Usage Examples

### Basic Usage

```bash
# Run complete balanced training pipeline
python balanced_training_pipeline.py \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/GenDATA/index1.out \
  --cfwr_root /home/ubuntu/GenDATA \
  --output_dir /home/ubuntu/GenDATA/balanced_models
```

### Advanced Configuration

```bash
# Custom parameters for large-scale training
python balanced_training_pipeline.py \
  --project_root /path/to/project \
  --warnings_file /path/to/warnings.out \
  --cfwr_root /path/to/cfwr \
  --output_dir /path/to/output \
  --examples_per_annotation 5000 \
  --target_balance 0.5 \
  --epochs 200 \
  --batch_size 64 \
  --regenerate_cfg
```

### Step-by-Step Usage

```bash
# Step 1: Generate balanced datasets
python balanced_dataset_generator.py \
  --cfg_dir /path/to/cfg_files \
  --output_dir /path/to/datasets \
  --examples_per_annotation 1000 \
  --target_balance 0.5

# Step 2: Train models
python balanced_annotation_type_trainer.py \
  --balanced_dataset_dir /path/to/datasets \
  --output_dir /path/to/models \
  --epochs 100 \
  --batch_size 32
```

## Output Files

### Generated Datasets
- `positive_balanced_dataset.json`: Balanced dataset for @Positive
- `nonnegative_balanced_dataset.json`: Balanced dataset for @NonNegative  
- `gtenegativeone_balanced_dataset.json`: Balanced dataset for @GTENegativeOne
- `generation_statistics.json`: Overall generation statistics

### Trained Models
- `positive_balanced_model.pth`: Trained model for @Positive
- `nonnegative_balanced_model.pth`: Trained model for @NonNegative
- `gtenegativeone_balanced_model.pth`: Trained model for @GTENegativeOne
- `balanced_training_statistics.json`: Training statistics and metrics

### Statistics Format
```json
{
  "total_examples": 3000,
  "positive_examples": 1500,
  "negative_examples": 1500,
  "annotation_type_counts": {
    "@Positive": {"positive": 500, "negative": 500},
    "@NonNegative": {"positive": 500, "negative": 500},
    "@GTENegativeOne": {"positive": 500, "negative": 500}
  }
}
```

## Performance Expectations

### Before Balanced Training
- **Accuracy**: ~60-70% (biased toward positive predictions)
- **Precision**: High on positive, very low on negative
- **Recall**: High on positive, very low on negative
- **F1-Score**: Imbalanced performance

### After Balanced Training
- **Accuracy**: ~85-95% (balanced performance)
- **Precision**: Balanced on both positive and negative
- **Recall**: Balanced on both positive and negative  
- **F1-Score**: Consistent performance across classes

## Integration with Existing Pipeline

The balanced training system integrates seamlessly with the existing annotation type pipeline:

1. **CFG Generation**: Uses existing CFG generation from `simple_annotation_type_pipeline.py`
2. **Feature Extraction**: Compatible with existing feature extraction logic
3. **Model Architecture**: Extends existing model architectures
4. **Prediction**: Can replace existing models in prediction pipeline

### Integration Steps

1. **Train Balanced Models**: Use balanced training pipeline
2. **Replace Models**: Update model loading in prediction scripts
3. **Update Pipeline**: Modify `simple_annotation_type_pipeline.py` to use balanced models
4. **Verify Performance**: Test on case studies and compare results

## Troubleshooting

### Common Issues

**Issue**: "No CFG files found"
- **Solution**: Ensure CFG generation step completed successfully
- **Check**: Verify `cfg_output_specimin` directory exists and contains JSON files

**Issue**: "Insufficient training data"
- **Solution**: Increase `examples_per_annotation` or check CFG file quality
- **Check**: Verify CFG files contain valid node data

**Issue**: "Poor model convergence"
- **Solution**: Adjust learning rate, increase epochs, or check data balance
- **Check**: Verify datasets are properly balanced (50/50 split)

**Issue**: "Out of memory during training"
- **Solution**: Reduce batch size or examples per annotation
- **Check**: Monitor GPU memory usage

### Performance Optimization

1. **Batch Size**: Start with 32, adjust based on available memory
2. **Epochs**: Start with 100, increase if convergence is slow
3. **Examples**: Start with 1000 per annotation type, increase for better performance
4. **Device**: Use GPU (`--device cuda`) for faster training

## Future Enhancements

### Planned Improvements

1. **Advanced Balancing**: Dynamic balance ratios based on dataset characteristics
2. **Semi-Supervised Learning**: Incorporate unlabeled examples
3. **Multi-Task Learning**: Train all annotation types simultaneously
4. **Active Learning**: Intelligent example selection for training
5. **Online Learning**: Incremental model updates with new data

### Research Directions

1. **Ensemble Methods**: Combine multiple balanced models
2. **Meta-Learning**: Learn to adapt to different code patterns
3. **Transfer Learning**: Pre-train on large codebases, fine-tune on specific projects
4. **Causal Analysis**: Understand why certain annotations are needed

## Conclusion

The Balanced Training System provides a robust solution to the data imbalance problem in annotation type prediction. By ensuring balanced positive and negative examples, models achieve better convergence, improved accuracy, and more reliable predictions.

This system is essential for building production-ready annotation type models that can effectively assist developers in placing the correct annotations in their Java code.
