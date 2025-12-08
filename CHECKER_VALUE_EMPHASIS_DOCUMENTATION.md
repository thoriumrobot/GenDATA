# Checker Value Emphasis: Automatic Learning Documentation

## Overview

This system implements automatic learning of relevant values to emphasize for each Checker Framework checker. Instead of manually specifying feature scaling (like the 1.5x-3.0x scaling for "could be zero" features), models learn during training which values boost performance and automatically emphasize them.

## Architecture

### Core Components

1. **Value Pattern Detector** (`value_pattern_detector.py`)
   - Extracts raw value patterns from CFG nodes
   - Detects checker-relevant patterns (0, -1, null, strings, etc.)
   - Returns feature dictionaries per checker

2. **Checker Value Attention** (`checker_value_attention.py`)
   - Learnable attention module that emphasizes relevant values
   - Multi-head self-attention for learning value importance
   - Replaces manual feature scaling with learned emphasis weights

3. **Checker-Specific Models** (`checker_specific_models.py`)
   - Separate model classes per checker
   - Integrates value attention with base models (GCN, HGT, GBT, etc.)
   - Factory function for creating checker-specific models

4. **Checker Configuration** (`checker_config.py`)
   - Defines checker-specific configurations
   - Maps checkers to target values and patterns
   - Centralized checker metadata

## Supported Checkers

### Lower Bound Checker (Index Checker)
- **Target Values**: 0, -1
- **Annotation Types**: @Positive, @NonNegative, @GTENegativeOne
- **Value Patterns**: zero, negative_one, positive, nonnegative, array_index, loop_variable, subtraction_result, compared_with_length, initialized_to_zero, offset_position

### Null Checker
- **Target Values**: null
- **Annotation Types**: @Nullable, @NonNull
- **Value Patterns**: null_literal, null_check, nullable_type, null_assignment, null_return, null_parameter, null_comparison, null_dereference

### Signature String Checker
- **Target Values**: string
- **Annotation Types**: @ClassGetName, @MethodGetName, @FullyQualifiedName
- **Value Patterns**: string_literal, string_operation, signature_pattern, class_name, method_name, fully_qualified_name, string_concatenation, string_comparison

### Interning Checker
- **Target Values**: interned_string
- **Annotation Types**: @Interned, @InternedDistinct
- **Value Patterns**: interned_string, string_constant, string_comparison, intern_method_call, string_literal, constant_string

### Lock Checker
- **Target Values**: lock
- **Annotation Types**: @GuardedBy, @Holding, @ReleasesNoLocks
- **Value Patterns**: lock_operation, synchronized_block, lock_variable, lock_acquire, lock_release, synchronization_pattern

### Regex Checker
- **Target Values**: regex_pattern
- **Annotation Types**: @Regex, @RegexBottom
- **Value Patterns**: regex_pattern, pattern_matching, string_pattern, regex_literal, pattern_compile, matcher_operation

## How It Works

### Training Process

1. **Feature Extraction**: Raw value patterns are extracted from CFG nodes using `ValuePatternDetector`
2. **Attention Learning**: Patterns are passed through `CheckerValueAttention` module
3. **Automatic Emphasis**: Attention weights learn which patterns to emphasize during backpropagation
4. **Model Training**: Checker-specific models train with emphasized features
5. **Interpretability**: Attention weights show which values are being emphasized

### Value Pattern Detection

For each checker, the system detects relevant patterns:

- **Lower Bound Checker**: Detects 0, -1, positive numbers, nonnegative patterns, array indices
- **Null Checker**: Detects null literals, null checks, nullable types
- **Signature String Checker**: Detects string literals, string operations, signature patterns
- **Interning Checker**: Detects interned strings, string constants
- **Lock Checker**: Detects lock operations, synchronization patterns
- **Regex Checker**: Detects regex patterns, pattern matching

### Attention Mechanism

The `CheckerValueAttention` module uses:
- **Multi-head self-attention**: Learns different aspects of value importance
- **Learnable emphasis weights**: Replaces manual 1.5x-3.0x scaling
- **Feed-forward refinement**: Processes attention output
- **Residual connections**: Preserves original pattern information

## Integration Points

### Feature Extraction

- **`improved_balanced_dataset_generator.py`**: Adds checker-aware feature extraction
- **`cfg_graph.py`**: Adds checker value patterns to graph node features
- **`dg2n_adapter.py`**, **`gcsn_adapter.py`**: Add checker value features to adapters
- **`enhanced_causal_model.py`**: Integrates checker-aware causal features

### Model Training

- **`improved_balanced_annotation_type_trainer.py`**: Supports checker types
- **`annotation_type_rl_*.py`**: Integrate checker-specific models
- **`train_checker_specific_models.py`**: Training script for checker-specific models

### Evaluation

- **`evaluate_checker_emphasis.py`**: Analyzes learned attention weights
- Visualizes which values are emphasized
- Compares with manual "could be zero" features

## Usage

### Training Checker-Specific Models

```bash
# Train models for all checkers
python train_checker_specific_models.py \
  --output_dir checker_specific_models \
  --balanced_dataset_dir real_balanced_datasets \
  --base_model_types gbt causal enhanced_causal \
  --epochs 200

# Train specific checkers
python train_checker_specific_models.py \
  --checkers INDEX NULLNESS SIGNATURE \
  --base_model_types gbt causal
```

### Evaluating Learned Emphasis

```bash
# Analyze attention weights
python evaluate_checker_emphasis.py \
  --results_file checker_specific_models/checker_training_results.json
```

### Using in Code

```python
from checker_config import CheckerType
from checker_specific_models import create_checker_specific_model

# Create Lower Bound Checker model
model = create_checker_specific_model(
    checker_type=CheckerType.INDEX,
    base_model_type='causal',
    input_dim=24,  # base features + pattern features
    hidden_dim=128,
    out_dim=2
)

# Get attention summary
attention_summary = model.get_attention_summary()
print(f"Learned emphasis: {attention_summary}")
```

## Expected Outcomes

1. **Automatic Value Discovery**: Models learn which values matter for each checker
2. **Improved Performance**: Better accuracy by emphasizing relevant values
3. **Checker-Specific Optimization**: Each checker gets optimized value emphasis
4. **Interpretability**: Attention weights show which values are emphasized
5. **Extensibility**: Easy to add new checkers or value patterns

## Comparison with Manual Features

### Manual "Could Be Zero" Features
- Fixed scaling: 1.5x - 3.0x
- Hand-crafted patterns
- Same for all models
- Requires domain knowledge

### Learned Value Emphasis
- Adaptive scaling: Learned during training
- Automatically discovered patterns
- Checker-specific emphasis
- No manual tuning required

## Files

### Core Implementation
- `checker_config.py`: Checker configurations
- `value_pattern_detector.py`: Pattern detection
- `checker_value_attention.py`: Attention mechanism
- `checker_specific_models.py`: Model architectures

### Integration
- `improved_balanced_dataset_generator.py`: Feature extraction
- `cfg_graph.py`: Graph feature integration
- `improved_balanced_annotation_type_trainer.py`: Training support
- `annotation_type_rl_*.py`: Trainer integration
- `dg2n_adapter.py`, `gcsn_adapter.py`: Adapter integration
- `enhanced_causal_model.py`: Causal model integration

### Scripts
- `train_checker_specific_models.py`: Training script
- `evaluate_checker_emphasis.py`: Evaluation script

## Future Work

1. **Graph-Based Models**: Extend to GCN, HGT, GCSN with graph attention
2. **Cross-Checker Learning**: Share patterns across related checkers
3. **Dynamic Emphasis**: Adjust emphasis based on context
4. **Pattern Discovery**: Automatically discover new value patterns
5. **Transfer Learning**: Transfer learned emphasis to new checkers

## References

- **"Could Be Zero" Features**: `COULD_BE_ZERO_FEATURES_DOCUMENTATION.md`
- **Enhanced Pipeline**: `ENHANCED_PIPELINE_DOCUMENTATION.md`
- **Ablation Studies**: `ABLATION_STUDY_RESULTS.md`

