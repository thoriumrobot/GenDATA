# Predictions Viewer Guide

This guide explains how to use the `show_predictions_with_context.py` script to visualize model predictions alongside their code context.

## Overview

The predictions viewer script helps you understand what the annotation type models are predicting by showing:
- The actual code where predictions were made
- Model predictions with confidence scores
- Line numbers and context around each prediction
- Filtering options for meaningful analysis

## Quick Start

### 1. Show Summary of All Predictions
```bash
python show_predictions_with_context.py --summary
```

### 2. Show Predictions for a Specific File
```bash
python show_predictions_with_context.py --file predictions_annotation_types/Collections.java.predictions.json
```

### 3. Show High Confidence Predictions Only
```bash
python show_predictions_with_context.py --confidence-threshold 0.8
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--predictions_dir` | Directory containing prediction files | `predictions_annotation_types` |
| `--file` | Show predictions for a specific file only | None (show all files) |
| `--no-context` | Don't show code context | False (show context) |
| `--context-lines` | Number of context lines to show around each prediction | 3 |
| `--confidence-threshold` | Minimum confidence threshold for showing predictions | 0.6 (60%) |
| `--filter-model` | Filter predictions by model type | None (show all models) |
| `--filter-annotation` | Filter predictions by annotation type | None (show all annotations) |
| `--summary` | Show prediction summary only | False |

## Usage Examples

### Basic Usage

```bash
# Show all predictions with code context
python show_predictions_with_context.py

# Show predictions for a specific file
python show_predictions_with_context.py --file predictions_annotation_types/String.java.predictions.json

# Show only high confidence predictions (80%+)
python show_predictions_with_context.py --confidence-threshold 0.8

# Show more context around each prediction
python show_predictions_with_context.py --context-lines 5
```

### Filtering Examples

```bash
# Show only predictions from enhanced_causal model
python show_predictions_with_context.py --filter-model enhanced_causal

# Show only @Positive annotation predictions
python show_predictions_with_context.py --filter-annotation @Positive

# Show only high confidence @NonNegative predictions from GCN model
python show_predictions_with_context.py --filter-model gcn --filter-annotation @NonNegative --confidence-threshold 0.7
```

### Analysis Examples

```bash
# Compare different models on the same file
python show_predictions_with_context.py --file predictions_annotation_types/Collections.java.predictions.json --filter-model enhanced_causal
python show_predictions_with_context.py --file predictions_annotation_types/Collections.java.predictions.json --filter-model gcn

# Find the most confident predictions across all files
python show_predictions_with_context.py --confidence-threshold 0.9

# Show predictions without code context (faster for large files)
python show_predictions_with_context.py --no-context
```

## Understanding the Output

### File Header
```
================================================================================
📄 FILE: /path/to/JavaFile.java
🎯 PREDICTIONS: 1279 total, 305 meaningful
================================================================================
```
- Shows the Java file being analyzed
- Total predictions vs meaningful predictions (filtered by confidence and content)

### Prediction Display
```
📍 LINE 146
----------------------------------------
  144: 
  145:     @Positive
→ 146:     public static <T> int binarySearch(List<? extends Comparable<? super T>> list, T key);
  147: 
  148:     @Positive

🎯 PREDICTIONS:
  🤖 enhanced_causal → @Positive on 'unknown' (confidence: 0.552)
```

- **📍 LINE 146**: The line number where the prediction was made
- **Context lines**: Shows surrounding code for context
- **→ 146**: Highlights the target line with an arrow
- **🎯 PREDICTIONS**: Shows the model predictions with:
  - Model type (enhanced_causal)
  - Annotation type (@Positive)
  - Confidence score (0.552 = 55.2%)

## Meaningful Prediction Filtering

The script automatically filters out predictions that are likely not meaningful:

### Excluded Predictions
- **Empty lines**: Lines with no content
- **Comment lines**: Lines starting with `//`, `*`, or `/*`
- **Package/import statements**: `package` and `import` declarations
- **Standalone annotations**: Lines containing only `@Positive`, `@NonNegative`, etc.
- **Low confidence**: Predictions below the confidence threshold

### Included Predictions
- **Method signatures**: Actual method declarations
- **Variable declarations**: Field and local variable declarations
- **Statements**: Code statements and expressions
- **High confidence**: Predictions above the confidence threshold

## Model Types and Annotation Types

### Available Model Types
- `enhanced_causal` - Enhanced causal model
- `enhanced_graph_causal` - Enhanced graph causal model
- `graph_causal` - Graph causal model
- `graphite` - GraphITE model
- `causal` - Basic causal model
- `hgt` - Heterogeneous Graph Transformer
- `gcn` - Graph Convolutional Network
- `gbt` - Gradient Boosting Tree
- `gcsn` - Graph Convolutional Skip Network
- `dg2n` - Deep Graph-to-Graph Network

### Available Annotation Types
- `@Positive` - Positive value expected
- `@NonNegative` - Non-negative value expected
- `@GTENegativeOne` - Greater than or equal to negative one

## Tips for Analysis

### 1. Start with Summary
```bash
python show_predictions_with_context.py --summary
```
This gives you an overview of all predictions across models and annotation types.

### 2. Focus on High Confidence Predictions
```bash
python show_predictions_with_context.py --confidence-threshold 0.8
```
High confidence predictions are more likely to be accurate and meaningful.

### 3. Compare Models
```bash
# Compare different models on the same file
python show_predictions_with_context.py --file predictions_annotation_types/Collections.java.predictions.json --filter-model enhanced_causal
python show_predictions_with_context.py --file predictions_annotation_types/Collections.java.predictions.json --filter-model gcn
```

### 4. Analyze Specific Annotation Types
```bash
# Focus on @Positive predictions
python show_predictions_with_context.py --filter-annotation @Positive --confidence-threshold 0.7
```

### 5. Use Appropriate Context
- **Large files**: Use `--context-lines 1` or `--no-context` for faster processing
- **Detailed analysis**: Use `--context-lines 5` for more context around each prediction

## Troubleshooting

### No Predictions Found
If you see "No meaningful predictions found":
1. Check if prediction files exist in the predictions directory
2. Lower the confidence threshold: `--confidence-threshold 0.3`
3. Check if the Java files are accessible for context

### File Not Found
If you get "file not found" errors:
1. Verify the file path is correct
2. Check if the predictions directory exists
3. Ensure you have read permissions on the files

### Performance Issues
For large prediction files:
1. Use `--no-context` to skip code context display
2. Use `--confidence-threshold` to filter predictions
3. Use `--context-lines 1` to reduce context display

## Demo Script

Run the demo script to see examples:
```bash
python demo_predictions_viewer.py
```

This will show various usage examples and help you get started with the predictions viewer.
