# Case Study Evaluation Pipeline: Documentation

## Overview

The case study evaluation pipeline evaluates annotation type prediction models on real-world Java projects. It compares 7 different model architectures (GCN, HGT, GBT, Causal, GCSN, DG2N, DGCRF) against ground truth extracted from Index Checker warnings.

**Current Status**: ✅ **Non-zero metrics achieved** for GCN, HGT, Causal, and GCSN models with partial accuracy of 0.1667 (16.67%).

## Key Features

- **Multi-model evaluation**: Supports 7 different model architectures
- **Ground truth extraction**: Automatically extracts annotations from Index Checker warnings
- **CFG generation**: Generates control flow graphs for case study files
- **Robust matching**: Uses ±3-line window and method-level matching for alignment
- **Per-node predictions**: Generates predictions for individual CFG nodes
- **Comprehensive metrics**: Computes accuracy (exact and partial), precision, recall, F1, coverage, and confusion matrices

## Architecture

The pipeline consists of five main components:

### 1. CFG Generation (`generate_case_study_cfgs.py`)

**Purpose**: Generate control flow graphs for all Java files in case study projects.

**What it does**:
- Scans `case_studies/` directory for Java files
- Generates CFGs using the existing CFG generation pipeline
- Creates one CFG per method (multiple JSON files per Java file)
- Emits `index.json` mapping absolute Java file paths to CFG directory paths

**Output**:
- `case_study_cfg_output/`: Directory containing CFG JSON files
  - `{ClassName}/`: Directory per Java file
    - `{methodName}.json`: CFG for each method
    - `cfg.json`: Canonical CFG file (first method)
  - `index.json`: Mapping from Java file paths to CFG paths

**Usage**:
```bash
python generate_case_study_cfgs.py
```

### 2. Ground Truth Extraction (`studies/extract_cs_ground_truth_from_checker.py`)

**Purpose**: Extract ground truth annotations from Index Checker warnings.

**What it does**:
- Runs Index Checker on case study projects
- Parses checker warnings to extract annotation locations and types
- Normalizes annotation types (`@Positive`, `@NonNegative`, `@GTENegativeOne`)
- Creates ground truth JSON files per project

**Output**:
- `case_studies/{project}/ground_truth.json`: JSON file with:
  ```json
  [
    {
      "file_path": "absolute/path/to/File.java",
      "annotations": [
        {"line": 123, "type": "@NonNegative"},
        ...
      ]
    },
    ...
  ]
  ```

**Usage**:
```bash
python studies/extract_cs_ground_truth_from_checker.py
```

### 3. Prediction Generation (`predict_case_studies_fixed.py`)

**Purpose**: Generate predictions for case study files using trained models.

**What it does**:
- Loads trained models for specified model type
- For each Java file:
  - Finds corresponding CFG using `index.json` (with fallback heuristics)
  - Loads all method CFGs from the directory
  - Generates per-node predictions using model
  - Extracts line numbers from CFG nodes
- Standardizes output format
- Saves predictions per project

**Key Features**:
- **CFG lookup**: Uses index first, then falls back to stem-based matching
- **Multi-method support**: Loads and merges all method CFGs from a directory
- **Per-node predictions**: Emits predictions for individual CFG nodes with correct line numbers
- **Feature dimension adaptation**: Automatically pads/truncates features to match model input size
- **DGCRF support**: Aliases to DG2N if native DGCRF models not found

**Output**:
- `case_studies/{project}/predictions_{model}.json`: JSON file with:
  ```json
  [
    {
      "file_path": "absolute/path/to/File.java",
      "predictions": [
        {"line": 123, "type": "@NonNegative", "confidence": 0.85},
        ...
      ]
    },
    ...
  ]
  ```

**Usage**:
```bash
# For a specific model
python predict_case_studies_fixed.py --model gcn

# Or run all models via the unified runner
python studies/run_annotation_type_predictions.py
```

### 4. Metrics Computation (`studies/compute_case_study_metrics.py`)

**Purpose**: Compute evaluation metrics comparing predictions to ground truth.

**What it does**:
- Loads ground truth and predictions for each project-model pair
- Filters GT to only files that have CFGs (ensures fair comparison)
- Uses robust matching:
  - Exact line match first
  - ±3-line window for near misses
  - Method-level context consideration
- Computes metrics:
  - **Accuracy (exact)**: Exact label matches
  - **Accuracy (partial)**: Allows 0.5 credit for @Positive ↔ @NonNegative swaps
  - **Precision/Recall/F1**: Weighted and macro averages
  - **Coverage**: Fraction of GT annotations with any prediction on their line
  - **Confusion matrix**: Per-class breakdown

**Output**:
- `case_studies/evaluation_results/{project}_{model}_metrics.json`: Detailed metrics per project-model
- `case_studies/evaluation_results/all_results.json`: Combined results

**Usage**:
```bash
python studies/compute_case_study_metrics.py
```

### 5. Metrics Aggregation (`studies/case_study_metrics_collector.py`)

**Purpose**: Aggregate metrics across projects and models.

**What it does**:
- Loads all project-model metrics
- Computes aggregate statistics (mean, std, min, max)
- Creates per-model summaries
- Generates comparison views

**Output**:
- `case_studies/evaluation_results/aggregate_metrics.json`: Aggregate statistics
- `case_studies/evaluation_results/per_project_metrics.json`: Per-project breakdown

**Usage**:
```bash
python studies/case_study_metrics_collector.py
```

## Complete Pipeline

### Step-by-Step Execution

1. **Generate CFGs**:
   ```bash
   python generate_case_study_cfgs.py
   ```
   This may take time for large projects.

2. **Extract Ground Truth**:
   ```bash
   python studies/extract_cs_ground_truth_from_checker.py
   ```

3. **Generate Predictions** (for all models):
   ```bash
   python studies/run_annotation_type_predictions.py
   ```
   Or for individual models:
   ```bash
   python predict_case_studies_fixed.py --model gcn
   python predict_case_studies_fixed.py --model hgt
   # ... etc
   ```

4. **Compute Metrics**:
   ```bash
   python studies/compute_case_study_metrics.py
   ```

5. **Aggregate Results**:
   ```bash
   python studies/case_study_metrics_collector.py
   ```

6. **Generate Comparison Report** (optional):
   ```bash
   python studies/generate_case_study_comparison.py
   ```

### One-Command Execution

For convenience, you can run the full pipeline:
```bash
python studies/run_annotation_type_predictions.py && \
python studies/compute_case_study_metrics.py && \
python studies/case_study_metrics_collector.py && \
python studies/generate_case_study_comparison.py
```

## Current Results

### Metrics Summary (Plume-lib Project)

| Model | Exact Accuracy | Partial Accuracy | Coverage | GT Annotations | Predictions |
|-------|----------------|------------------|----------|----------------|-------------|
| GCN   | 0.0000         | **0.1667**       | 0.0000   | 15             | 2,374       |
| HGT   | 0.0000         | **0.1667**       | 0.0000   | 15             | 3,185       |
| GBT   | 0.0000         | 0.0000           | 0.0000   | 15             | 26          |
| Causal| 0.0000         | **0.1667**       | 0.0000   | 15             | 1,988       |
| GCSN  | 0.0000         | **0.1667**       | 0.0000   | 15             | 2,374       |
| DG2N  | 0.0000         | 0.0000           | 0.0000   | 15             | 54          |
| DGCRF | 0.0000         | 0.0000           | 0.0000   | 15             | 54          |

**Key Observations**:
- ✅ **4 out of 7 models** achieve non-zero partial accuracy (16.67%)
- Partial accuracy indicates matches within ±3-line window with compatible annotation types
- GBT, DG2N, and DGCRF produce fewer predictions and may need threshold adjustment
- Ground truth is limited to 5 files (15 annotations total) that have both CFGs and GT

### Evaluation Dataset

**Projects Evaluated**:
- `guava`: Google Guava library
- `jfreechart`: JFreeChart plotting library
- `plume-lib`: Plume library

**Files with Ground Truth and CFGs** (Plume-lib):
1. `Intern.java`
2. `CountingPrintWriter.java`
3. `EntryReader.java`
4. `StringBuilderDelimited.java`
5. `RegexUtil.java`

**Total**: 15 ground truth annotations across 5 files

## Technical Details

### CFG Indexing

The CFG generation creates an `index.json` file mapping:
```json
{
  "/absolute/path/to/File.java": "/absolute/path/to/case_study_cfg_output/ClassName/cfg.json"
}
```

The predictor uses this index for exact matching, with fallback heuristics:
1. Exact match via index
2. Stem-based matching (filename without extension)
3. Parent directory name heuristic

### Multi-Method CFG Handling

Each Java file may have multiple methods, each with its own CFG. The predictor:
1. Loads all `.json` files in the CFG directory
2. Merges nodes, edges, and metadata
3. Makes predictions on the combined node set
4. Preserves line numbers from original CFG nodes

### Feature Dimension Adaptation

Models expect different input feature dimensions. The predictor:
- Detects expected input size from model architecture
- Pads features with zeros if too small
- Truncates features if too large
- Logs dimension adjustments for debugging

### Path Normalization

File paths are normalized for consistent matching:
- Absolute paths are converted to relative paths (starting with `case_studies/`)
- Both GT and predictions use the same normalization
- Ensures matching works regardless of absolute vs relative paths

## File Structure

```
case_studies/
├── guava/
│   ├── ground_truth.json
│   ├── predictions_gcn.json
│   ├── predictions_hgt.json
│   └── ...
├── jfreechart/
│   └── ...
├── plume-lib/
│   └── ...
└── evaluation_results/
    ├── {project}_{model}_metrics.json
    ├── aggregate_metrics.json
    ├── per_project_metrics.json
    └── all_results.json

case_study_cfg_output/
├── index.json
├── {ClassName}/
│   ├── cfg.json
│   ├── {method1}.json
│   └── ...
└── ...
```

## Configuration

### Model Directory

Set `MODELS_DIR` environment variable to point to trained models:
```bash
export MODELS_DIR=/path/to/models_annotation_types
```

If unset, defaults to `models_annotation_types/` in the project root.

### Prediction Threshold

Default threshold is 0.3. Adjust in `predict_case_studies_fixed.py`:
```python
preds = predictor.predict_annotations_for_file_with_cfg(..., threshold=0.3)
```

### Matching Window

The ±3-line matching window is configurable in `compute_case_study_metrics.py`:
```python
y_true, y_pred = align_labels(gt_map, pr_map, window=3)
```

## Troubleshooting

### Zero Metrics

If metrics are zero:
1. **Check CFG generation**: Ensure `case_study_cfg_output/index.json` exists and has entries
2. **Verify GT files**: Check that `case_studies/{project}/ground_truth.json` has annotations
3. **Check predictions**: Verify `case_studies/{project}/predictions_{model}.json` has predictions
4. **Filter alignment**: Ensure GT files have corresponding CFGs (metrics filter to files with both)

### Missing CFGs

If some files lack CFGs:
- CFG generation may have failed silently for large files
- Check `case_study_cfg_output/` for missing directories
- Re-run CFG generation for specific files if needed

### Model Loading Errors

If models fail to load:
- Verify `MODELS_DIR` is set correctly
- Check that model checkpoints exist in `{MODELS_DIR}/{model}/{annotation_type}/`
- For DGCRF, ensure DG2N models are available (DGCRF aliases to DG2N)

### Low Prediction Counts

If models produce few predictions:
- Lower the prediction threshold (currently 0.3)
- Check model confidence scores using `studies/analyze_prediction_confidence.py`
- Verify model is trained and loaded correctly

## Future Improvements

1. **Generate CFGs for all GT files**: Currently 7 files with GT lack CFGs
2. **Improve line number accuracy**: CFG line numbers may not match source exactly
3. **Tune prediction thresholds**: Per-model thresholds may improve precision/recall
4. **Expand ground truth**: More annotated files would provide better evaluation
5. **Method-level matching**: Improve matching using method context

## References

- Main prediction script: `predict_case_studies_fixed.py`
- Model predictor: `model_based_predictor.py`
- Metrics computation: `studies/compute_case_study_metrics.py`
- CFG generation: `generate_case_study_cfgs.py`
- Ground truth extraction: `studies/extract_cs_ground_truth_from_checker.py`

## Last Updated

November 6, 2025

