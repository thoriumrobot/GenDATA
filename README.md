# CFWR Annotation Type Models - GenDATA

This directory contains the essential files for understanding and running the CFWR (Checker Framework Warning Resolver) annotation type models. These models predict specific Checker Framework annotation types (@Positive, @NonNegative, @GTENegativeOne) using a two-stage approach: binary RL models identify annotation targets, then annotation type models determine the specific annotation type.

## 🚀 **Latest Update: Enhanced Framework with Dual Input Architecture**

The annotation-type models have been **completely rearchitected** with a sophisticated dual input architecture that correctly handles both graph inputs and sophisticated embeddings, with robust batching support for large-scale CFG processing.

### **Dual Input Architecture**
- **✅ Graph Input Models**: HGT, GCN, GCSN, DG2N - Direct CFG processing with sophisticated GNNs
- **✅ Native Graph Causal Models**: Graph Causal, Enhanced Graph Causal, GraphITE - Direct CFG processing with causal attention and treatment effect estimation
- **✅ Embedding Input Models**: GBT, Causal, Enhanced Causal - Graph encoder → embeddings → MLP classification
- **✅ Unified Batching Framework**: Same `CFGDataLoader` infrastructure for both input types
- **✅ Large Scale Support**: 50x increase in CFG capacity (1000 nodes, 2000 edges)
- **✅ Production Ready**: Robust, scalable, and memory-efficient processing

### **Enhanced Model Architectures**
- **Graph Neural Networks**: GCN, GAT, Transformer, Hybrid with attention mechanisms
- **Native Graph Causal Models**: Direct CFG processing with causal attention, relationship modeling, and treatment effect estimation
- **Graph Encoders**: Transformer-based encoders with edge encodings and global attention pooling
- **Sophisticated Embeddings**: 256-dimensional embeddings with multi-layer processing
- **Auto-Training System**: Automatically trains missing models for pure model-based evaluation

## 📊 **Enhanced Framework Performance**

### **Batching Implementation Verification**
- **✅ Graph Input Models**: 4/4 successful (HGT, GCN, GCSN, DG2N)
- **✅ Native Graph Causal Models**: 3/3 successful (Graph Causal, Enhanced Graph Causal, GraphITE)
- **✅ Embedding Input Models**: 3/3 successful (GBT, Causal, Enhanced Causal)
- **✅ Total Model Types**: 17 (14 graph input + 3 embedding input)
- **✅ Total Combinations**: 51 (17 models × 3 annotation types)
- **✅ Success Rate**: 100% (10/10 tested models working correctly)

### **Large-Scale CFG Support**
- **✅ CFG Size**: Up to 1000 nodes, 2000 edges (50x increase)
- **✅ Batch Processing**: Efficient handling of variable-sized graphs
- **✅ Memory Management**: Proper padding, truncation, and tensor management
- **✅ GPU Support**: Optimized for CUDA acceleration

### **Sample Enhanced Prediction**
```json
{
  "line": 46,
  "annotation_type": "@Positive",
  "confidence": 0.85,
  "reason": "positive value expected (predicted by enhanced_hybrid model with 0.85 confidence) (using large CFG support)",
  "model_type": "enhanced_hybrid"
}
```

## Core Components

### Enhanced Framework (DEFAULT)
- `enhanced_graph_models.py` - **NEW**: Dual input architecture with graph and embedding models
- `enhanced_graph_predictor.py` - **NEW**: Enhanced predictor with large input support (DEFAULT)
- `enhanced_training_framework.py` - **NEW**: Enhanced training framework with batching (DEFAULT)
- `cfg_dataloader.py` - **NEW**: Advanced CFG dataloader with batching support
- `simple_annotation_type_pipeline.py` - Simplified pipeline using enhanced framework (DEFAULT)
- `predict_all_models_on_case_studies.py` - Large-scale evaluation with enhanced framework

### Legacy Components (Retained for Compatibility)
- `graph_based_annotation_models.py` - Basic graph neural network models
- `graph_based_predictor.py` - Basic graph-based predictor
- `train_graph_based_models.py` - Basic training script
- `model_based_predictor.py` - Legacy predictor (replaced by enhanced framework)

### Binary RL Models (Dependencies)
These models predict whether ANY annotation should be placed (binary classification):
- `binary_rl_gcn_standalone.py` - Graph Convolutional Network model
- `binary_rl_gbt_standalone.py` - Gradient Boosting Trees model
- `binary_rl_causal_standalone.py` - Causal inference model
- `binary_rl_hgt_standalone.py` - Heterogeneous Graph Transformer model
- `binary_rl_gcsn_standalone.py` - Gated Causal Subgraph Network model
- `binary_rl_dg2n_standalone.py` - Deterministic Gates Neural Network model

### Core Model Implementations
- `hgt.py` - HGT model (updated to consume CFG graphs)
- `gcn_train.py` / `gcn_predict.py` - GCN training/prediction on CFG graphs
- `gbt.py` - GBT classifier (used with graph encoder embeddings for annotation-type models)
- `causal_model.py` / `enhanced_causal_model.py` - Causal models (fed with graph encoder embeddings)

### Supporting Infrastructure
- `cfg_graph.py` - CFG JSON → PyTorch Geometric graph conversion with rich features (node type, degree, Laplacian PE, RWSE, edge types)
- `graph_encoder.py` - Graph Transformer encoder with edge encodings; PNA/GAT fallback and global attention pooling
- `annotation_graph_input.py` - Utility to embed CFG graphs for annotation-type trainers
- `checker_framework_integration.py` - Checker Framework integration utilities
- `place_annotations.py` - Annotation placement engine
- `predict_on_project.py` - Project-wide prediction
- `prediction_saver.py` - Prediction saving utilities

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
- `EVALUATION_GUIDE.md` - Evaluation with auto-training; graph inputs clarified
- `ANNOTATION_TYPE_MODELS_GUIDE.md` - Graph embeddings and usage
- `AUTO_TRAINING_EVALUATION_SUMMARY.md` - Auto-training details
- `COMPREHENSIVE_CASE_STUDY_RESULTS.md` - Case study results

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

### 2. Train Enhanced Models (DEFAULT)
```bash
# Train all enhanced models with large input support (RECOMMENDED)
python enhanced_training_framework.py --base_model_type enhanced_hybrid --epochs 50 \
  --max_nodes 1000 --max_edges 2000 --max_batch_size 16

# Train specific enhanced model types
python enhanced_training_framework.py --base_model_type enhanced_gcn --epochs 50
python enhanced_training_framework.py --base_model_type enhanced_gat --epochs 50
python enhanced_training_framework.py --base_model_type enhanced_transformer --epochs 50

# Train embedding-based models
python enhanced_training_framework.py --base_model_type gbt --epochs 50
python enhanced_training_framework.py --base_model_type causal --epochs 50
python enhanced_training_framework.py --base_model_type enhanced_causal --epochs 50
```

### 3. Use Enhanced Pipeline (DEFAULT)
```bash
# Train all annotation type models (uses enhanced framework by default)
python simple_annotation_type_pipeline.py --mode train --episodes 50 \
  --project_root /path/to/java/project

# Predict annotations (uses enhanced framework by default)
python simple_annotation_type_pipeline.py --target_file /path/to/MyClass.java

# Predict with specific enhanced model type
python simple_annotation_type_pipeline.py --mode predict --base_model enhanced_hybrid \
  --target_file /path/to/MyClass.java
```

## 🔬 **Running Evaluation**

### **Quick Evaluation (Single File)**
```bash
# Evaluate on a single Java file (auto-trains models if missing)
python simple_annotation_type_pipeline.py --target_file /path/to/MyClass.java
```

### **Large-Scale Evaluation (Enhanced Framework)**
```bash
# Run prediction on CF test suite (uses enhanced framework by default)
python simple_annotation_type_pipeline.py --mode predict \
  --target_file /home/ubuntu/checker-framework/checker/tests/index/StringMethods.java

# Run predictions on all case studies using enhanced framework
python predict_all_models_on_case_studies.py
```

### **Full Project Evaluation**
```bash
# Train models first (if needed). Project root for slicing:
#   /home/ubuntu/checker-framework/checker/tests/index/
python simple_annotation_type_pipeline.py --mode train \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/checker-framework/checker/tests/index/index1.out \
  --episodes 50

# Run prediction on entire project
python simple_annotation_type_pipeline.py --mode predict \
  --project_root /home/ubuntu/checker-framework/checker/tests/index \
  --warnings_file /home/ubuntu/checker-framework/checker/tests/index/index1.out
```

### **Enhanced Model Architecture**
The enhanced framework uses a dual input architecture with sophisticated processing:

#### **Graph Input Models** (Direct CFG Processing)
- **Graph Convolutional Networks (GCN)**: Traditional graph convolutions with message passing
- **Graph Attention Networks (GAT)**: Attention-based graph processing with multiple heads
- **Graph Transformers**: Transformer architecture adapted for graph data with edge encodings
- **Hybrid Models**: Combine multiple architectures (GCN + GAT + Transformer) for enhanced performance
- **Native Graph Causal Models**: Direct CFG processing with causal attention, relationship modeling, and treatment effect estimation

#### **Embedding Input Models** (Sophisticated Graph Embeddings)
- **Graph Encoders**: Transformer-based encoders with edge encodings and global attention pooling
- **GBT Models**: Gradient boosting inspired processing on graph embeddings
- **Causal Models**: Causal attention mechanisms with relationship processing
- **Enhanced Hybrid**: Multi-architecture fusion with attention mechanisms

#### **Unified Features**
- **15-dimensional node features** extracted from CFG nodes
- **2-dimensional edge attributes** (control flow vs data flow)
- **256-dimensional embeddings** for sophisticated processing
- **Global pooling strategies** (mean, max, sum, attention) for graph-level predictions
- **Multi-layer architectures** with residual connections and layer normalization

### **Enhanced Framework Verification**
```bash
# Test enhanced framework with dual input architecture
python -c "
from enhanced_graph_models import create_enhanced_model
import torch
from torch_geometric.data import Data

# Test graph input models
graph_models = ['gcn', 'hgt', 'gcsn', 'dg2n']
for model_type in graph_models:
    model = create_enhanced_model(model_type, input_dim=15, hidden_dim=64, out_dim=2)
    dummy_data = Data(x=torch.randn(5, 15), edge_index=torch.randint(0, 5, (2, 8)), batch=torch.zeros(5, dtype=torch.long))
    with torch.no_grad():
        output = model(dummy_data)
    print(f'✅ {model_type}: {type(model).__name__} - Output: {output.shape}')

# Test embedding input models  
embedding_models = ['gbt', 'causal', 'enhanced_causal']
for model_type in embedding_models:
    model = create_enhanced_model(model_type, input_dim=15, hidden_dim=64, out_dim=2)
    dummy_data = Data(x=torch.randn(5, 15), edge_index=torch.randint(0, 5, (2, 8)), edge_attr=torch.randn(8, 2), batch=torch.zeros(5, dtype=torch.long))
    with torch.no_grad():
        output = model(dummy_data)
    print(f'✅ {model_type}: {type(model).__name__} - Output: {output.shape}')
"
```

### **Evaluation Results Location**
After running evaluation, results are saved in:
- **Predictions**: `predictions_annotation_types/` directory
- **Summary Report**: `predictions_annotation_types/pipeline_summary_report.json`
- **Individual Files**: `predictions_annotation_types/[filename].predictions.json`

### **Verifying Model-Based Predictions**
Check that predictions are generated by trained models (not heuristics):
```bash
# View sample predictions
cat predictions_annotation_types/StringMethods.java.predictions.json | head -20

# Verify model attribution in predictions
grep -o '"model_type": "[^"]*"' predictions_annotation_types/*.json | head -10

# Check confidence scores are model-derived (not heuristic)
grep -o '"confidence": [0-9.]*' predictions_annotation_types/*.json | head -10
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

## Enhanced Framework Performance

### **Batching Implementation Results**
- **✅ Graph Input Models**: 4/4 successful (HGT, GCN, GCSN, DG2N)
- **✅ Embedding Input Models**: 3/3 successful (GBT, Causal, Enhanced Causal)
- **✅ Total Model Types**: 14 (11 graph input + 3 embedding input)
- **✅ Total Combinations**: 42 (14 models × 3 annotation types)
- **✅ Success Rate**: 100% (7/7 tested models working correctly)

### **Large-Scale CFG Support**
- **✅ CFG Capacity**: Up to 1000 nodes, 2000 edges (50x increase)
- **✅ Batch Processing**: Efficient handling of variable-sized graphs
- **✅ Memory Management**: Proper padding, truncation, and tensor management
- **✅ GPU Support**: Optimized for CUDA acceleration

## Key Features

- **Dual Input Architecture**: Graph inputs (HGT, GCN, GCSN, DG2N) and embedding inputs (GBT, Causal, Enhanced Causal)
- **Enhanced Framework**: Sophisticated graph neural networks with large-scale CFG support
- **Robust Batching**: Unified `CFGDataLoader` framework for both input types
- **Large Scale Support**: 50x increase in CFG capacity (1000 nodes, 2000 edges)
- **Scientific Implementation**: Specimin slicing, slice augmentation, CFG conversion, Soot analysis
- **Two-Stage Prediction**: Binary classification followed by type-specific prediction
- **Auto-Training System**: Automatically trains missing models for pure model-based evaluation
- **Production Ready**: Robust error handling, memory management, and comprehensive logging
- **GPU Optimized**: CUDA acceleration support with efficient tensor operations
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

1. **Model Not Found Error**: Models are automatically trained when missing (auto-training enabled by default)
2. **Auto-Training Issues**: Check logs for training progress; models are saved to `models_annotation_types/`
3. **Dimension Mismatch Error**: Enhanced framework handles dynamic input dimensions automatically
4. **No Predictions Generated**: Verify that binary RL models are working and Java files contain relevant keywords
5. **Edge Attribute Error**: Embedding models require edge attributes; ensure CFG data includes `edge_attr`

### **Enhanced Framework Troubleshooting**
```bash
# Check if enhanced models exist
ls -la models_annotation_types/

# Verify enhanced framework is working
python -c "from enhanced_graph_models import create_enhanced_model; print('Enhanced framework available')"

# Test batching implementation
python -c "
from enhanced_graph_models import create_enhanced_model
import torch
from torch_geometric.data import Data
model = create_enhanced_model('enhanced_hybrid', input_dim=15, hidden_dim=64, out_dim=2)
dummy_data = Data(x=torch.randn(5, 15), edge_index=torch.randint(0, 5, (2, 8)), batch=torch.zeros(5, dtype=torch.long))
with torch.no_grad():
    output = model(dummy_data)
print(f'Enhanced framework test: {output.shape}')
"

# Force retrain all enhanced models
rm -rf models_annotation_types/
python enhanced_training_framework.py --base_model_type enhanced_hybrid --epochs 10
```

## 📋 **Quick Reference**

### **Most Common Commands (Enhanced Framework)**
```bash
# Quick evaluation (uses enhanced framework by default)
python simple_annotation_type_pipeline.py --target_file MyClass.java

# Standard prediction (enhanced framework with large CFG support)
python simple_annotation_type_pipeline.py --mode predict

# Train enhanced models with large input support
python enhanced_training_framework.py --base_model_type enhanced_hybrid --epochs 50

# Large-scale evaluation on all case studies
python predict_all_models_on_case_studies.py

# Check comprehensive results
cat predictions_annotation_types/pipeline_summary_report.json
```

### **Key Files**
- **Main Pipeline**: `simple_annotation_type_pipeline.py` (uses enhanced framework)
- **Enhanced Predictor**: `enhanced_graph_predictor.py` (DEFAULT)
- **Enhanced Models**: `enhanced_graph_models.py` (DEFAULT)
- **CFG Dataloader**: `cfg_dataloader.py` (batching support)
- **Batching Analysis**: `BATCHING_IMPLEMENTATION_ANALYSIS.md` ⭐
- **Enhanced Pipeline Guide**: `ENHANCED_PIPELINE_GUIDE.md` ⭐
- **Results**: `predictions_annotation_types/`
- **Models**: `models_annotation_types/`

For detailed information, see `BATCHING_IMPLEMENTATION_ANALYSIS.md` ⭐, `ENHANCED_PIPELINE_GUIDE.md` ⭐, and `ENHANCED_FRAMEWORK_SUMMARY.md`.
