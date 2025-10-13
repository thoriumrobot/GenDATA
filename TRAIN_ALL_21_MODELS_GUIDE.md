# Train All 21 Models with Enhanced Pipeline Guide

## Overview

This guide explains how to train all 21 annotation type models using the enhanced GenDATA pipeline with semantic-preserving augmentation, augment-first approach, and enhanced Soot slicing.

## 🎯 **21 Models to Train**

### **Model Structure**
- **7 Base Model Types**: gcn, gbt, causal, enhanced_causal, hgt, gcsn, dg2n
- **3 Annotation Types**: @Positive, @NonNegative, @GTENegativeOne
- **Total**: 7 × 3 = 21 models

### **Model Combinations**
```
@Positive:     positive_gcn, positive_gbt, positive_causal, positive_enhanced_causal, positive_hgt, positive_gcsn, positive_dg2n
@NonNegative:  nonnegative_gcn, nonnegative_gbt, nonnegative_causal, nonnegative_enhanced_causal, nonnegative_hgt, nonnegative_gcsn, nonnegative_dg2n
@GTENegativeOne: gtenegativeone_gcn, gtenegativeone_gbt, gtenegativeone_causal, gtenegativeone_enhanced_causal, gtenegativeone_hgt, gtenegativeone_gcsn, gtenegativeone_dg2n
```

## 🚀 **Enhanced Pipeline Features**

### **Default Configuration**
The enhanced pipeline now uses these defaults:
- **Slicer**: Enhanced Soot with forward/backward/combined slicing
- **Augmentation**: Semantic-preserving transformations
- **Approach**: Augment-first (augment code then slice each variant)
- **Mode**: Combined slicing (forward + backward)

### **Enhanced Features**
1. **Semantic-Preserving Augmentation**
   - Loop conversions (for ↔ while)
   - Guard reversals (if-else condition flipping)
   - Mathematical properties (commutativity, identity operations)
   - De Morgan's laws
   - Ternary ↔ if/else conversions
   - Switch ↔ if/else chain conversions
   - Variable inlining/extraction

2. **Augment-First Approach**
   - Original code is augmented first with semantic transformations
   - Each augmented variant is then sliced
   - Provides greater semantic diversity in training data

3. **Enhanced Soot Slicing**
   - Forward slicing: finds statements influenced by target
   - Backward slicing: finds statements that influence target
   - Combined slicing: merges both for complete analysis
   - Comprehensive data flow and control flow analysis

## 📋 **Training Methods**

### **Method 1: Automated Training Script (Recommended)**

Use the updated `train_all_21_models.py` script:

```bash
cd /home/ubuntu/GenDATA
python train_all_21_models.py
```

**Features:**
- Trains all 21 models automatically
- Uses enhanced pipeline defaults
- Progress tracking and error handling
- Model verification and reporting

### **Method 2: Individual Model Training**

Train models individually using the enhanced pipeline:

```bash
# Train @Positive models
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model gcn
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model gbt
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model causal
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model enhanced_causal
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model hgt
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model gcsn
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model dg2n

# Train @NonNegative models
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model gcn
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model gbt
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model causal
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model enhanced_causal
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model hgt
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model gcsn
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model dg2n

# Train @GTENegativeOne models
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model gcn
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model gbt
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model causal
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model enhanced_causal
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model hgt
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model gcsn
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model dg2n
```

### **Method 3: Using Simple Pipeline**

Use the enhanced simple pipeline:

```bash
# Train all annotation types with enhanced pipeline
python simple_annotation_type_pipeline.py --mode train --episodes 100
```

## ⚙️ **Configuration Options**

### **Enhanced Pipeline Defaults**
```python
# These are now the defaults - no need to specify
augment_first = True          # Augment code before slicing
slicer_type = 'soot'         # Enhanced Soot slicer
slice_mode = 'combined'      # Forward + backward slicing
augmentation_type = 'semantic' # Semantic-preserving transformations
```

### **Custom Configuration**
```bash
# Use traditional approach (if needed)
python train_all_21_models.py --no_augment_first

# Custom episodes
python train_all_21_models.py --episodes 50

# Custom device
python train_all_21_models.py --device cuda
```

## 📊 **Training Process**

### **Step-by-Step Process**
1. **Semantic Augmentation**: Apply semantic-preserving transformations to original code
2. **Augment-First Slicing**: Slice each augmented variant using enhanced Soot slicer
3. **CFG Generation**: Convert slices to Control Flow Graphs using Checker Framework's CFG Builder
4. **Model Training**: Train each of the 21 models on the generated CFG data
5. **Model Verification**: Verify trained models and save results

### **Expected Output**
```
🎯 GenDATA - Train All 21 Models with Enhanced Pipeline
================================================================================
Training 21 annotation type models:
- 7 base model types: gcn, gbt, causal, enhanced_causal, hgt, gcsn, dg2n
- 3 annotation types: @Positive, @NonNegative, @GTENegativeOne
- Total: 7 × 3 = 21 models
- Using enhanced pipeline: Semantic Augmentation → Enhanced Soot Slicing → CFG Builder → Training
- Enhanced features: Augment-first approach, semantic-preserving transformations, forward/backward slicing
================================================================================

📋 Training models for @Positive...
🚀 Training positive_gcn model using enhanced pipeline...
✅ Successfully trained positive_gcn
📊 Progress: 1/21

🚀 Training positive_gbt model using enhanced pipeline...
✅ Successfully trained positive_gbt
📊 Progress: 2/21

... (continues for all 21 models)

🎯 TRAINING COMPLETE!
✅ Successfully trained: 21/21 models
```

## 📁 **Output Files**

### **Model Files**
Trained models are saved in `models_annotation_types/`:
```
models_annotation_types/
├── positive_gcn_model.pth
├── positive_gbt_model.pth
├── positive_causal_model.pth
├── positive_enhanced_causal_model.pth
├── positive_hgt_model.pth
├── positive_gcsn_model.pth
├── positive_dg2n_model.pth
├── nonnegative_gcn_model.pth
├── nonnegative_gbt_model.pth
├── nonnegative_causal_model.pth
├── nonnegative_enhanced_causal_model.pth
├── nonnegative_hgt_model.pth
├── nonnegative_gcsn_model.pth
├── nonnegative_dg2n_model.pth
├── gtenegativeone_gcn_model.pth
├── gtenegativeone_gbt_model.pth
├── gtenegativeone_causal_model.pth
├── gtenegativeone_enhanced_causal_model.pth
├── gtenegativeone_hgt_model.pth
├── gtenegativeone_gcsn_model.pth
└── gtenegativeone_dg2n_model.pth
```

### **Training Statistics**
Training statistics are saved as JSON files:
```
models_annotation_types/
├── positive_gcn_stats.json
├── positive_gbt_stats.json
├── ... (for all 21 models)
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Out of Memory**
   ```bash
   # Use CPU instead of GPU
   python train_all_21_models.py --device cpu
   ```

2. **Training Timeout**
   ```bash
   # Reduce episodes for faster training
   python train_all_21_models.py --episodes 50
   ```

3. **Model Training Failures**
   - Check logs for specific error messages
   - Verify CFG generation completed successfully
   - Ensure sufficient disk space for model files

### **Verification**
```bash
# Check trained models
ls -la models_annotation_types/*.pth | wc -l
# Should show 21 model files

# Check training statistics
ls -la models_annotation_types/*.json | wc -l
# Should show 21 stats files
```

## 🎯 **Next Steps**

After training all 21 models:

1. **Test Models**: Use the trained models for prediction
2. **Evaluate Performance**: Run case studies on real projects
3. **Compare Results**: Analyze performance across different model types
4. **Optimize**: Fine-tune hyperparameters based on results

### **Prediction Usage**
```bash
# Use trained models for prediction
python simple_annotation_type_pipeline.py --mode predict --target_file /path/to/TestClass.java
```

## 📈 **Expected Benefits**

### **Enhanced Training Quality**
- **Better Data Diversity**: Semantic augmentation provides meaningful code variations
- **Improved Slicing**: Enhanced Soot slicer captures complete dependencies
- **Robust Models**: Augment-first approach creates more resilient training data

### **Performance Improvements**
- **Higher Accuracy**: Better training data leads to more accurate models
- **Better Generalization**: Diverse training data improves model robustness
- **Consistent Results**: Enhanced pipeline provides reliable training process

## 🎉 **Conclusion**

The enhanced pipeline provides a comprehensive, automated way to train all 21 annotation type models with:

- **Semantic-preserving augmentation** for better training data quality
- **Augment-first approach** for maximum data diversity
- **Enhanced Soot slicing** for comprehensive dependency analysis
- **Automated training** with progress tracking and error handling

This ensures that all models are trained with the highest quality data and most advanced techniques available in the GenDATA pipeline.



