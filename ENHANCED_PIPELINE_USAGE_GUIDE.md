# Enhanced Pipeline Usage Guide

## 🎯 **Updated Configuration**

The GenDATA pipeline has been updated with new defaults:

### **New Default Settings**
- **Augmentation Factor**: 10 (reduced from 50 for faster training)
- **Slicer**: Enhanced Soot with forward/backward/combined slicing
- **Augmentation**: Semantic-preserving transformations (training only)
- **Approach**: Augment-first (training only)
- **Prediction**: No augmentation, direct slicing

## 🚀 **Training All 21 Models**

### **Method 1: Automated Training (Recommended)**
```bash
# Train all 21 models with default settings (augmentation factor = 10)
python train_all_21_models.py
```

### **Method 2: Individual Training**
```bash
# Train specific annotation type models
python annotation_type_rl_positive.py --mode train --episodes 100 --base_model enhanced_causal
python annotation_type_rl_nonnegative.py --mode train --episodes 100 --base_model enhanced_causal
python annotation_type_rl_gtenegativeone.py --mode train --episodes 100 --base_model enhanced_causal
```

### **Method 3: Simple Pipeline**
```bash
# Use the simple pipeline with enhanced defaults
python simple_annotation_type_pipeline.py --mode train --episodes 100
```

## 🔮 **Prediction (No Augmentation)**

### **Method 1: Enhanced Prediction Pipeline (Recommended)**
```bash
# Predict on all case studies
python predict_with_enhanced_pipeline.py

# Predict on specific file
python predict_with_enhanced_pipeline.py --target_file /path/to/TestClass.java

# Predict on custom case studies directory
python predict_with_enhanced_pipeline.py --case_studies_dir /path/to/custom/case_studies
```

### **Method 2: Simple Pipeline Prediction**
```bash
# Use the simple pipeline for prediction (no augmentation)
python simple_annotation_type_pipeline.py --mode predict --target_file /path/to/TestClass.java
```

### **Method 3: Case Studies**
```bash
# Run predictions on case studies
python run_case_studies.py
```

## ⚙️ **Configuration Details**

### **Training Configuration**
```python
# Default training settings
augmentation_factor = 10      # Number of variants per file
slicer_type = 'soot'         # Enhanced Soot slicer
slice_mode = 'combined'      # Forward + backward slicing
augmentation_type = 'semantic' # Semantic-preserving transformations
augment_first = True         # Augment code before slicing
episodes = 100              # Training episodes
base_model = 'enhanced_causal' # Default base model
```

### **Prediction Configuration**
```python
# Prediction settings (no augmentation)
augment_first = False        # No augmentation during prediction
slicer_type = 'soot'         # Enhanced Soot slicer
slice_mode = 'combined'      # Forward + backward slicing
direct_slicing = True        # Direct slicing of target files
```

## 📊 **Pipeline Flow**

### **Training Flow**
```
Original Code (Read-Only)
    ↓
Step 1: Semantic Augmentation (10 variants per file)
    ↓
Augmented Variants (Full Project Context)
    ↓
Step 2: Enhanced Soot Slicing (Forward/Backward/Combined)
    ↓
Slices from Each Variant
    ↓
Step 3: CFG Generation
    ↓
Step 4: Model Training (21 models)
    ↓
Trained Models
```

### **Prediction Flow**
```
Target Files
    ↓
Step 1: Direct Enhanced Soot Slicing (No Augmentation)
    ↓
Slices
    ↓
Step 2: CFG Generation
    ↓
Step 3: Model Prediction
    ↓
Annotation Placements
```

## 🎯 **Benefits of New Configuration**

### **Training Benefits**
- **Faster Training**: Reduced augmentation factor (10 vs 50) speeds up training
- **Better Quality**: Semantic augmentation provides meaningful code variations
- **Enhanced Slicing**: Forward/backward slicing captures complete dependencies
- **Original Code Protection**: Read-only access with checksum verification

### **Prediction Benefits**
- **No Augmentation**: Direct slicing for faster prediction
- **Enhanced Slicing**: Same advanced slicing as training
- **Consistent Results**: Same slicer used in training and prediction
- **Better Performance**: No overhead from augmentation during prediction

## 📁 **Directory Structure**

### **Training Outputs**
```
/home/ubuntu/GenDATA/
├── augmented_code/           # Augmented variants (10 per file)
├── slices_augmented_first/   # Slices from augmented variants
├── cfg_output_augmented_first/ # CFGs from slices
└── models_annotation_types/  # Trained models (21 models)
```

### **Prediction Outputs**
```
/home/ubuntu/GenDATA/
├── prediction_slices/        # Direct slices from target files
├── cfg_output_prediction/    # CFGs from prediction slices
└── predictions_annotation_types/ # Prediction results
```

## 🔧 **Customization Options**

### **Training Customization**
```bash
# Custom augmentation factor
python train_all_21_models.py --augmentation_factor 5

# Custom episodes
python train_all_21_models.py --episodes 50

# Custom base model
python train_all_21_models.py --base_model gcn
```

### **Prediction Customization**
```bash
# Custom case studies directory
python predict_with_enhanced_pipeline.py --case_studies_dir /path/to/custom

# Custom models directory
python predict_with_enhanced_pipeline.py --models_dir /path/to/models

# Custom output directory
python predict_with_enhanced_pipeline.py --output_dir /path/to/output
```

## 📈 **Performance Expectations**

### **Training Performance**
- **Augmentation Factor 10**: ~5x faster than factor 50
- **Enhanced Soot Slicing**: More accurate dependency tracking
- **Semantic Augmentation**: Better training data quality
- **21 Models**: Complete coverage of all annotation types

### **Prediction Performance**
- **No Augmentation**: ~10x faster prediction
- **Direct Slicing**: Immediate processing of target files
- **Enhanced Slicing**: Same quality as training
- **Batch Processing**: Efficient handling of multiple files

## 🎉 **Quick Start**

### **1. Train All Models**
```bash
python train_all_21_models.py
```

### **2. Run Predictions**
```bash
python predict_with_enhanced_pipeline.py
```

### **3. Check Results**
```bash
ls -la predictions_annotation_types/
```

## 🔍 **Verification**

### **Check Configuration**
```bash
python pipeline_config.py
```

### **Verify Models**
```bash
ls -la models_annotation_types/*.pth | wc -l
# Should show 21 model files
```

### **Test Prediction**
```bash
python predict_with_enhanced_pipeline.py --target_file /path/to/single/file.java
```

## 🎯 **Summary**

The enhanced pipeline now provides:
- ✅ **Augmentation factor of 10** for faster training
- ✅ **No augmentation during prediction** for faster inference
- ✅ **Enhanced Soot slicer** with forward/backward slicing
- ✅ **Semantic-preserving transformations** for better training data
- ✅ **Original code protection** with read-only access
- ✅ **21 trained models** for complete annotation coverage

The pipeline is optimized for both training speed and prediction performance while maintaining high quality results.
