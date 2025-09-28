#!/usr/bin/env python3
"""
GenDATA Pipeline Completeness Verification Script

This script verifies that all necessary components are present in GenDATA
for running the annotation type pipeline with all required models.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ MISSING {description}: {filepath}")
        return False

def main():
    """Main verification function"""
    print("🔍 Verifying GenDATA Pipeline Completeness")
    print("=" * 50)
    
    gendata_root = Path("/home/thoriumrobot/project/GenDATA")
    all_present = True
    
    # Required Binary RL Models (6 models)
    print("\n📋 Binary RL Models (6 required):")
    binary_rl_models = [
        "binary_rl_gcn_standalone.py",
        "binary_rl_gbt_standalone.py", 
        "binary_rl_causal_standalone.py",
        "binary_rl_hgt_standalone.py",
        "binary_rl_gcsn_standalone.py",
        "binary_rl_dg2n_standalone.py"
    ]
    
    for model in binary_rl_models:
        if not check_file_exists(gendata_root / model, f"Binary RL Model ({model.split('_')[2].upper()})"):
            all_present = False
    
    # Required Annotation Type Models (3 types)
    print("\n📋 Annotation Type Models (3 required):")
    annotation_types = [
        "annotation_type_rl_positive.py",
        "annotation_type_rl_nonnegative.py", 
        "annotation_type_rl_gtenegativeone.py"
    ]
    
    for model in annotation_types:
        annotation_type = model.split('_')[2].replace('.py', '').upper()
        if not check_file_exists(gendata_root / model, f"Annotation Type Model (@{annotation_type})"):
            all_present = False
    
    # Core Model Implementations (6 models)
    print("\n📋 Core Model Implementations (6 required):")
    core_models = [
        ("hgt.py", "HGT Model"),
        ("gbt.py", "GBT Model"),
        ("causal_model.py", "Causal Model"),
        ("gcn_train.py", "GCN Training"),
        ("gcn_predict.py", "GCN Prediction"),
        ("gcsn_adapter.py", "GCSN Adapter"),
        ("dg2n_adapter.py", "DG2N Adapter"),
        ("dgcrf_model.py", "DG-CRF Model"),
        ("train_dgcrf.py", "DG-CRF Training"),
        ("predict_dgcrf.py", "DG-CRF Prediction"),
        ("sg_cfgnet.py", "SG-CFGNet Model"),
        ("sg_cfgnet_train.py", "SG-CFGNet Training"),
        ("sg_cfgnet_predict.py", "SG-CFGNet Prediction")
    ]
    
    for model_file, description in core_models:
        if not check_file_exists(gendata_root / model_file, description):
            all_present = False
    
    # Pipeline Infrastructure
    print("\n📋 Pipeline Infrastructure:")
    infrastructure_files = [
        ("pipeline.py", "Main Pipeline"),
        ("cfg.py", "CFG Generation"),
        ("augment_slices.py", "Slice Augmentation"),
        ("simple_annotation_type_pipeline.py", "Simple Annotation Pipeline"),
        ("annotation_type_pipeline.py", "Full Annotation Pipeline"),
        ("predict_and_annotate.py", "Integrated Prediction & Annotation"),
        ("predict_on_project.py", "Project-wide Prediction"),
        ("place_annotations.py", "Annotation Placement"),
        ("checker_framework_integration.py", "Checker Framework Integration"),
        ("prediction_saver.py", "Prediction Saving")
    ]
    
    for file_path, description in infrastructure_files:
        if not check_file_exists(gendata_root / file_path, description):
            all_present = False
    
    # Evaluation and Testing
    print("\n📋 Evaluation and Testing:")
    evaluation_files = [
        ("run_case_studies.py", "Binary RL Case Studies"),
        ("annotation_type_case_studies.py", "Annotation Type Case Studies"),
        ("comprehensive_annotation_type_evaluation.py", "Comprehensive Evaluation"),
        ("annotation_type_evaluation.py", "Annotation Type Evaluation"),
        ("annotation_type_prediction.py", "Annotation Type Prediction")
    ]
    
    for file_path, description in evaluation_files:
        if not check_file_exists(gendata_root / file_path, description):
            all_present = False
    
    # Training and Optimization
    print("\n📋 Training and Optimization:")
    training_files = [
        ("enhanced_rl_training.py", "Enhanced RL Training"),
        ("rl_annotation_type_training.py", "RL Annotation Type Training"),
        ("rl_pipeline.py", "RL Pipeline"),
        ("hyperparameter_search_annotation_types.py", "Hyperparameter Search"),
        ("simple_hyperparameter_search_annotation_types.py", "Simple Hyperparameter Search")
    ]
    
    for file_path, description in training_files:
        if not check_file_exists(gendata_root / file_path, description):
            all_present = False
    
    # Java Components
    print("\n📋 Java Components:")
    java_files = [
        ("src/main/java/cfwr/CheckerFrameworkWarningResolver.java", "Warning Resolver"),
        ("src/main/java/cfwr/CheckerFrameworkSlicer.java", "CF Slicer"),
        ("src/main/java/cfwr/SootSlicer.java", "Soot Slicer"),
        ("src/main/java/cfwr/WalaSliceCLI.java", "WALA Slicer"),
        ("build.gradle", "Gradle Build File"),
        ("gradlew", "Gradle Wrapper"),
        ("tools/soot_slicer.sh", "Soot Slicer Script")
    ]
    
    for file_path, description in java_files:
        if not check_file_exists(gendata_root / file_path, description):
            all_present = False
    
    # Configuration and Data
    print("\n📋 Configuration and Data:")
    config_files = [
        ("requirements.txt", "Python Dependencies"),
        ("annotation_type_config.json", "Annotation Type Config"),
        ("index1.out", "Sample Warnings File"),
        ("index1.small.out", "Small Sample Warnings"),
        ("hyperparameter_search_annotation_types_results_20250927_224114.json", "Hyperparameter Results"),
        ("simple_hyperparameter_search_annotation_types_results_20250927_224445.json", "Simple Hyperparameter Results")
    ]
    
    for file_path, description in config_files:
        if not check_file_exists(gendata_root / file_path, description):
            all_present = False
    
    # Documentation
    print("\n📋 Documentation:")
    doc_files = [
        ("README.md", "Main README"),
        ("ANNOTATION_TYPE_MODELS_GUIDE.md", "Annotation Type Guide"),
        ("COMPREHENSIVE_CASE_STUDY_RESULTS.md", "Case Study Results")
    ]
    
    for file_path, description in doc_files:
        if not check_file_exists(gendata_root / file_path, description):
            all_present = False
    
    # Directories
    print("\n📋 Required Directories:")
    required_dirs = [
        ("models_annotation_types/", "Annotation Type Models Directory"),
        ("predictions_annotation_types/", "Predictions Directory"),
        ("src/", "Java Source Directory"),
        ("tools/", "Tools Directory"),
        ("gradle/", "Gradle Directory")
    ]
    
    for dir_path, description in required_dirs:
        if os.path.exists(gendata_root / dir_path):
            print(f"✅ {description}: {dir_path}")
        else:
            print(f"❌ MISSING {description}: {dir_path}")
            all_present = False
    
    # Final Summary
    print("\n" + "=" * 50)
    if all_present:
        print("🎉 SUCCESS: GenDATA is a complete, self-contained annotation type pipeline!")
        print("\n✅ All required components are present:")
        print("   • 6 Binary RL Models (HGT, GBT, Causal, GCN, GCSN, DG2N)")
        print("   • 3 Annotation Type Models (@Positive, @NonNegative, @GTENegativeOne)")
        print("   • Complete pipeline infrastructure")
        print("   • Java components for warning resolution and slicing")
        print("   • Evaluation and testing frameworks")
        print("   • Documentation and configuration files")
        print("\n🚀 Ready to train and predict annotation types!")
    else:
        print("❌ FAILURE: Some required components are missing.")
        print("Please check the missing files above and copy them from CFWR.")
    
    return all_present

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

