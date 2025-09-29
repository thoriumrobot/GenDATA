#!/usr/bin/env python3
"""
Test script to verify enhanced causal model integration in GenDATA
"""

import os
import sys
import tempfile
import shutil

def test_enhanced_causal_import():
    """Test that enhanced causal model can be imported"""
    try:
        from enhanced_causal_model import EnhancedCausalModel, extract_enhanced_causal_features
        print("✅ Enhanced causal model import successful")
        return True
    except ImportError as e:
        print(f"❌ Enhanced causal model import failed: {e}")
        return False

def test_annotation_type_trainer():
    """Test that annotation type trainers support enhanced causal"""
    try:
        from annotation_type_rl_positive import AnnotationTypeTrainer
        from annotation_type_rl_nonnegative import AnnotationTypeTrainer as NonNegativeTrainer
        from annotation_type_rl_gtenegativeone import AnnotationTypeTrainer as GTENegativeOneTrainer
        
        # Test positive trainer
        positive_trainer = AnnotationTypeTrainer(base_model_type='enhanced_causal')
        print(f"✅ @Positive trainer with enhanced causal: {type(positive_trainer.model).__name__}")
        
        # Test nonnegative trainer
        nonnegative_trainer = NonNegativeTrainer(base_model_type='enhanced_causal')
        print(f"✅ @NonNegative trainer with enhanced causal: {type(nonnegative_trainer.model).__name__}")
        
        # Test gtenegativeone trainer
        gtenegativeone_trainer = GTENegativeOneTrainer(base_model_type='enhanced_causal')
        print(f"✅ @GTENegativeOne trainer with enhanced causal: {type(gtenegativeone_trainer.model).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Annotation type trainer test failed: {e}")
        return False

def test_feature_extraction():
    """Test enhanced causal feature extraction"""
    try:
        from enhanced_causal_model import extract_enhanced_causal_features
        
        # Create mock data
        mock_node = {
            'label': 'int count = 0;',
            'node_type': 'variable_declaration',
            'line': 10
        }
        
        mock_cfg_data = {
            'nodes': [mock_node],
            'edges': []
        }
        
        features = extract_enhanced_causal_features(mock_node, mock_cfg_data)
        
        if len(features) == 32:
            print(f"✅ Enhanced causal feature extraction successful: {len(features)} features")
            return True
        else:
            print(f"❌ Expected 32 features, got {len(features)}")
            return False
            
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_command_line_interface():
    """Test that command line interfaces support enhanced_causal"""
    try:
        import subprocess
        
        # Test positive script
        result = subprocess.run([
            sys.executable, 'annotation_type_rl_positive.py', '--help'
        ], capture_output=True, text=True)
        
        if 'enhanced_causal' in result.stdout:
            print("✅ @Positive script supports enhanced_causal")
        else:
            print("❌ @Positive script does not support enhanced_causal")
            return False
        
        # Test nonnegative script
        result = subprocess.run([
            sys.executable, 'annotation_type_rl_nonnegative.py', '--help'
        ], capture_output=True, text=True)
        
        if 'enhanced_causal' in result.stdout:
            print("✅ @NonNegative script supports enhanced_causal")
        else:
            print("❌ @NonNegative script does not support enhanced_causal")
            return False
        
        # Test gtenegativeone script
        result = subprocess.run([
            sys.executable, 'annotation_type_rl_gtenegativeone.py', '--help'
        ], capture_output=True, text=True)
        
        if 'enhanced_causal' in result.stdout:
            print("✅ @GTENegativeOne script supports enhanced_causal")
        else:
            print("❌ @GTENegativeOne script does not support enhanced_causal")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Command line interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Causal Model Integration in GenDATA")
    print("=" * 60)
    
    tests = [
        ("Enhanced Causal Import", test_enhanced_causal_import),
        ("Annotation Type Trainers", test_annotation_type_trainer),
        ("Feature Extraction", test_feature_extraction),
        ("Command Line Interface", test_command_line_interface)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced causal model is successfully integrated.")
        print("\n📖 Usage Examples:")
        print("   # Train @Positive with enhanced causal model")
        print("   python annotation_type_rl_positive.py --base_model enhanced_causal --episodes 50")
        print("   # Train @NonNegative with enhanced causal model")
        print("   python annotation_type_rl_nonnegative.py --base_model enhanced_causal --episodes 50")
        print("   # Train @GTENegativeOne with enhanced causal model")
        print("   python annotation_type_rl_gtenegativeone.py --base_model enhanced_causal --episodes 50")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
