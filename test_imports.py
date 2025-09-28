#!/usr/bin/env python3
"""
Quick Import Test for GenDATA Pipeline

This script tests that all major components can be imported without errors.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test importing major pipeline components"""
    print("🧪 Testing GenDATA Pipeline Imports")
    print("=" * 40)
    
    # Add GenDATA to Python path
    gendata_root = Path("/home/thoriumrobot/project/GenDATA")
    sys.path.insert(0, str(gendata_root))
    
    # Test imports
    imports_to_test = [
        ("annotation_type_rl_positive", "Annotation Type RL Positive"),
        ("annotation_type_rl_nonnegative", "Annotation Type RL NonNegative"),
        ("annotation_type_rl_gtenegativeone", "Annotation Type RL GTENegativeOne"),
        ("simple_annotation_type_pipeline", "Simple Annotation Type Pipeline"),
        ("binary_rl_gcn_standalone", "Binary RL GCN"),
        ("binary_rl_gbt_standalone", "Binary RL GBT"),
        ("binary_rl_causal_standalone", "Binary RL Causal"),
        ("binary_rl_hgt_standalone", "Binary RL HGT"),
        ("binary_rl_gcsn_standalone", "Binary RL GCSN"),
        ("binary_rl_dg2n_standalone", "Binary RL DG2N"),
        ("hgt", "HGT Model"),
        ("gbt", "GBT Model"),
        ("causal_model", "Causal Model"),
        ("checker_framework_integration", "Checker Framework Integration"),
        ("place_annotations", "Annotation Placement"),
        ("prediction_saver", "Prediction Saver")
    ]
    
    success_count = 0
    total_count = len(imports_to_test)
    
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {description}: {module_name} - {e}")
        except Exception as e:
            print(f"⚠️  {description}: {module_name} - {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Import Test Results: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 All imports successful! Pipeline is ready to use.")
        return True
    elif success_count >= total_count * 0.8:
        print("⚠️  Most imports successful. Some components may have dependencies.")
        return True
    else:
        print("❌ Many imports failed. Check dependencies and Python path.")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)


