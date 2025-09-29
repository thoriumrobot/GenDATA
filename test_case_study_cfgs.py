#!/usr/bin/env python3
"""
Test script to generate CFGs for a small subset of case study files
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cfg_generation():
    """Test CFG generation on a few case study files"""
    
    case_studies_dir = '/home/ubuntu/GenDATA/case_studies'
    output_dir = '/home/ubuntu/GenDATA/test_case_study_cfg_output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find a few Java files to test with
    test_files = []
    for root, dirs, files in os.walk(case_studies_dir):
        for file in files:
            if file.endswith('.java') and len(test_files) < 3:
                test_files.append(os.path.join(root, file))
    
    logger.info(f"Testing with {len(test_files)} files")
    
    # Import the CFG generation functions
    from cfg import generate_control_flow_graphs, save_cfgs
    
    for java_file in test_files:
        try:
            basename = os.path.splitext(os.path.basename(java_file))[0]
            cfg_output_dir = os.path.join(output_dir, basename)
            os.makedirs(cfg_output_dir, exist_ok=True)
            
            logger.info(f"Generating CFG for {basename}...")
            cfgs = generate_control_flow_graphs(java_file, cfg_output_dir)
            
            # Save the CFGs
            if cfgs:
                save_cfgs(cfgs, cfg_output_dir)
                logger.info(f"Saved CFGs for {basename}")
                
                # Create a canonical cfg.json file that the predictor expects
                import shutil
                json_files = [f for f in os.listdir(cfg_output_dir) if f.endswith('.json')]
                if json_files:
                    # Use the first JSON file as the canonical cfg.json
                    src_file = os.path.join(cfg_output_dir, json_files[0])
                    dst_file = os.path.join(cfg_output_dir, 'cfg.json')
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Created canonical cfg.json for {basename}")
            
            # Check results
            cfg_files = [f for f in os.listdir(cfg_output_dir) if f.endswith('.json')]
            logger.info(f"Generated {len(cfg_files)} CFG files for {basename}")
            
        except Exception as e:
            logger.error(f"Failed to generate CFG for {java_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    total_dirs = len([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])
    logger.info(f"Test completed. Created {total_dirs} CFG directories")

if __name__ == '__main__':
    test_cfg_generation()
