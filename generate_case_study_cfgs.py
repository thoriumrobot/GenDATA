#!/usr/bin/env python3
"""
Generate CFGs for case study files
This script creates slices and CFGs specifically for the case study Java files
so that the prediction pipeline can work properly.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_case_study_cfgs():
    """Generate CFGs for case study files using the existing pipeline infrastructure"""
    
    case_studies_dir = '/home/ubuntu/GenDATA/case_studies'
    output_dir = '/home/ubuntu/GenDATA/case_study_cfg_output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Java files in case studies
    java_files = []
    for root, dirs, files in os.walk(case_studies_dir):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(java_files)} Java files in case studies")
    
    # Generate CFGs for each Java file using the existing CFG generation pipeline
    from pipeline import run_cfg_generation
    
    # Create a temporary directory structure that mimics what the pipeline expects
    temp_slices_dir = '/home/ubuntu/GenDATA/temp_case_study_slices'
    os.makedirs(temp_slices_dir, exist_ok=True)
    
    processed_count = 0
    for java_file in java_files:
        try:
            # Copy the Java file to the temp slices directory with a unique name
            basename = os.path.splitext(os.path.basename(java_file))[0]
            temp_java_file = os.path.join(temp_slices_dir, f"{basename}_case_study.java")
            
            # Copy the file
            import shutil
            shutil.copy2(java_file, temp_java_file)
            
            # Generate CFG for this file
            cfg_output_dir = os.path.join(output_dir, basename)
            os.makedirs(cfg_output_dir, exist_ok=True)
            
            # Use the existing CFG generation from cfg.py
            from cfg import generate_control_flow_graphs, save_cfgs
            
            try:
                # Generate CFGs for this Java file
                cfgs = generate_control_flow_graphs(temp_java_file, cfg_output_dir)
                
                # Save the CFGs to JSON files
                if cfgs:
                    save_cfgs(cfgs, cfg_output_dir)
                    
                    # Create a canonical cfg.json file that the predictor expects
                    import shutil
                    json_files = [f for f in os.listdir(cfg_output_dir) if f.endswith('.json')]
                    if json_files:
                        # Use the first JSON file as the canonical cfg.json
                        src_file = os.path.join(cfg_output_dir, json_files[0])
                        dst_file = os.path.join(cfg_output_dir, 'cfg.json')
                        shutil.copy2(src_file, dst_file)
                        
                        processed_count += 1
                        logger.info(f"Generated CFG for {basename} ({len(json_files)} files)")
                    else:
                        logger.warning(f"No CFG files generated for {basename}")
                else:
                    logger.warning(f"No CFGs generated for {basename}")
            except Exception as e:
                logger.warning(f"Failed to generate CFG for {basename}: {e}")
            
            # Clean up temp file
            os.remove(temp_java_file)
            
        except Exception as e:
            logger.error(f"Error processing {java_file}: {e}")
    
    logger.info(f"Successfully generated CFGs for {processed_count}/{len(java_files)} case study files")
    return output_dir

def main():
    """Main function to generate case study CFGs"""
    logger.info("Starting case study CFG generation...")
    
    output_dir = generate_case_study_cfgs()
    
    logger.info(f"Case study CFGs generated in: {output_dir}")
    
    # Verify some CFGs were created
    cfg_count = len([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))])
    logger.info(f"Created {cfg_count} CFG directories")
    
    if cfg_count > 0:
        logger.info("✅ Case study CFG generation completed successfully!")
        return 0
    else:
        logger.error("❌ No CFGs were generated")
        return 1

if __name__ == '__main__':
    sys.exit(main())
