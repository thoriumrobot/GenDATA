#!/usr/bin/env python3
"""
Generate CFGs for qualifying projects
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from test_lower_bound_warnings import LowerBoundWarningTester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_cfgs_for_project(project_dir: Path, project_name: str, cfg_output_dir: Path):
    """Generate CFGs for all Java files in a project"""
    logger.info(f"Generating CFGs for {project_name}")
    
    # Find Java files
    tester = LowerBoundWarningTester()
    java_files = tester.find_java_files(project_dir, max_files=None)  # Get all files
    
    if not java_files:
        logger.warning(f"No Java files found in {project_dir}")
        return 0
    
    logger.info(f"Found {len(java_files)} Java files")
    
    # Create project-specific CFG directory
    project_cfg_dir = cfg_output_dir / project_name
    project_cfg_dir.mkdir(parents=True, exist_ok=True)
    
    generated_count = 0
    for java_file in java_files:
        try:
            # Generate CFG for this file
            result = subprocess.run(
                ['python3', 'cfg.py', '--java_file', java_file, '--out_dir', str(project_cfg_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                generated_count += 1
                if generated_count % 10 == 0:
                    logger.info(f"Generated CFGs for {generated_count}/{len(java_files)} files")
            else:
                logger.debug(f"Failed to generate CFG for {java_file}: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"CFG generation timed out for {java_file}")
        except Exception as e:
            logger.debug(f"Error generating CFG for {java_file}: {e}")
    
    logger.info(f"Generated CFGs for {generated_count}/{len(java_files)} files in {project_name}")
    return generated_count


def main():
    # Load qualifying projects
    with open('project_discovery_manual/lower_bound_project_candidates.json', 'r') as f:
        data = json.load(f)
    
    qualifying = []
    for project in data.get('ranked_projects', []):
        if project.get('compilation_success') and project.get('warning_count', 0) > 0:
            qualifying.append(project)
    
    logger.info(f"Generating CFGs for {len(qualifying)} qualifying projects")
    
    # CFG output directory
    cfg_output_dir = Path('/home/ubuntu/GenDATA/cfg_output_adaptive_specimin_lower_bound')
    cfg_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone and generate CFGs
    tester = LowerBoundWarningTester(temp_dir=Path('/tmp/cfg_generation'))
    
    for project in qualifying:
        project_name = project['project_name']
        project_url = project['project_url']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {project_name}")
        logger.info(f"{'='*80}")
        
        # Clone project
        project_dir = tester.clone_repository(project_url, project_name)
        if not project_dir:
            logger.error(f"Failed to clone {project_name}")
            continue
        
        # Generate CFGs
        count = generate_cfgs_for_project(project_dir, project_name, cfg_output_dir)
        logger.info(f"Completed {project_name}: {count} CFGs generated")
    
    logger.info(f"\n{'='*80}")
    logger.info("CFG generation complete!")
    logger.info(f"CFGs saved to: {cfg_output_dir}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()

