#!/usr/bin/env python3
"""
Prepare Outline Projects for Evaluation

This script downloads and prepares the evaluation projects mentioned in GenDATA outline.md:
- Agrona
- Hipparchus
- Eclipse Collections

It sets up the project structure similar to existing case studies (Guava, JFreeChart, Plume-lib).
"""

import os
import subprocess
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project GitHub URLs
PROJECT_URLS = {
    'agrona': 'https://github.com/real-logic/agrona.git',
    'hipparchus': 'https://github.com/Hipparchus-Math/hipparchus.git',
    'eclipse-collections': 'https://github.com/eclipse/eclipse-collections.git'
}

# Project names for directory structure
PROJECT_NAMES = {
    'agrona': 'agrona',
    'hipparchus': 'hipparchus',
    'eclipse-collections': 'eclipse-collections'
}

def clone_project(name, url, target_dir):
    """Clone a project from GitHub"""
    project_path = Path(target_dir) / name
    
    if project_path.exists():
        logger.info(f"Project {name} already exists at {project_path}, skipping clone")
        return project_path
    
    logger.info(f"Cloning {name} from {url}...")
    try:
        subprocess.run(['git', 'clone', url, str(project_path)], check=True)
        logger.info(f"Successfully cloned {name}")
        return project_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone {name}: {e}")
        return None

def create_ground_truth_template(project_path, project_name):
    """Create a template ground_truth.json file"""
    ground_truth_path = project_path / 'ground_truth.json'
    
    if ground_truth_path.exists():
        logger.info(f"ground_truth.json already exists for {project_name}")
        return
    
    # Create empty ground truth template
    template = {
        "project": project_name,
        "annotations": [],
        "note": "Ground truth annotations for evaluation. Format: [{\"file\": \"path/to/file.java\", \"line\": 42, \"annotation\": \"@Positive\", \"target\": \"parameter\"}]"
    }
    
    with open(ground_truth_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created ground_truth.json template for {project_name}")

def prepare_project(name, url, case_studies_dir):
    """Prepare a single project"""
    logger.info(f"Preparing project: {name}")
    
    # Clone project
    project_path = clone_project(name, url, case_studies_dir)
    if not project_path:
        return False
    
    # Create ground truth template
    create_ground_truth_template(project_path, name)
    
    logger.info(f"Successfully prepared {name}")
    return True

def main():
    """Main function"""
    case_studies_dir = Path('/home/ubuntu/GenDATA/case_studies')
    case_studies_dir.mkdir(exist_ok=True)
    
    logger.info("Preparing outline projects for evaluation...")
    logger.info(f"Target directory: {case_studies_dir}")
    
    success_count = 0
    for name, url in PROJECT_URLS.items():
        if prepare_project(name, url, case_studies_dir):
            success_count += 1
    
    logger.info(f"Successfully prepared {success_count}/{len(PROJECT_URLS)} projects")
    
    if success_count == len(PROJECT_URLS):
        logger.info("All projects prepared successfully!")
        return 0
    else:
        logger.warning(f"Only {success_count}/{len(PROJECT_URLS)} projects prepared")
        return 1

if __name__ == '__main__':
    exit(main())

