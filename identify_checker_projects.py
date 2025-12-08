#!/usr/bin/env python3
"""
Identify Suitable Projects for Checker Evaluation

This script identifies projects suitable for evaluation with different checkers
by searching for relevant code patterns.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Set
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CASE_STUDIES_DIR = Path('/home/ubuntu/GenDATA/case_studies')

def find_sql_related_code(project_path: Path) -> Dict[str, any]:
    """Find SQL-related code patterns in a project."""
    logger.info(f"Searching for SQL-related code in {project_path.name}...")
    
    patterns = {
        'executeQuery': 0,
        'executeUpdate': 0,
        'execute': 0,
        'PreparedStatement': 0,
        'Statement': 0,
        'Connection': 0,
        'string_concatenation': 0,  # String concatenation with SQL
        'sql_queries': 0  # Files with SQL queries
    }
    
    sql_files = []
    
    try:
        for java_file in project_path.rglob('*.java'):
            try:
                with open(java_file, 'r', errors='ignore') as f:
                    content = f.read()
                    
                    # Check for SQL-related patterns
                    if 'executeQuery' in content:
                        patterns['executeQuery'] += content.count('executeQuery')
                    if 'executeUpdate' in content:
                        patterns['executeUpdate'] += content.count('executeUpdate')
                    if 'execute(' in content and 'Statement' in content:
                        patterns['execute'] += 1
                    if 'PreparedStatement' in content:
                        patterns['PreparedStatement'] += content.count('PreparedStatement')
                    if 'Statement' in content and 'execute' in content:
                        patterns['Statement'] += 1
                    if 'Connection' in content and ('execute' in content or 'prepare' in content):
                        patterns['Connection'] += 1
                    
                    # Check for string concatenation with SQL-like patterns
                    if ('SELECT' in content or 'INSERT' in content or 'UPDATE' in content or 'DELETE' in content) and '+' in content:
                        patterns['string_concatenation'] += 1
                    
                    # Count files with SQL queries
                    if any(keyword in content for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE']):
                        patterns['sql_queries'] += 1
                        sql_files.append(str(java_file))
            except Exception as e:
                logger.debug(f"Error reading {java_file}: {e}")
                continue
    except Exception as e:
        logger.warning(f"Error searching project: {e}")
    
    total_score = sum(patterns.values())
    
    return {
        'patterns': patterns,
        'sql_files': sql_files[:10],  # Limit to first 10
        'total_score': total_score,
        'suitable': total_score > 5  # Threshold for suitability
    }

def find_signature_related_code(project_path: Path) -> Dict[str, any]:
    """Find signature/reflection-related code patterns in a project."""
    logger.info(f"Searching for signature/reflection code in {project_path.name}...")
    
    patterns = {
        'Class.forName': 0,
        'Class.getName': 0,
        'getMethod': 0,
        'getDeclaredMethod': 0,
        'MethodDescriptor': 0,
        'FieldDescriptor': 0,
        'reflection': 0,
        'signature_files': 0
    }
    
    signature_files = []
    
    try:
        for java_file in project_path.rglob('*.java'):
            try:
                with open(java_file, 'r', errors='ignore') as f:
                    content = f.read()
                    
                    # Check for reflection/signature patterns
                    if 'Class.forName' in content:
                        patterns['Class.forName'] += content.count('Class.forName')
                    if 'Class.getName' in content or '.getName()' in content:
                        patterns['Class.getName'] += content.count('.getName()')
                    if 'getMethod' in content:
                        patterns['getMethod'] += content.count('getMethod')
                    if 'getDeclaredMethod' in content:
                        patterns['getDeclaredMethod'] += content.count('getDeclaredMethod')
                    if 'MethodDescriptor' in content:
                        patterns['MethodDescriptor'] += content.count('MethodDescriptor')
                    if 'FieldDescriptor' in content or 'L' in content and ';' in content:  # Field descriptor pattern
                        patterns['FieldDescriptor'] += 1
                    if 'java.lang.reflect' in content:
                        patterns['reflection'] += 1
                    
                    # Count files with signature-related code
                    if any(pattern in content for pattern in ['Class.forName', 'Class.getName', 'getMethod', 'MethodDescriptor']):
                        patterns['signature_files'] += 1
                        signature_files.append(str(java_file))
            except Exception as e:
                logger.debug(f"Error reading {java_file}: {e}")
                continue
    except Exception as e:
        logger.warning(f"Error searching project: {e}")
    
    total_score = sum(patterns.values())
    
    return {
        'patterns': patterns,
        'signature_files': signature_files[:10],  # Limit to first 10
        'total_score': total_score,
        'suitable': total_score > 3  # Threshold for suitability
    }

def identify_projects_for_checkers() -> Dict[str, Dict[str, any]]:
    """Identify suitable projects for each checker."""
    logger.info("=" * 80)
    logger.info("Identifying Suitable Projects for Checker Evaluation")
    logger.info("=" * 80)
    
    results = {
        'sql_quotes': {'suitable': [], 'unsuitable': []},
        'signature_string': {'suitable': [], 'unsuitable': []}
    }
    
    if not CASE_STUDIES_DIR.exists():
        logger.error(f"Case studies directory not found: {CASE_STUDIES_DIR}")
        return results
    
    # Get all projects
    projects = [d for d in CASE_STUDIES_DIR.iterdir() if d.is_dir()]
    logger.info(f"Found {len(projects)} projects to analyze")
    
    for project_path in projects:
        project_name = project_path.name
        logger.info(f"\nAnalyzing project: {project_name}")
        
        # Check SQL Quotes suitability
        sql_analysis = find_sql_related_code(project_path)
        if sql_analysis['suitable']:
            results['sql_quotes']['suitable'].append({
                'name': project_name,
                'score': sql_analysis['total_score'],
                'patterns': sql_analysis['patterns'],
                'files_found': len(sql_analysis['sql_files'])
            })
            logger.info(f"  ✅ Suitable for SQL Quotes Checker (score: {sql_analysis['total_score']})")
        else:
            results['sql_quotes']['unsuitable'].append({
                'name': project_name,
                'score': sql_analysis['total_score']
            })
            logger.info(f"  ❌ Not suitable for SQL Quotes Checker (score: {sql_analysis['total_score']})")
        
        # Check Signature String suitability
        signature_analysis = find_signature_related_code(project_path)
        if signature_analysis['suitable']:
            results['signature_string']['suitable'].append({
                'name': project_name,
                'score': signature_analysis['total_score'],
                'patterns': signature_analysis['patterns'],
                'files_found': len(signature_analysis['signature_files'])
            })
            logger.info(f"  ✅ Suitable for Signature String Checker (score: {signature_analysis['total_score']})")
        else:
            results['signature_string']['unsuitable'].append({
                'name': project_name,
                'score': signature_analysis['total_score']
            })
            logger.info(f"  ❌ Not suitable for Signature String Checker (score: {signature_analysis['total_score']})")
    
    return results

def main():
    """Main function"""
    results = identify_projects_for_checkers()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    logger.info("\nSQL Quotes Checker - Suitable Projects:")
    for project in results['sql_quotes']['suitable']:
        logger.info(f"  - {project['name']} (score: {project['score']}, files: {project['files_found']})")
    
    logger.info("\nSignature String Checker - Suitable Projects:")
    for project in results['signature_string']['suitable']:
        logger.info(f"  - {project['name']} (score: {project['score']}, files: {project['files_found']})")
    
    # Save results
    import json
    results_file = Path('/home/ubuntu/GenDATA/checker_project_identification.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    return results

if __name__ == '__main__':
    main()

