#!/usr/bin/env python3
"""
Code Pattern Analyzer

Analyzes Java code for patterns that are likely to trigger Lower Bound Checker warnings.
Clones repositories and scans Java files for array access, loop variables, comparisons, etc.
"""

import os
import json
import re
import subprocess
import tempfile
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lower Bound Checker warning patterns
LOWER_BOUND_PATTERNS = {
    'array_access': r'\[.*?\]',  # Array access: array[index]
    'loop_variables': r'for\s*\([^)]*\b([ij]|index|idx|pos)\s*=',  # Loop variables i, j, index, etc.
    'array_length': r'\.length\b',  # Array length access
    'comparison_zero': r'[><=!]+\s*0\b',  # Comparisons with 0
    'comparison_neg_one': r'[><=!]+\s*-1\b',  # Comparisons with -1
    'array_creation': r'new\s+\w+\[',  # Array creation: new Type[size]
    'index_variables': r'\b(index|idx|i|j|k|pos|offset)\s*[=+\-*]',  # Index variable assignments
    'array_bounds_check': r'\.length\s*[><=]',  # Array bounds checks
    'negative_index': r'\[.*?-\s*\d+\]',  # Negative index access
    'parameter_array_access': r'\w+\s*\[.*?\w+.*?\]',  # Parameter used in array access
}

@dataclass
class PatternStats:
    """Statistics about code patterns"""
    total_lines: int
    total_files: int
    pattern_counts: Dict[str, int]
    pattern_density: Dict[str, float]  # Patterns per line
    high_density_files: List[Tuple[str, float]]  # (file_path, density)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_lines': self.total_lines,
            'total_files': self.total_files,
            'pattern_counts': self.pattern_counts,
            'pattern_density': self.pattern_density,
            'high_density_files': [(f, d) for f, d in self.high_density_files]
        }

@dataclass
class ProjectPatternAnalysis:
    """Pattern analysis for a project"""
    project_name: str
    project_url: str
    clone_success: bool
    stats: Optional[PatternStats]
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'project_name': self.project_name,
            'project_url': self.project_url,
            'clone_success': self.clone_success,
            'stats': self.stats.to_dict() if self.stats else None,
            'error_message': self.error_message
        }

class CodePatternAnalyzer:
    """Analyzes Java code for Lower Bound Checker patterns"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize pattern analyzer
        
        Args:
            temp_dir: Temporary directory for cloning repositories
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / 'github_analysis'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.patterns = LOWER_BOUND_PATTERNS
    
    def clone_repository(self, clone_url: str, project_name: str) -> Optional[Path]:
        """
        Clone a repository (shallow clone)
        
        Args:
            clone_url: Git clone URL
            project_name: Project name for directory
            
        Returns:
            Path to cloned repository or None if failed
        """
        repo_dir = self.temp_dir / project_name.replace('/', '_')
        
        # Remove existing directory if it exists
        if repo_dir.exists():
            logger.info(f"Removing existing directory: {repo_dir}")
            shutil.rmtree(repo_dir)
        
        try:
            logger.info(f"Cloning {clone_url} to {repo_dir}")
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', clone_url, str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned {project_name}")
                return repo_dir
            else:
                logger.error(f"Failed to clone {project_name}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout cloning {project_name}")
            return None
        except Exception as e:
            logger.error(f"Error cloning {project_name}: {e}")
            return None
    
    def find_java_files(self, repo_dir: Path) -> List[Path]:
        """Find all Java files in repository"""
        java_files = []
        
        # Exclude common directories
        exclude_dirs = {
            '.git', 'target', 'build', 'bin', 'out', 'dist',
            'test', 'tests', 'test-src', 'test-sources',
            'generated', 'generated-sources', '.gradle', '.mvn'
        }
        
        for root, dirs, files in os.walk(repo_dir):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if file.endswith('.java'):
                    java_files.append(Path(root) / file)
        
        return java_files
    
    def analyze_file(self, java_file: Path) -> Dict[str, int]:
        """
        Analyze a Java file for patterns
        
        Args:
            java_file: Path to Java file
            
        Returns:
            Dictionary of pattern counts
        """
        pattern_counts = {pattern: 0 for pattern in self.patterns.keys()}
        
        try:
            with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Count patterns
                for pattern_name, pattern_regex in self.patterns.items():
                    matches = re.findall(pattern_regex, content, re.MULTILINE)
                    pattern_counts[pattern_name] = len(matches)
        
        except Exception as e:
            logger.warning(f"Error analyzing {java_file}: {e}")
        
        return pattern_counts
    
    def analyze_project(self, project_name: str, clone_url: str) -> ProjectPatternAnalysis:
        """
        Analyze a project for Lower Bound Checker patterns
        
        Args:
            project_name: Name of the project
            clone_url: Git clone URL
            
        Returns:
            ProjectPatternAnalysis object
        """
        logger.info(f"Analyzing patterns in {project_name}")
        
        # Clone repository
        repo_dir = self.clone_repository(clone_url, project_name)
        
        if not repo_dir:
            return ProjectPatternAnalysis(
                project_name=project_name,
                project_url=clone_url,
                clone_success=False,
                stats=None,
                error_message="Failed to clone repository"
            )
        
        try:
            # Find Java files
            java_files = self.find_java_files(repo_dir)
            
            if not java_files:
                return ProjectPatternAnalysis(
                    project_name=project_name,
                    project_url=clone_url,
                    clone_success=True,
                    stats=None,
                    error_message="No Java files found"
                )
            
            logger.info(f"Found {len(java_files)} Java files")
            
            # Analyze files
            total_pattern_counts = {pattern: 0 for pattern in self.patterns.keys()}
            total_lines = 0
            file_densities = []
            
            for java_file in java_files:
                pattern_counts = self.analyze_file(java_file)
                
                # Count lines
                try:
                    with open(java_file, 'r', encoding='utf-8', errors='ignore') as f:
                        file_lines = len(f.readlines())
                        total_lines += file_lines
                except Exception:
                    file_lines = 0
                
                # Accumulate pattern counts
                for pattern_name, count in pattern_counts.items():
                    total_pattern_counts[pattern_name] += count
                
                # Calculate file density
                if file_lines > 0:
                    file_pattern_count = sum(pattern_counts.values())
                    file_density = file_pattern_count / file_lines
                    file_densities.append((str(java_file.relative_to(repo_dir)), file_density))
            
            # Calculate overall density
            pattern_density = {}
            for pattern_name, count in total_pattern_counts.items():
                pattern_density[pattern_name] = count / total_lines if total_lines > 0 else 0.0
            
            # Overall density (total patterns per line)
            total_patterns = sum(total_pattern_counts.values())
            overall_density = total_patterns / total_lines if total_lines > 0 else 0.0
            
            # Find high-density files (top 10%)
            file_densities.sort(key=lambda x: x[1], reverse=True)
            high_density_count = max(1, len(file_densities) // 10)
            high_density_files = file_densities[:high_density_count]
            
            stats = PatternStats(
                total_lines=total_lines,
                total_files=len(java_files),
                pattern_counts=total_pattern_counts,
                pattern_density=pattern_density,
                high_density_files=high_density_files
            )
            
            logger.info(f"Analysis complete: {total_lines} lines, {total_patterns} patterns, density: {overall_density:.4f}")
            
            return ProjectPatternAnalysis(
                project_name=project_name,
                project_url=clone_url,
                clone_success=True,
                stats=stats,
                error_message=None
            )
        
        except Exception as e:
            logger.error(f"Error analyzing {project_name}: {e}")
            return ProjectPatternAnalysis(
                project_name=project_name,
                project_url=clone_url,
                clone_success=True,
                stats=None,
                error_message=str(e)
            )
        
        finally:
            # Clean up cloned repository
            if repo_dir.exists():
                try:
                    shutil.rmtree(repo_dir)
                    logger.debug(f"Cleaned up {repo_dir}")
                except Exception:
                    pass  # Ignore cleanup errors

def load_projects(input_file: str) -> List[Dict[str, Any]]:
    """Load projects from JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)
        return data.get('projects', [])

def save_analysis(analyses: List[ProjectPatternAnalysis], output_file: str):
    """Save pattern analysis to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    analyses_dict = {
        'metadata': {
            'total_projects': len(analyses),
            'successful_analyses': sum(1 for a in analyses if a.stats),
            'generated_at': __import__('datetime').datetime.now().isoformat()
        },
        'analyses': [analysis.to_dict() for analysis in analyses]
    }
    
    with open(output_path, 'w') as f:
        json.dump(analyses_dict, f, indent=2)
    
    logger.info(f"Saved {len(analyses)} analyses to {output_file}")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Analyze Java projects for Lower Bound Checker patterns')
    parser.add_argument('--input', required=True, help='Input JSON file with projects')
    parser.add_argument('--output', default='pattern_analysis.json', help='Output JSON file')
    parser.add_argument('--max-projects', type=int, help='Maximum number of projects to analyze')
    parser.add_argument('--temp-dir', help='Temporary directory for cloning')
    
    args = parser.parse_args()
    
    # Load projects
    projects = load_projects(args.input)
    
    if args.max_projects:
        projects = projects[:args.max_projects]
    
    logger.info(f"Analyzing {len(projects)} projects")
    
    # Analyze projects
    analyzer = CodePatternAnalyzer(temp_dir=args.temp_dir)
    analyses = []
    
    for i, project in enumerate(projects, 1):
        logger.info(f"Processing project {i}/{len(projects)}: {project.get('name', 'unknown')}")
        
        clone_url = project.get('clone_url') or project.get('url', '')
        project_name = project.get('full_name') or project.get('name', 'unknown')
        
        if not clone_url:
            logger.warning(f"No clone URL for project {project_name}")
            continue
        
        analysis = analyzer.analyze_project(project_name, clone_url)
        analyses.append(analysis)
    
    # Save results
    save_analysis(analyses, args.output)
    
    successful = sum(1 for a in analyses if a.stats)
    logger.info(f"Completed: {successful}/{len(analyses)} successful analyses")
    
    return 0

if __name__ == '__main__':
    exit(main())

