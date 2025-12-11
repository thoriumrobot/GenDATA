#!/usr/bin/env python3
"""
Project Scorer

Scores and ranks projects based on compilation success, warning count,
pattern density, and other criteria. Combines results from all analysis phases.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProjectScore:
    """Score for a project"""
    project_name: str
    project_url: str
    total_score: float
    compilation_score: float
    warning_score: float
    pattern_score: float
    size_score: float
    compilation_success: bool
    warning_count: int
    pattern_density: float
    java_files: int
    build_system: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ProjectScorer:
    """Scores and ranks projects"""
    
    def __init__(self,
                 min_warnings: int = 10,
                 max_warnings: int = 1000,
                 ideal_warning_range: tuple = (50, 500)):
        """
        Initialize scorer
        
        Args:
            min_warnings: Minimum warnings to be considered
            max_warnings: Maximum warnings to be considered
            ideal_warning_range: Ideal warning count range (min, max)
        """
        self.min_warnings = min_warnings
        self.max_warnings = max_warnings
        self.ideal_warning_min, self.ideal_warning_max = ideal_warning_range
    
    def score_compilation(self, compilation_success: bool) -> float:
        """
        Score based on compilation success
        
        Args:
            compilation_success: Whether project compiles
            
        Returns:
            Score (0-30 points)
        """
        if not compilation_success:
            return 0.0
        return 30.0
    
    def score_warnings(self, warning_count: int) -> float:
        """
        Score based on warning count
        
        Args:
            warning_count: Number of Lower Bound warnings
            
        Returns:
            Score (0-30 points)
        """
        if warning_count < self.min_warnings:
            return 10.0  # Low score for too few warnings
        
        if warning_count > self.max_warnings:
            return 10.0  # Low score for too many warnings
        
        # Ideal range: 50-500
        if self.ideal_warning_min <= warning_count <= self.ideal_warning_max:
            return 30.0
        elif self.min_warnings <= warning_count < self.ideal_warning_min:
            return 20.0
        elif self.ideal_warning_max < warning_count <= self.max_warnings:
            return 25.0
        
        return 15.0
    
    def score_patterns(self, pattern_density: Optional[float]) -> float:
        """
        Score based on pattern density
        
        Args:
            pattern_density: Overall pattern density (patterns per line)
            
        Returns:
            Score (0-20 points)
        """
        if pattern_density is None:
            return 5.0
        
        if pattern_density > 0.1:
            return 20.0
        elif pattern_density > 0.05:
            return 15.0
        elif pattern_density > 0.02:
            return 10.0
        else:
            return 5.0
    
    def score_size(self, java_files: int) -> float:
        """
        Score based on project size
        
        Args:
            java_files: Number of Java files
            
        Returns:
            Score (0-10 points)
        """
        # Ideal: 100-1000 files
        if 100 <= java_files <= 1000:
            return 10.0
        elif 50 <= java_files < 100:
            return 7.0
        elif 1000 < java_files <= 5000:
            return 8.0
        elif java_files > 5000:
            return 3.0
        else:
            return 3.0
    
    def score_project(self,
                     project_data: Dict[str, Any]) -> Optional[ProjectScore]:
        """
        Score a project based on all available data
        
        Args:
            project_data: Combined project data from all phases
            
        Returns:
            ProjectScore object or None if insufficient data
        """
        project_name = (
            project_data.get('project_name') or
            project_data.get('name') or
            project_data.get('full_name', 'unknown')
        )
        project_url = (
            project_data.get('project_url') or
            project_data.get('url') or
            project_data.get('clone_url', '')
        )
        
        # Get compilation data
        compilation_success = project_data.get('compilation_success', False)
        build_system = project_data.get('build_system', 'unknown')
        java_files = project_data.get('java_files_found', 0)
        
        # Get warning data
        warning_count = project_data.get('warning_count', 0)
        # If a warning_stats block exists, prefer its total_warnings
        if 'warning_stats' in project_data and project_data['warning_stats']:
            warning_count = project_data['warning_stats'].get('total_warnings', warning_count)
        # Do NOT override with pattern stats; those don't contain warnings
        
        # Get pattern data
        pattern_density = None
        if 'stats' in project_data and project_data['stats']:
            pattern_stats = project_data['stats']
            if isinstance(pattern_stats, dict):
                # Calculate overall density from pattern counts
                pattern_counts = pattern_stats.get('pattern_counts', {})
                total_lines = pattern_stats.get('total_lines', 0)
                if total_lines > 0:
                    total_patterns = sum(pattern_counts.values())
                    pattern_density = total_patterns / total_lines
        elif 'pattern_density' in project_data:
            pattern_density = project_data['pattern_density']
        
        # Calculate scores
        compilation_score = self.score_compilation(compilation_success)
        warning_score = self.score_warnings(warning_count)
        pattern_score = self.score_patterns(pattern_density)
        size_score = self.score_size(java_files)
        
        total_score = compilation_score + warning_score + pattern_score + size_score
        
        return ProjectScore(
            project_name=project_name,
            project_url=project_url,
            total_score=total_score,
            compilation_score=compilation_score,
            warning_score=warning_score,
            pattern_score=pattern_score,
            size_score=size_score,
            compilation_success=compilation_success,
            warning_count=warning_count,
            pattern_density=pattern_density or 0.0,
            java_files=java_files,
            build_system=build_system
        )
    
    def merge_project_data(self,
                          github_projects: List[Dict],
                          pattern_analyses: List[Dict],
                          compilation_results: List[Dict],
                          warning_results: List[Dict]) -> List[Dict]:
        """
        Merge data from all phases into unified project records
        
        Args:
            github_projects: Projects from GitHub search
            pattern_analyses: Pattern analysis results
            compilation_results: Compilation test results
            warning_results: Warning test results
            
        Returns:
            List of merged project dictionaries
        """
        # Create lookup dictionaries
        projects_by_name = {}
        
        # Add GitHub projects
        for project in github_projects:
            name = project.get('full_name') or project.get('name', '')
            if name:
                projects_by_name[name] = project.copy()
                short = name.split('/')[-1]
                # Map short name to same record for easier merging
                if short not in projects_by_name:
                    projects_by_name[short] = projects_by_name[name]
        
        # Merge pattern analyses
        for analysis in pattern_analyses:
            name = analysis.get('project_name', '')
            if name in projects_by_name:
                projects_by_name[name].update({
                    # Also store under 'stats' so downstream scoring sees density
                    'stats': analysis.get('stats'),
                    'pattern_stats': analysis.get('stats'),
                    'pattern_analysis': analysis
                })
            else:
                # Try matching by short repo name if full_name not present
                short = name.split('/')[-1] if name else ''
                for key in list(projects_by_name.keys()):
                    if key.endswith('/' + short) or key == short:
                        projects_by_name[key].update({
                            'stats': analysis.get('stats'),
                            'pattern_stats': analysis.get('stats'),
                            'pattern_analysis': analysis
                        })
                        break
        
        # Merge compilation results
        for result in compilation_results:
            name = result.get('project_name', '')
            if name in projects_by_name:
                projects_by_name[name].update({
                    'compilation_success': result.get('compilation_success', False),
                    'build_system': result.get('build_system', 'unknown'),
                    'java_files_found': result.get('java_files_found', 0),
                    'compilation_result': result
                })
        
        # Merge warning results
        for result in warning_results:
            name = result.get('project_name', '')
            if name in projects_by_name:
                projects_by_name[name].update({
                    'warning_stats': result.get('stats'),
                    'warning_count': result.get('stats', {}).get('total_warnings', 0) if result.get('stats') else 0,
                    'warning_result': result
                })
            else:
                short = name.split('/')[-1] if name else ''
                for key in list(projects_by_name.keys()):
                    if key.endswith('/' + short) or key == short:
                        projects_by_name[key].update({
                            'warning_stats': result.get('stats'),
                            'warning_count': result.get('stats', {}).get('total_warnings', 0) if result.get('stats') else 0,
                            'warning_result': result
                        })
                        break
        
        # Deduplicate projects by full_name/url
        unique = []
        seen_keys = set()
        for proj in projects_by_name.values():
            key = proj.get('full_name') or proj.get('project_url') or proj.get('url') or proj.get('name')
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(proj)
        return unique
    
    def rank_projects(self, scores: List[ProjectScore]) -> List[ProjectScore]:
        """
        Rank projects by total score
        
        Args:
            scores: List of project scores
            
        Returns:
            Sorted list (highest score first)
        """
        return sorted(scores, key=lambda s: s.total_score, reverse=True)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_ranked_projects(scores: List[ProjectScore], output_file: str):
    """Save ranked projects to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'metadata': {
            'total_projects': len(scores),
            'high_score_projects': sum(1 for s in scores if s.total_score >= 70),
            'generated_at': __import__('datetime').datetime.now().isoformat()
        },
        'ranked_projects': [score.to_dict() for score in scores]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Saved {len(scores)} ranked projects to {output_file}")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Score and rank projects')
    parser.add_argument('--github-projects', help='GitHub projects JSON file')
    parser.add_argument('--pattern-analysis', help='Pattern analysis JSON file')
    parser.add_argument('--compilation-results', help='Compilation results JSON file')
    parser.add_argument('--warning-results', help='Warning test results JSON file')
    parser.add_argument('--combined-input', help='Combined input JSON file (alternative to separate files)')
    parser.add_argument('--output', default='ranked_projects.json', help='Output JSON file')
    parser.add_argument('--min-warnings', type=int, default=10, help='Minimum warnings')
    parser.add_argument('--max-warnings', type=int, default=1000, help='Maximum warnings')
    parser.add_argument('--min-score', type=float, default=70.0, help='Minimum score to include')
    
    args = parser.parse_args()
    
    scorer = ProjectScorer(
        min_warnings=args.min_warnings,
        max_warnings=args.max_warnings
    )
    
    # Load data
    if args.combined_input:
        # Load from single combined file
        combined_data = load_json_file(args.combined_input)
        projects = combined_data.get('projects', [])
    else:
        # Load from separate files and merge
        github_projects = []
        pattern_analyses = []
        compilation_results = []
        warning_results = []
        
        if args.github_projects:
            github_data = load_json_file(args.github_projects)
            github_projects = github_data.get('projects', [])
        
        if args.pattern_analysis:
            pattern_data = load_json_file(args.pattern_analysis)
            pattern_analyses = pattern_data.get('analyses', [])
        
        if args.compilation_results:
            compilation_data = load_json_file(args.compilation_results)
            compilation_results = compilation_data.get('results', [])
        
        if args.warning_results:
            warning_data = load_json_file(args.warning_results)
            warning_results = warning_data.get('results', [])
        
        projects = scorer.merge_project_data(
            github_projects,
            pattern_analyses,
            compilation_results,
            warning_results
        )
    
    logger.info(f"Scoring {len(projects)} projects")
    
    # Score projects
    scores = []
    for project in projects:
        score = scorer.score_project(project)
        if score:
            scores.append(score)
    
    # Rank projects
    ranked = scorer.rank_projects(scores)
    
    # Filter by minimum score
    filtered = [s for s in ranked if s.total_score >= args.min_score]
    
    logger.info(f"Scored {len(scores)} projects, {len(filtered)} meet minimum score of {args.min_score}")
    
    # Print top 10
    logger.info("\nTop 10 projects:")
    for i, score in enumerate(filtered[:10], 1):
        logger.info(f"{i}. {score.project_name}: {score.total_score:.1f} points "
                   f"(warnings: {score.warning_count}, density: {score.pattern_density:.4f})")
    
    # Save results
    save_ranked_projects(filtered, args.output)
    
    return 0

if __name__ == '__main__':
    exit(main())

