#!/usr/bin/env python3
"""
Find Lower Bound Checker Projects

Main orchestration script that runs the complete pipeline:
1. Search GitHub for Java projects
2. Analyze code patterns
3. Test compilation
4. Test Lower Bound warnings
5. Score and rank projects

This script coordinates all phases of the project discovery pipeline.
"""

import os
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectDiscoveryPipeline:
    """Orchestrates the complete project discovery pipeline"""
    
    def __init__(self, work_dir: Optional[str] = None):
        """
        Initialize pipeline
        
        Args:
            work_dir: Working directory for intermediate files
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / 'project_discovery'
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Intermediate file paths
        self.github_projects_file = self.work_dir / 'github_projects.json'
        self.pattern_analysis_file = self.work_dir / 'pattern_analysis.json'
        self.compilation_results_file = self.work_dir / 'compilation_results.json'
        self.warning_results_file = self.work_dir / 'warning_test_results.json'
        self.final_output_file = self.work_dir / 'lower_bound_project_candidates.json'
    
    def run_phase(self, phase_name: str, script_path: str, args: List[str]) -> bool:
        """
        Run a pipeline phase
        
        Args:
            phase_name: Name of the phase
            script_path: Path to script to run
            args: Arguments to pass to script
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"=== Running Phase: {phase_name} ===")
        
        cmd = ['python3', script_path] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per phase
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {phase_name} completed successfully")
                return True
            else:
                logger.error(f"❌ {phase_name} failed:")
                logger.error(result.stderr)
                return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {phase_name} timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running {phase_name}: {e}")
            return False
    
    def phase1_github_search(self,
                            max_projects: int = 100,
                            min_stars: int = 10,
                            github_token: Optional[str] = None,
                            use_all_queries: bool = True) -> bool:
        """Phase 1: Search GitHub for Java projects"""
        args = [
            '--output', str(self.github_projects_file),
            '--max-results', str(max_projects),
            '--min-stars', str(min_stars)
        ]
        
        if use_all_queries:
            args.append('--use-all-queries')
        else:
            args.extend(['--query', 'language:java stars:>10'])
        
        if github_token:
            args.extend(['--github-token', github_token])
        
        return self.run_phase(
            'GitHub Search',
            'github_project_finder.py',
            args
        )
    
    def phase2_pattern_analysis(self,
                               max_projects: Optional[int] = None) -> bool:
        """Phase 2: Analyze code patterns"""
        args = [
            '--input', str(self.github_projects_file),
            '--output', str(self.pattern_analysis_file)
        ]
        
        if max_projects:
            args.extend(['--max-projects', str(max_projects)])
        
        return self.run_phase(
            'Pattern Analysis',
            'analyze_code_patterns.py',
            args
        )
    
    def phase3_compilation_test(self,
                               max_projects: Optional[int] = None,
                               timeout: int = 600) -> bool:
        """Phase 3: Test compilation"""
        args = [
            '--input', str(self.pattern_analysis_file),
            '--output', str(self.compilation_results_file),
            '--timeout', str(timeout)
        ]
        
        if max_projects:
            args.extend(['--max-projects', str(max_projects)])
        
        return self.run_phase(
            'Compilation Test',
            'test_project_compilation.py',
            args
        )
    
    def phase4_warning_test(self,
                           max_projects: Optional[int] = None,
                           max_files: int = 100,
                           timeout: int = 600,
                           checker_cp: Optional[str] = None) -> bool:
        """Phase 4: Test Lower Bound warnings"""
        args = [
            '--input', str(self.compilation_results_file),
            '--output', str(self.warning_results_file),
            '--max-files', str(max_files),
            '--timeout', str(timeout)
        ]
        
        if max_projects:
            args.extend(['--max-projects', str(max_projects)])
        
        if checker_cp:
            args.extend(['--checker-cp', checker_cp])
        
        return self.run_phase(
            'Warning Test',
            'test_lower_bound_warnings.py',
            args
        )
    
    def phase5_scoring(self,
                      min_score: float = 70.0,
                      min_warnings: int = 10,
                      max_warnings: int = 1000) -> bool:
        """Phase 5: Score and rank projects"""
        args = [
            '--github-projects', str(self.github_projects_file),
            '--pattern-analysis', str(self.pattern_analysis_file),
            '--compilation-results', str(self.compilation_results_file),
            '--warning-results', str(self.warning_results_file),
            '--output', str(self.final_output_file),
            '--min-score', str(min_score),
            '--min-warnings', str(min_warnings),
            '--max-warnings', str(max_warnings)
        ]
        
        return self.run_phase(
            'Scoring and Ranking',
            'score_projects.py',
            args
        )
    
    def run_pipeline(self,
                    max_projects: int = 100,
                    min_stars: int = 10,
                    min_score: float = 70.0,
                    min_warnings: int = 10,
                    max_warnings: int = 1000,
                    github_token: Optional[str] = None,
                    checker_cp: Optional[str] = None,
                    skip_phases: Optional[List[int]] = None) -> bool:
        """
        Run the complete pipeline
        
        Args:
            max_projects: Maximum projects to process
            min_stars: Minimum GitHub stars
            min_score: Minimum score to include in final output
            min_warnings: Minimum warnings required
            max_warnings: Maximum warnings allowed
            github_token: GitHub API token
            checker_cp: Checker Framework classpath
            skip_phases: List of phase numbers to skip (1-5)
            
        Returns:
            True if pipeline completed successfully
        """
        skip_phases = skip_phases or []
        
        logger.info("=" * 60)
        logger.info("Starting Lower Bound Checker Project Discovery Pipeline")
        logger.info("=" * 60)
        
        # Phase 1: GitHub Search
        if 1 not in skip_phases:
            if not self.phase1_github_search(
                max_projects=max_projects,
                min_stars=min_stars,
                github_token=github_token
            ):
                logger.error("Pipeline failed at Phase 1")
                return False
        else:
            logger.info("Skipping Phase 1: GitHub Search")
        
        # Phase 2: Pattern Analysis
        if 2 not in skip_phases:
            if not self.phase2_pattern_analysis(max_projects=max_projects):
                logger.error("Pipeline failed at Phase 2")
                return False
        else:
            logger.info("Skipping Phase 2: Pattern Analysis")
        
        # Phase 3: Compilation Test
        if 3 not in skip_phases:
            if not self.phase3_compilation_test(max_projects=max_projects):
                logger.error("Pipeline failed at Phase 3")
                return False
        else:
            logger.info("Skipping Phase 3: Compilation Test")
        
        # Phase 4: Warning Test
        if 4 not in skip_phases:
            if not self.phase4_warning_test(
                max_projects=max_projects,
                checker_cp=checker_cp
            ):
                logger.error("Pipeline failed at Phase 4")
                return False
        else:
            logger.info("Skipping Phase 4: Warning Test")
        
        # Phase 5: Scoring
        if 5 not in skip_phases:
            if not self.phase5_scoring(
                min_score=min_score,
                min_warnings=min_warnings,
                max_warnings=max_warnings
            ):
                logger.error("Pipeline failed at Phase 5")
                return False
        else:
            logger.info("Skipping Phase 5: Scoring")
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final results saved to: {self.final_output_file}")
        logger.info("=" * 60)
        
        return True
    
    def print_summary(self):
        """Print summary of results"""
        if not self.final_output_file.exists():
            logger.warning("Final output file not found")
            return
        
        with open(self.final_output_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        projects = data.get('ranked_projects', [])
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Total projects found: {metadata.get('total_projects', 0)}")
        logger.info(f"High-score projects (≥70): {metadata.get('high_score_projects', 0)}")
        logger.info(f"\nTop 10 Projects:")
        
        for i, project in enumerate(projects[:10], 1):
            logger.info(f"{i}. {project['project_name']}")
            logger.info(f"   Score: {project['total_score']:.1f}")
            logger.info(f"   Warnings: {project['warning_count']}")
            logger.info(f"   Pattern Density: {project['pattern_density']:.4f}")
            logger.info(f"   URL: {project['project_url']}")
            logger.info("")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Find GitHub projects for Lower Bound Checker evaluation'
    )
    parser.add_argument('--max-projects', type=int, default=100,
                       help='Maximum projects to process')
    parser.add_argument('--min-stars', type=int, default=10,
                       help='Minimum GitHub stars')
    parser.add_argument('--min-score', type=float, default=70.0,
                       help='Minimum score to include')
    parser.add_argument('--min-warnings', type=int, default=10,
                       help='Minimum warnings required')
    parser.add_argument('--max-warnings', type=int, default=1000,
                       help='Maximum warnings allowed')
    parser.add_argument('--work-dir', help='Working directory for intermediate files')
    parser.add_argument('--github-token', help='GitHub API token')
    parser.add_argument('--checker-cp', help='Checker Framework classpath')
    parser.add_argument('--skip-phases', nargs='+', type=int,
                       help='Phase numbers to skip (1-5)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary, do not run pipeline')
    
    args = parser.parse_args()
    
    pipeline = ProjectDiscoveryPipeline(work_dir=args.work_dir)
    
    if args.summary_only:
        pipeline.print_summary()
        return 0
    
    success = pipeline.run_pipeline(
        max_projects=args.max_projects,
        min_stars=args.min_stars,
        min_score=args.min_score,
        min_warnings=args.min_warnings,
        max_warnings=args.max_warnings,
        github_token=args.github_token,
        checker_cp=args.checker_cp or os.environ.get('CHECKERFRAMEWORK_CP'),
        skip_phases=args.skip_phases
    )
    
    if success:
        pipeline.print_summary()
        return 0
    else:
        logger.error("Pipeline failed")
        return 1

if __name__ == '__main__':
    exit(main())

