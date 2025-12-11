#!/usr/bin/env python3
"""
Lower Bound Checker Warning Tester

Tests projects with the Lower Bound Checker (Index Checker) to generate warnings.
Parses warnings and provides statistics.
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

@dataclass
class WarningStats:
    """Statistics about Lower Bound Checker warnings"""
    total_warnings: int
    warnings_by_type: Dict[str, int]
    files_with_warnings: int
    warning_lines: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_warnings': self.total_warnings,
            'warnings_by_type': self.warnings_by_type,
            'files_with_warnings': self.files_with_warnings,
            'warning_lines': self.warning_lines[:100]  # Limit to first 100
        }

@dataclass
class WarningTestResult:
    """Result of warning test"""
    project_name: str
    project_url: str
    compilation_success: bool
    checker_success: bool
    stats: Optional[WarningStats]
    error_message: Optional[str]
    checker_output: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'project_name': self.project_name,
            'project_url': self.project_url,
            'compilation_success': self.compilation_success,
            'checker_success': self.checker_success,
            'stats': self.stats.to_dict() if self.stats else None,
            'error_message': self.error_message,
            'checker_output': self.checker_output[:5000]  # Limit output size
        }

class LowerBoundWarningTester:
    """Tests projects with Lower Bound Checker"""
    
    def __init__(self, 
                 checker_cp: Optional[str] = None,
                 temp_dir: Optional[str] = None,
                 timeout: int = 600):
        """
        Initialize warning tester
        
        Args:
            checker_cp: Checker Framework classpath
            temp_dir: Temporary directory for cloning
            timeout: Timeout in seconds
        """
        self.checker_cp = checker_cp or os.environ.get('CHECKERFRAMEWORK_CP', '')
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / 'warning_test'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        if not self.checker_cp:
            logger.warning("CHECKERFRAMEWORK_CP not set. Checker Framework may not work.")
    
    def clone_repository(self, clone_url: str, project_name: str) -> Optional[Path]:
        """Clone a repository (shallow clone)"""
        repo_dir = self.temp_dir / project_name.replace('/', '_')
        
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        
        try:
            logger.info(f"Cloning {clone_url}")
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', clone_url, str(repo_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return repo_dir
            else:
                logger.error(f"Failed to clone: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error cloning: {e}")
            return None
    
    def find_java_files(self, repo_dir: Path, max_files: Optional[int] = None) -> List[str]:
        """Find Java files in repository"""
        java_files = []
        exclude_dirs = {
            '.git', 'target', 'build', 'bin', 'out', 'dist',
            'test', 'tests', 'test-src', 'test-sources',
            'generated', 'generated-sources', '.gradle', '.mvn'
        }
        
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
                    if max_files and len(java_files) >= max_files:
                        break
            
            if max_files and len(java_files) >= max_files:
                break
        
        return java_files
    
    def run_lower_bound_checker(self, repo_dir: Path, java_files: List[str]) -> Tuple[bool, str]:
        """
        Run Lower Bound Checker on Java files
        
        Args:
            repo_dir: Repository directory
            java_files: List of Java file paths
            
        Returns:
            (success, output) tuple
        """
        if not java_files:
            return False, "No Java files found"
        
        try:
            # Create output directory
            output_dir = repo_dir / 'checker_output'
            output_dir.mkdir(exist_ok=True)
            
            # Build javac command
            cmd = [
                'javac',
                '-cp', self.checker_cp,
                '-processor', 'org.checkerframework.checker.index.IndexChecker',
                '-Xmaxwarns', '10000',
                '-d', str(output_dir),
                '-sourcepath', str(repo_dir)
            ]
            
            # Add Java files
            cmd.extend(java_files)
            
            logger.info(f"Running Lower Bound Checker on {len(java_files)} files")
            
            result = subprocess.run(
                cmd,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Combine stdout and stderr
            output = result.stdout + result.stderr
            
            # Success if checker ran (even if warnings were generated)
            success = True  # Checker runs even if there are warnings
            
            return success, output
        
        except subprocess.TimeoutExpired:
            return False, "Checker timed out"
        except Exception as e:
            return False, str(e)
    
    def parse_warnings(self, output: str) -> WarningStats:
        """
        Parse warnings from checker output
        
        Args:
            output: Checker output string
            
        Returns:
            WarningStats object
        """
        warnings_by_type = {}
        warning_lines = []
        files_with_warnings = set()
        
        # Pattern for Checker Framework warnings
        # Format: file:line:column: compiler.err/warn.proc.messager: [checker.message] message
        warning_pattern = re.compile(
            r'^(.+?\.java):(\d+)(?::(\d+))?:\s*(compiler\.(?:err|warn)\.proc\.messager|error|warning):\s*\[(.+?)\]\s*(.+)$',
            re.MULTILINE
        )
        
        matches = warning_pattern.findall(output)
        
        for match in matches:
            file_path, line_num, col_num, level, checker_msg, message = match
            
            # Extract warning type from checker message
            warning_type = checker_msg.split('.')[-1] if '.' in checker_msg else checker_msg
            
            # Track warning type
            warnings_by_type[warning_type] = warnings_by_type.get(warning_type, 0) + 1
            
            # Track file
            files_with_warnings.add(file_path)
            
            # Store warning line
            warning_line = f"{file_path}:{line_num}:{col_num or '0'}: [{checker_msg}] {message}"
            warning_lines.append(warning_line)
        
        # Also try simpler pattern for variations
        simple_pattern = re.compile(
            r'^(.+?\.java):(\d+):\s*(error|warning):\s*(.+)$',
            re.MULTILINE
        )
        
        simple_matches = simple_pattern.findall(output)
        for match in simple_matches:
            file_path, line_num, level, message = match
            
            # Skip if already captured
            warning_line = f"{file_path}:{line_num}: {level}: {message}"
            if warning_line not in warning_lines:
                warnings_by_type['other'] = warnings_by_type.get('other', 0) + 1
                files_with_warnings.add(file_path)
                warning_lines.append(warning_line)
        
        return WarningStats(
            total_warnings=len(warning_lines),
            warnings_by_type=warnings_by_type,
            files_with_warnings=len(files_with_warnings),
            warning_lines=warning_lines
        )
    
    def test_warnings(self, 
                     project_name: str,
                     clone_url: str,
                     max_files: Optional[int] = 100) -> WarningTestResult:
        """
        Test project for Lower Bound Checker warnings
        
        Args:
            project_name: Name of the project
            clone_url: Git clone URL
            max_files: Maximum number of Java files to check
            
        Returns:
            WarningTestResult object
        """
        logger.info(f"Testing Lower Bound warnings for {project_name}")
        
        # Clone repository
        repo_dir = self.clone_repository(clone_url, project_name)
        
        if not repo_dir:
            return WarningTestResult(
                project_name=project_name,
                project_url=clone_url,
                compilation_success=False,
                checker_success=False,
                stats=None,
                error_message="Failed to clone repository",
                checker_output=""
            )
        
        try:
            # Find Java files
            java_files = self.find_java_files(repo_dir, max_files=max_files)
            
            if not java_files:
                return WarningTestResult(
                    project_name=project_name,
                    project_url=clone_url,
                    compilation_success=False,
                    checker_success=False,
                    stats=None,
                    error_message="No Java files found",
                    checker_output=""
                )
            
            # Run Lower Bound Checker
            checker_success, checker_output = self.run_lower_bound_checker(repo_dir, java_files)
            
            if not checker_success:
                return WarningTestResult(
                    project_name=project_name,
                    project_url=clone_url,
                    compilation_success=False,
                    checker_success=False,
                    stats=None,
                    error_message="Failed to run checker",
                    checker_output=checker_output
                )
            
            # Parse warnings
            stats = self.parse_warnings(checker_output)
            
            logger.info(f"Found {stats.total_warnings} warnings in {stats.files_with_warnings} files")
            
            return WarningTestResult(
                project_name=project_name,
                project_url=clone_url,
                compilation_success=True,
                checker_success=True,
                stats=stats,
                error_message=None,
                checker_output=checker_output
            )
        
        except Exception as e:
            logger.error(f"Error testing warnings: {e}")
            return WarningTestResult(
                project_name=project_name,
                project_url=clone_url,
                compilation_success=False,
                checker_success=False,
                stats=None,
                error_message=str(e),
                checker_output=""
            )
        
        finally:
            # Clean up
            if repo_dir.exists():
                try:
                    shutil.rmtree(repo_dir)
                except Exception:
                    pass

def load_projects(input_file: str) -> List[Dict[str, Any]]:
    """Load projects from JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)
        # Handle different formats
        if 'results' in data:
            return data['results']
        elif 'analyses' in data:
            return data['analyses']
        elif 'projects' in data:
            return data['projects']
        else:
            return data

def save_results(results: List[WarningTestResult], output_file: str):
    """Save warning test results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'metadata': {
            'total_projects': len(results),
            'successful_tests': sum(1 for r in results if r.checker_success),
            'projects_with_warnings': sum(1 for r in results if r.stats and r.stats.total_warnings > 0),
            'generated_at': __import__('datetime').datetime.now().isoformat()
        },
        'results': [result.to_dict() for result in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Saved {len(results)} warning test results to {output_file}")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Test projects with Lower Bound Checker')
    parser.add_argument('--input', required=True, help='Input JSON file with projects')
    parser.add_argument('--output', default='warning_test_results.json', help='Output JSON file')
    parser.add_argument('--max-projects', type=int, help='Maximum number of projects to test')
    parser.add_argument('--max-files', type=int, default=100, help='Maximum Java files per project')
    parser.add_argument('--checker-cp', help='Checker Framework classpath')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds')
    parser.add_argument('--temp-dir', help='Temporary directory for cloning')
    
    args = parser.parse_args()
    
    # Load projects
    projects = load_projects(args.input)
    
    if args.max_projects:
        projects = projects[:args.max_projects]
    
    # Filter to only compiled projects
    compiled_projects = [
        p for p in projects 
        if p.get('compilation_success', False) or p.get('build_system') != 'unknown'
    ]
    
    logger.info(f"Testing Lower Bound warnings for {len(compiled_projects)} compiled projects")
    
    # Test warnings
    tester = LowerBoundWarningTester(
        checker_cp=args.checker_cp,
        temp_dir=args.temp_dir,
        timeout=args.timeout
    )
    
    results = []
    
    for i, project in enumerate(compiled_projects, 1):
        project_name = project.get('project_name') or project.get('name') or project.get('full_name', 'unknown')
        clone_url = project.get('clone_url') or project.get('url') or project.get('project_url', '')
        
        if not clone_url:
            logger.warning(f"No clone URL for project {project_name}")
            continue
        
        logger.info(f"Processing {i}/{len(compiled_projects)}: {project_name}")
        result = tester.test_warnings(project_name, clone_url, max_files=args.max_files)
        results.append(result)
    
    # Save results
    save_results(results, args.output)
    
    successful = sum(1 for r in results if r.checker_success)
    with_warnings = sum(1 for r in results if r.stats and r.stats.total_warnings > 0)
    logger.info(f"Completed: {successful}/{len(results)} successful tests, {with_warnings} with warnings")
    
    return 0

if __name__ == '__main__':
    exit(main())

