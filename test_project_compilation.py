#!/usr/bin/env python3
"""
Project Compilation Tester

Tests whether GitHub projects can be compiled successfully.
Detects build system (Maven, Gradle, Ant) and attempts compilation.
"""

import os
import json
import subprocess
import tempfile
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BuildSystem(Enum):
    """Build system types"""
    MAVEN = "maven"
    GRADLE = "gradle"
    ANT = "ant"
    MAKE = "make"
    UNKNOWN = "unknown"

@dataclass
class CompilationResult:
    """Result of compilation attempt"""
    project_name: str
    project_url: str
    build_system: str
    compilation_success: bool
    java_files_found: int
    compilation_output: str
    error_message: Optional[str]
    compilation_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class ProjectCompilationTester:
    """Tests project compilation"""
    
    def __init__(self, temp_dir: Optional[str] = None, timeout: int = 600):
        """
        Initialize compilation tester
        
        Args:
            temp_dir: Temporary directory for cloning repositories
            timeout: Compilation timeout in seconds
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / 'compilation_test'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
    
    def clone_repository(self, clone_url: str, project_name: str) -> Optional[Path]:
        """Clone a repository (shallow clone)"""
        repo_dir = self.temp_dir / project_name.replace('/', '_')
        
        if repo_dir.exists():
            logger.info(f"Removing existing directory: {repo_dir}")
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
    
    def detect_build_system(self, repo_dir: Path) -> BuildSystem:
        """Detect build system used by project"""
        # Check for Maven
        if (repo_dir / 'pom.xml').exists():
            return BuildSystem.MAVEN
        
        # Check for Gradle
        if (repo_dir / 'build.gradle').exists() or (repo_dir / 'build.gradle.kts').exists():
            return BuildSystem.GRADLE
        
        # Check for Gradle wrapper
        if (repo_dir / 'gradlew').exists() or (repo_dir / 'gradlew.bat').exists():
            return BuildSystem.GRADLE
        
        # Check for Ant
        if (repo_dir / 'build.xml').exists():
            return BuildSystem.ANT
        
        # Check for Makefile
        if (repo_dir / 'Makefile').exists() or (repo_dir / 'makefile').exists():
            return BuildSystem.MAKE
        
        return BuildSystem.UNKNOWN
    
    def count_java_files(self, repo_dir: Path) -> int:
        """Count Java files in repository"""
        java_files = 0
        exclude_dirs = {'.git', 'target', 'build', 'bin', 'out', 'dist', 'test', 'tests'}
        
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            for file in files:
                if file.endswith('.java'):
                    java_files += 1
        
        return java_files
    
    def compile_maven(self, repo_dir: Path) -> Tuple[bool, str]:
        """Compile Maven project"""
        try:
            logger.info("Compiling with Maven")
            result = subprocess.run(
                ['mvn', 'clean', 'compile', '-DskipTests'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except Exception as e:
            return False, str(e)
    
    def compile_gradle(self, repo_dir: Path) -> Tuple[bool, str]:
        """Compile Gradle project"""
        try:
            # Use gradlew if available, otherwise use gradle
            gradle_cmd = 'gradlew' if (repo_dir / 'gradlew').exists() else 'gradle'
            
            if gradle_cmd == 'gradlew':
                # Make gradlew executable
                gradlew_path = repo_dir / 'gradlew'
                if gradlew_path.exists():
                    os.chmod(gradlew_path, 0o755)
                cmd = ['./gradlew', 'build', '-x', 'test']
            else:
                cmd = ['gradle', 'build', '-x', 'test']
            
            logger.info(f"Compiling with Gradle ({gradle_cmd})")
            result = subprocess.run(
                cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except Exception as e:
            return False, str(e)
    
    def compile_ant(self, repo_dir: Path) -> Tuple[bool, str]:
        """Compile Ant project"""
        try:
            logger.info("Compiling with Ant")
            result = subprocess.run(
                ['ant', 'compile'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except Exception as e:
            return False, str(e)
    
    def compile_make(self, repo_dir: Path) -> Tuple[bool, str]:
        """Compile Make project"""
        try:
            logger.info("Compiling with Make")
            result = subprocess.run(
                ['make'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except Exception as e:
            return False, str(e)
    
    def test_compilation(self, project_name: str, clone_url: str) -> CompilationResult:
        """
        Test compilation of a project
        
        Args:
            project_name: Name of the project
            clone_url: Git clone URL
            
        Returns:
            CompilationResult object
        """
        import time
        start_time = time.time()
        
        logger.info(f"Testing compilation of {project_name}")
        
        # Clone repository
        repo_dir = self.clone_repository(clone_url, project_name)
        
        if not repo_dir:
            return CompilationResult(
                project_name=project_name,
                project_url=clone_url,
                build_system=BuildSystem.UNKNOWN.value,
                compilation_success=False,
                java_files_found=0,
                compilation_output="",
                error_message="Failed to clone repository",
                compilation_time_seconds=time.time() - start_time
            )
        
        try:
            # Count Java files
            java_files = self.count_java_files(repo_dir)
            
            # Detect build system
            build_system = self.detect_build_system(repo_dir)
            
            if build_system == BuildSystem.UNKNOWN:
                return CompilationResult(
                    project_name=project_name,
                    project_url=clone_url,
                    build_system=build_system.value,
                    compilation_success=False,
                    java_files_found=java_files,
                    compilation_output="",
                    error_message="Unknown build system",
                    compilation_time_seconds=time.time() - start_time
                )
            
            # Compile based on build system
            success = False
            output = ""
            error_msg = None
            
            if build_system == BuildSystem.MAVEN:
                success, output = self.compile_maven(repo_dir)
            elif build_system == BuildSystem.GRADLE:
                success, output = self.compile_gradle(repo_dir)
            elif build_system == BuildSystem.ANT:
                success, output = self.compile_ant(repo_dir)
            elif build_system == BuildSystem.MAKE:
                success, output = self.compile_make(repo_dir)
            
            if not success:
                error_msg = "Compilation failed"
            
            compilation_time = time.time() - start_time
            
            return CompilationResult(
                project_name=project_name,
                project_url=clone_url,
                build_system=build_system.value,
                compilation_success=success,
                java_files_found=java_files,
                compilation_output=output[:5000],  # Limit output size
                error_message=error_msg,
                compilation_time_seconds=compilation_time
            )
        
        except Exception as e:
            logger.error(f"Error testing compilation: {e}")
            return CompilationResult(
                project_name=project_name,
                project_url=clone_url,
                build_system=BuildSystem.UNKNOWN.value,
                compilation_success=False,
                java_files_found=0,
                compilation_output="",
                error_message=str(e),
                compilation_time_seconds=time.time() - start_time
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
        # Handle both pattern analysis format and github projects format
        if 'analyses' in data:
            return data['analyses']
        elif 'projects' in data:
            return data['projects']
        else:
            return data

def save_results(results: List[CompilationResult], output_file: str):
    """Save compilation results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'metadata': {
            'total_projects': len(results),
            'successful_compilations': sum(1 for r in results if r.compilation_success),
            'generated_at': __import__('datetime').datetime.now().isoformat()
        },
        'results': [result.to_dict() for result in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Saved {len(results)} compilation results to {output_file}")

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Test compilation of Java projects')
    parser.add_argument('--input', required=True, help='Input JSON file with projects')
    parser.add_argument('--output', default='compilation_results.json', help='Output JSON file')
    parser.add_argument('--max-projects', type=int, help='Maximum number of projects to test')
    parser.add_argument('--timeout', type=int, default=600, help='Compilation timeout in seconds')
    parser.add_argument('--temp-dir', help='Temporary directory for cloning')
    
    args = parser.parse_args()
    
    # Load projects
    projects = load_projects(args.input)
    
    if args.max_projects:
        projects = projects[:args.max_projects]
    
    logger.info(f"Testing compilation of {len(projects)} projects")
    
    # Test compilation
    tester = ProjectCompilationTester(temp_dir=args.temp_dir, timeout=args.timeout)
    results = []
    
    for i, project in enumerate(projects, 1):
        project_name = project.get('project_name') or project.get('name') or project.get('full_name', 'unknown')
        clone_url = project.get('clone_url') or project.get('url', '')
        
        if not clone_url:
            logger.warning(f"No clone URL for project {project_name}")
            continue
        
        logger.info(f"Processing {i}/{len(projects)}: {project_name}")
        result = tester.test_compilation(project_name, clone_url)
        results.append(result)
    
    # Save results
    save_results(results, args.output)
    
    successful = sum(1 for r in results if r.compilation_success)
    logger.info(f"Completed: {successful}/{len(results)} successful compilations")
    
    return 0

if __name__ == '__main__':
    exit(main())

