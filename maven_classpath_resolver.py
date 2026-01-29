#!/usr/bin/env python3
"""
Maven Classpath Resolver

Resolves Maven project dependencies to build a complete classpath
for running the Checker Framework on Maven projects.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class MavenCompilationResult:
    """Result of Maven compilation"""
    success: bool
    classpath: str
    error_message: Optional[str] = None
    target_dirs: List[str] = None
    
    def __post_init__(self):
        if self.target_dirs is None:
            self.target_dirs = []


class MavenClasspathResolver:
    """
    Resolves Maven project dependencies to build javac classpath.
    
    This enables the Checker Framework to run on Maven projects by:
    1. Compiling the project with Maven to resolve dependencies
    2. Extracting the dependency classpath
    3. Building a complete classpath for javac
    """
    
    def __init__(self, timeout: int = 300):
        """
        Initialize Maven classpath resolver.
        
        Args:
            timeout: Timeout for Maven commands in seconds
        """
        self.timeout = timeout
        self._maven_available = None
    
    def is_maven_available(self) -> bool:
        """Check if Maven is available on the system"""
        if self._maven_available is None:
            try:
                result = subprocess.run(
                    ['mvn', '--version'],
                    capture_output=True,
                    timeout=30
                )
                self._maven_available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._maven_available = False
        return self._maven_available
    
    def is_maven_project(self, project_dir: Path) -> bool:
        """
        Check if directory contains a Maven project.
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            True if pom.xml exists
        """
        return (Path(project_dir) / 'pom.xml').exists()
    
    def is_multimodule_project(self, project_dir: Path) -> bool:
        """
        Check if this is a multi-module Maven project.
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            True if project has multiple pom.xml files (modules)
        """
        project_dir = Path(project_dir)
        pom_files = list(project_dir.rglob('pom.xml'))
        return len(pom_files) > 1
    
    def compile_project(self, project_dir: Path, quiet: bool = True) -> Tuple[bool, str]:
        """
        Compile Maven project to resolve dependencies.
        
        For multi-module projects, uses 'mvn install' to ensure inter-module
        dependencies are available in the local repository.
        
        Args:
            project_dir: Path to project directory
            quiet: If True, suppress Maven output
            
        Returns:
            Tuple of (success, error_message)
        """
        project_dir = Path(project_dir)
        
        if not self.is_maven_project(project_dir):
            return False, "Not a Maven project (no pom.xml)"
        
        if not self.is_maven_available():
            return False, "Maven is not available on this system"
        
        # For multi-module projects, use 'install' to install modules to local repo
        # This ensures inter-module dependencies can be resolved
        if self.is_multimodule_project(project_dir):
            cmd = ['mvn', 'install', '-DskipTests']
        else:
            cmd = ['mvn', 'compile', '-DskipTests']
        
        # Add flags to skip problematic Maven plugins
        cmd.extend(['-Drat.skip=true', '-Denforcer.skip=true', '-Dcheckstyle.skip=true'])
        
        if quiet:
            cmd.append('-q')
        
        logger.info(f"Compiling Maven project: {project_dir}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Maven compilation successful for {project_dir}")
                return True, ""
            else:
                error_msg = result.stderr[:1000] if result.stderr else result.stdout[:1000]
                logger.error(f"Maven compilation failed: {error_msg}")
                return False, f"Maven compilation failed: {error_msg}"
                
        except subprocess.TimeoutExpired:
            logger.error(f"Maven compilation timed out after {self.timeout}s")
            return False, f"Maven compilation timed out after {self.timeout}s"
        except Exception as e:
            logger.error(f"Error running Maven: {e}")
            return False, f"Error running Maven: {e}"
    
    def get_dependency_classpath(self, project_dir: Path) -> Tuple[bool, str]:
        """
        Get the Maven dependency classpath.
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            Tuple of (success, classpath_or_error)
        """
        project_dir = Path(project_dir)
        
        if not self.is_maven_project(project_dir):
            return False, "Not a Maven project"
        
        if not self.is_maven_available():
            return False, "Maven is not available"
        
        # Use dependency:build-classpath to get resolved dependencies
        cmd = [
            'mvn', 'dependency:build-classpath',
            '-Dmdep.outputFile=/dev/stdout',
            '-q'
        ]
        
        logger.debug(f"Getting dependency classpath: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Extract classpath from output (even if Maven returned error)
            # For multi-module projects, some modules might fail but we can still
            # extract classpaths from the ones that succeeded
            output = result.stdout + result.stderr
            
            # Remove ANSI color codes that Maven might output
            import re
            output = re.sub(r'\x1b\[[0-9;]*m', '', output)
            output = re.sub(r'\[\[[^\]]*\]\]', '', output)  # Remove [[INFO]] etc
            
            # Look for jar paths in the output
            all_jars = set()
            
            # Match paths that look like jar files
            jar_pattern = re.compile(r'(/[^\s:]+\.jar)')
            for match in jar_pattern.finditer(output):
                jar_path = match.group(1)
                if os.path.exists(jar_path):
                    all_jars.add(jar_path)
            
            if all_jars:
                classpath = ':'.join(sorted(all_jars))
                logger.info(f"Got dependency classpath ({len(all_jars)} jars, {len(classpath)} chars)")
                return True, classpath
            
            # If we didn't find any classpath but command succeeded, return empty
            if result.returncode == 0:
                return True, ""
            
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            logger.warning(f"Could not extract classpath: {error_msg}")
            return False, f"Failed to get classpath: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, "Timeout getting classpath"
        except Exception as e:
            return False, f"Error getting classpath: {e}"
    
    def get_target_directories(self, project_dir: Path) -> List[str]:
        """
        Get all target/classes directories in a project.
        
        For multi-module projects, this includes all module outputs.
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            List of target/classes directory paths (absolute paths)
        """
        # Ensure project_dir is absolute
        project_dir = Path(project_dir).resolve()
        target_dirs = []
        
        # Find all pom.xml files (each represents a module)
        for pom in project_dir.rglob('pom.xml'):
            target_classes = (pom.parent / 'target' / 'classes').resolve()
            if target_classes.exists():
                abs_path = str(target_classes)
                if abs_path not in target_dirs:
                    target_dirs.append(abs_path)
        
        # Also check for standard Maven target directory
        main_target = (project_dir / 'target' / 'classes').resolve()
        if main_target.exists() and str(main_target) not in target_dirs:
            target_dirs.append(str(main_target))
        
        return target_dirs
    
    def get_full_classpath(self, project_dir: Path, checker_cp: str) -> Tuple[bool, str]:
        """
        Get the complete classpath for running Checker Framework.
        
        Combines:
        - Checker Framework classpath
        - Maven dependency classpath
        - Project target/classes directories
        
        Args:
            project_dir: Path to project directory
            checker_cp: Checker Framework classpath
            
        Returns:
            Tuple of (success, full_classpath_or_error)
        """
        project_dir = Path(project_dir)
        classpaths = []
        
        # Start with Checker Framework classpath
        if checker_cp:
            classpaths.append(checker_cp)
        
        # Get Maven dependency classpath
        success, maven_cp = self.get_dependency_classpath(project_dir)
        if success and maven_cp:
            classpaths.append(maven_cp)
        elif not success:
            logger.warning(f"Could not get Maven classpath: {maven_cp}")
        
        # Add target/classes directories
        target_dirs = self.get_target_directories(project_dir)
        classpaths.extend(target_dirs)
        
        # Combine all classpaths
        full_cp = ':'.join(filter(None, classpaths))
        
        if not full_cp:
            return False, "Could not build classpath"
        
        logger.info(f"Built full classpath with {len(classpaths)} components")
        return True, full_cp
    
    def prepare_project(self, project_dir: Path, checker_cp: str) -> MavenCompilationResult:
        """
        Prepare a Maven project for Checker Framework analysis.
        
        This is the main entry point that:
        1. Compiles the project
        2. Resolves dependencies
        3. Builds the full classpath
        
        Args:
            project_dir: Path to project directory
            checker_cp: Checker Framework classpath
            
        Returns:
            MavenCompilationResult with success status and classpath
        """
        project_dir = Path(project_dir)
        
        if not self.is_maven_project(project_dir):
            logger.info(f"{project_dir} is not a Maven project, using default classpath")
            return MavenCompilationResult(
                success=True,
                classpath=checker_cp,
                target_dirs=[]
            )
        
        # Step 1: Compile the project
        compile_success, compile_error = self.compile_project(project_dir)
        if not compile_success:
            return MavenCompilationResult(
                success=False,
                classpath="",
                error_message=compile_error
            )
        
        # Step 2: Get full classpath
        cp_success, full_cp = self.get_full_classpath(project_dir, checker_cp)
        if not cp_success:
            return MavenCompilationResult(
                success=False,
                classpath="",
                error_message=full_cp
            )
        
        # Step 3: Get target directories
        target_dirs = self.get_target_directories(project_dir)
        
        return MavenCompilationResult(
            success=True,
            classpath=full_cp,
            target_dirs=target_dirs
        )
    
    def clean_project(self, project_dir: Path) -> bool:
        """
        Clean Maven project (remove target directories).
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            True if clean succeeded
        """
        if not self.is_maven_project(project_dir):
            return True
        
        try:
            result = subprocess.run(
                ['mvn', 'clean', '-q'],
                cwd=str(project_dir),
                capture_output=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Error cleaning project: {e}")
            return False


def main():
    """Test the Maven classpath resolver"""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python maven_classpath_resolver.py <project_dir>")
        sys.exit(1)
    
    project_dir = Path(sys.argv[1])
    checker_cp = os.environ.get('CHECKERFRAMEWORK_CP', '')
    
    resolver = MavenClasspathResolver()
    
    print(f"Project: {project_dir}")
    print(f"Is Maven project: {resolver.is_maven_project(project_dir)}")
    print(f"Is multi-module: {resolver.is_multimodule_project(project_dir)}")
    print(f"Maven available: {resolver.is_maven_available()}")
    
    if resolver.is_maven_project(project_dir):
        print("\nPreparing project...")
        result = resolver.prepare_project(project_dir, checker_cp)
        
        print(f"Success: {result.success}")
        if result.error_message:
            print(f"Error: {result.error_message}")
        print(f"Target dirs: {result.target_dirs}")
        print(f"Classpath length: {len(result.classpath)}")


if __name__ == '__main__':
    main()
