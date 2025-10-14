#!/usr/bin/env python3
"""
Checker Framework Runner Module

This module provides reusable functionality for running the Checker Framework's
Lower Bound Checker (Index Checker) on target projects and generating warning files
for use in the GenDATA prediction pipeline.
"""

import os
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

class CheckerFrameworkRunner:
    """Runs Checker Framework's Lower Bound Checker on Java projects"""
    
    def __init__(self, 
                 checker_home: Optional[str] = None,
                 checker_cp: Optional[str] = None,
                 max_warnings: int = 1000,
                 processor: str = 'org.checkerframework.checker.index.IndexChecker'):
        """
        Initialize Checker Framework runner
        
        Args:
            checker_home: Path to Checker Framework installation
            checker_cp: Classpath for Checker Framework jars
            max_warnings: Maximum number of warnings to generate
            processor: Checker Framework processor to use
        """
        self.checker_home = checker_home or os.environ.get('CHECKERFRAMEWORK_HOME', '/home/ubuntu/checker-framework-3.42.0')
        self.checker_cp = checker_cp or os.environ.get('CHECKERFRAMEWORK_CP', '')
        self.max_warnings = max_warnings
        self.processor = processor
        
        logger.info(f"Initialized CheckerFrameworkRunner with:")
        logger.info(f"  Checker Home: {self.checker_home}")
        logger.info(f"  Processor: {self.processor}")
        logger.info(f"  Max Warnings: {self.max_warnings}")
    
    def find_java_files(self, project_root: str, 
                       exclude_dirs: Optional[List[str]] = None,
                       max_files: Optional[int] = None) -> List[str]:
        """
        Find Java files in the project
        
        Args:
            project_root: Root directory of the project
            exclude_dirs: Directories to exclude (default: test, tests, target, build, .git)
            max_files: Maximum number of files to return (for testing)
            
        Returns:
            List of Java file paths
        """
        if exclude_dirs is None:
            exclude_dirs = ['test', 'tests', 'target', 'build', '.git', '.gradle', '.mvn']
        
        java_files = []
        
        for root, dirs, files in os.walk(project_root):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
                    
                    # Limit files if max_files is specified
                    if max_files and len(java_files) >= max_files:
                        break
            
            if max_files and len(java_files) >= max_files:
                break
        
        logger.info(f"Found {len(java_files)} Java files in {project_root}")
        return java_files
    
    def run_checker_on_project(self, project_root: str, output_file: str,
                              max_files: Optional[int] = None,
                              additional_args: Optional[List[str]] = None) -> bool:
        """
        Run Checker Framework's Lower Bound Checker on a project
        
        Args:
            project_root: Root directory of the target project
            output_file: Path to save the warnings output
            max_files: Maximum number of Java files to process (for testing)
            additional_args: Additional arguments to pass to javac
            
        Returns:
            True if checker ran successfully, False otherwise
        """
        logger.info(f"Running Lower Bound Checker on project: {project_root}")
        logger.info(f"Output will be saved to: {output_file}")
        
        # Validate project root
        if not os.path.exists(project_root):
            logger.error(f"Project root does not exist: {project_root}")
            return False
        
        # Find Java files
        java_files = self.find_java_files(project_root, max_files=max_files)
        
        if not java_files:
            logger.warning(f"No Java files found in {project_root}")
            # Create empty output file
            with open(output_file, 'w') as f:
                f.write("# No Java files found in project\n")
            return True
        
        # Set up environment
        env = os.environ.copy()
        
        # Build the javac command with Checker Framework
        cmd = [
            'javac',
            '-cp', self.checker_cp,
            '-processor', self.processor,
            '-Xmaxwarns', str(self.max_warnings),
            '-d', '/tmp/checker_output',
            '-sourcepath', project_root
        ]
        
        # Add additional arguments if provided
        if additional_args:
            cmd.extend(additional_args)
        
        # Add Java files to command
        cmd.extend(java_files)
        
        try:
            # Create output directory
            os.makedirs('/tmp/checker_output', exist_ok=True)
            
            # Create output file directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            logger.info(f"Running command: {' '.join(cmd[:10])}{'...' if len(cmd) > 10 else ''}")
            logger.info(f"Processing {len(java_files)} Java files")
            
            # Run the checker
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=project_root,
                timeout=300  # 5 minute timeout
            )
            
            # Save warnings to output file
            with open(output_file, 'w') as f:
                f.write(f"# Checker Framework Lower Bound Checker Output\n")
                f.write(f"# Project: {project_root}\n")
                f.write(f"# Files processed: {len(java_files)}\n")
                f.write(f"# Command: {' '.join(cmd[:10])}{'...' if len(cmd) > 10 else ''}\n")
                f.write(f"# Return code: {result.returncode}\n\n")
                
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            if result.returncode == 0:
                logger.info(f"Checker Framework completed successfully. Warnings saved to {output_file}")
                return True
            else:
                logger.warning(f"Checker Framework completed with warnings/errors (return code: {result.returncode})")
                logger.info(f"Output saved to {output_file}")
                return True  # Still return True as we got output
                
        except subprocess.TimeoutExpired:
            logger.error("Checker Framework timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Error running Checker Framework: {e}")
            return False
    
    def parse_warnings_file(self, warnings_file: str) -> Dict[str, Any]:
        """
        Parse a warnings file to extract warning information
        
        Args:
            warnings_file: Path to the warnings file
            
        Returns:
            Dictionary with parsed warning information
        """
        if not os.path.exists(warnings_file):
            logger.error(f"Warnings file does not exist: {warnings_file}")
            return {}
        
        warnings_info = {
            'total_warnings': 0,
            'warning_types': {},
            'files_with_warnings': set(),
            'warning_lines': []
        }
        
        try:
            with open(warnings_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Look for warning patterns
                    # Format: /path/to/file.java:line:col: compiler.err/warn.proc.messager: [checker] message
                    if ': compiler.' in line and ('err' in line or 'warn' in line):
                        warnings_info['total_warnings'] += 1
                        warnings_info['warning_lines'].append(line)
                        
                        # Extract file path
                        if ':' in line:
                            file_path = line.split(':')[0]
                            warnings_info['files_with_warnings'].add(file_path)
                        
                        # Extract warning type
                        if '[index]' in line.lower():
                            warnings_info['warning_types']['index'] = warnings_info['warning_types'].get('index', 0) + 1
                        elif '[assignment]' in line.lower():
                            warnings_info['warning_types']['assignment'] = warnings_info['warning_types'].get('assignment', 0) + 1
                        else:
                            warnings_info['warning_types']['other'] = warnings_info['warning_types'].get('other', 0) + 1
            
            # Convert set to list for JSON serialization
            warnings_info['files_with_warnings'] = list(warnings_info['files_with_warnings'])
            
            logger.info(f"Parsed {warnings_info['total_warnings']} warnings from {len(warnings_info['files_with_warnings'])} files")
            
        except Exception as e:
            logger.error(f"Error parsing warnings file: {e}")
        
        return warnings_info
    
    def validate_checker_environment(self) -> bool:
        """
        Validate that Checker Framework is properly installed and configured
        
        Returns:
            True if environment is valid, False otherwise
        """
        logger.info("Validating Checker Framework environment...")
        
        # Check if javac is available
        try:
            result = subprocess.run(['javac', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("javac not found or not working")
                return False
            logger.info(f"javac version: {result.stderr.strip()}")
        except FileNotFoundError:
            logger.error("javac command not found")
            return False
        
        # Check Checker Framework home
        if not os.path.exists(self.checker_home):
            logger.warning(f"Checker Framework home not found: {self.checker_home}")
            logger.info("This may be normal if Checker Framework is installed elsewhere")
        
        # Check classpath
        if not self.checker_cp:
            logger.warning("CHECKERFRAMEWORK_CP environment variable not set")
            logger.info("This may cause issues if Checker Framework jars are not in default locations")
        
        logger.info("Checker Framework environment validation completed")
        return True


def run_checker_framework_on_project(project_root: str, output_file: str,
                                   max_files: Optional[int] = None,
                                   checker_home: Optional[str] = None,
                                   checker_cp: Optional[str] = None) -> bool:
    """
    Convenience function to run Checker Framework on a project
    
    Args:
        project_root: Root directory of the target project
        output_file: Path to save the warnings output
        max_files: Maximum number of Java files to process (for testing)
        checker_home: Path to Checker Framework installation
        checker_cp: Classpath for Checker Framework jars
        
    Returns:
        True if checker ran successfully, False otherwise
    """
    runner = CheckerFrameworkRunner(
        checker_home=checker_home,
        checker_cp=checker_cp
    )
    
    return runner.run_checker_on_project(
        project_root=project_root,
        output_file=output_file,
        max_files=max_files
    )


def main():
    """Command-line interface for Checker Framework runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Checker Framework Lower Bound Checker on a project')
    parser.add_argument('project_root', help='Root directory of the Java project')
    parser.add_argument('output_file', help='Output file for warnings')
    parser.add_argument('--max-files', type=int, help='Maximum number of Java files to process')
    parser.add_argument('--checker-home', help='Path to Checker Framework installation')
    parser.add_argument('--checker-cp', help='Classpath for Checker Framework jars')
    parser.add_argument('--validate-only', action='store_true', help='Only validate environment, do not run checker')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    runner = CheckerFrameworkRunner(
        checker_home=args.checker_home,
        checker_cp=args.checker_cp
    )
    
    if args.validate_only:
        success = runner.validate_checker_environment()
    else:
        success = runner.run_checker_on_project(
            project_root=args.project_root,
            output_file=args.output_file,
            max_files=args.max_files
        )
    
    if success:
        print("✅ Checker Framework operation completed successfully")
        return 0
    else:
        print("❌ Checker Framework operation failed")
        return 1


if __name__ == '__main__':
    exit(main())
