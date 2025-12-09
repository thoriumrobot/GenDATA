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
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

class CheckerFrameworkRunner:
    """Runs Checker Framework checkers on Java projects (checker-agnostic)"""
    
    def __init__(self, 
                 checker_home: Optional[str] = None,
                 checker_cp: Optional[str] = None,
                 max_warnings: int = 1000,
                 processor: Optional[str] = None,
                 checker_name: Optional[str] = None):
        """
        Initialize Checker Framework runner
        
        Args:
            checker_home: Path to Checker Framework installation
            checker_cp: Classpath for Checker Framework jars
            max_warnings: Maximum number of warnings to generate
            processor: Checker Framework processor to use (overrides checker_name if provided)
            checker_name: Name of checker to use (e.g., 'lower_bound', 'sql_quotes', 'signature_string')
        """
        self.checker_home = checker_home or os.environ.get('CHECKERFRAMEWORK_HOME', '/home/ubuntu/checker-framework-3.42.0')
        self.checker_cp = checker_cp or os.environ.get('CHECKERFRAMEWORK_CP', '')
        self.max_warnings = max_warnings
        
        # Determine processor from checker_name or use provided processor
        self.checker_name = checker_name
        self.checker_interface = None
        
        if processor:
            self.processor = processor
        elif checker_name:
            # Try to get checker from registry
            try:
                from checker_registry import get_checker
                self.checker_interface = get_checker(checker_name)
                if self.checker_interface:
                    self.processor = self.checker_interface.get_checker_processor()
                    logger.info(f"Loaded checker '{checker_name}' from registry")
                else:
                    # Fallback to default processor based on checker name
                    self.processor = self._get_default_processor(checker_name)
            except Exception as e:
                logger.warning(f"Could not load checker '{checker_name}' from registry: {e}")
                self.processor = self._get_default_processor(checker_name)
        else:
            # Default to Lower Bound Checker for backward compatibility
            self.processor = 'org.checkerframework.checker.index.IndexChecker'
            self.checker_name = 'lower_bound'
        
        logger.info(f"Initialized CheckerFrameworkRunner with:")
        logger.info(f"  Checker Home: {self.checker_home}")
        logger.info(f"  Processor: {self.processor}")
        logger.info(f"  Checker Name: {self.checker_name}")
        logger.info(f"  Max Warnings: {self.max_warnings}")
    
    def _get_default_processor(self, checker_name: str) -> str:
        """Get default processor class name for a checker"""
        processor_map = {
            'lower_bound': 'org.checkerframework.checker.index.IndexChecker',
            'sql_quotes': 'org.checkerframework.checker.quotes.QuotesChecker',
            'signature_string': 'org.checkerframework.checker.signature.qual.SignatureChecker',
        }
        return processor_map.get(checker_name.lower(), 'org.checkerframework.checker.index.IndexChecker')
    
    def find_java_files(self, project_root: str, 
                       exclude_dirs: Optional[List[str]] = None,
                       max_files: Optional[int] = None) -> List[str]:
        """
        Find Java files in the project, excluding problematic directories
        
        Args:
            project_root: Root directory of the project
            exclude_dirs: Directories to exclude (default includes test, benchmark, etc.)
            max_files: Maximum number of files to return (for testing)
            
        Returns:
            List of Java file paths
        """
        if exclude_dirs is None:
            # Exclude test, benchmark, and build directories that often have compilation issues
            exclude_dirs = [
                'test', 'tests', 'test-src', 'test-sources',
                'target', 'build', '.git', '.gradle', '.mvn',
                'benchmark', 'benchmarks', 'agrona-benchmarks',
                'jmh-tests', 'jmh', 'performance-tests',
                'example', 'examples', 'demo', 'demos',
                'generated', 'generated-sources', 'generated-test-sources'
            ]
        
        java_files = []
        
        for root, dirs, files in os.walk(project_root):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            # Also exclude if path contains excluded patterns
            root_lower = root.lower()
            if any(excluded in root_lower for excluded in ['benchmark', 'jmh', 'performance']):
                continue
            
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
                    
                    # Limit files if max_files is specified
                    if max_files and len(java_files) >= max_files:
                        break
            
            if max_files and len(java_files) >= max_files:
                break
        
        logger.info(f"Found {len(java_files)} Java files in {project_root} (excluding test/benchmark directories)")
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
        checker_display_name = self.checker_name.replace('_', ' ').title() if self.checker_name else "Checker"
        logger.info(f"Running {checker_display_name} on project: {project_root}")
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
            # Format should match index1.out: file:line:column: level: [checker.message] message
            # Extract only warning/error lines and normalize paths to be relative to project_root
            project_root_path = Path(project_root).resolve()
            
            warning_lines = []
            
            # Process stdout and stderr to extract warning lines
            for output in [result.stdout, result.stderr]:
                for line in output.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip header comments and section markers
                    if line.startswith('#') or line.startswith('==='):
                        continue
                    
                    # Check if this looks like a warning/error line
                    # Format: file:line:column: level: [checker.message] message
                    # Or: file:line: level: [checker.message] message (no column)
                    warning_pattern = re.compile(r'^(.+?):(\d+)(?::(\d+))?:\s*(error|warning|compiler\.(?:err|warn)\.proc\.messager):\s*\[(.+?)\]\s*(.+)$')
                    match = warning_pattern.match(line)
                    if match:
                        file_path, line_num, col_num, level, checker_msg, message = match.groups()
                        
                        # Convert absolute paths to relative paths
                        file_path_obj = Path(file_path)
                        if file_path_obj.is_absolute():
                            try:
                                # Try to make it relative to project_root
                                relative_path = file_path_obj.relative_to(project_root_path)
                                file_path = str(relative_path)
                            except ValueError:
                                # If not under project_root, keep as is but try to extract just filename
                                file_path = file_path_obj.name
                        
                        # Reconstruct warning line in standard format
                        if col_num:
                            warning_line = f"{file_path}:{line_num}:{col_num}: {level}: [{checker_msg}] {message}"
                        else:
                            # If no column, use 0 as default
                            warning_line = f"{file_path}:{line_num}:0: {level}: [{checker_msg}] {message}"
                        
                        warning_lines.append(warning_line)
                    elif 'error:' in line or 'warning:' in line:
                        # Try to parse as warning even if format is slightly different
                        # This handles variations in javac output
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            file_path = parts[0]
                            try:
                                line_num = int(parts[1])
                                rest = ':'.join(parts[2:])
                                
                                # Convert to relative path
                                file_path_obj = Path(file_path)
                                if file_path_obj.is_absolute():
                                    try:
                                        relative_path = file_path_obj.relative_to(project_root_path)
                                        file_path = str(relative_path)
                                    except ValueError:
                                        file_path = file_path_obj.name
                                
                                warning_line = f"{file_path}:{line_num}:0: {rest}"
                                warning_lines.append(warning_line)
                            except ValueError:
                                pass
            
            # Write warnings to file (no header comments, just warnings)
            with open(output_file, 'w') as f:
                for warning_line in warning_lines:
                    f.write(warning_line + '\n')
            
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
        Parse a warnings file to extract warning information (checker-aware)
        
        Args:
            warnings_file: Path to the warnings file
            
        Returns:
            Dictionary with parsed warning information
        """
        if not os.path.exists(warnings_file):
            logger.error(f"Warnings file does not exist: {warnings_file}")
            return {}
        
        # Use checker-specific parser if available
        if self.checker_interface:
            try:
                parsed_warnings = self.checker_interface.parse_warnings(warnings_file)
                warnings_info = {
                    'total_warnings': len(parsed_warnings),
                    'total_compilation_errors': 0,
                    'warning_types': {},
                    'files_with_warnings': set(),
                    'warning_lines': [],
                    'compilation_error_lines': [],
                    'parsed_warnings': parsed_warnings
                }
                
                # Extract information from parsed warnings
                for warning in parsed_warnings:
                    warnings_info['files_with_warnings'].add(warning.get('file', ''))
                    ann_type = warning.get('annotation_type', 'unknown')
                    warnings_info['warning_types'][ann_type] = warnings_info['warning_types'].get(ann_type, 0) + 1
                    warnings_info['warning_lines'].append(f"{warning.get('file', '')}:{warning.get('line', 0)}: {warning.get('message', '')}")
                
                warnings_info['files_with_warnings'] = list(warnings_info['files_with_warnings'])
                logger.info(f"Parsed {warnings_info['total_warnings']} checker warnings using checker-specific parser")
                return warnings_info
            except Exception as e:
                logger.warning(f"Checker-specific parser failed, falling back to generic parser: {e}")
        
        # Fallback to generic parser
        warnings_info = {
            'total_warnings': 0,
            'total_compilation_errors': 0,
            'warning_types': {},
            'files_with_warnings': set(),
            'warning_lines': [],
            'compilation_error_lines': []
        }
        
        # Get checker-specific warning patterns
        warning_patterns = []
        if self.checker_interface:
            warning_patterns = self.checker_interface.get_warning_patterns()
        else:
            # Default patterns for Lower Bound Checker
            warning_patterns = ['[index]', '[assignment]']
        
        try:
            with open(warnings_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Distinguish between compilation errors and checker warnings
                    is_compilation_error = (
                        'error:' in line.lower() and 
                        ('cannot find symbol' in line.lower() or 
                         'package' in line.lower() and 'does not exist' in line.lower() or
                         'symbol:' in line.lower())
                    )
                    
                    # Check if this line matches any checker-specific warning patterns
                    is_checker_warning = False
                    if warning_patterns:
                        is_checker_warning = any(pattern.lower() in line.lower() for pattern in warning_patterns)
                    else:
                        # Generic checker warning detection
                        is_checker_warning = (
                            ('warning:' in line.lower() and '[' in line.lower() and ']' in line.lower()) or
                            ('compiler.warn.proc.messager' in line.lower()) or
                            (': compiler.' in line and 'warn' in line and not is_compilation_error)
                        )
                    
                    if is_compilation_error:
                        warnings_info['total_compilation_errors'] += 1
                        warnings_info['compilation_error_lines'].append(line)
                    elif is_checker_warning:
                        warnings_info['total_warnings'] += 1
                        warnings_info['warning_lines'].append(line)
                        
                        # Extract file path
                        if ':' in line:
                            file_path = line.split(':')[0]
                            warnings_info['files_with_warnings'].add(file_path)
                        
                        # Extract warning type based on patterns
                        for pattern in warning_patterns:
                            if pattern.lower() in line.lower():
                                warnings_info['warning_types'][pattern] = warnings_info['warning_types'].get(pattern, 0) + 1
                                break
                        else:
                            warnings_info['warning_types']['other'] = warnings_info['warning_types'].get('other', 0) + 1
            
            # Convert set to list for JSON serialization
            warnings_info['files_with_warnings'] = list(warnings_info['files_with_warnings'])
            
            logger.info(f"Parsed {warnings_info['total_warnings']} checker warnings and {warnings_info['total_compilation_errors']} compilation errors from {len(warnings_info['files_with_warnings'])} files")
            
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


def count_checker_warnings(warnings_file: str) -> int:
    """
    Count actual checker warnings (not compilation errors) in a warnings file.
    
    Args:
        warnings_file: Path to warnings file
        
    Returns:
        Number of actual checker warnings found
    """
    if not os.path.exists(warnings_file):
        return 0
    
    checker_warning_count = 0
    
    try:
        with open(warnings_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Check if this is a compilation error (not a checker warning)
                is_compilation_error = (
                    'error:' in line.lower() and 
                    ('cannot find symbol' in line.lower() or 
                     'package' in line.lower() and 'does not exist' in line.lower() or
                     'symbol:' in line.lower())
                )
                
                # Check if this is a checker warning
                is_checker_warning = (
                    ('warning:' in line.lower() and '[index]' in line.lower()) or
                    ('compiler.warn.proc.messager' in line.lower()) or
                    (': compiler.' in line and 'warn' in line and not is_compilation_error)
                )
                
                if is_checker_warning:
                    checker_warning_count += 1
    except Exception as e:
        logger.debug(f"Error counting warnings: {e}")
    
    return checker_warning_count


def run_checker_framework_on_project(project_root: str, output_file: str,
                                   max_files: Optional[int] = None,
                                   checker_home: Optional[str] = None,
                                   checker_cp: Optional[str] = None,
                                   checker_name: Optional[str] = None,
                                   processor: Optional[str] = None) -> bool:
    """
    Convenience function to run Checker Framework on a project
    
    Args:
        project_root: Root directory of the target project
        output_file: Path to save the warnings output
        max_files: Maximum number of Java files to process (for testing)
        checker_home: Path to Checker Framework installation
        checker_cp: Classpath for Checker Framework jars
        checker_name: Name of checker to use (e.g., 'lower_bound', 'sql_quotes')
        processor: Checker Framework processor class (overrides checker_name if provided)
        
    Returns:
        True if checker ran successfully, False otherwise
    """
    runner = CheckerFrameworkRunner(
        checker_home=checker_home,
        checker_cp=checker_cp,
        checker_name=checker_name,
        processor=processor
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
