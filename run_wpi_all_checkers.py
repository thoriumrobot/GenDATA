#!/usr/bin/env python3
"""
Unified WPI (Whole Program Inference) Script for All Checkers

Runs Checker Framework's Whole Program Inference on projects for:
- Lower Bound Checker (IndexChecker)
- SQL Quotes Checker (SqlQuotesChecker)
- Signature String Checker (SignatureChecker)

Properly manages backups to ensure original code is never modified.

Usage:
    python run_wpi_all_checkers.py --checker lower_bound
    python run_wpi_all_checkers.py --checker sql_quotes
    python run_wpi_all_checkers.py --checker signature_string
    python run_wpi_all_checkers.py --all
"""

import os
import sys
import json
import shutil
import subprocess
import logging
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_wpi_all_checkers.log')
    ]
)
logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_BACKUP = GEN_DATA_ROOT / 'case_studies_backup'
ANNOTATION_EVAL_BACKUPS = GEN_DATA_ROOT / 'annotation_evaluation' / 'backups'
WPI_WORK_DIR = GEN_DATA_ROOT / 'wpi_work'
WPI_OUTPUT_DIR = GEN_DATA_ROOT / 'wpi_output'
CHECKER_FRAMEWORK_HOME = Path('/home/ubuntu/checker-framework')

# Checker processor mapping
CHECKER_PROCESSORS = {
    'lower_bound': 'org.checkerframework.checker.index.IndexChecker',
    'sql_quotes': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
    'signature_string': 'org.checkerframework.checker.signature.SignatureChecker',
}

# Projects per checker for WPI evaluation (3 real GitHub projects each, no training sets)
WPI_PROJECTS = {
    'lower_bound': [
        {'name': 'pom-tuner', 'modules': ['pom-tuner'], 'is_tycho': False},
        {'name': 'commons-lang', 'modules': None, 'is_tycho': False},
        {'name': 'commons-io', 'modules': None, 'is_tycho': False},
    ],
    'sql_quotes': [
        {'name': 'commons-dbcp', 'modules': None, 'is_tycho': False},
        {'name': 'mybatis-3', 'modules': None, 'is_tycho': False},
        {'name': 'commons-dbutils', 'modules': None, 'is_tycho': False},
    ],
    'signature_string': [
        {'name': 'javassist', 'modules': None, 'is_tycho': False},
        {'name': 'reflections', 'modules': None, 'is_tycho': False},
        {'name': 'guice', 'modules': ['core'], 'is_tycho': False},
    ],
}

# Backup directories - NEVER modify
BACKUP_DIRECTORIES = [
    CASE_STUDIES_BACKUP,
    ANNOTATION_EVAL_BACKUPS,
    GEN_DATA_ROOT / 'annotated_projects_backup',
]


@dataclass
class WPIResult:
    """Result of running WPI on a project"""
    project_name: str
    checker_name: str
    success: bool
    baseline_warnings: int
    after_wpi_warnings: int
    reduction_percentage: float
    ajava_files_count: int
    iterations: int
    execution_time_seconds: float
    error_message: Optional[str] = None


try:
    from backup_safety import verify_not_backup_dir, restore_from_backup as backup_restore
    
    def restore_from_backup(project_name: str, target_dir: Path) -> bool:
        """Restore a project using the shared backup_safety module"""
        try:
            return backup_restore(project_name, target_dir, force=True)
        except Exception as e:
            logger.error(f"Error restoring {project_name}: {e}")
            return False

except ImportError:
    # Fallback if backup_safety module not available
    def verify_not_backup_dir(path: Path) -> bool:
        """Verify that a path is not inside a backup directory (safety check)"""
        for backup_dir in BACKUP_DIRECTORIES:
            if backup_dir.exists():
                try:
                    path.relative_to(backup_dir)
                    logger.error(f"SAFETY: Attempted to modify backup directory: {path}")
                    return False
                except ValueError:
                    continue
        return True
    
    def restore_from_backup(project_name: str, target_dir: Path) -> bool:
        """
        Restore a project from backup to target directory.
        
        Args:
            project_name: Name of the project
            target_dir: Target directory to restore to (NOT a backup!)
            
        Returns:
            True if restore successful
        """
        # Safety check
        if not verify_not_backup_dir(target_dir):
            return False
        
        # Check backup locations
        backup_sources = [
            ANNOTATION_EVAL_BACKUPS / project_name,
            CASE_STUDIES_BACKUP / project_name,
        ]
        
        source_backup = None
        for backup in backup_sources:
            if backup.exists():
                source_backup = backup
                break
        
        if not source_backup:
            logger.error(f"No backup found for {project_name}")
            return False
        
        try:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_backup, target_dir)
            logger.info(f"Restored {project_name} from {source_backup}")
            return True
        except Exception as e:
            logger.error(f"Error restoring {project_name}: {e}")
            return False


def find_java_files(directory: Path, max_files: int = 200) -> List[Path]:
    """Find Java files in a directory, excluding tests and build directories"""
    java_files = []
    
    exclude_patterns = [
        '/test/', '/tests/', '/target/', '/build/', 
        '/generated/', '/.git/', '/benchmark/', '/javatests/'
    ]
    
    # Prioritize src/main files first
    main_sources = list(directory.rglob('src/main/**/*.java'))
    
    # Also check core/src for guice-like structures and submodules
    if not main_sources:
        main_sources = list(directory.rglob('**/src/**/*.java'))
    
    for java_file in main_sources:
        path_str = str(java_file)
        if not any(pattern in path_str for pattern in exclude_patterns):
            java_files.append(java_file)
            if len(java_files) >= max_files:
                break
    
    return java_files


def run_checker(project_dir: Path, checker_name: str, java_files: List[Path]) -> Tuple[int, str]:
    """Run a Checker Framework checker on Java files with Maven classpath resolution"""
    if not java_files:
        return 0, "No Java files to check"
    
    processor = CHECKER_PROCESSORS.get(checker_name)
    if not processor:
        return -1, f"Unknown checker: {checker_name}"
    
    checker_javac = CHECKER_FRAMEWORK_HOME / 'checker' / 'bin' / 'javac'
    checker_cp = f"{CHECKER_FRAMEWORK_HOME}/checker/dist/checker-qual.jar:{CHECKER_FRAMEWORK_HOME}/checker/dist/checker.jar"
    
    # Try to get Maven classpath
    maven_classpath = ""
    try:
        from maven_classpath_resolver import MavenClasspathResolver
        resolver = MavenClasspathResolver(timeout=300)
        result = resolver.prepare_project(project_dir, checker_cp)
        if result.success:
            maven_classpath = result.classpath
    except Exception as e:
        logger.debug(f"Maven classpath resolution failed: {e}")
    
    # Build full classpath
    if maven_classpath:
        full_classpath = maven_classpath
    else:
        full_classpath = checker_cp
    
    cmd = [
        str(checker_javac),
        '-processor', processor,
        '-cp', full_classpath,
        '-Xlint:-processing',
        '-Awarns',
    ] + [str(f) for f in java_files[:200]]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_dir)
        )
        output = result.stdout + result.stderr
        
        # Count warnings
        warning_count = 0
        for line in output.split('\n'):
            if 'error:' in line.lower() or 'warning:' in line.lower():
                if '[' in line and ']' in line:
                    if not any(w in line for w in ['[deprecation]', '[removal]', '[unchecked]', '[rawtypes]', '[path]', '[options]']):
                        warning_count += 1
        
        return warning_count, output
        
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)


def get_wpi_pom_additions(checker_name: str, wpi_output_path: str, checker_version: str = '3.42.0') -> str:
    """
    Generate the POM XML additions for Checker Framework WPI.
    
    Args:
        checker_name: Name of the checker
        wpi_output_path: Path where WPI should output ajava files
        checker_version: Checker Framework version
        
    Returns:
        XML string to add to POM
    """
    processor = CHECKER_PROCESSORS.get(checker_name, 'org.checkerframework.checker.index.IndexChecker')
    
    return f'''
    <!-- Checker Framework WPI Configuration -->
    <properties>
        <checkerframework.version>{checker_version}</checkerframework.version>
        <wpi.output.dir>{wpi_output_path}</wpi.output.dir>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.checkerframework</groupId>
            <artifactId>checker-qual</artifactId>
            <version>${{checkerframework.version}}</version>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.14.1</version>
                <configuration>
                    <fork>true</fork>
                    <showWarnings>true</showWarnings>
                    <annotationProcessorPaths>
                        <path>
                            <groupId>org.checkerframework</groupId>
                            <artifactId>checker</artifactId>
                            <version>${{checkerframework.version}}</version>
                        </path>
                    </annotationProcessorPaths>
                    <annotationProcessors>
                        <annotationProcessor>{processor}</annotationProcessor>
                    </annotationProcessors>
                    <compilerArgs>
                        <arg>-Ainfer=ajava</arg>
                        <arg>-Awarns</arg>
                        <arg>-AshowPrefixInWarningMessages</arg>
                        <arg>-Aajava={wpi_output_path}</arg>
                        <arg>-Xmaxerrs</arg>
                        <arg>10000</arg>
                        <arg>-Xmaxwarns</arg>
                        <arg>10000</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.model=ALL-UNNAMED</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.processing=ALL-UNNAMED</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED</arg>
                        <arg>-J--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED</arg>
                        <arg>-J--add-opens=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED</arg>
                    </compilerArgs>
                </configuration>
            </plugin>
        </plugins>
    </build>
'''


class WPIAllCheckersRunner:
    """Runs WPI for all checkers on configured projects"""
    
    def __init__(self, 
                 work_dir: Path = WPI_WORK_DIR,
                 output_dir: Path = WPI_OUTPUT_DIR,
                 checker_version: str = '3.42.0',
                 timeout: int = 1800):
        """
        Initialize WPI runner.
        
        Args:
            work_dir: Working directory for WPI (projects are copied here)
            output_dir: Directory for WPI output files
            checker_version: Checker Framework version
            timeout: Timeout for WPI operations
        """
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.checker_version = checker_version
        self.timeout = timeout
        
        # Ensure directories exist
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized WPIAllCheckersRunner")
        logger.info(f"  Work directory: {self.work_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def get_project_work_dir(self, checker_name: str, project_name: str) -> Path:
        """Get working directory for a project (NOT a backup!)"""
        return self.work_dir / checker_name / project_name
    
    def get_wpi_output_dir(self, checker_name: str, project_name: str) -> Path:
        """Get WPI output directory for a project"""
        return self.output_dir / checker_name / project_name
    
    def modify_pom_for_wpi(self, pom_path: Path, checker_name: str, wpi_output_path: str) -> bool:
        """
        Modify a POM file to add Checker Framework WPI configuration.
        
        Args:
            pom_path: Path to pom.xml
            checker_name: Checker name
            wpi_output_path: Path for WPI output
            
        Returns:
            True if modification successful
        """
        try:
            # Read existing POM
            with open(pom_path, 'r') as f:
                content = f.read()
            
            # Check if already modified
            if 'checkerframework' in content.lower():
                logger.info(f"POM already contains Checker Framework config")
                return True
            
            # Find </project> and insert before it
            additions = get_wpi_pom_additions(checker_name, wpi_output_path, self.checker_version)
            
            if '</project>' in content:
                content = content.replace('</project>', f'{additions}\n</project>')
            else:
                logger.warning(f"Could not find </project> in {pom_path}")
                return False
            
            # Write modified POM
            with open(pom_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Modified POM for WPI: {pom_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying POM: {e}")
            return False
    
    def run_wpi_iterations(self, project_dir: Path, max_iterations: int = 10) -> Tuple[int, int]:
        """
        Run WPI iterations until convergence.
        
        Args:
            project_dir: Project directory
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (iterations_run, final_warning_count)
        """
        prev_warnings = -1
        iterations = 0
        
        for i in range(max_iterations):
            iterations = i + 1
            logger.info(f"  WPI iteration {iterations}")
            
            try:
                # Run Maven compile with Checker Framework
                result = subprocess.run(
                    ['mvn', 'compile', '-DskipTests', '-Dcheckstyle.skip=true', '-Dspotless.check.skip=true'],
                    cwd=str(project_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                output = result.stdout + result.stderr
                
                # Count warnings
                warning_count = 0
                for line in output.split('\n'):
                    if 'warning:' in line.lower() or 'error:' in line.lower():
                        if '[' in line and ']' in line:
                            warning_count += 1
                
                logger.info(f"    Warnings: {warning_count}")
                
                # Check for convergence
                if warning_count == prev_warnings:
                    logger.info(f"  WPI converged after {iterations} iterations")
                    return iterations, warning_count
                
                prev_warnings = warning_count
                
            except subprocess.TimeoutExpired:
                logger.error(f"  WPI iteration timed out")
                return iterations, -1
            except Exception as e:
                logger.error(f"  WPI iteration error: {e}")
                return iterations, -1
        
        logger.info(f"  WPI completed {max_iterations} iterations (may not have converged)")
        return iterations, prev_warnings
    
    def count_ajava_files(self, output_dir: Path) -> int:
        """Count .ajava files in output directory"""
        if not output_dir.exists():
            return 0
        return len(list(output_dir.rglob('*.ajava')))
    
    def run_wpi_for_project(self, project_name: str, checker_name: str, 
                           project_config: Dict) -> WPIResult:
        """
        Run WPI for a single project.
        
        Args:
            project_name: Project name
            checker_name: Checker name
            project_config: Project configuration dict
            
        Returns:
            WPIResult
        """
        start_time = time.time()
        
        project_dir = self.get_project_work_dir(checker_name, project_name)
        wpi_output = self.get_wpi_output_dir(checker_name, project_name)
        
        # Skip Tycho projects
        if project_config.get('is_tycho', False):
            return WPIResult(
                project_name=project_name,
                checker_name=checker_name,
                success=False,
                baseline_warnings=0,
                after_wpi_warnings=0,
                reduction_percentage=0.0,
                ajava_files_count=0,
                iterations=0,
                execution_time_seconds=0.0,
                error_message="Tycho/Eclipse projects not supported"
            )
        
        # Restore from backup
        if not restore_from_backup(project_name, project_dir):
            return WPIResult(
                project_name=project_name,
                checker_name=checker_name,
                success=False,
                baseline_warnings=0,
                after_wpi_warnings=0,
                reduction_percentage=0.0,
                ajava_files_count=0,
                iterations=0,
                execution_time_seconds=time.time() - start_time,
                error_message="Failed to restore from backup"
            )
        
        # Get baseline warnings
        java_files = find_java_files(project_dir)
        baseline_warnings, _ = run_checker(project_dir, checker_name, java_files)
        
        if baseline_warnings < 0:
            baseline_warnings = 0
        
        logger.info(f"Baseline warnings for {project_name}: {baseline_warnings}")
        
        # Find and modify POM files
        pom_path = project_dir / 'pom.xml'
        if not pom_path.exists():
            # Try to find pom.xml in subdirectory
            for p in project_dir.rglob('pom.xml'):
                pom_path = p
                break
        
        if not pom_path.exists():
            return WPIResult(
                project_name=project_name,
                checker_name=checker_name,
                success=False,
                baseline_warnings=baseline_warnings,
                after_wpi_warnings=baseline_warnings,
                reduction_percentage=0.0,
                ajava_files_count=0,
                iterations=0,
                execution_time_seconds=time.time() - start_time,
                error_message="No pom.xml found"
            )
        
        # Create WPI output directory
        wpi_output.mkdir(parents=True, exist_ok=True)
        
        # Modify POM for WPI
        if not self.modify_pom_for_wpi(pom_path, checker_name, str(wpi_output)):
            return WPIResult(
                project_name=project_name,
                checker_name=checker_name,
                success=False,
                baseline_warnings=baseline_warnings,
                after_wpi_warnings=baseline_warnings,
                reduction_percentage=0.0,
                ajava_files_count=0,
                iterations=0,
                execution_time_seconds=time.time() - start_time,
                error_message="Failed to modify POM"
            )
        
        # Run WPI iterations
        iterations, final_warnings = self.run_wpi_iterations(project_dir)
        
        if final_warnings < 0:
            final_warnings = baseline_warnings
        
        # Count ajava files
        ajava_count = self.count_ajava_files(wpi_output)
        
        # Calculate reduction
        reduction_pct = 0.0
        if baseline_warnings > 0:
            reduction_pct = ((baseline_warnings - final_warnings) / baseline_warnings) * 100.0
        
        return WPIResult(
            project_name=project_name,
            checker_name=checker_name,
            success=True,
            baseline_warnings=baseline_warnings,
            after_wpi_warnings=final_warnings,
            reduction_percentage=reduction_pct,
            ajava_files_count=ajava_count,
            iterations=iterations,
            execution_time_seconds=time.time() - start_time
        )
    
    def run_wpi_for_checker(self, checker_name: str) -> List[WPIResult]:
        """
        Run WPI for all projects for a checker.
        
        Args:
            checker_name: Checker name
            
        Returns:
            List of WPIResult
        """
        projects = WPI_PROJECTS.get(checker_name, [])
        
        if not projects:
            logger.error(f"No projects configured for WPI with {checker_name}")
            return []
        
        results = []
        for project_config in projects:
            project_name = project_config['name']
            logger.info(f"\n--- Running WPI for {project_name} with {checker_name} ---")
            
            result = self.run_wpi_for_project(project_name, checker_name, project_config)
            results.append(result)
            
            # Log result
            if result.success:
                logger.info(f"  Baseline: {result.baseline_warnings}, After: {result.after_wpi_warnings}")
                logger.info(f"  Reduction: {result.reduction_percentage:.1f}%")
                logger.info(f"  AJAVA files: {result.ajava_files_count}")
            else:
                logger.warning(f"  Failed: {result.error_message}")
        
        return results
    
    def save_results(self, results: Dict[str, List[WPIResult]], output_file: Path) -> None:
        """Save WPI results to JSON"""
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'checker_version': self.checker_version,
            },
            'results': {}
        }
        
        for checker_name, checker_results in results.items():
            output_data['results'][checker_name] = [
                asdict(r) for r in checker_results
            ]
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run WPI for all checkers')
    parser.add_argument('--checker', choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       help='Specific checker to run WPI for')
    parser.add_argument('--all', action='store_true', help='Run WPI for all checkers')
    parser.add_argument('--output', default='wpi_output/wpi_all_checkers_report.json',
                       help='Output file for results')
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    if not args.checker and not args.all:
        parser.error("Either --checker or --all must be specified")
    
    runner = WPIAllCheckersRunner(timeout=args.timeout)
    
    checkers_to_run = []
    if args.all:
        checkers_to_run = ['lower_bound', 'sql_quotes', 'signature_string']
    else:
        checkers_to_run = [args.checker]
    
    all_results = {}
    for checker_name in checkers_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running WPI for {checker_name} checker")
        logger.info(f"{'='*60}")
        
        results = runner.run_wpi_for_checker(checker_name)
        all_results[checker_name] = results
    
    # Save results
    output_path = GEN_DATA_ROOT / args.output
    runner.save_results(all_results, output_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("WPI SUMMARY")
    logger.info("="*60)
    
    for checker_name, results in all_results.items():
        logger.info(f"\n{checker_name}:")
        for r in results:
            status = "OK" if r.success else "FAILED"
            logger.info(f"  {r.project_name}: {status} - {r.reduction_percentage:.1f}% reduction")
    
    logger.info("\nWPI complete!")


if __name__ == '__main__':
    main()
