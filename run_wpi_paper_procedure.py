#!/usr/bin/env python3
"""
WPI Paper Procedure Implementation

Implements the experimental procedure from kelloggm/wpi-paper to run
Whole Program Inference on Maven projects by directly modifying build files.
"""

import os
import sys
import re
import shutil
import subprocess
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# XML namespace for Maven POM
POM_NS = '{http://maven.apache.org/POM/4.0.0}'


@dataclass
class WPIResult:
    """Result of running WPI on a project"""
    project_name: str
    success: bool
    iterations: int
    final_warnings: int
    ajava_files_count: int
    execution_time_seconds: float
    error_message: Optional[str] = None


class WPIPaperProcedure:
    """
    Implements the WPI paper experimental procedure for Maven projects.
    
    This approach directly modifies pom.xml files to add Checker Framework
    processor configuration, avoiding DLJC compatibility issues.
    """
    
    def __init__(self, 
                 base_dir: str = '/home/ubuntu/GenDATA',
                 checker_version: str = '3.42.0'):
        """
        Initialize WPI paper procedure runner.
        
        Args:
            base_dir: Base directory for WPI work
            checker_version: Checker Framework version to use
        """
        self.base_dir = Path(base_dir)
        self.wpi_projects_dir = self.base_dir / 'wpi_projects'
        self.wpi_output_dir = self.base_dir / 'wpi_output'
        self.checker_version = checker_version
        
        # Projects to process
        self.projects = {
            'pom-tuner': {
                'wpi_dir': 'pom-tuner-wpi',
                'modules': ['pom-tuner'],  # Only main module, skip tests
                'is_tycho': False,
            },
            'sortpom': {
                'wpi_dir': 'sortpom-wpi',
                'modules': ['sorter'],  # Main sorter module
                'is_tycho': False,
            },
            'eclipse-external-annotations-m2e-plugin': {
                'wpi_dir': 'eclipse-wpi',
                'modules': ['eclipse-external-annotations-m2e-plugin.core'],
                'is_tycho': True,  # Uses Tycho/Eclipse build - different handling needed
            }
        }
        
        logger.info(f"WPI Paper Procedure initialized")
        logger.info(f"  Projects dir: {self.wpi_projects_dir}")
        logger.info(f"  Output dir: {self.wpi_output_dir}")
        logger.info(f"  Checker version: {self.checker_version}")
    
    def get_checker_framework_pom_additions(self, wpi_output_path: str) -> str:
        """
        Generate the POM XML additions for Checker Framework WPI.
        
        Args:
            wpi_output_path: Path where WPI should output ajava files
            
        Returns:
            XML string to add to POM
        """
        return f'''
    <!-- Checker Framework WPI Configuration -->
    <properties>
        <checkerframework.version>{self.checker_version}</checkerframework.version>
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
                        <annotationProcessor>org.checkerframework.checker.index.IndexChecker</annotationProcessor>
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

    def modify_pom_for_wpi(self, pom_path: Path, wpi_output_path: str) -> bool:
        """
        Modify a pom.xml to add Checker Framework WPI configuration.
        
        Args:
            pom_path: Path to pom.xml file
            wpi_output_path: Path for WPI output
            
        Returns:
            True if modification was successful
        """
        try:
            # Read the original POM
            with open(pom_path, 'r') as f:
                content = f.read()
            
            # Backup original
            backup_path = pom_path.with_suffix('.xml.backup')
            with open(backup_path, 'w') as f:
                f.write(content)
            
            # Check if already modified
            if 'checkerframework' in content.lower():
                logger.info(f"  POM already has Checker Framework config: {pom_path}")
                return True
            
            # Add Checker Framework version property
            if '<properties>' in content:
                # Add to existing properties
                content = content.replace(
                    '<properties>',
                    f'<properties>\n        <checkerframework.version>{self.checker_version}</checkerframework.version>'
                )
            else:
                # Add properties section before dependencies or build
                insert_point = content.find('<dependencies>')
                if insert_point == -1:
                    insert_point = content.find('<build>')
                if insert_point == -1:
                    insert_point = content.find('</project>')
                
                if insert_point != -1:
                    prop_section = f'''
    <properties>
        <checkerframework.version>{self.checker_version}</checkerframework.version>
    </properties>

'''
                    content = content[:insert_point] + prop_section + content[insert_point:]
            
            # Add checker-qual dependency
            checker_qual_dep = f'''
        <dependency>
            <groupId>org.checkerframework</groupId>
            <artifactId>checker-qual</artifactId>
            <version>${{checkerframework.version}}</version>
        </dependency>
'''
            if '<dependencies>' in content:
                content = content.replace('<dependencies>', '<dependencies>' + checker_qual_dep)
            else:
                # Add dependencies section
                insert_point = content.find('<build>')
                if insert_point == -1:
                    insert_point = content.find('</project>')
                if insert_point != -1:
                    dep_section = f'''
    <dependencies>
{checker_qual_dep}
    </dependencies>

'''
                    content = content[:insert_point] + dep_section + content[insert_point:]
            
            # Add or modify maven-compiler-plugin
            compiler_plugin_config = f'''
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
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
                        <annotationProcessor>org.checkerframework.checker.index.IndexChecker</annotationProcessor>
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
'''
            
            # Check if there's already a maven-compiler-plugin in plugins
            if '<artifactId>maven-compiler-plugin</artifactId>' in content:
                # Need to modify existing plugin - this is complex, so we'll add to pluginManagement
                logger.warning(f"  Existing maven-compiler-plugin found, adding to profile instead")
            
            # Add a WPI profile
            wpi_profile = f'''
    <profiles>
        <profile>
            <id>wpi</id>
            <build>
                <plugins>
{compiler_plugin_config}
                </plugins>
            </build>
        </profile>
    </profiles>
'''
            
            # Find where to insert the profile
            if '<profiles>' in content:
                # Add to existing profiles
                content = content.replace('</profiles>', f'''
        <profile>
            <id>wpi</id>
            <build>
                <plugins>
{compiler_plugin_config}
                </plugins>
            </build>
        </profile>
    </profiles>''')
            else:
                # Add profiles section before </project>
                content = content.replace('</project>', wpi_profile + '</project>')
            
            # Remove -Werror if present
            content = re.sub(r'<arg>-Werror</arg>\s*', '', content)
            
            # Write modified POM
            with open(pom_path, 'w') as f:
                f.write(content)
            
            logger.info(f"  Modified POM: {pom_path}")
            return True
            
        except Exception as e:
            logger.error(f"  Error modifying POM {pom_path}: {e}")
            return False

    def create_wpi_script(self, project_dir: Path, project_name: str, 
                          wpi_output_path: str, modules: List[str]) -> Path:
        """
        Create the WPI iteration script for a project.
        
        Args:
            project_dir: Project root directory
            project_name: Name of the project
            wpi_output_path: Path for WPI output
            modules: List of module names to build
            
        Returns:
            Path to created script
        """
        # Determine WPIOUTDIR - for Maven, it's typically in target/
        # The Checker Framework creates build/whole-program-inference
        if modules:
            wpi_out_dirs = ' '.join([f'{m}/build/whole-program-inference' for m in modules])
            wpi_out_dir = f'{modules[0]}/build/whole-program-inference'
        else:
            wpi_out_dirs = 'build/whole-program-inference'
            wpi_out_dir = 'build/whole-program-inference'
        
        script_content = f'''#!/bin/bash

# WPI iteration script for {project_name}
# Generated by run_wpi_paper_procedure.py

set -e

# Build command - use WPI profile, skip tests and style checks
BUILD_CMD="mvn compile -Pwpi -DskipTests -Deditorconfig.skip=true -Dspotless.check.skip=true -Denforcer.skip=true -Dcheckstyle.skip=true -Dlicense.skip=true -Dformatter.skip=true -Dimpsort.skip=true -q"
CLEAN_CMD="mvn clean -q"

# WPI output directories
WPITEMPDIR="{wpi_output_path}"
WPIOUTDIR="{wpi_out_dir}"

DEBUG=1
MAX_ITERATIONS=10

# Ensure output directory exists
rm -rf "${{WPITEMPDIR}}"
mkdir -p "${{WPITEMPDIR}}"

# Initial build to create WPIOUTDIR
echo "Initial build..."
${{BUILD_CMD}} 2>&1 | tee wpi_build.log || true

count=1
while [ $count -le $MAX_ITERATIONS ]; do
    if [[ ${{DEBUG}} == 1 ]]; then
        echo "============================================"
        echo "Entering iteration ${{count}}"
        echo "============================================"
    fi
    
    # Build and capture output
    ${{BUILD_CMD}} 2>&1 | tee -a wpi_build.log || true
    
    # Check if WPIOUTDIR was created
    if [ ! -d "${{WPIOUTDIR}}" ]; then
        echo "Warning: WPIOUTDIR not created yet, trying alternative locations..."
        # Try to find the WPI output
        for candidate in build/whole-program-inference target/build/whole-program-inference */build/whole-program-inference; do
            if [ -d "$candidate" ]; then
                WPIOUTDIR="$candidate"
                echo "Found WPI output at: ${{WPIOUTDIR}}"
                break
            fi
        done
    fi
    
    if [ ! -d "${{WPIOUTDIR}}" ]; then
        echo "WPIOUTDIR still not found after iteration ${{count}}"
        if [ $count -ge 3 ]; then
            echo "Giving up after ${{count}} iterations without WPI output"
            break
        fi
        ((count++))
        continue
    fi
    
    # Create temp dir if needed
    mkdir -p "${{WPITEMPDIR}}"
    
    # Compare directories
    DIFF_RESULT=$(diff -r "${{WPITEMPDIR}}" "${{WPIOUTDIR}}" 2>/dev/null || true)
    
    if [[ ${{DEBUG}} == 1 ]]; then
        echo "Diff result length: ${{#DIFF_RESULT}}"
        echo "${{DIFF_RESULT}}" > "iteration${{count}}.diff"
    fi
    
    # If no difference, WPI has converged
    if [[ -z "${{DIFF_RESULT}}" ]] && [ -d "${{WPIOUTDIR}}" ] && [ "$(ls -A ${{WPIOUTDIR}} 2>/dev/null)" ]; then
        echo "WPI converged after ${{count}} iterations"
        break
    fi
    
    # Copy new output to temp
    rm -rf "${{WPITEMPDIR}}"
    if [ -d "${{WPIOUTDIR}}" ]; then
        cp -r "${{WPIOUTDIR}}" "${{WPITEMPDIR}}"
    else
        mkdir -p "${{WPITEMPDIR}}"
    fi
    
    # Clean for next iteration
    ${{CLEAN_CMD}} 2>/dev/null || true
    
    ((count++))
done

# Final build to get warnings
echo "============================================"
echo "Final build to collect warnings"
echo "============================================"
${{BUILD_CMD}} 2>&1 | tee typecheck.out

# Count warnings
echo "============================================"
echo "Results:"
echo "============================================"
echo "Iterations: $((count - 1))"
echo "Ajava files in output:"
find "${{WPITEMPDIR}}" -name "*.ajava" 2>/dev/null | wc -l || echo "0"
echo "Warnings in final build:"
grep -c "warning:" typecheck.out 2>/dev/null || echo "0"
'''
        
        script_path = project_dir / 'wpi.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"  Created WPI script: {script_path}")
        return script_path

    def run_wpi(self, project_name: str, timeout: int = 1800) -> WPIResult:
        """
        Run WPI on a project.
        
        Args:
            project_name: Name of the project
            timeout: Timeout in seconds
            
        Returns:
            WPIResult with execution details
        """
        project_config = self.projects.get(project_name)
        if not project_config:
            return WPIResult(
                project_name=project_name,
                success=False,
                iterations=0,
                final_warnings=0,
                ajava_files_count=0,
                execution_time_seconds=0,
                error_message=f"Unknown project: {project_name}"
            )
        
        project_dir = self.wpi_projects_dir / project_name
        wpi_output_path = str(self.wpi_output_dir / project_config['wpi_dir'])
        
        if not project_dir.exists():
            return WPIResult(
                project_name=project_name,
                success=False,
                iterations=0,
                final_warnings=0,
                ajava_files_count=0,
                execution_time_seconds=0,
                error_message=f"Project directory not found: {project_dir}"
            )
        
        logger.info(f"Running WPI on {project_name}...")
        logger.info(f"  Project dir: {project_dir}")
        logger.info(f"  WPI output: {wpi_output_path}")
        
        # Check if wpi.sh exists
        wpi_script = project_dir / 'wpi.sh'
        if not wpi_script.exists():
            return WPIResult(
                project_name=project_name,
                success=False,
                iterations=0,
                final_warnings=0,
                ajava_files_count=0,
                execution_time_seconds=0,
                error_message=f"WPI script not found: {wpi_script}"
            )
        
        start_time = time.time()
        
        try:
            # Run the WPI script
            result = subprocess.run(
                ['bash', str(wpi_script)],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            output = result.stdout + result.stderr
            
            # Save output
            output_file = project_dir / 'wpi_output.log'
            with open(output_file, 'w') as f:
                f.write(output)
            
            # Parse results
            iterations = 0
            for match in re.finditer(r'Entering iteration (\d+)', output):
                iterations = max(iterations, int(match.group(1)))
            
            # Count warnings
            typecheck_file = project_dir / 'typecheck.out'
            final_warnings = 0
            if typecheck_file.exists():
                with open(typecheck_file, 'r') as f:
                    typecheck_content = f.read()
                final_warnings = len(re.findall(r'warning:', typecheck_content))
            
            # Count ajava files
            ajava_count = 0
            wpi_dir = Path(wpi_output_path)
            if wpi_dir.exists():
                ajava_count = len(list(wpi_dir.rglob('*.ajava')))
            
            logger.info(f"  WPI completed in {execution_time:.1f}s")
            logger.info(f"  Iterations: {iterations}")
            logger.info(f"  Final warnings: {final_warnings}")
            logger.info(f"  Ajava files: {ajava_count}")
            
            return WPIResult(
                project_name=project_name,
                success=True,
                iterations=iterations,
                final_warnings=final_warnings,
                ajava_files_count=ajava_count,
                execution_time_seconds=execution_time
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return WPIResult(
                project_name=project_name,
                success=False,
                iterations=0,
                final_warnings=0,
                ajava_files_count=0,
                execution_time_seconds=execution_time,
                error_message=f"Timeout after {timeout}s"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return WPIResult(
                project_name=project_name,
                success=False,
                iterations=0,
                final_warnings=0,
                ajava_files_count=0,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )

    def setup_project(self, project_name: str) -> bool:
        """
        Set up a project for WPI by modifying its POM and creating the WPI script.
        
        Args:
            project_name: Name of the project
            
        Returns:
            True if setup was successful
        """
        project_config = self.projects.get(project_name)
        if not project_config:
            logger.error(f"Unknown project: {project_name}")
            return False
        
        project_dir = self.wpi_projects_dir / project_name
        wpi_output_path = str(self.wpi_output_dir / project_config['wpi_dir'])
        modules = project_config.get('modules', [])
        
        logger.info(f"Setting up {project_name} for WPI...")
        
        # Modify parent POM
        parent_pom = project_dir / 'pom.xml'
        if parent_pom.exists():
            if not self.modify_pom_for_wpi(parent_pom, wpi_output_path):
                return False
        
        # Modify module POMs if specified
        for module in modules:
            module_pom = project_dir / module / 'pom.xml'
            if module_pom.exists():
                if not self.modify_pom_for_wpi(module_pom, wpi_output_path):
                    logger.warning(f"  Failed to modify module POM: {module_pom}")
        
        # Create WPI script
        self.create_wpi_script(project_dir, project_name, wpi_output_path, modules)
        
        return True

    def run_all(self) -> List[WPIResult]:
        """
        Run WPI on all configured projects.
        
        Returns:
            List of WPIResult for each project
        """
        results = []
        
        for project_name in self.projects:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {project_name}")
            logger.info('='*60)
            
            # Setup project
            if not self.setup_project(project_name):
                results.append(WPIResult(
                    project_name=project_name,
                    success=False,
                    iterations=0,
                    final_warnings=0,
                    ajava_files_count=0,
                    execution_time_seconds=0,
                    error_message="Setup failed"
                ))
                continue
            
            # Run WPI
            result = self.run_wpi(project_name)
            results.append(result)
        
        return results

    def generate_report(self, results: List[WPIResult]) -> None:
        """Generate a report of WPI results."""
        timestamp = datetime.now().isoformat()
        
        # JSON report
        json_report = {
            'metadata': {
                'timestamp': timestamp,
                'checker_version': self.checker_version,
                'procedure': 'wpi-paper'
            },
            'results': [
                {
                    'project': r.project_name,
                    'success': r.success,
                    'iterations': r.iterations,
                    'final_warnings': r.final_warnings,
                    'ajava_files': r.ajava_files_count,
                    'execution_time': r.execution_time_seconds,
                    'error': r.error_message
                }
                for r in results
            ]
        }
        
        json_path = self.wpi_output_dir / 'wpi_paper_results.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"JSON report saved to: {json_path}")
        
        # Markdown report
        md_lines = [
            "# WPI Paper Procedure Results",
            "",
            f"**Generated**: {timestamp}",
            f"**Checker Framework Version**: {self.checker_version}",
            "",
            "## Summary",
            "",
            "| Project | Success | Iterations | Warnings | Ajava Files | Time |",
            "|---------|---------|------------|----------|-------------|------|"
        ]
        
        for r in results:
            status = "Yes" if r.success else "No"
            time_str = f"{r.execution_time_seconds:.1f}s"
            md_lines.append(
                f"| {r.project_name} | {status} | {r.iterations} | "
                f"{r.final_warnings} | {r.ajava_files_count} | {time_str} |"
            )
        
        md_lines.extend([
            "",
            "## Detailed Results",
            ""
        ])
        
        for r in results:
            md_lines.extend([
                f"### {r.project_name}",
                "",
                f"- **Success**: {r.success}",
                f"- **Iterations**: {r.iterations}",
                f"- **Final Warnings**: {r.final_warnings}",
                f"- **Ajava Files Generated**: {r.ajava_files_count}",
                f"- **Execution Time**: {r.execution_time_seconds:.1f}s",
            ])
            
            if r.error_message:
                md_lines.append(f"- **Error**: {r.error_message}")
            
            md_lines.append("")
        
        md_path = self.wpi_output_dir / 'wpi_paper_results.md'
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"Markdown report saved to: {md_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run WPI using the wpi-paper experimental procedure'
    )
    parser.add_argument(
        '--project', '-p',
        type=str,
        help='Run on specific project only'
    )
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only set up projects, do not run WPI'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=1800,
        help='Timeout per project in seconds (default: 1800)'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = WPIPaperProcedure()
    
    if args.project:
        if args.project not in runner.projects:
            logger.error(f"Unknown project: {args.project}")
            logger.error(f"Available: {list(runner.projects.keys())}")
            sys.exit(1)
        
        if args.setup_only:
            runner.setup_project(args.project)
        else:
            runner.setup_project(args.project)
            result = runner.run_wpi(args.project, timeout=args.timeout)
            runner.generate_report([result])
    else:
        if args.setup_only:
            for project_name in runner.projects:
                runner.setup_project(project_name)
        else:
            results = runner.run_all()
            runner.generate_report(results)
    
    logger.info(f"\nResults saved to: {runner.wpi_output_dir}")


if __name__ == '__main__':
    main()
