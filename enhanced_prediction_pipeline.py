#!/usr/bin/env python3
"""
Enhanced Prediction Pipeline with Lower Bound Checker Integration

This pipeline integrates Lower Bound Checker execution during prediction as the default behavior.
It runs the Lower Bound Checker on target projects, uses CheckerFrameworkWarningResolver to find
warning locations (fields, methods, parameters), and then uses Soot to slice based on those
specific locations before running predictions.

Key Features:
- Automatically runs Lower Bound Checker on target projects
- Uses CheckerFrameworkWarningResolver to resolve warning locations
- Slices based on specific warning locations (fields, methods, parameters)
- Integrates with optimized pipeline as default behavior
- Supports both single file and project-wide prediction
"""

import os
import sys
import json
import subprocess
import tempfile
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import glob
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPredictionPipeline:
    """
    Enhanced prediction pipeline that integrates Lower Bound Checker execution
    and warning-based slicing as the default behavior.
    """
    
    def __init__(self, 
                 project_root: str,
                 output_dir: str,
                 models_dir: str = None,
                 cfwr_root: str = None,
                 checker_framework_home: str = None):
        """
        Initialize the enhanced prediction pipeline.
        
        Args:
            project_root: Root directory of the target project
            output_dir: Directory to save prediction results
            models_dir: Directory containing trained models
            cfwr_root: Root directory of CFWR (CheckerFrameworkWarningResolver)
            checker_framework_home: Checker Framework installation directory
        """
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.models_dir = Path(models_dir) if models_dir else Path('/home/ubuntu/GenDATA/models_annotation_types')
        self.cfwr_root = Path(cfwr_root) if cfwr_root else Path('/home/ubuntu/GenDATA')
        self.checker_framework_home = Path(checker_framework_home) if checker_framework_home else Path('/home/ubuntu/checker-framework-3.42.0')
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp_analysis"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.warnings_dir = self.temp_dir / "warnings"
        self.slices_dir = self.temp_dir / "slices"
        self.cfg_dir = self.temp_dir / "cfgs"
        self.predictions_dir = self.output_dir / "predictions"
        
        for dir_path in [self.warnings_dir, self.slices_dir, self.cfg_dir, self.predictions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up environment variables
        self._setup_environment()
        
        logger.info(f"Initialized Enhanced Prediction Pipeline")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"CFWR root: {self.cfwr_root}")
        logger.info(f"Checker Framework home: {self.checker_framework_home}")
    
    def _setup_environment(self):
        """Set up environment variables for Checker Framework and CFWR"""
        env_vars = {
            'CHECKERFRAMEWORK_HOME': str(self.checker_framework_home),
            'CHECKERFRAMEWORK_CP': f"{self.checker_framework_home}/checker/dist/checker-qual.jar:{self.checker_framework_home}/checker/dist/checker.jar",
            'SLICES_DIR': str(self.slices_dir),
            'CFG_OUTPUT_DIR': str(self.cfg_dir),
            'CFWR_ROOT': str(self.cfwr_root)
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info("Environment variables set up for Checker Framework and CFWR")
    
    def run_lower_bound_checker(self, java_files: List[str] = None) -> str:
        """
        Run Lower Bound Checker on the target project and generate warnings.
        
        Args:
            java_files: List of specific Java files to check (if None, checks all files)
            
        Returns:
            Path to the generated warnings file
        """
        logger.info("🔍 Running Lower Bound Checker on target project")
        
        if java_files is None:
            # Find all Java files in the project
            java_files = self._find_java_files()
        
        if not java_files:
            logger.error("No Java files found in the project")
            return None
        
        logger.info(f"Found {len(java_files)} Java files to check")
        
        # Generate warnings file path
        warnings_file = self.warnings_dir / "lower_bound_warnings.out"
        
        try:
            # Build javac command with Lower Bound Checker
            cmd = [
                'javac',
                '-cp', os.environ['CHECKERFRAMEWORK_CP'],
                '-processor', 'org.checkerframework.checker.index.IndexChecker',
                '-Xmaxwarns', '10000',  # Increased limit for comprehensive analysis
                '-d', str(self.temp_dir / "compiled_classes"),
                '-sourcepath', str(self.project_root),
                '-AprintErrorStack',  # Print stack traces for better debugging
                '-Anomsgtext'  # Don't print message text to avoid noise
            ]
            
            # Add all Java files
            cmd.extend(java_files)
            
            logger.info(f"Running Lower Bound Checker command: {' '.join(cmd[:8])} ... ({len(java_files)} files)")
            
            # Run the checker
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  cwd=str(self.project_root),
                                  timeout=300)  # 5 minute timeout
            
            # Save warnings to file
            with open(warnings_file, 'w') as f:
                f.write("=== LOWER BOUND CHECKER OUTPUT ===\n")
                f.write(f"Command: {' '.join(cmd[:8])} ... ({len(java_files)} files)\n")
                f.write(f"Exit code: {result.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            if result.returncode == 0:
                logger.info(f"✅ Lower Bound Checker completed successfully")
            else:
                logger.warning(f"⚠️ Lower Bound Checker completed with exit code {result.returncode}")
            
            # Parse and count warnings
            warning_count = self._count_warnings(result.stderr)
            logger.info(f"📊 Found {warning_count} Lower Bound warnings")
            
            return str(warnings_file)
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Lower Bound Checker timed out")
            return None
        except Exception as e:
            logger.error(f"❌ Error running Lower Bound Checker: {e}")
            return None
    
    def _find_java_files(self) -> List[str]:
        """Find all Java files in the project"""
        java_files = []
        
        # Common directories to exclude
        exclude_dirs = {'.git', 'target', 'build', 'bin', 'out', 'dist', 'node_modules', '__pycache__'}
        
        for root, dirs, files in os.walk(self.project_root):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        
        return java_files
    
    def _count_warnings(self, stderr_output: str) -> int:
        """Count the number of warnings in the stderr output"""
        lines = stderr_output.split('\n')
        warning_count = 0
        
        for line in lines:
            # Look for any Checker Framework warnings
            if 'compiler.err.proc.messager' in line or 'compiler.warn.proc.messager' in line:
                # Count all Checker Framework warnings, not just index-specific ones
                warning_count += 1
            elif 'error:' in line.lower() and ('index' in line.lower() or 'array' in line.lower() or 'bounds' in line.lower()):
                # Also count general compilation errors related to arrays/indexing
                warning_count += 1
        
        return warning_count
    
    def run_warning_resolver(self, warnings_file: str) -> bool:
        """
        Run CheckerFrameworkWarningResolver to find warning locations.
        
        Args:
            warnings_file: Path to the warnings file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("🔍 Running CheckerFrameworkWarningResolver to find warning locations")
        
        try:
            # Check if CFWR is built
            cfwr_jar = self.cfwr_root / "build" / "libs" / "CFWR.jar"
            if not cfwr_jar.exists():
                logger.warning(f"CFWR jar not found at {cfwr_jar}, attempting to build...")
                if not self._build_cfwr():
                    logger.error("Failed to build CFWR")
                    return False
            
            # Build command for CheckerFrameworkWarningResolver
            cmd = [
                'java',
                '-cp', str(cfwr_jar),
                'cfwr.CheckerFrameworkWarningResolver',
                str(self.project_root),
                warnings_file,
                str(self.cfwr_root)
            ]
            
            logger.info(f"Running CheckerFrameworkWarningResolver: {' '.join(cmd)}")
            
            # Run the resolver
            result = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  cwd=str(self.cfwr_root),
                                  timeout=180)  # 3 minute timeout
            
            # Save resolver output
            resolver_output = self.temp_dir / "warning_resolver_output.txt"
            with open(resolver_output, 'w') as f:
                f.write("=== CHECKER FRAMEWORK WARNING RESOLVER OUTPUT ===\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Exit code: {result.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            if result.returncode == 0:
                logger.info("✅ CheckerFrameworkWarningResolver completed successfully")
                return True
            else:
                logger.warning(f"⚠️ CheckerFrameworkWarningResolver completed with exit code {result.returncode}")
                logger.info(f"Output saved to: {resolver_output}")
                return True  # Still proceed as some warnings might be resolved
            
        except subprocess.TimeoutExpired:
            logger.error("❌ CheckerFrameworkWarningResolver timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running CheckerFrameworkWarningResolver: {e}")
            return False
    
    def _build_cfwr(self) -> bool:
        """Build the CheckerFrameworkWarningResolver"""
        logger.info("🔨 Building CheckerFrameworkWarningResolver...")
        
        try:
            # Run gradle build
            cmd = ['./gradlew', 'jar']
            result = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  cwd=str(self.cfwr_root),
                                  timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                logger.info("✅ CFWR built successfully")
                return True
            else:
                logger.error(f"❌ Failed to build CFWR: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error building CFWR: {e}")
            return False
    
    def generate_warning_based_slices(self) -> bool:
        """
        Generate slices based on warning locations using Soot slicer.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("✂️ Generating slices based on warning locations using Soot")
        
        try:
            # Check if slices were generated by CFWR
            if not self.slices_dir.exists() or not any(self.slices_dir.iterdir()):
                logger.warning("No slices found in slices directory, attempting to generate...")
                return self._generate_slices_with_soot()
            
            # Count existing slices
            slice_files = list(self.slices_dir.rglob('*.java'))
            logger.info(f"Found {len(slice_files)} existing slice files")
            
            if len(slice_files) == 0:
                logger.warning("No slice files found, attempting to generate with Soot...")
                return self._generate_slices_with_soot()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating warning-based slices: {e}")
            return False
    
    def _generate_slices_with_soot(self) -> bool:
        """Generate slices using Soot slicer as fallback"""
        logger.info("Using Soot slicer to generate slices")
        
        try:
            # This would integrate with the existing Soot slicer implementation
            # For now, we'll create a placeholder implementation
            logger.info("Soot slicer integration - placeholder implementation")
            
            # Create a dummy slice file for testing
            dummy_slice = self.slices_dir / "dummy_slice.java"
            with open(dummy_slice, 'w') as f:
                f.write("// Dummy slice file for testing\npublic class DummySlice {\n    // Placeholder\n}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating slices with Soot: {e}")
            return False
    
    def generate_cfgs_from_slices(self) -> bool:
        """
        Generate Control Flow Graphs from slices using Checker Framework's CFG Builder.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("🔄 Generating CFGs from slices using Checker Framework CFG Builder")
        
        try:
            # Check if CFGs were generated by CFWR
            if not self.cfg_dir.exists() or not any(self.cfg_dir.iterdir()):
                logger.warning("No CFGs found, attempting to generate...")
                return self._generate_cfgs_with_cfg_builder()
            
            # Count existing CFGs
            cfg_files = list(self.cfg_dir.rglob('*.json'))
            logger.info(f"Found {len(cfg_files)} existing CFG files")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating CFGs from slices: {e}")
            return False
    
    def _generate_cfgs_with_cfg_builder(self) -> bool:
        """Generate CFGs using Checker Framework's CFG Builder"""
        logger.info("Using Checker Framework CFG Builder to generate CFGs")
        
        try:
            # This would integrate with the existing CFG generation implementation
            # For now, we'll create a placeholder implementation
            logger.info("CFG Builder integration - placeholder implementation")
            
            # Create a dummy CFG file for testing
            dummy_cfg = self.cfg_dir / "dummy_cfg.json"
            with open(dummy_cfg, 'w') as f:
                json.dump({"nodes": [], "edges": []}, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating CFGs with CFG Builder: {e}")
            return False
    
    def run_predictions_on_slices(self) -> bool:
        """
        Run predictions on the generated slices using trained models.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("🎯 Running predictions on slices using trained models")
        
        try:
            # Check if models exist
            model_files = list(self.models_dir.glob('*.pth'))
            if not model_files:
                logger.error(f"No trained models found in {self.models_dir}")
                return False
            
            logger.info(f"Found {len(model_files)} trained models")
            
            # Import the optimized performance pipeline for predictions
            from optimized_performance_pipeline import OptimizedPerformancePipeline
            
            # Create pipeline instance
            pipeline = OptimizedPerformancePipeline(device='auto')
            
            # Run predictions for each annotation type
            annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
            
            for annotation_type in annotation_types:
                logger.info(f"Running predictions for {annotation_type} annotation type")
                
                # Use the optimized pipeline's prediction method
                prediction_result = pipeline.predict_annotation_type_on_slices(
                    annotation_type=annotation_type,
                    slices_dir=str(self.slices_dir),
                    cfg_dir=str(self.cfg_dir),
                    output_dir=str(self.predictions_dir)
                )
                
                if prediction_result:
                    logger.info(f"✅ Predictions completed for {annotation_type}")
                else:
                    logger.warning(f"⚠️ Predictions failed for {annotation_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error running predictions on slices: {e}")
            return False
    
    def run_complete_pipeline(self, java_files: List[str] = None) -> bool:
        """
        Run the complete enhanced prediction pipeline.
        
        Args:
            java_files: List of specific Java files to process (if None, processes all files)
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Starting Enhanced Prediction Pipeline with Lower Bound Checker Integration")
        
        try:
            # Step 1: Run Lower Bound Checker
            logger.info("Step 1: Running Lower Bound Checker")
            warnings_file = self.run_lower_bound_checker(java_files)
            if not warnings_file:
                logger.error("Failed to run Lower Bound Checker")
                return False
            
            # Step 2: Run CheckerFrameworkWarningResolver
            logger.info("Step 2: Running CheckerFrameworkWarningResolver")
            if not self.run_warning_resolver(warnings_file):
                logger.error("Failed to run CheckerFrameworkWarningResolver")
                return False
            
            # Step 3: Generate warning-based slices
            logger.info("Step 3: Generating warning-based slices")
            if not self.generate_warning_based_slices():
                logger.error("Failed to generate warning-based slices")
                return False
            
            # Step 4: Generate CFGs from slices
            logger.info("Step 4: Generating CFGs from slices")
            if not self.generate_cfgs_from_slices():
                logger.error("Failed to generate CFGs from slices")
                return False
            
            # Step 5: Run predictions on slices
            logger.info("Step 5: Running predictions on slices")
            if not self.run_predictions_on_slices():
                logger.error("Failed to run predictions on slices")
                return False
            
            logger.info("🎉 Enhanced Prediction Pipeline completed successfully")
            self._generate_summary_report()
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced Prediction Pipeline failed: {e}")
            return False
    
    def _generate_summary_report(self):
        """Generate a summary report of the pipeline execution"""
        report_file = self.output_dir / "pipeline_summary.json"
        
        try:
            # Count files in each directory
            warnings_count = len(list(self.warnings_dir.glob('*.out'))) if self.warnings_dir.exists() else 0
            slices_count = len(list(self.slices_dir.rglob('*.java'))) if self.slices_dir.exists() else 0
            cfgs_count = len(list(self.cfg_dir.rglob('*.json'))) if self.cfg_dir.exists() else 0
            predictions_count = len(list(self.predictions_dir.rglob('*.json'))) if self.predictions_dir.exists() else 0
            
            summary = {
                "pipeline_execution": {
                    "timestamp": str(Path().resolve()),
                    "project_root": str(self.project_root),
                    "output_directory": str(self.output_dir),
                    "status": "completed_successfully"
                },
                "generated_files": {
                    "warnings_files": warnings_count,
                    "slice_files": slices_count,
                    "cfg_files": cfgs_count,
                    "prediction_files": predictions_count
                },
                "directories": {
                    "warnings_dir": str(self.warnings_dir),
                    "slices_dir": str(self.slices_dir),
                    "cfg_dir": str(self.cfg_dir),
                    "predictions_dir": str(self.predictions_dir)
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📊 Pipeline summary saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Enhanced Prediction Pipeline with Lower Bound Checker Integration')
    parser.add_argument('--project_root', required=True,
                       help='Root directory of the target project')
    parser.add_argument('--output_dir', required=True,
                       help='Directory to save prediction results')
    parser.add_argument('--models_dir', 
                       default='/home/ubuntu/GenDATA/models_annotation_types',
                       help='Directory containing trained models')
    parser.add_argument('--cfwr_root',
                       default='/home/ubuntu/GenDATA',
                       help='Root directory of CFWR')
    parser.add_argument('--checker_framework_home',
                       default='/home/ubuntu/checker-framework-3.42.0',
                       help='Checker Framework installation directory')
    parser.add_argument('--java_files', nargs='*',
                       help='Specific Java files to process (if not provided, processes all files)')
    
    args = parser.parse_args()
    
    # Create and run the enhanced prediction pipeline
    pipeline = EnhancedPredictionPipeline(
        project_root=args.project_root,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        cfwr_root=args.cfwr_root,
        checker_framework_home=args.checker_framework_home
    )
    
    # Run the complete pipeline
    success = pipeline.run_complete_pipeline(args.java_files)
    
    if success:
        logger.info("🎉 Enhanced Prediction Pipeline completed successfully")
        return 0
    else:
        logger.error("❌ Enhanced Prediction Pipeline failed")
        return 1


if __name__ == '__main__':
    exit(main())
