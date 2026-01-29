#!/usr/bin/env python3
"""
Evaluate Annotation Placement Effectiveness by Base Model

Tests how much each base model's annotation placement reduces Lower Bound Checker warnings.
For each qualifying project, tests each base model separately and measures warning reduction.
"""

import os
import json
import shutil
import subprocess
import tempfile
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from filtered_multi_checker_predictor import FilteredMultiCheckerPredictor
from place_annotations import ComprehensiveAnnotationPlacer
from test_lower_bound_warnings import LowerBoundWarningTester, WarningStats
from checker_evaluation_config import get_checker_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResult:
    """Result for a single base model evaluation"""
    base_model: str
    annotations_placed: int
    warnings_after: int
    warning_reduction: int
    reduction_percentage: float
    placement_success: bool
    compilation_success: bool
    error_message: Optional[str] = None


@dataclass
class ProjectEvaluationResult:
    """Result for a single project evaluation"""
    project_name: str
    project_url: str
    baseline_warnings: int
    model_results: List[ModelEvaluationResult]
    error_message: Optional[str] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    metadata: Dict[str, Any]
    results: List[ProjectEvaluationResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'metadata': self.metadata,
            'results': [asdict(result) for result in self.results]
        }


class AnnotationPlacementEvaluator:
    """Evaluates annotation placement effectiveness by base model"""
    
    # Checker processor mapping
    CHECKER_PROCESSORS = {
        'lower_bound': 'org.checkerframework.checker.index.IndexChecker',
        'sql_quotes': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
        'signature_string': 'org.checkerframework.checker.signature.SignatureChecker',
    }
    
    # Backup directories that should never be modified
    BACKUP_DIRECTORIES = [
        Path('/home/ubuntu/GenDATA/case_studies_backup'),
        Path('/home/ubuntu/GenDATA/annotation_evaluation/backups'),
        Path('/home/ubuntu/GenDATA/annotated_projects_backup'),
    ]
    
    def __init__(self, 
                 work_dir: str = './annotation_evaluation',
                 checker_cp: Optional[str] = None,
                 cfg_dir: Optional[str] = None,
                 timeout: int = 600,
                 checker_name: str = 'lower_bound'):
        """
        Initialize evaluator.
        
        Args:
            work_dir: Working directory for evaluation
            checker_cp: Checker Framework classpath
            cfg_dir: Directory containing CFG files (if None, will try to find/generate)
            timeout: Timeout for compilation/checker runs
            checker_name: Name of the checker ('lower_bound', 'sql_quotes', 'signature_string')
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.checker_cp = checker_cp or os.environ.get('CHECKERFRAMEWORK_CP', '')
        self.cfg_dir = cfg_dir
        self.timeout = timeout
        self.checker_name = checker_name.lower()
        
        # Validate checker name
        if self.checker_name not in self.CHECKER_PROCESSORS:
            raise ValueError(f"Unknown checker: {checker_name}. "
                           f"Supported: {list(self.CHECKER_PROCESSORS.keys())}")
        
        # Base models to test
        config = get_checker_config(self.checker_name)
        self.base_models = config.get('base_models', ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n'])
        
        # Initialize warning tester (still uses LowerBoundWarningTester but we override the processor)
        self.warning_tester = LowerBoundWarningTester(
            checker_cp=self.checker_cp,
            temp_dir=str(self.work_dir / 'temp_repos'),
            timeout=self.timeout
        )
        
        logger.info(f"Initialized AnnotationPlacementEvaluator")
        logger.info(f"  Checker: {self.checker_name}")
        logger.info(f"  Work directory: {self.work_dir}")
        logger.info(f"  Base models to test: {self.base_models}")
    
    def verify_not_backup_dir(self, path: Path) -> bool:
        """Verify that a path is not inside a backup directory (safety check)"""
        for backup_dir in self.BACKUP_DIRECTORIES:
            if backup_dir.exists():
                try:
                    path.relative_to(backup_dir)
                    logger.error(f"SAFETY: Attempted to modify backup directory: {path}")
                    return False
                except ValueError:
                    continue
        return True
    
    def load_qualifying_projects(self, candidates_file: str) -> List[Dict[str, Any]]:
        """
        Load qualifying projects from candidates file.
        
        Qualifying projects: compiled successfully AND have warnings > 0
        
        Args:
            candidates_file: Path to lower_bound_project_candidates.json
            
        Returns:
            List of qualifying project dictionaries
        """
        with open(candidates_file, 'r') as f:
            data = json.load(f)
        
        qualifying = []
        for project in data.get('ranked_projects', []):
            if project.get('compilation_success') and project.get('warning_count', 0) > 0:
                qualifying.append(project)
        
        logger.info(f"Found {len(qualifying)} qualifying projects:")
        for proj in qualifying:
            logger.info(f"  - {proj['project_name']}: {proj['warning_count']} warnings")
        
        return qualifying
    
    def verify_project_compilable(self, project_dir: Path, project_name: str) -> Tuple[bool, str]:
        """
        Verify that a project can compile and be analyzed by the Checker Framework.
        
        This pre-flight check ensures:
        1. Maven projects compile successfully with dependencies resolved
        2. The Checker Framework can run without crashing
        3. Files are actually processed (not 0 due to errors)
        
        Args:
            project_dir: Path to project directory
            project_name: Name of the project
            
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Pre-flight check for {project_name}...")
        
        # Step 1: Check if Maven project and compile it
        try:
            from maven_classpath_resolver import MavenClasspathResolver
            resolver = MavenClasspathResolver(timeout=self.timeout)
            
            if resolver.is_maven_project(project_dir):
                logger.info(f"  Compiling Maven project...")
                success, error = resolver.compile_project(project_dir)
                if not success:
                    return False, f"Maven compilation failed: {error}"
                logger.info(f"  Maven compilation successful")
        except ImportError:
            logger.debug("Maven resolver not available")
        except Exception as e:
            logger.warning(f"  Maven check failed: {e}")
        
        # Step 2: Run checker on a sample of files
        try:
            java_files = self.warning_tester.find_java_files(project_dir, max_files=5)
            if not java_files:
                return False, "No Java files found"
            
            logger.info(f"  Testing checker on {len(java_files)} sample files...")
            success, output = self.warning_tester.run_lower_bound_checker(project_dir, java_files)
            
            # Step 3: Verify checker actually ran
            try:
                from checker_crash_detector import detect_checker_crash
                # Pass return code (0 if success, 1 if not)
                returncode = 0 if success else 1
                crash_result = detect_checker_crash(output, returncode=returncode)
                
                if crash_result.crashed:
                    return False, f"Checker crashed: {crash_result.crash_reason}"
                
                if crash_result.no_files_processed:
                    return False, "Checker did not process any files"
                
                if not crash_result.checker_analysis_succeeded():
                    if crash_result.has_compilation_errors:
                        return False, f"Compilation errors prevent analysis ({crash_result.compilation_error_count} errors)"
                    return False, "Checker analysis did not complete successfully"
                
                logger.info(f"  Pre-flight check passed")
                return True, "Project compiles and checker runs successfully"
                
            except ImportError:
                # If crash detector not available, just check success flag
                if success:
                    return True, "Checker ran (crash detection not available)"
                return False, f"Checker failed: {output[:200]}"
                
        except Exception as e:
            return False, f"Pre-flight check error: {e}"
    
    def backup_project(self, project_dir: Path, project_name: str) -> Path:
        """
        Create a backup of the project.
        
        Args:
            project_dir: Directory to backup
            project_name: Name of the project
            
        Returns:
            Path to backup directory
        """
        backup_dir = self.work_dir / 'backups' / project_name
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        logger.info(f"Backing up {project_name} to {backup_dir}")
        shutil.copytree(project_dir, backup_dir)
        
        # Verify backup was successful
        if not self.verify_backup(backup_dir, project_dir):
            raise RuntimeError(f"Backup verification failed for {project_name}")
        
        return backup_dir
    
    def verify_backup(self, backup_dir: Path, original_dir: Path) -> bool:
        """
        Verify backup integrity by checking key files exist.
        
        Args:
            backup_dir: Backup directory
            original_dir: Original directory
            
        Returns:
            True if backup appears valid
        """
        try:
            # Check that backup directory exists and is not empty
            if not backup_dir.exists():
                logger.error(f"Backup directory does not exist: {backup_dir}")
                return False
            
            # Check for at least some Java files in backup
            java_files_backup = list(backup_dir.rglob('*.java'))
            java_files_original = list(original_dir.rglob('*.java'))
            
            if len(java_files_backup) == 0:
                logger.error(f"No Java files found in backup")
                return False
            
            if len(java_files_backup) < len(java_files_original) * 0.9:  # Allow some tolerance
                logger.warning(f"Backup has fewer Java files ({len(java_files_backup)}) than original ({len(java_files_original)})")
            
            logger.info(f"Backup verified: {len(java_files_backup)} Java files")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False
    
    def restore_project(self, project_dir: Path, backup_dir: Path):
        """
        Restore project from backup.
        
        Args:
            project_dir: Directory to restore to
            backup_dir: Backup directory to restore from
        """
        if not backup_dir.exists():
            raise ValueError(f"Backup directory does not exist: {backup_dir}")
        
        # Verify backup before restoring
        if not self.verify_backup(backup_dir, backup_dir):  # Self-check backup
            raise ValueError(f"Backup verification failed: {backup_dir}")
        
        # Remove existing directory
        if project_dir.exists():
            logger.info(f"Removing existing project directory: {project_dir}")
            shutil.rmtree(project_dir)
        
        logger.info(f"Restoring project from {backup_dir} to {project_dir}")
        shutil.copytree(backup_dir, project_dir)
        
        # Verify restore was successful
        if not project_dir.exists():
            raise RuntimeError(f"Restore failed: {project_dir} does not exist after restore")
        
        java_files_restored = list(project_dir.rglob('*.java'))
        if len(java_files_restored) == 0:
            raise RuntimeError(f"Restore appears incomplete: no Java files found in {project_dir}")
        
        logger.info(f"Restore verified: {len(java_files_restored)} Java files")
    
    def get_baseline_warnings(self, project_dir: Path, project_name: str, 
                             fallback_count: Optional[int] = None) -> int:
        """
        Get baseline warning count for a project.
        
        Args:
            project_dir: Project directory
            project_name: Name of the project
            fallback_count: Fallback warning count if checker fails
            
        Returns:
            Baseline warning count
        """
        logger.info(f"Getting baseline warnings for {project_name}")
        
        # Find Java files
        java_files = self.warning_tester.find_java_files(project_dir, max_files=50)
        
        if not java_files:
            logger.warning(f"No Java files found in {project_dir}")
            if fallback_count is not None:
                logger.info(f"Using fallback warning count: {fallback_count}")
                return fallback_count
            return 0
        
        # Run checker
        success, output = self.warning_tester.run_lower_bound_checker(project_dir, java_files)
        
        # FIXED: Verify checker actually ran using crash detector before parsing warnings
        try:
            from checker_crash_detector import detect_checker_crash
            returncode = 0 if success else 1
            crash_result = detect_checker_crash(output, returncode=returncode)
            
            if crash_result.crashed or crash_result.no_files_processed:
                logger.error(f"Checker failed for {project_name}: {crash_result.crash_reason}")
                if fallback_count is not None:
                    logger.info(f"Using fallback warning count: {fallback_count}")
                    return fallback_count
                # Raise error instead of returning 0 silently
                raise RuntimeError(f"Checker failed: {crash_result.crash_reason}")
            
            # Check if checker analysis actually succeeded
            if not crash_result.checker_analysis_succeeded():
                logger.warning(f"Checker analysis may not have run for {project_name}: "
                             f"compilation_errors={crash_result.has_compilation_errors}, "
                             f"success_indicators={crash_result.has_success_indicators}")
                if fallback_count is not None and crash_result.has_compilation_errors:
                    logger.info(f"Using fallback due to compilation errors: {fallback_count}")
                    return fallback_count
        except ImportError:
            logger.debug("Crash detector not available")
        except RuntimeError:
            raise  # Re-raise our own errors
        except Exception as e:
            logger.warning(f"Error checking crash status: {e}")
        
        # #region agent log
        import json
        with open('/home/ubuntu/GenDATA/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'baseline-warnings',
                'hypothesisId': 'D',
                'location': 'evaluate_annotation_placement.py:253',
                'message': 'Baseline checker execution',
                'data': {
                    'project_name': project_name,
                    'success': success,
                    'output_length': len(output) if output else 0,
                    'output_preview': output[:500] if output else '',
                    'java_files_count': len(java_files)
                },
                'timestamp': int(__import__('time').time() * 1000)
            }) + '\n')
        # #endregion
        
        if not success:
            logger.warning(f"Failed to run checker for baseline: {output[:200]}")
            # #region agent log
            with open('/home/ubuntu/GenDATA/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'baseline-warnings',
                    'hypothesisId': 'B',
                    'location': 'evaluate_annotation_placement.py:256',
                    'message': 'Checker execution failed',
                    'data': {
                        'project_name': project_name,
                        'error_output': output[:500] if output else '',
                        'using_fallback': fallback_count is not None,
                        'fallback_count': fallback_count
                    },
                    'timestamp': int(__import__('time').time() * 1000)
                }) + '\n')
            # #endregion
            if fallback_count is not None:
                logger.info(f"Using fallback warning count: {fallback_count}")
                return fallback_count
            return 0
        
        # Parse warnings
        stats = self.warning_tester.parse_warnings(output)
        
        # #region agent log
        with open('/home/ubuntu/GenDATA/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'baseline-warnings',
                'hypothesisId': 'C',
                'location': 'evaluate_annotation_placement.py:263',
                'message': 'Baseline warning parsing',
                'data': {
                    'project_name': project_name,
                    'total_warnings': stats.total_warnings,
                    'warning_lines_count': len(stats.warning_lines),
                    'sample_warnings': stats.warning_lines[:3] if stats.warning_lines else []
                },
                'timestamp': int(__import__('time').time() * 1000)
            }) + '\n')
        # #endregion
        
        # Check if we got valid warnings (not just compilation errors)
        # Filter out compilation errors - look for actual checker warnings
        actual_warnings = 0
        if stats.warning_lines:
            # Checker Framework warnings typically contain "[checker.name]" or "error:" for type errors
            for line in stats.warning_lines:
                # Skip compilation errors (package does not exist, cannot find symbol, etc.)
                if any(err in line.lower() for err in ['package', 'does not exist', 'cannot find symbol', 
                                                       'error:', 'compiler.err']):
                    continue
                # Count actual Lower Bound Checker warnings
                # These include: [argument], [array.access...], [samelen], [lowerbound], [index...]
                # Format: file:line:col: [checker-key] message
                if '[' in line and ']' in line:
                    # Extract checker key: look for [something] pattern
                    import re
                    match = re.search(r'\[([^\]]+)\]', line)
                    if match:
                        checker_key = match.group(1).lower()
                        # Count all Index/Lower Bound checker warnings
                        if any(key in checker_key for key in ['index', 'lower', 'array', 'argument', 
                                                               'samelen', 'substringgfrom', 'offset', 
                                                               'length', 'searchfrom', 'dep-ann']):
                            actual_warnings += 1
                        elif not any(err in checker_key for err in ['error', 'compiler']):
                            # Count other checker warnings that aren't compilation errors
                            actual_warnings += 1
        
        # If we have actual warnings, use them; otherwise use total (which might include errors)
        warning_count = actual_warnings if actual_warnings > 0 else stats.total_warnings
        
        # If still 0 and we have a fallback, use it
        if warning_count == 0 and fallback_count is not None:
            logger.warning(f"No warnings parsed from checker output, using fallback: {fallback_count}")
            return fallback_count
        
        logger.info(f"Baseline warnings for {project_name}: {warning_count} (parsed from checker output)")
        return warning_count
    
    def _clean_checker_output(self, project_dir: Path):
        """Clean up checker output files to ensure clean backup"""
        try:
            checker_output_dir = project_dir / 'checker_output'
            if checker_output_dir.exists():
                shutil.rmtree(checker_output_dir)
                logger.debug(f"Cleaned up checker output directory: {checker_output_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning checker output: {e}")
    
    def find_cfg_directory(self, project_dir: Path, project_name: str) -> Optional[Path]:
        """
        Find or determine CFG directory for a project.
        
        Args:
            project_dir: Project directory
            project_name: Name of the project
            
        Returns:
            Path to CFG directory, or None if not found
        """
        # If explicitly provided, use it
        if self.cfg_dir:
            cfg_path = Path(self.cfg_dir)
            if cfg_path.exists():
                logger.info(f"Using provided CFG directory: {cfg_path}")
                if self._verify_cfg_directory(cfg_path, project_name):
                    return cfg_path
                else:
                    logger.warning(f"Provided CFG directory failed verification: {cfg_path}")
        
        # Try common CFG directory patterns
        base_cfg_dir = Path('/home/ubuntu/GenDATA')
        common_patterns = [
            # Project-specific CFG directories (by project name)
            base_cfg_dir / f'cfg_output_adaptive_specimin_lower_bound' / project_name,
            base_cfg_dir / f'cfg_output_adaptive_specimin_lower_bound' / project_name.replace('-', '_'),
            base_cfg_dir / f'cfg_output_adaptive_specimin_lower_bound' / project_name.replace('_', '-'),
            # General CFG directories
            base_cfg_dir / f'cfg_output_adaptive_specimin_lower_bound',
            base_cfg_dir / 'cfg_output',
            # Project-local directories
            project_dir / 'cfg',
            project_dir / 'cfg_output',
            project_dir / '.cfg',
        ]
        
        for pattern in common_patterns:
            if pattern.exists():
                if self._verify_cfg_directory(pattern, project_name):
                    logger.info(f"Found CFG directory: {pattern}")
                    return pattern
                else:
                    logger.debug(f"CFG directory exists but failed verification: {pattern}")
        
        logger.warning(f"CFG directory not found for {project_name}. Predictions may fail.")
        logger.info(f"Searched patterns: {[str(p) for p in common_patterns[:5]]}")
        return None
    
    def _verify_cfg_directory(self, cfg_dir: Path, project_name: str) -> bool:
        """
        Verify that CFG directory contains valid CFG files.
        
        Args:
            cfg_dir: CFG directory to verify
            project_name: Name of the project (for context)
            
        Returns:
            True if directory appears to contain valid CFG files
        """
        try:
            if not cfg_dir.exists() or not cfg_dir.is_dir():
                return False
            
            # Look for JSON files (CFG files are JSON)
            json_files = list(cfg_dir.rglob('*.json'))
            
            if len(json_files) == 0:
                logger.debug(f"No JSON files found in {cfg_dir}")
                return False
            
            # Check if at least one JSON file looks like a CFG file
            # CFG files typically have 'nodes' key
            valid_cfg_count = 0
            for json_file in json_files[:5]:  # Check first 5 files
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and ('nodes' in data or 'edges' in data):
                            valid_cfg_count += 1
                except:
                    pass
            
            if valid_cfg_count > 0:
                logger.debug(f"Found {valid_cfg_count} valid CFG files in {cfg_dir}")
                return True
            else:
                logger.debug(f"No valid CFG files found in {cfg_dir} (checked {len(json_files)} JSON files)")
                return False
                
        except Exception as e:
            logger.debug(f"Error verifying CFG directory {cfg_dir}: {e}")
            return False
    
    def generate_predictions(self, project_dir: Path, cfg_dir: Path, 
                             base_model: str, output_file: Path) -> bool:
        """
        Generate predictions for a project using a specific base model.
        
        Args:
            project_dir: Project directory
            cfg_dir: CFG directory
            base_model: Base model to use
            output_file: Path to save predictions JSON
            
        Returns:
            True if predictions were generated successfully
        """
        try:
            logger.info(f"Generating predictions using {base_model} model")
            
            # Create filtered predictor for the configured checker
            predictor = FilteredMultiCheckerPredictor(
                checker_name=self.checker_name,
                base_model_filter=base_model
            )
            
            # Load models
            if not predictor.load_checker_models():
                logger.warning(f"No models loaded for {base_model}")
                return False
            
            # Find Java files
            java_files = self.warning_tester.find_java_files(project_dir, max_files=50)
            
            if not java_files:
                logger.warning("No Java files found")
                return False
            
            # Generate predictions for all files
            all_predictions = []
            for java_file in java_files:
                java_file_path = Path(java_file)
                # Ensure we use absolute path
                java_file_abs = str(java_file_path.resolve())
                
                # Verify file exists before predicting
                if not Path(java_file_abs).exists():
                    logger.warning(f"Java file does not exist: {java_file_abs}")
                    continue
                
                predictions = predictor.predict_for_file(java_file_abs, str(cfg_dir))
                
                # Ensure file_path is set correctly in predictions and verify paths
                for pred in predictions:
                    if 'file_path' not in pred or not pred['file_path']:
                        pred['file_path'] = java_file_abs
                    else:
                        # Verify the path exists
                        pred_path = Path(pred['file_path'])
                        if not pred_path.exists():
                            # Try to resolve relative to project_dir
                            resolved = (project_dir / pred['file_path']).resolve()
                            if resolved.exists():
                                pred['file_path'] = str(resolved)
                            else:
                                # Use absolute path from java_file
                                pred['file_path'] = java_file_abs
                
                all_predictions.extend(predictions)
            
            # Save predictions
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            
            logger.info(f"Generated {len(all_predictions)} predictions using {base_model}")
            return len(all_predictions) > 0
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return False
    
    def place_annotations(self, project_dir: Path, predictions_file: Path, 
                         output_dir: Path) -> Dict[str, int]:
        """
        Place annotations in the project.
        
        Args:
            project_dir: Project directory (will be modified)
            predictions_file: Path to predictions JSON
            output_dir: Output directory for placement logs
            
        Returns:
            Placement statistics dictionary
        """
        try:
            logger.info(f"Placing annotations from {predictions_file}")
            
            # Verify project directory exists
            if not project_dir.exists():
                raise ValueError(f"Project directory does not exist: {project_dir}")
            
            # Verify predictions file exists
            if not predictions_file.exists():
                raise ValueError(f"Predictions file does not exist: {predictions_file}")
            
            # Safety check: never modify backups
            if not self.verify_not_backup_dir(project_dir):
                raise ValueError(f"Safety check failed: {project_dir} is a backup directory")
            
            placer = ComprehensiveAnnotationPlacer(
                project_root=str(project_dir.resolve()),  # Use resolved absolute path
                output_dir=str(output_dir),
                checker_name=self.checker_name,
                backup=False,  # We handle backups ourselves
                perfect_placement=True
            )
            
            # Load predictions
            predictions = placer.load_predictions(str(predictions_file))
            
            if not predictions:
                logger.warning("No predictions to place")
                return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
            
            # Verify predictions have valid file paths
            valid_predictions = []
            for pred in predictions:
                pred_path = Path(pred.file_path)
                if pred_path.exists():
                    valid_predictions.append(pred)
                else:
                    # Try resolving relative to project_dir
                    resolved = (project_dir / pred.file_path).resolve()
                    if resolved.exists():
                        pred.file_path = str(resolved)
                        valid_predictions.append(pred)
                    else:
                        logger.warning(f"Prediction file path does not exist: {pred.file_path}")
            
            if not valid_predictions:
                logger.warning("No valid predictions after path verification")
                return {'total': len(predictions), 'successful': 0, 'failed': 0, 'skipped': len(predictions)}
            
            logger.info(f"Placing {len(valid_predictions)} valid predictions (out of {len(predictions)} total)")
            
            # Place annotations
            stats = placer.process_predictions(valid_predictions)
            
            logger.info(f"Placed annotations: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error placing annotations: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0, 'error': str(e)}
    
    def evaluate_model(self, project_dir: Path, backup_dir: Path, 
                      project_name: str, base_model: str, 
                      baseline_warnings: int, cfg_dir: Path) -> ModelEvaluationResult:
        """
        Evaluate a single base model for a project.
        
        Args:
            project_dir: Project directory (will be modified)
            backup_dir: Backup directory to restore from
            project_name: Name of the project
            base_model: Base model to evaluate
            baseline_warnings: Baseline warning count
            cfg_dir: CFG directory
            
        Returns:
            ModelEvaluationResult
        """
        # Verify backup exists before starting
        if not backup_dir.exists():
            return ModelEvaluationResult(
                base_model=base_model,
                annotations_placed=0,
                warnings_after=baseline_warnings,
                warning_reduction=0,
                reduction_percentage=0.0,
                placement_success=False,
                compilation_success=False,
                error_message=f"Backup directory does not exist: {backup_dir}"
            )
        
        # Restore project from backup
        try:
            self.restore_project(project_dir, backup_dir)
        except Exception as e:
            logger.error(f"Failed to restore project for {base_model}: {e}")
            return ModelEvaluationResult(
                base_model=base_model,
                annotations_placed=0,
                warnings_after=baseline_warnings,
                warning_reduction=0,
                reduction_percentage=0.0,
                placement_success=False,
                compilation_success=False,
                error_message=f"Restore failed: {e}"
            )
        
        try:
            # Generate predictions
            predictions_file = self.work_dir / 'predictions' / project_name / f'{base_model}_predictions.json'
            if not self.generate_predictions(project_dir, cfg_dir, base_model, predictions_file):
                return ModelEvaluationResult(
                    base_model=base_model,
                    annotations_placed=0,
                    warnings_after=baseline_warnings,
                    warning_reduction=0,
                    reduction_percentage=0.0,
                    placement_success=False,
                    compilation_success=False,
                    error_message="Failed to generate predictions"
                )
            
            # Place annotations
            placement_output = self.work_dir / 'placement_output' / project_name / base_model
            placement_stats = self.place_annotations(project_dir, predictions_file, placement_output)
            
            annotations_placed = placement_stats.get('successful', 0)
            
            # Get warning count after placement
            # FIXED: Track actual compilation success instead of hardcoding True
            compilation_success = True
            checker_crashed = False
            
            try:
                # #region agent log
                import json
                with open('/home/ubuntu/GenDATA/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'post-placement',
                        'hypothesisId': 'E',
                        'location': 'evaluate_annotation_placement.py:613',
                        'message': 'Getting warnings after annotation placement',
                        'data': {
                            'project_name': project_name,
                            'base_model': base_model,
                            'annotations_placed': annotations_placed,
                            'baseline_warnings': baseline_warnings
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
                
                warnings_after = self.get_baseline_warnings(project_dir, project_name)
                
                # FIXED: Verify the result is valid using crash detector
                # Run a quick check to see if the checker actually ran successfully
                java_files = self.warning_tester.find_java_files(project_dir, max_files=10)
                if java_files:
                    success, output = self.warning_tester.run_lower_bound_checker(project_dir, java_files)
                    try:
                        from checker_crash_detector import detect_checker_crash
                        crash_result = detect_checker_crash(output)
                        
                        if crash_result.crashed or crash_result.no_files_processed:
                            # Checker failed - don't claim warning reduction
                            logger.error(f"Checker verification failed: {crash_result.crash_reason}")
                            warnings_after = baseline_warnings
                            compilation_success = False
                            checker_crashed = True
                        elif crash_result.has_compilation_errors and warnings_after == 0:
                            # Suspicious: compilation errors but 0 warnings
                            # This likely means checker never ran analysis
                            if crash_result.compilation_error_count > 0:
                                logger.warning(f"Suspicious result: {crash_result.compilation_error_count} "
                                             f"compilation errors but 0 warnings")
                                warnings_after = baseline_warnings
                                compilation_success = False
                    except ImportError:
                        pass  # Crash detector not available
                
                # #region agent log
                with open('/home/ubuntu/GenDATA/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'post-placement',
                        'hypothesisId': 'E',
                        'location': 'evaluate_annotation_placement.py:615',
                        'message': 'Warnings after placement result',
                        'data': {
                            'project_name': project_name,
                            'base_model': base_model,
                            'warnings_after': warnings_after,
                            'baseline_warnings': baseline_warnings,
                            'reduction': baseline_warnings - warnings_after,
                            'is_zero': warnings_after == 0,
                            'compilation_success': compilation_success
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
            except Exception as e:
                logger.error(f"Failed to get warnings after placement for {base_model}: {e}")
                # FIXED: Check if this is a checker crash vs other error
                error_str = str(e).lower()
                if 'timeout' in error_str or 'crash' in error_str or 'exception in thread' in error_str:
                    checker_crashed = True
                    compilation_success = False
                
                # #region agent log
                import json
                with open('/home/ubuntu/GenDATA/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'post-placement',
                        'hypothesisId': 'B',
                        'location': 'evaluate_annotation_placement.py:617',
                        'message': 'Exception getting warnings after placement',
                        'data': {
                            'project_name': project_name,
                            'base_model': base_model,
                            'error': str(e),
                            'checker_crashed': checker_crashed,
                            'using_baseline_as_fallback': True
                        },
                        'timestamp': int(__import__('time').time() * 1000)
                    }) + '\n')
                # #endregion
                warnings_after = baseline_warnings  # Assume no improvement on error
            
            # Calculate reduction
            warning_reduction = baseline_warnings - warnings_after
            reduction_percentage = (warning_reduction / baseline_warnings * 100) if baseline_warnings > 0 else 0.0
            
            return ModelEvaluationResult(
                base_model=base_model,
                annotations_placed=annotations_placed,
                warnings_after=warnings_after,
                warning_reduction=warning_reduction,
                reduction_percentage=reduction_percentage,
                placement_success=annotations_placed > 0,
                compilation_success=compilation_success  # FIXED: Use actual value
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {base_model} for {project_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Try to restore project to clean state on error
            try:
                self.restore_project(project_dir, backup_dir)
            except Exception as restore_error:
                logger.error(f"Failed to restore project after error: {restore_error}")
            
            return ModelEvaluationResult(
                base_model=base_model,
                annotations_placed=0,
                warnings_after=baseline_warnings,
                warning_reduction=0,
                reduction_percentage=0.0,
                placement_success=False,
                compilation_success=False,
                error_message=str(e)
            )
    
    def evaluate_project(self, project: Dict[str, Any]) -> ProjectEvaluationResult:
        """
        Evaluate all base models for a single project.
        
        Args:
            project: Project dictionary from candidates file
            
        Returns:
            ProjectEvaluationResult
        """
        project_name = project['project_name']
        project_url = project['project_url']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating project: {project_name}")
        logger.info(f"{'='*80}")
        
        # Clone project
        project_dir = None
        backup_dir = None
        try:
            project_dir = self.warning_tester.clone_repository(project_url, project_name)
            if not project_dir:
                return ProjectEvaluationResult(
                    project_name=project_name,
                    project_url=project_url,
                    baseline_warnings=0,
                    model_results=[],
                    error_message="Failed to clone repository"
                )
            
            # Verify clone was successful
            if not project_dir.exists():
                return ProjectEvaluationResult(
                    project_name=project_name,
                    project_url=project_url,
                    baseline_warnings=0,
                    model_results=[],
                    error_message=f"Cloned directory does not exist: {project_dir}"
                )
        
        except Exception as e:
            logger.error(f"Error cloning repository {project_name}: {e}")
            return ProjectEvaluationResult(
                project_name=project_name,
                project_url=project_url,
                baseline_warnings=0,
                model_results=[],
                error_message=f"Clone error: {e}"
            )
        
        try:
            # FIXED: Run pre-flight check to ensure project compiles before evaluation
            compilable, message = self.verify_project_compilable(project_dir, project_name)
            if not compilable:
                logger.error(f"Pre-flight check failed for {project_name}: {message}")
                return ProjectEvaluationResult(
                    project_name=project_name,
                    project_url=project_url,
                    baseline_warnings=0,
                    model_results=[],
                    error_message=f"Pre-flight check failed: {message}"
                )
            
            # Get baseline warnings BEFORE backup (to ensure clean state)
            # Note: This may create checker output files, but we'll clean them up
            # Use warning count from candidates file as fallback
            fallback_warning_count = project.get('warning_count', 0)
            try:
                baseline_warnings = self.get_baseline_warnings(
                    project_dir, project_name, fallback_count=fallback_warning_count
                )
            except Exception as e:
                logger.error(f"Failed to get baseline warnings for {project_name}: {e}")
                # Use fallback count
                baseline_warnings = fallback_warning_count
                logger.info(f"Using fallback warning count from candidates file: {baseline_warnings}")
            
            # Clean up any checker output files before backup
            self._clean_checker_output(project_dir)
            
            # Backup project (after baseline, but before model tests)
            try:
                backup_dir = self.backup_project(project_dir, project_name)
            except Exception as e:
                logger.error(f"Failed to backup project {project_name}: {e}")
                return ProjectEvaluationResult(
                    project_name=project_name,
                    project_url=project_url,
                    baseline_warnings=baseline_warnings,
                    model_results=[],
                    error_message=f"Backup failed: {e}"
                )
            
            # Find CFG directory
            cfg_dir = self.find_cfg_directory(project_dir, project_name)
            if not cfg_dir:
                return ProjectEvaluationResult(
                    project_name=project_name,
                    project_url=project_url,
                    baseline_warnings=baseline_warnings,
                    model_results=[],
                    error_message="CFG directory not found"
                )
            
            # Evaluate each base model
            model_results = []
            for base_model in self.base_models:
                try:
                    logger.info(f"\nEvaluating {base_model} model for {project_name}")
                    result = self.evaluate_model(
                        project_dir, backup_dir, project_name, base_model,
                        baseline_warnings, cfg_dir
                    )
                    model_results.append(result)
                    
                    logger.info(f"  Annotations placed: {result.annotations_placed}")
                    logger.info(f"  Warnings after: {result.warnings_after}")
                    logger.info(f"  Reduction: {result.warning_reduction} ({result.reduction_percentage:.2f}%)")
                    
                    if result.error_message:
                        logger.warning(f"  Error: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"Unexpected error evaluating {base_model} for {project_name}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    model_results.append(ModelEvaluationResult(
                        base_model=base_model,
                        annotations_placed=0,
                        warnings_after=baseline_warnings,
                        warning_reduction=0,
                        reduction_percentage=0.0,
                        placement_success=False,
                        compilation_success=False,
                        error_message=f"Unexpected error: {e}"
                    ))
            
            # FIXED: Restore project to clean state after all model evaluations
            try:
                self.restore_project(project_dir, backup_dir)
                logger.info(f"Restored {project_name} to clean state after evaluation")
            except Exception as restore_error:
                logger.error(f"Failed to restore {project_name} after evaluation: {restore_error}")
            
            return ProjectEvaluationResult(
                project_name=project_name,
                project_url=project_url,
                baseline_warnings=baseline_warnings,
                model_results=model_results
            )
            
        except Exception as e:
            logger.error(f"Error evaluating project {project_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # FIXED: Try to restore project on error as well
            try:
                if backup_dir and backup_dir.exists():
                    self.restore_project(project_dir, backup_dir)
                    logger.info(f"Restored {project_name} after error")
            except Exception as restore_error:
                logger.warning(f"Could not restore {project_name} after error: {restore_error}")
            
            # Try to preserve backup even on error
            if backup_dir and backup_dir.exists():
                logger.info(f"Backup preserved at: {backup_dir}")
            
            return ProjectEvaluationResult(
                project_name=project_name,
                project_url=project_url,
                baseline_warnings=baseline_warnings if 'baseline_warnings' in locals() else 0,
                model_results=[],
                error_message=str(e)
            )
    
    def run_evaluation(self, candidates_file: str, output_file: str) -> EvaluationReport:
        """
        Run complete evaluation.
        
        Args:
            candidates_file: Path to lower_bound_project_candidates.json
            output_file: Path to save evaluation report
            
        Returns:
            EvaluationReport
        """
        # Load qualifying projects
        projects = self.load_qualifying_projects(candidates_file)
        
        if not projects:
            logger.error("No qualifying projects found")
            return EvaluationReport(
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'projects_evaluated': 0,
                    'base_models_tested': len(self.base_models)
                },
                results=[]
            )
        
        # Evaluate each project
        results = []
        for project in projects:
            result = self.evaluate_project(project)
            results.append(result)
        
        # Create report
        report = EvaluationReport(
            metadata={
                'timestamp': datetime.now().isoformat(),
                'projects_evaluated': len(results),
                'base_models_tested': len(self.base_models),
                'base_models': self.base_models
            },
            results=results
        )
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluation complete. Report saved to: {output_file}")
        logger.info(f"{'='*80}")
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def print_summary(self, report: EvaluationReport):
        """Print evaluation summary"""
        logger.info("\nEvaluation Summary:")
        logger.info(f"  Projects evaluated: {len(report.results)}")
        logger.info(f"  Base models tested: {len(report.metadata['base_models'])}")
        
        for result in report.results:
            logger.info(f"\n  {result.project_name}:")
            logger.info(f"    Baseline warnings: {result.baseline_warnings}")
            
            if result.model_results:
                best_model = max(result.model_results, 
                               key=lambda r: r.reduction_percentage if r.placement_success else -1)
                logger.info(f"    Best model: {best_model.base_model} "
                          f"({best_model.reduction_percentage:.2f}% reduction)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate annotation placement effectiveness by base model'
    )
    parser.add_argument('--candidates-file', 
                       default='project_discovery_manual/lower_bound_project_candidates.json',
                       help='Path to project candidates JSON file')
    parser.add_argument('--output-file',
                       default='annotation_evaluation/evaluation_report.json',
                       help='Path to save evaluation report')
    parser.add_argument('--work-dir',
                       default='./annotation_evaluation',
                       help='Working directory for evaluation')
    parser.add_argument('--checker-cp',
                       help='Checker Framework classpath (defaults to CHECKERFRAMEWORK_CP env var)')
    parser.add_argument('--cfg-dir',
                       help='CFG directory (will try to auto-detect if not provided)')
    parser.add_argument('--timeout',
                       type=int,
                       default=600,
                       help='Timeout for compilation/checker runs (seconds)')
    parser.add_argument('--checker',
                       choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       default='lower_bound',
                       help='Checker to evaluate (default: lower_bound)')
    
    args = parser.parse_args()
    
    # Create evaluator with specified checker
    evaluator = AnnotationPlacementEvaluator(
        work_dir=args.work_dir,
        checker_cp=args.checker_cp,
        cfg_dir=args.cfg_dir,
        timeout=args.timeout,
        checker_name=args.checker
    )
    
    # Run evaluation
    report = evaluator.run_evaluation(args.candidates_file, args.output_file)
    
    return 0 if report.results else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

