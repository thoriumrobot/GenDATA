#!/usr/bin/env python3
"""
Unified Multi-Checker Evaluation Script

Evaluates annotation placement effectiveness for all three checkers:
- Lower Bound Checker
- SQL Quotes Checker  
- Signature String Checker

Uses trained models via MultiCheckerPredictor and properly manages backups
to ensure original code is never modified.

Usage:
    python evaluate_all_checkers.py --checker lower_bound
    python evaluate_all_checkers.py --checker sql_quotes
    python evaluate_all_checkers.py --checker signature_string
    python evaluate_all_checkers.py --all
"""

import os
import sys
import json
import shutil
import subprocess
import logging
import argparse
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
        logging.FileHandler('evaluate_all_checkers.log')
    ]
)
logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
ANNOTATION_EVAL_DIR = GEN_DATA_ROOT / 'annotation_evaluation'
CASE_STUDIES_BACKUP = GEN_DATA_ROOT / 'case_studies_backup'
CHECKER_FRAMEWORK_HOME = Path('/home/ubuntu/checker-framework')

# Checker processor mapping
CHECKER_PROCESSORS = {
    'lower_bound': 'org.checkerframework.checker.index.IndexChecker',
    'sql_quotes': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
    'signature_string': 'org.checkerframework.checker.signature.SignatureChecker',
}

# Projects per checker for evaluation (3 real GitHub projects each, no training sets)
# Selected for having >= 5 baseline warnings for each checker
EVALUATION_PROJECTS = {
    'lower_bound': [
        {'name': 'pom-tuner', 'url': 'https://github.com/l2x6/pom-tuner'},  # 35 warnings
        {'name': 'commons-lang', 'url': 'https://github.com/apache/commons-lang'},  # 100 warnings
        {'name': 'commons-io', 'url': 'https://github.com/apache/commons-io'},  # 100 warnings
    ],
    'sql_quotes': [
        {'name': 'commons-dbcp', 'url': 'https://github.com/apache/commons-dbcp'},  # 49 warnings
        {'name': 'mybatis-3', 'url': 'https://github.com/mybatis/mybatis-3'},  # 6 warnings
        {'name': 'commons-dbutils', 'url': 'https://github.com/apache/commons-dbutils'},  # 3 warnings (annotated)
    ],
    'signature_string': [
        {'name': 'javassist', 'url': 'https://github.com/jboss-javassist/javassist'},  # 14 warnings
        {'name': 'reflections', 'url': 'https://github.com/ronmamo/reflections'},  # 9 warnings
        {'name': 'kryo', 'url': 'https://github.com/EsotericSoftware/kryo'},  # 7 warnings
    ],
}


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
    checker_name: str
    baseline_warnings: int
    model_results: List[Dict[str, Any]]
    error_message: Optional[str] = None


try:
    from backup_safety import verify_not_backup_dir, restore_from_backup as backup_restore
except ImportError:
    # Fallback if backup_safety module not available
    def verify_not_backup_dir(path: Path) -> bool:
        """Fallback implementation"""
        backup_dirs = [
            ANNOTATION_EVAL_DIR / 'backups',
            CASE_STUDIES_BACKUP,
            GEN_DATA_ROOT / 'annotated_projects_backup',
        ]
        for backup_dir in backup_dirs:
            if backup_dir.exists():
                try:
                    path.relative_to(backup_dir)
                    logger.error(f"SAFETY CHECK FAILED: {path}")
                    return False
                except ValueError:
                    continue
        return True
    
    def backup_restore(project_name: str, target_dir: Path, force: bool = True) -> bool:
        """Fallback implementation"""
        return restore_from_backup_impl(project_name, target_dir)


def restore_from_backup(project_name: str, target_dir: Path, checker_name: str = None) -> bool:
    """
    Restore a project from backup to target directory.
    
    Uses the shared backup_safety module for safety checks.
    For SQL Quotes and Signature String, prefers evaluation_ready directory
    which has entry-point annotations for proper baseline warnings.
    
    Args:
        project_name: Name of the project
        target_dir: Target directory to restore to
        checker_name: Checker name (affects backup source priority)
        
    Returns:
        True if restore successful
    """
    try:
        return backup_restore(project_name, target_dir, force=True, checker_name=checker_name)
    except Exception as e:
        logger.error(f"Error restoring {project_name}: {e}")
        return False


def restore_from_backup_impl(project_name: str, target_dir: Path) -> bool:
    """Fallback restore implementation if backup_safety not available"""
    # Safety check
    if not verify_not_backup_dir(target_dir):
        return False
    
    # Check backup locations
    backup_sources = [
        ANNOTATION_EVAL_DIR / 'backups' / project_name,
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
        '/generated/', '/.git/', '/benchmark/'
    ]
    
    # Prioritize src/main first, then other directories
    main_sources = list(directory.glob('src/main/**/*.java'))
    other_sources = [f for f in directory.rglob('*.java') if f not in main_sources]
    all_sources = main_sources + other_sources
    
    for java_file in all_sources:
        path_str = str(java_file)
        if not any(pattern in path_str for pattern in exclude_patterns):
            java_files.append(java_file)
            if len(java_files) >= max_files:
                break
    
    return java_files


def run_checker(project_dir: Path, checker_name: str, java_files: List[Path]) -> Tuple[int, str]:
    """
    Run a Checker Framework checker on Java files.
    
    Args:
        project_dir: Project directory
        checker_name: Name of the checker
        java_files: List of Java files to check
        
    Returns:
        Tuple of (warning_count, raw_output)
    """
    if not java_files:
        return 0, "No Java files to check"
    
    processor = CHECKER_PROCESSORS.get(checker_name)
    if not processor:
        return -1, f"Unknown checker: {checker_name}"
    
    checker_javac = CHECKER_FRAMEWORK_HOME / 'checker' / 'bin' / 'javac'
    checker_cp = f"{CHECKER_FRAMEWORK_HOME}/checker/dist/checker-qual.jar:{CHECKER_FRAMEWORK_HOME}/checker/dist/checker.jar"
    
    # Try to resolve Maven dependencies if pom.xml exists
    full_classpath = checker_cp
    pom_file = project_dir / 'pom.xml'
    if pom_file.exists():
        try:
            from maven_classpath_resolver import MavenClasspathResolver
            resolver = MavenClasspathResolver(timeout=300)
            result = resolver.prepare_project(project_dir, checker_cp)
            if result.success:
                full_classpath = result.classpath
                logger.debug(f"Using Maven classpath for {project_dir.name}")
        except ImportError:
            logger.debug("Maven classpath resolver not available")
        except Exception as e:
            logger.debug(f"Maven classpath resolution failed: {e}")
    
    cmd = [
        str(checker_javac),
        '-processor', processor,
        '-cp', full_classpath,
        '-Xlint:-processing',
        '-Awarns',
    ] + [str(f) for f in java_files[:200]]  # Analyze more files for better coverage
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_dir)
        )
        output = result.stdout + result.stderr
        
        # Count checker-specific warnings
        warning_count = 0
        for line in output.split('\n'):
            if 'error:' in line.lower() or 'warning:' in line.lower():
                if '[' in line and ']' in line:
                    # Skip generic warnings
                    if not any(w in line for w in ['[deprecation]', '[removal]', '[unchecked]', '[rawtypes]', '[path]', '[options]']):
                        warning_count += 1
        
        return warning_count, output
        
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)


class MultiCheckerEvaluator:
    """Evaluates annotation placement for all checkers"""
    
    def __init__(self, 
                 work_dir: Path = ANNOTATION_EVAL_DIR,
                 timeout: int = 600):
        """
        Initialize evaluator.
        
        Args:
            work_dir: Working directory for evaluation
            timeout: Timeout for operations
        """
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        # Base models to test
        self.base_models = ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n']
        
        logger.info(f"Initialized MultiCheckerEvaluator")
        logger.info(f"  Work directory: {self.work_dir}")
    
    def get_temp_project_dir(self, checker_name: str, project_name: str) -> Path:
        """Get temporary project directory path (never a backup)"""
        return self.work_dir / 'temp_repos' / checker_name / project_name
    
    def get_predictions_dir(self, checker_name: str, project_name: str) -> Path:
        """Get predictions directory path"""
        return self.work_dir / 'predictions' / checker_name / project_name
    
    def generate_predictions(self, project_dir: Path, checker_name: str, 
                            base_model: str, output_file: Path) -> bool:
        """
        Generate predictions for a project using a specific base model.
        
        Args:
            project_dir: Project directory
            checker_name: Checker name
            base_model: Base model to use
            output_file: Path to save predictions JSON
            
        Returns:
            True if predictions were generated successfully
        """
        try:
            from filtered_multi_checker_predictor import FilteredMultiCheckerPredictor
            
            logger.info(f"Generating predictions using {base_model} for {checker_name}")
            
            # Create predictor for specific checker and model
            predictor = FilteredMultiCheckerPredictor(
                checker_name=checker_name,
                base_model_filter=base_model
            )
            
            # Load models
            if not predictor.load_checker_models():
                logger.warning(f"No models loaded for {base_model} / {checker_name}")
                return False
            
            # Find Java files
            java_files = find_java_files(project_dir, max_files=50)
            
            if not java_files:
                logger.warning("No Java files found")
                return False
            
            # Get CFG directory - try checker-specific first
            cfg_dirs = [
                GEN_DATA_ROOT / f'cfg_output_adaptive_specimin_{checker_name}',
                GEN_DATA_ROOT / 'cfg_output_adaptive_specimin',
                GEN_DATA_ROOT / 'cfg_output_specimin',
            ]
            
            cfg_dir = None
            for cd in cfg_dirs:
                if cd.exists():
                    cfg_dir = cd
                    break
            
            if not cfg_dir:
                logger.warning(f"No CFG directory found for {checker_name}")
                return False
            
            # Generate predictions
            all_predictions = []
            for java_file in java_files:
                predictions = predictor.predict_for_file(str(java_file), str(cfg_dir))
                for pred in predictions:
                    if 'file_path' not in pred or not pred['file_path']:
                        pred['file_path'] = str(java_file)
                all_predictions.extend(predictions)
            
            # Save predictions
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            
            logger.info(f"Generated {len(all_predictions)} predictions")
            return len(all_predictions) > 0
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def place_annotations(self, project_dir: Path, predictions_file: Path,
                         checker_name: str) -> Dict[str, int]:
        """
        Place annotations in the project.
        
        Args:
            project_dir: Project directory (will be modified)
            predictions_file: Path to predictions JSON
            checker_name: Checker name
            
        Returns:
            Placement statistics dictionary
        """
        # Safety check
        if not verify_not_backup_dir(project_dir):
            return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0, 'error': 'Safety check failed'}
        
        try:
            from place_annotations import ComprehensiveAnnotationPlacer
            
            placer = ComprehensiveAnnotationPlacer(
                project_root=str(project_dir),
                output_dir=str(self.work_dir / 'placement_output'),
                checker_name=checker_name,
                backup=False,  # We handle backups ourselves
                perfect_placement=True
            )
            
            predictions = placer.load_predictions(str(predictions_file))
            
            if not predictions:
                return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
            
            stats = placer.process_predictions(predictions)
            logger.info(f"Placed annotations: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error placing annotations: {e}")
            return {'total': 0, 'successful': 0, 'failed': 0, 'skipped': 0, 'error': str(e)}
    
    def evaluate_model(self, project_name: str, checker_name: str,
                      base_model: str, baseline_warnings: int) -> ModelEvaluationResult:
        """
        Evaluate a single model for a project.
        
        Args:
            project_name: Project name
            checker_name: Checker name
            base_model: Base model to evaluate
            baseline_warnings: Baseline warning count
            
        Returns:
            ModelEvaluationResult
        """
        project_dir = self.get_temp_project_dir(checker_name, project_name)
        
        # Restore from backup (never modifies backups)
        if not restore_from_backup(project_name, project_dir, checker_name):
            return ModelEvaluationResult(
                base_model=base_model,
                annotations_placed=0,
                warnings_after=baseline_warnings,
                warning_reduction=0,
                reduction_percentage=0.0,
                placement_success=False,
                compilation_success=False,
                error_message="Failed to restore from backup"
            )
        
        try:
            # Generate predictions
            predictions_dir = self.get_predictions_dir(checker_name, project_name)
            predictions_file = predictions_dir / f'{base_model}_predictions.json'
            
            if not self.generate_predictions(project_dir, checker_name, base_model, predictions_file):
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
            stats = self.place_annotations(project_dir, predictions_file, checker_name)
            annotations_placed = stats.get('successful', 0)
            
            # Get warning count after placement
            java_files = find_java_files(project_dir)
            warnings_after, _ = run_checker(project_dir, checker_name, java_files)
            
            if warnings_after < 0:
                warnings_after = baseline_warnings
                compilation_success = False
            else:
                compilation_success = True
            
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
                compilation_success=compilation_success
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {base_model}: {e}")
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
    
    def evaluate_project(self, project_name: str, checker_name: str) -> ProjectEvaluationResult:
        """
        Evaluate all models for a project.
        
        Args:
            project_name: Project name
            checker_name: Checker name
            
        Returns:
            ProjectEvaluationResult
        """
        logger.info(f"Evaluating {project_name} with {checker_name} checker")
        
        # First restore project to get baseline
        project_dir = self.get_temp_project_dir(checker_name, project_name)
        
        if not restore_from_backup(project_name, project_dir, checker_name):
            return ProjectEvaluationResult(
                project_name=project_name,
                checker_name=checker_name,
                baseline_warnings=0,
                model_results=[],
                error_message="Failed to restore project from backup"
            )
        
        # Get baseline warnings
        java_files = find_java_files(project_dir)
        baseline_warnings, baseline_output = run_checker(project_dir, checker_name, java_files)
        
        if baseline_warnings < 0:
            return ProjectEvaluationResult(
                project_name=project_name,
                checker_name=checker_name,
                baseline_warnings=0,
                model_results=[],
                error_message=f"Failed to get baseline warnings: {baseline_output}"
            )
        
        logger.info(f"Baseline warnings for {project_name}: {baseline_warnings}")
        
        # Evaluate each model
        model_results = []
        for base_model in self.base_models:
            logger.info(f"  Evaluating model: {base_model}")
            result = self.evaluate_model(project_name, checker_name, base_model, baseline_warnings)
            model_results.append(asdict(result))
        
        return ProjectEvaluationResult(
            project_name=project_name,
            checker_name=checker_name,
            baseline_warnings=baseline_warnings,
            model_results=model_results
        )
    
    def evaluate_checker(self, checker_name: str) -> List[ProjectEvaluationResult]:
        """
        Evaluate all projects for a checker.
        
        Args:
            checker_name: Checker name
            
        Returns:
            List of ProjectEvaluationResult
        """
        projects = EVALUATION_PROJECTS.get(checker_name, [])
        
        if not projects:
            logger.error(f"No projects configured for {checker_name}")
            return []
        
        results = []
        for project in projects:
            project_name = project['name']
            result = self.evaluate_project(project_name, checker_name)
            results.append(result)
        
        return results
    
    def save_results(self, results: Dict[str, List[ProjectEvaluationResult]], 
                    output_file: Path) -> None:
        """Save evaluation results to JSON"""
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'base_models': self.base_models,
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
    parser = argparse.ArgumentParser(description='Evaluate annotation placement for all checkers')
    parser.add_argument('--checker', choices=['lower_bound', 'sql_quotes', 'signature_string'],
                       help='Specific checker to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all checkers')
    parser.add_argument('--output', default='annotation_evaluation/evaluation_report_all_checkers.json',
                       help='Output file for results')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    if not args.checker and not args.all:
        parser.error("Either --checker or --all must be specified")
    
    evaluator = MultiCheckerEvaluator(timeout=args.timeout)
    
    checkers_to_evaluate = []
    if args.all:
        checkers_to_evaluate = ['lower_bound', 'sql_quotes', 'signature_string']
    else:
        checkers_to_evaluate = [args.checker]
    
    all_results = {}
    for checker_name in checkers_to_evaluate:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {checker_name} checker")
        logger.info(f"{'='*60}")
        
        results = evaluator.evaluate_checker(checker_name)
        all_results[checker_name] = results
        
        # Print summary
        for result in results:
            logger.info(f"\n{result.project_name}:")
            logger.info(f"  Baseline warnings: {result.baseline_warnings}")
            for mr in result.model_results:
                logger.info(f"  {mr['base_model']}: {mr['reduction_percentage']:.1f}% reduction")
    
    # Save results
    output_path = GEN_DATA_ROOT / args.output
    evaluator.save_results(all_results, output_path)
    
    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
