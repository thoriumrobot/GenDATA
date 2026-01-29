#!/usr/bin/env python3
"""
Placement Pipeline Runner

This script orchestrates the full annotation placement pipeline:
1. Generate warnings from CF test suites (for training)
2. Run placement on annotated projects
3. Count warnings before/after to measure reduction

Usage:
    python run_placement_pipeline.py [--train] [--evaluate] [--all]
    
    nohup python run_placement_pipeline.py --all > pipeline.log 2>&1 &
"""

import os
import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from place_sql_quotes_annotations import SqlQuotesAnnotationPlacer, SqlQuotesAnnotationType
from place_signature_annotations import SignatureAnnotationPlacer, SignatureAnnotationType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('placement_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Base directories
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'
CASE_STUDIES_BACKUP_DIR = GEN_DATA_ROOT / 'case_studies_backup'
CHECKER_FRAMEWORK_HOME = Path('/home/ubuntu/checker-framework')
CHECKER_JAVAC = CHECKER_FRAMEWORK_HOME / 'checker' / 'bin' / 'javac'

# Checker configurations
# Training: Use CF test suites (have intentional type errors) - separate from evaluation
# Evaluation: Use 3 real GitHub projects per checker (no training sets)
CHECKER_CONFIGS = {
    'sql_quotes': {
        'processor': 'org.checkerframework.checker.sqlquotes.SqlQuotesChecker',
        'test_suite': 'cf_sqlquotes_tests',
        'evaluation_projects': [
            # 3 real GitHub projects for evaluation
            {'name': 'commons-dbutils', 'type': 'backup', 'url': 'https://github.com/apache/commons-dbutils'},
            {'name': 'commons-dbcp', 'type': 'backup', 'url': 'https://github.com/apache/commons-dbcp'},
            {'name': 'mybatis-3', 'type': 'backup', 'url': 'https://github.com/mybatis/mybatis-3'},
        ],
        'training_examples': 'training_sql_quotes',  # Used only for training, not evaluation
        'use_model_prediction': True,
    },
    'signature_string': {
        'processor': 'org.checkerframework.checker.signature.SignatureChecker',
        'test_suite': 'cf_signature_tests',
        'evaluation_projects': [
            # 3 real GitHub projects for evaluation
            {'name': 'kryo', 'type': 'backup', 'url': 'https://github.com/EsotericSoftware/kryo'},
            {'name': 'guice', 'type': 'backup', 'url': 'https://github.com/google/guice'},
            {'name': 'cglib', 'type': 'backup', 'url': 'https://github.com/cglib/cglib'},
        ],
        'training_examples': 'training_signature',  # Used only for training, not evaluation
        'use_model_prediction': True,
    }
}

# Backup directories - NEVER modify these
BACKUP_DIRECTORIES = [
    CASE_STUDIES_BACKUP_DIR,
    GEN_DATA_ROOT / 'annotation_evaluation' / 'backups',
    GEN_DATA_ROOT / 'annotated_projects_backup',
]


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


def find_java_files(directory: Path) -> List[Path]:
    """Find all Java files in a directory"""
    return list(directory.rglob('*.java'))


def clone_fresh_project(project_name: str, git_url: str) -> bool:
    """
    Clone a fresh copy of a project from GitHub to get unannotated version
    
    Args:
        project_name: Name of the project directory
        git_url: GitHub URL to clone from
        
    Returns:
        True if cloned successfully, False otherwise
    """
    project_dir = CASE_STUDIES_DIR / project_name
    
    try:
        # Remove existing project directory
        if project_dir.exists():
            shutil.rmtree(project_dir)
            logger.debug(f"Removed existing: {project_dir}")
        
        # Clone from GitHub (shallow clone for speed)
        logger.info(f"Cloning {project_name} from {git_url}...")
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', git_url, str(project_dir)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            return False
        
        logger.info(f"Cloned {project_name} successfully")
        return project_dir.exists()
        
    except subprocess.TimeoutExpired:
        logger.error(f"Git clone timed out for {project_name}")
        return False
    except Exception as e:
        logger.error(f"Error cloning {project_name}: {e}")
        return False


def parse_warnings(checker_output: str, project_dir: Path) -> List[Dict]:
    """
    Parse checker output to extract warning locations
    
    Args:
        checker_output: Raw output from the checker
        project_dir: Base project directory for relative paths
        
    Returns:
        List of warning dictionaries with file_path and line_number
    """
    import re
    
    warnings = []
    
    # Pattern: /path/to/File.java:123: error: [type] message
    warning_pattern = re.compile(r'^(.+\.java):(\d+):\s*(error|warning):\s*\[([^\]]+)\](.*)$')
    
    for line in checker_output.split('\n'):
        match = warning_pattern.match(line.strip())
        if match:
            file_path = match.group(1)
            line_number = int(match.group(2))
            severity = match.group(3)
            warning_type = match.group(4)
            message = match.group(5).strip()
            
            # Skip generic warnings
            if warning_type in ['deprecation', 'removal', 'unchecked', 'rawtypes', 'serial', 'path', 'options']:
                continue
            
            # Get relative path if possible
            try:
                rel_path = Path(file_path).relative_to(project_dir)
            except ValueError:
                rel_path = Path(file_path)
            
            warnings.append({
                'file_path': str(file_path),
                'relative_path': str(rel_path),
                'line_number': line_number,
                'severity': severity,
                'warning_type': warning_type,
                'message': message
            })
    
    return warnings


def restore_from_backup(project_name: str, target_dir: Path = None) -> bool:
    """
    Restore a project from backup to get a fresh copy
    
    Args:
        project_name: Name of the project to restore
        target_dir: Optional target directory (defaults to case_studies)
        
    Returns:
        True if restored successfully, False otherwise
    """
    # Check multiple backup locations
    backup_dirs_to_check = [
        CASE_STUDIES_BACKUP_DIR / project_name,
        GEN_DATA_ROOT / 'annotation_evaluation' / 'backups' / project_name,
    ]
    
    backup_dir = None
    for bd in backup_dirs_to_check:
        if bd.exists():
            backup_dir = bd
            break
    
    if not backup_dir:
        logger.warning(f"Backup not found for {project_name}")
        logger.warning(f"  Checked: {[str(b) for b in backup_dirs_to_check]}")
        return False
    
    project_dir = target_dir if target_dir else (CASE_STUDIES_DIR / project_name)
    
    # Safety check: never write to backup directories
    if not verify_not_backup_dir(project_dir):
        return False
    
    try:
        # Remove existing project directory
        if project_dir.exists():
            shutil.rmtree(project_dir)
            logger.debug(f"Removed existing: {project_dir}")
        
        # Copy from backup
        project_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_dir, project_dir)
        logger.info(f"Restored {project_name} from {backup_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error restoring {project_name}: {e}")
        return False


def place_annotations_for_checker(
    project_dir: Path, 
    checker_type: str,
    java_files: List[Path]
) -> Tuple[int, List[Dict]]:
    """
    Place annotations on Java files using heuristic-based placement
    
    Args:
        project_dir: Project directory
        checker_type: Type of checker (sql_quotes or signature_string)
        java_files: List of Java files to process
        
    Returns:
        Tuple of (annotations_placed, placement_details)
    """
    total_placed = 0
    placement_details = []
    
    for java_file in java_files:
        try:
            if checker_type == 'sql_quotes':
                placer = SqlQuotesAnnotationPlacer(str(java_file))
                placements = placer.analyze_and_place()
                if placements:
                    placer.add_imports()
                    placer.save_file()
                    total_placed += len(placements)
                    placement_details.append({
                        'file': str(java_file.relative_to(project_dir)),
                        'placements': len(placements),
                        'annotations': [p.annotation.value for p in placements]
                    })
            else:  # signature_string
                placer = SignatureAnnotationPlacer(str(java_file))
                placements = placer.analyze_and_place()
                if placements:
                    annotations_used = list(set(p.annotation for p in placements))
                    placer.add_imports(annotations_used)
                    placer.save_file()
                    total_placed += len(placements)
                    placement_details.append({
                        'file': str(java_file.relative_to(project_dir)),
                        'placements': len(placements),
                        'annotations': [p.annotation.value for p in placements]
                    })
        except Exception as e:
            logger.warning(f"Error processing {java_file}: {e}")
    
    return total_placed, placement_details


def place_annotations_at_warnings(
    warnings: List[Dict],
    checker_type: str,
    project_dir: Path
) -> Tuple[int, List[Dict]]:
    """
    Place annotations at specific warning locations
    
    Args:
        warnings: List of warning dictionaries with file_path and line_number
        checker_type: Type of checker (sql_quotes or signature_string)
        project_dir: Base project directory
        
    Returns:
        Tuple of (annotations_placed, placement_details)
    """
    total_placed = 0
    placement_details = []
    
    # Group warnings by file
    files_to_process = {}
    for warning in warnings:
        file_path = warning['file_path']
        if file_path not in files_to_process:
            files_to_process[file_path] = []
        files_to_process[file_path].append(warning)
    
    for file_path, file_warnings in files_to_process.items():
        if not Path(file_path).exists():
            logger.warning(f"Warning file not found: {file_path}")
            continue
        
        try:
            # Get line numbers to annotate (sorted in reverse to maintain positions)
            line_numbers = sorted(set(w['line_number'] for w in file_warnings), reverse=True)
            
            if checker_type == 'sql_quotes':
                placer = SqlQuotesAnnotationPlacer(file_path)
                placed_count = 0
                for line_num in line_numbers:
                    # Find the declaration line (may be above the warning line)
                    target_line = find_declaration_line(placer.lines, line_num)
                    if target_line and placer.is_valid_annotation_target(target_line):
                        annotation = placer.analyze_string_literal(placer.lines[target_line - 1])
                        if annotation is None:
                            annotation = SqlQuotesAnnotationType.SQL_EVEN_QUOTES
                        if placer.place_annotation(target_line, annotation):
                            placed_count += 1
                
                if placed_count > 0:
                    placer.add_imports()
                    placer.save_file()
                    total_placed += placed_count
                    
                    try:
                        rel_path = Path(file_path).relative_to(project_dir)
                    except ValueError:
                        rel_path = Path(file_path)
                    
                    placement_details.append({
                        'file': str(rel_path),
                        'warnings_targeted': len(file_warnings),
                        'annotations_placed': placed_count,
                        'lines': [w['line_number'] for w in file_warnings]
                    })
            
            else:  # signature_string
                placer = SignatureAnnotationPlacer(file_path)
                placed_count = 0
                annotations_used = []
                
                for line_num in line_numbers:
                    target_line = find_declaration_line(placer.lines, line_num)
                    if target_line and placer.is_valid_annotation_target(target_line):
                        annotation = placer.infer_annotation_from_context(placer.lines[target_line - 1])
                        if placer.place_annotation(target_line, annotation):
                            placed_count += 1
                            annotations_used.append(annotation)
                
                if placed_count > 0:
                    unique_annotations = list(set(annotations_used))
                    placer.add_imports(unique_annotations)
                    placer.save_file()
                    total_placed += placed_count
                    
                    try:
                        rel_path = Path(file_path).relative_to(project_dir)
                    except ValueError:
                        rel_path = Path(file_path)
                    
                    placement_details.append({
                        'file': str(rel_path),
                        'warnings_targeted': len(file_warnings),
                        'annotations_placed': placed_count,
                        'lines': [w['line_number'] for w in file_warnings]
                    })
                    
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
    
    return total_placed, placement_details


def find_declaration_line(lines: List[str], warning_line: int) -> Optional[int]:
    """
    Find the declaration line for a warning (may be above the warning line)
    
    The warning might be on a method call, but we need to annotate the variable declaration.
    """
    if warning_line < 1 or warning_line > len(lines):
        return None
    
    # First check the warning line itself
    line = lines[warning_line - 1].strip()
    if 'String ' in line or 'final String' in line:
        return warning_line
    
    # Look backwards for a variable declaration that might be used at the warning line
    # Extract variable name from warning line if it's a method call
    import re
    
    # Pattern for method call: methodName(varName) or methodName(varName, ...)
    var_match = re.search(r'\w+\s*\(\s*(\w+)', line)
    if var_match:
        var_name = var_match.group(1)
        
        # Search backwards for declaration of this variable
        for i in range(warning_line - 2, max(0, warning_line - 50), -1):
            check_line = lines[i].strip()
            # Look for String varName = or final String varName =
            if f'String {var_name}' in check_line or f'String {var_name} =' in check_line:
                return i + 1
    
    # If we can't find a declaration, return the warning line if it's valid
    if 'String' in line:
        return warning_line
    
    return None


def place_annotations_with_models(
    project_dir: Path,
    checker_type: str,
    base_model: str = 'hgt'
) -> Tuple[int, List[Dict]]:
    """
    Place annotations using trained models via MultiCheckerPredictor.
    
    This replaces the heuristic-based placement with model-based prediction.
    
    Args:
        project_dir: Project directory (will be modified - NOT a backup!)
        checker_type: Type of checker (sql_quotes or signature_string)
        base_model: Base model to use for prediction
        
    Returns:
        Tuple of (annotations_placed, placement_details)
    """
    # Safety check: never modify backups
    if not verify_not_backup_dir(project_dir):
        logger.error("Safety check failed: attempted to modify backup directory")
        return 0, []
    
    try:
        from filtered_multi_checker_predictor import FilteredMultiCheckerPredictor
        from place_annotations import ComprehensiveAnnotationPlacer
        import json
        import tempfile
        
        logger.info(f"Using model-based prediction for {checker_type} with {base_model}")
        
        # Create predictor
        predictor = FilteredMultiCheckerPredictor(
            checker_name=checker_type,
            base_model_filter=base_model
        )
        
        # Load models
        if not predictor.load_checker_models():
            logger.warning(f"No models loaded for {base_model}, falling back to heuristic placement")
            java_files = find_java_files(project_dir)
            return place_annotations_for_checker(project_dir, checker_type, java_files)
        
        # Find Java files
        java_files = find_java_files(project_dir)
        if not java_files:
            logger.warning("No Java files found")
            return 0, []
        
        # Get CFG directory
        cfg_dirs = [
            GEN_DATA_ROOT / f'cfg_output_adaptive_specimin_{checker_type}',
            GEN_DATA_ROOT / 'cfg_output_adaptive_specimin',
            GEN_DATA_ROOT / 'cfg_output_specimin',
        ]
        
        cfg_dir = None
        for cd in cfg_dirs:
            if cd.exists():
                cfg_dir = cd
                break
        
        if not cfg_dir:
            logger.warning(f"No CFG directory found, falling back to heuristic placement")
            return place_annotations_for_checker(project_dir, checker_type, java_files)
        
        # Generate predictions
        all_predictions = []
        for java_file in java_files[:50]:  # Limit files
            predictions = predictor.predict_for_file(str(java_file), str(cfg_dir))
            for pred in predictions:
                if 'file_path' not in pred or not pred['file_path']:
                    pred['file_path'] = str(java_file)
            all_predictions.extend(predictions)
        
        if not all_predictions:
            logger.info("No predictions generated, trying heuristic placement")
            return place_annotations_for_checker(project_dir, checker_type, java_files)
        
        # Save predictions to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(all_predictions, f, indent=2)
            predictions_file = f.name
        
        try:
            # Place annotations
            placer = ComprehensiveAnnotationPlacer(
                project_root=str(project_dir),
                output_dir=str(GEN_DATA_ROOT / 'placement_output'),
                checker_name=checker_type,
                backup=False,
                perfect_placement=True
            )
            
            predictions = placer.load_predictions(predictions_file)
            stats = placer.process_predictions(predictions)
            
            total_placed = stats.get('successful', 0)
            placement_details = [{
                'method': 'model_based',
                'model': base_model,
                'predictions_generated': len(all_predictions),
                'annotations_placed': total_placed
            }]
            
            return total_placed, placement_details
            
        finally:
            # Clean up temp file
            Path(predictions_file).unlink(missing_ok=True)
        
    except ImportError as e:
        logger.warning(f"Model-based prediction not available: {e}")
        logger.warning("Falling back to heuristic placement")
        java_files = find_java_files(project_dir)
        return place_annotations_for_checker(project_dir, checker_type, java_files)
    except Exception as e:
        logger.error(f"Error in model-based placement: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 0, []


def run_checker(java_files: List[Path], processor: str) -> Tuple[int, str]:
    """
    Run a Checker Framework checker on Java files
    
    Returns:
        Tuple of (warning_count, output)
    """
    if not java_files:
        return 0, "No Java files to check"
    
    # Build classpath
    checker_cp = f"{CHECKER_FRAMEWORK_HOME}/checker/dist/checker-qual.jar:{CHECKER_FRAMEWORK_HOME}/checker/dist/checker.jar"
    
    cmd = [
        str(CHECKER_JAVAC),
        '-processor', processor,
        '-cp', checker_cp,
        '-Xlint:-processing',
        '-Awarns',
    ] + [str(f) for f in java_files]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        output = result.stdout + result.stderr
        
        # Count warnings
        warning_count = 0
        for line in output.split('\n'):
            if 'error:' in line.lower() or 'warning:' in line.lower():
                if '[' in line and ']' in line:
                    # Skip generic warnings
                    if not any(w in line for w in ['[deprecation]', '[removal]', '[unchecked]', '[rawtypes]']):
                        warning_count += 1
        
        return warning_count, output
        
    except subprocess.TimeoutExpired:
        return -1, "Timeout"
    except Exception as e:
        return -1, str(e)


def run_training_phase(checker_type: str) -> Dict:
    """
    Run training phase for a checker type using CF test suite
    
    Returns:
        Dictionary with training results
    """
    config = CHECKER_CONFIGS[checker_type]
    test_suite_dir = CASE_STUDIES_DIR / config['test_suite']
    
    logger.info(f"=== Training Phase: {checker_type} ===")
    
    results = {
        'checker_type': checker_type,
        'test_suite': str(test_suite_dir),
        'timestamp': datetime.now().isoformat(),
    }
    
    if not test_suite_dir.exists():
        logger.error(f"Test suite not found: {test_suite_dir}")
        results['status'] = 'error'
        results['error'] = 'Test suite not found'
        return results
    
    # Find Java files
    java_files = find_java_files(test_suite_dir)
    logger.info(f"Found {len(java_files)} Java files in test suite")
    results['java_files'] = len(java_files)
    
    # Run checker to count warnings
    warning_count, output = run_checker(java_files, config['processor'])
    
    logger.info(f"Checker warnings: {warning_count}")
    results['warning_count'] = warning_count
    results['status'] = 'success' if warning_count >= 0 else 'error'
    
    # Save output
    output_file = GEN_DATA_ROOT / f'{checker_type}_training_warnings.out'
    with open(output_file, 'w') as f:
        f.write(output)
    results['output_file'] = str(output_file)
    
    return results


def run_evaluation_phase(checker_type: str) -> Dict:
    """
    Run evaluation phase for a checker type
    
    Complete flow:
    1. Clone fresh project from GitHub (to get unannotated code)
    2. Count baseline warnings (before placement)
    3. Parse warning locations
    4. Place annotations at warning locations
    5. Count warnings after placement
    6. Calculate and report reduction
    
    Returns:
        Dictionary with evaluation results
    """
    config = CHECKER_CONFIGS[checker_type]
    
    logger.info(f"=== Evaluation Phase: {checker_type} ===")
    
    results = {
        'checker_type': checker_type,
        'timestamp': datetime.now().isoformat(),
        'projects': {}
    }
    
    for project_info in config['evaluation_projects']:
        project_name = project_info['name']
        project_type = project_info.get('type', 'git')
        git_url = project_info.get('url', '')
        project_dir = CASE_STUDIES_DIR / project_name
        
        logger.info(f"Evaluating project: {project_name}")
        
        project_results = {
            'project': project_name,
            'path': str(project_dir),
            'type': project_type,
        }
        
        # Step 1: Get fresh copy of project
        if project_type == 'local':
            # Restore from backup (training examples with entry-point annotations)
            logger.info(f"  Step 1: Restoring from backup...")
            if not restore_from_backup(project_name):
                if not project_dir.exists():
                    logger.error(f"Project not found: {project_dir}")
                    project_results['status'] = 'not_found'
                    results['projects'][project_name] = project_results
                    continue
                logger.warning(f"  No backup, using existing project")
        elif project_type == 'backup':
            # Restore from backup directory (real projects)
            logger.info(f"  Step 1: Restoring from backup...")
            if not restore_from_backup(project_name):
                logger.error(f"Failed to restore {project_name} from backup")
                project_results['status'] = 'restore_failed'
                results['projects'][project_name] = project_results
                continue
            project_results['source'] = 'backup'
        else:
            # Clone fresh from GitHub
            logger.info(f"  Step 1: Cloning fresh from GitHub...")
            if not clone_fresh_project(project_name, git_url):
                logger.error(f"Failed to clone {project_name}")
                project_results['status'] = 'clone_failed'
                results['projects'][project_name] = project_results
                continue
            project_results['git_url'] = git_url
        
        # Find Java files
        java_files = find_java_files(project_dir)
        project_results['java_files'] = len(java_files)
        logger.info(f"  Found {len(java_files)} Java files")
        
        # Step 2: Count baseline warnings (before placement)
        logger.info(f"  Step 2: Counting baseline warnings...")
        baseline_count, baseline_output = run_checker(java_files, config['processor'])
        project_results['baseline_warnings'] = baseline_count
        logger.info(f"  Baseline warnings: {baseline_count}")
        
        # Save baseline output
        baseline_file = GEN_DATA_ROOT / f'{checker_type}_{project_name}_baseline.out'
        with open(baseline_file, 'w') as f:
            f.write(baseline_output)
        project_results['baseline_file'] = str(baseline_file)
        
        # Check if we have positive baseline warnings
        if baseline_count <= 0:
            logger.warning(f"  No baseline warnings found, skipping placement")
            project_results['status'] = 'no_warnings'
            project_results['after_warnings'] = 0
            project_results['reduction'] = 0
            project_results['reduction_pct'] = 0.0
            results['projects'][project_name] = project_results
            continue
        
        # Step 3: Parse warning locations
        logger.info(f"  Step 3: Parsing warning locations...")
        warnings = parse_warnings(baseline_output, project_dir)
        project_results['warnings_parsed'] = len(warnings)
        logger.info(f"  Parsed {len(warnings)} warning locations")
        
        # Step 4: Place annotations
        # Use model-based placement if configured, otherwise use heuristic placement
        use_models = config.get('use_model_prediction', False)
        
        if use_models:
            logger.info(f"  Step 4: Placing annotations using trained models...")
            annotations_placed, placement_details = place_annotations_with_models(
                project_dir, checker_type, base_model='hgt'
            )
            project_results['placement_method'] = 'model_based'
        else:
            logger.info(f"  Step 4: Placing annotations at warning locations (heuristic)...")
            annotations_placed, placement_details = place_annotations_at_warnings(
                warnings, checker_type, project_dir
            )
            project_results['placement_method'] = 'heuristic'
        
        project_results['annotations_placed'] = annotations_placed
        project_results['placement_details'] = placement_details
        logger.info(f"  Annotations placed: {annotations_placed}")
        
        # Step 5: Count warnings after placement
        logger.info(f"  Step 5: Counting warnings after placement...")
        # Re-find Java files (in case any were modified)
        java_files = find_java_files(project_dir)
        after_count, after_output = run_checker(java_files, config['processor'])
        project_results['after_warnings'] = after_count
        logger.info(f"  After warnings: {after_count}")
        
        # Save after output
        after_file = GEN_DATA_ROOT / f'{checker_type}_{project_name}_after.out'
        with open(after_file, 'w') as f:
            f.write(after_output)
        project_results['after_file'] = str(after_file)
        
        # Step 6: Calculate reduction
        reduction = baseline_count - after_count
        reduction_pct = (reduction / baseline_count) * 100.0 if baseline_count > 0 else 0.0
        
        project_results['reduction'] = reduction
        project_results['reduction_pct'] = round(reduction_pct, 2)
        project_results['status'] = 'success'
        
        logger.info(f"  Reduction: {reduction} ({reduction_pct:.1f}%)")
        
        results['projects'][project_name] = project_results
    
    # Also check training examples (without modification)
    training_dir = CASE_STUDIES_DIR / config['training_examples']
    if training_dir.exists():
        java_files = find_java_files(training_dir)
        warning_count, output = run_checker(java_files, config['processor'])
        results['training_examples'] = {
            'path': str(training_dir),
            'java_files': len(java_files),
            'warning_count': warning_count,
        }
        logger.info(f"Training examples warnings: {warning_count}")
    
    return results


def run_full_pipeline() -> Dict:
    """
    Run the full pipeline for all checkers
    
    Returns:
        Dictionary with all results
    """
    logger.info("=" * 60)
    logger.info("Starting Full Placement Pipeline")
    logger.info("=" * 60)
    
    all_results = {
        'start_time': datetime.now().isoformat(),
        'training': {},
        'evaluation': {},
    }
    
    # Run training phase for each checker
    for checker_type in CHECKER_CONFIGS:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {checker_type}")
        logger.info(f"{'='*40}")
        
        # Training
        training_results = run_training_phase(checker_type)
        all_results['training'][checker_type] = training_results
        
        # Evaluation
        eval_results = run_evaluation_phase(checker_type)
        all_results['evaluation'][checker_type] = eval_results
    
    all_results['end_time'] = datetime.now().isoformat()
    
    # Save results
    results_file = GEN_DATA_ROOT / 'placement_pipeline_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    return all_results


def print_summary(results: Dict):
    """Print a summary of the pipeline results"""
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    
    print("\n### Training Phase ###")
    for checker_type, data in results.get('training', {}).items():
        status = data.get('status', 'unknown')
        warnings = data.get('warning_count', 'N/A')
        print(f"  {checker_type}: {status}, {warnings} warnings")
    
    print("\n### Evaluation Phase ###")
    for checker_type, data in results.get('evaluation', {}).items():
        print(f"\n  {checker_type}:")
        for project, proj_data in data.get('projects', {}).items():
            status = proj_data.get('status', 'unknown')
            
            # Check for new fields (before/after)
            if 'baseline_warnings' in proj_data:
                baseline = proj_data.get('baseline_warnings', 'N/A')
                after = proj_data.get('after_warnings', 'N/A')
                placed = proj_data.get('annotations_placed', 0)
                reduction_pct = proj_data.get('reduction_pct', 0)
                print(f"    - {project}: {status}")
                print(f"        Before: {baseline} warnings")
                print(f"        Annotations placed: {placed}")
                print(f"        After: {after} warnings")
                print(f"        Reduction: {reduction_pct}%")
            else:
                # Old format fallback
                warnings = proj_data.get('warning_count', 'N/A')
                print(f"    - {project}: {status}, {warnings} warnings")
        
        if 'training_examples' in data:
            te = data['training_examples']
            print(f"    - training_examples: {te.get('warning_count', 'N/A')} warnings")
    
    print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run annotation placement pipeline')
    parser.add_argument('--train', action='store_true', help='Run training phase only')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation phase only')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--checker', choices=['sql_quotes', 'signature_string'],
                       help='Run for specific checker only')
    
    args = parser.parse_args()
    
    # Default to full pipeline if no options specified
    if not (args.train or args.evaluate or args.all):
        args.all = True
    
    if args.all:
        results = run_full_pipeline()
        print_summary(results)
    else:
        checkers = [args.checker] if args.checker else list(CHECKER_CONFIGS.keys())
        
        results = {'training': {}, 'evaluation': {}}
        
        for checker_type in checkers:
            if args.train:
                results['training'][checker_type] = run_training_phase(checker_type)
            if args.evaluate:
                results['evaluation'][checker_type] = run_evaluation_phase(checker_type)
        
        print_summary(results)
    
    return 0


if __name__ == '__main__':
    exit(main())
