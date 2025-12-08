#!/usr/bin/env python3
"""
Collect and format predictions for outline projects.

This script collects predictions from predictions_annotation_types/ and formats
them as predictions_{model}.json files in case_studies/{project}/ directory.

Input Format (from predictions_annotation_types/):
- Various formats from prediction pipeline output

Output Format (to case_studies/{project}/predictions_{model}.json):
[
  {
    "file_path": "/path/to/file.java",
    "predictions": [
      {"line": 42, "type": "@Positive", "confidence": 0.85}
    ]
  }
]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model name mappings (handle both enhanced_causal and dgcrf)
MODEL_NAME_MAPPING = {
    'dgcrf': 'enhanced_causal',
    'enhanced_causal': 'enhanced_causal',
    'gcn': 'gcn',
    'hgt': 'hgt',
    'gbt': 'gbt',
    'causal': 'causal',
    'gcsn': 'gcsn',
    'dg2n': 'dg2n'
}

# Annotation type suffixes
ANNOTATION_TYPES = ['positive', 'nonnegative', 'gtenegativeone']


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file if it exists."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        logger.debug(f"Failed to load {path}: {e}")
    return None


def normalize_file_path(file_path: str, project_name: str) -> str:
    """Normalize file path to absolute path."""
    if not file_path:
        return file_path
    
    # If already absolute, return as is
    if file_path.startswith('/'):
        return file_path
    
    # Try to resolve relative to case_studies
    case_studies_root = Path('/home/ubuntu/GenDATA/case_studies')
    project_path = case_studies_root / project_name
    
    # Try relative path
    if (project_path / file_path).exists():
        return str((project_path / file_path).resolve())
    
    # Try to find file in project
    for java_file in project_path.rglob('*.java'):
        if java_file.name == Path(file_path).name or str(java_file).endswith(file_path):
            return str(java_file.resolve())
    
    # Return as absolute if contains project name
    if project_name in file_path:
        return file_path
    
    return file_path


def collect_predictions_from_file(pred_file: Path, project_name: str) -> Dict[str, List[Dict]]:
    """Collect predictions from a single prediction file."""
    data = load_json(pred_file)
    if not data:
        return {}
    
    predictions_by_file: Dict[str, List[Dict]] = {}
    
    # Handle different input formats
    if isinstance(data, dict):
        # Format 1: Per-file format {"file": "...", "predictions": [...]}
        if 'file' in data and 'predictions' in data:
            file_path = normalize_file_path(data['file'], project_name)
            preds = []
            for p in data.get('predictions', []):
                line = p.get('line') or p.get('line_number') or p.get('lineno')
                ann_type = p.get('type') or p.get('annotation_type') or p.get('annotation')
                conf = p.get('confidence') or p.get('score') or p.get('prob', 0.0)
                
                if line is not None and ann_type:
                    preds.append({
                        'line': int(line),
                        'type': str(ann_type),
                        'confidence': float(conf)
                    })
            
            if preds:
                predictions_by_file[file_path] = preds
        
        # Format 2: Batch format {"files": [...]}
        elif 'files' in data:
            for entry in data['files']:
                file_path = normalize_file_path(entry.get('file_path') or entry.get('file'), project_name)
                preds = []
                for p in entry.get('predictions', []):
                    line = p.get('line') or p.get('line_number') or p.get('lineno')
                    ann_type = p.get('type') or p.get('annotation_type') or p.get('annotation')
                    conf = p.get('confidence') or p.get('score') or p.get('prob', 0.0)
                    
                    if line is not None and ann_type:
                        preds.append({
                            'line': int(line),
                            'type': str(ann_type),
                            'confidence': float(conf)
                        })
                
                if preds:
                    predictions_by_file[file_path] = preds
        
        # Format 3: Flat format {file_path: [predictions]}
        else:
            for key, value in data.items():
                if isinstance(value, list):
                    file_path = normalize_file_path(key, project_name)
                    preds = []
                    for p in value:
                        line = p.get('line') or p.get('line_number') or p.get('lineno')
                        ann_type = p.get('type') or p.get('annotation_type') or p.get('annotation')
                        conf = p.get('confidence') or p.get('score') or p.get('prob', 0.0)
                        
                        if line is not None and ann_type:
                            preds.append({
                                'line': int(line),
                                'type': str(ann_type),
                                'confidence': float(conf)
                            })
                    
                    if preds:
                        predictions_by_file[file_path] = preds
    
    elif isinstance(data, list):
        # Format 4: List format [{file_path, predictions}]
        for entry in data:
            file_path = normalize_file_path(entry.get('file_path') or entry.get('file'), project_name)
            preds = []
            for p in entry.get('predictions', []):
                line = p.get('line') or p.get('line_number') or p.get('lineno')
                ann_type = p.get('type') or p.get('annotation_type') or p.get('annotation')
                conf = p.get('confidence') or p.get('score') or p.get('prob', 0.0)
                
                if line is not None and ann_type:
                    preds.append({
                        'line': int(line),
                        'type': str(ann_type),
                        'confidence': float(conf)
                    })
            
            if preds:
                predictions_by_file[file_path] = preds
    
    return predictions_by_file


def collect_predictions_for_project(project_name: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Collect all predictions for a project, organized by model."""
    predictions_dir = Path('/home/ubuntu/GenDATA/predictions_annotation_types')
    
    if not predictions_dir.exists():
        logger.warning(f"Predictions directory not found: {predictions_dir}")
        return {}
    
    # Collect predictions from all JSON files
    all_predictions_by_model: Dict[str, Dict[str, List[Dict]]] = {}
    
    # Search for prediction files
    for pred_file in predictions_dir.rglob('*.json'):
        try:
            # Try to extract model name from filename
            filename = pred_file.stem.lower()
            model_name = None
            
            # Check for model names in filename
            for model_key, normalized_model in MODEL_NAME_MAPPING.items():
                if model_key in filename:
                    model_name = normalized_model
                    break
            
            # Check for annotation type in filename
            annotation_type = None
            for ann_type in ANNOTATION_TYPES:
                if ann_type in filename:
                    annotation_type = ann_type
                    break
            
            # Collect predictions
            file_predictions = collect_predictions_from_file(pred_file, project_name)
            
            if file_predictions:
                # Group by model and annotation type
                if model_name:
                    model_key = f"{model_name}_{annotation_type}" if annotation_type else model_name
                else:
                    model_key = "unknown"
                
                if model_key not in all_predictions_by_model:
                    all_predictions_by_model[model_key] = {}
                
                # Merge predictions
                for file_path, preds in file_predictions.items():
                    if file_path not in all_predictions_by_model[model_key]:
                        all_predictions_by_model[model_key][file_path] = []
                    all_predictions_by_model[model_key][file_path].extend(preds)
        
        except Exception as e:
            logger.debug(f"Error processing {pred_file}: {e}")
            continue
    
    return all_predictions_by_model


def format_predictions_by_model(predictions_by_file: Dict[str, List[Dict]]) -> List[Dict]:
    """Format predictions as predictions_{model}.json format."""
    formatted = []
    
    for file_path, preds in predictions_by_file.items():
        formatted.append({
            'file_path': file_path,
            'predictions': preds
        })
    
    return formatted


def save_predictions(project_dir: Path, model_name: str, predictions: List[Dict]) -> Path:
    """Save formatted predictions to case_studies/{project}/predictions_{model}.json."""
    output_file = project_dir / f'predictions_{model_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Saved {len(predictions)} file predictions for model {model_name} to {output_file}")
    return output_file


def collect_and_save_predictions(project_name: str) -> Dict[str, Path]:
    """Collect predictions for a project and save them in correct format."""
    project_dir = Path('/home/ubuntu/GenDATA/case_studies') / project_name
    
    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        return {}
    
    logger.info(f"Collecting predictions for project: {project_name}")
    
    # Collect all predictions
    all_predictions = collect_predictions_for_project(project_name)
    
    if not all_predictions:
        logger.warning(f"No predictions found for project {project_name}")
        return {}
    
    # Save predictions for each model
    saved_files = {}
    
    for model_key, predictions_by_file in all_predictions.items():
        formatted = format_predictions_by_model(predictions_by_file)
        
        if formatted:
            output_file = save_predictions(project_dir, model_key, formatted)
            saved_files[model_key] = output_file
    
    logger.info(f"Saved predictions for {len(saved_files)} models")
    return saved_files


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python collect_outline_project_predictions.py <project_name>")
        print("Example: python collect_outline_project_predictions.py agrona")
        sys.exit(1)
    
    project_name = sys.argv[1]
    saved_files = collect_and_save_predictions(project_name)
    
    if saved_files:
        print(f"Successfully collected and saved predictions for {len(saved_files)} models")
        for model, path in saved_files.items():
            print(f"  {model}: {path}")
    else:
        print(f"No predictions found or saved for {project_name}")
        sys.exit(1)


if __name__ == '__main__':
    main()

