#!/usr/bin/env python3
"""
Integrated Prediction and Annotation Pipeline

This script combines the CFWR prediction pipeline with comprehensive annotation placement,
providing an end-to-end solution for predicting and placing Checker Framework annotations
with full support for Lower Bound Checker annotations.

Features:
- Integrates with existing prediction pipeline
- Supports all Checker Framework annotation types
- Places multiple annotations at the same location
- Validates annotations after placement
- Generates comprehensive reports
- Best practices defaults throughout
"""

import os
import sys
import json
import argparse
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Import our modules
from predict_on_project import generate_warning_based_slices_with_dataflow
from place_annotations import ComprehensiveAnnotationPlacer
from checker_framework_integration import CheckerType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedPipeline:
    """Integrated prediction and annotation pipeline"""
    
    def __init__(self, project_root: str, output_dir: str, models_dir: str = None):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.models_dir = Path(models_dir) if models_dir else Path(os.environ.get('MODELS_DIR', 'models'))
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        self.annotations_dir = self.output_dir / "annotated_project"
        self.annotations_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized pipeline for project: {self.project_root}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_predictions(self, models: List[str] = None, slicer: str = 'cf', 
                       use_augmented_slices: bool = True, dataflow_cfgs: bool = True) -> Dict[str, str]:
        """Run predictions using the CFWR pipeline"""
        if models is None:
            models = ['hgt', 'gbt', 'causal']
        
        logger.info(f"Running predictions with models: {models}")
        logger.info(f"Using slicer: {slicer}")
        logger.info(f"Augmented slices: {use_augmented_slices}")
        logger.info(f"Dataflow CFGs: {dataflow_cfgs}")
        
        # Create temporary directory for pipeline outputs
        temp_dir = self.output_dir / "temp_pipeline"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Generate slices and CFGs using existing pipeline
            slices_dir, cfg_dir = generate_warning_based_slices_with_dataflow(
                project_root=str(self.project_root),
                warnings_file=str(temp_dir / "warnings.out"),  # Create dummy warnings file
                output_dir=str(temp_dir),
                slicer_type=slicer,
                use_augmented_slices=use_augmented_slices
            )
            
            if not slices_dir or not os.path.exists(slices_dir):
                raise RuntimeError("Failed to generate slices")
            
            logger.info(f"Generated slices in: {slices_dir}")
            logger.info(f"Generated CFGs in: {cfg_dir}")
            
            # Run predictions for each model
            prediction_files = {}
            
            for model_type in models:
                logger.info(f"Running {model_type.upper()} predictions...")
                
                # Determine model file path
                model_path = self.models_dir / f"{model_type}_model.{'pth' if model_type == 'hgt' else 'joblib'}"
                
                if model_type in ('hgt','gbt','causal'):
                    if not model_path.exists():
                        logger.warning(f"Model file not found: {model_path}")
                        continue
                
                # Output file for this model's predictions
                output_file = self.predictions_dir / f"{model_type}_predictions_dataflow.json"
                
                # Run prediction based on model type
                success = self._run_model_prediction(
                    model_type, str(model_path), slices_dir, str(output_file), cfg_dir
                )
                
                if success:
                    prediction_files[model_type] = str(output_file)
                    logger.info(f"{model_type.upper()} predictions saved to: {output_file}")
                else:
                    logger.error(f"Failed to run {model_type.upper()} predictions")
            
            return prediction_files
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            raise
    
    def _run_model_prediction(self, model_type: str, model_path: str, slices_dir: str, 
                            output_file: str, cfg_dir: str) -> bool:
        """Run prediction for a specific model type"""
        import subprocess
        
        try:
            # Prepare command based on model type
            if model_type == 'hgt':
                cmd = [
                    sys.executable, 'predict_hgt.py',
                    '--slices_dir', slices_dir,
                    '--model_path', model_path,
                    '--out_path', output_file,
                    '--cfg_output_dir', cfg_dir
                ]
            elif model_type == 'gbt':
                cmd = [
                    sys.executable, 'predict_gbt.py',
                    '--slices_dir', slices_dir,
                    '--model_path', model_path,
                    '--out_path', output_file,
                    '--cfg_output_dir', cfg_dir
                ]
            elif model_type == 'causal':
                cmd = [
                    sys.executable, 'predict_causal.py',
                    '--slices_dir', slices_dir,
                    '--model_path', model_path,
                    '--out_path', output_file,
                    '--cfg_output_dir', cfg_dir
                ]
            elif model_type == 'nullgtn':
                # Use wrapper; requires env-provided paths
                artifact_dir = os.environ.get('NULLGTN_ARTIFACT_DIR', os.path.join(os.getcwd(), 'nullgtn-artifact'))
                work_dir = os.environ.get('NULLGTN_WORK_DIR', os.path.join(self.output_dir, 'temp_pipeline', 'nullgtn_work'))
                os.makedirs(work_dir, exist_ok=True)
                model_key = os.environ.get('NULLGTN_MODEL_KEY', 'default')
                cmd = [
                    sys.executable, 'predict_nullgtn.py',
                    '--artifact_dir', artifact_dir,
                    '--model_key', model_key,
                    '--work_dir', work_dir,
                    '--out_path', output_file,
                ]
            elif model_type == 'gcn':
                cmd = [
                    sys.executable, 'gcn_predict.py',
                    '--java_file', str(self.project_root),  # For project predictions, prefer predict_on_project.py usage; here we skip
                    '--model_path', str(self.models_dir / 'gcn' / 'best_gcn.pth'),
                    '--out_path', output_file,
                    '--cfg_output_dir', str(cfg_dir)
                ]
            elif model_type == 'dg2n':
                # dg2n not executed at full-project granularity here; prefer predict_on_project.py
                logger.warning('DG2N not supported in this integrated project-level prediction path; use predict_on_project.py')
                continue
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            # Run prediction
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                logger.debug(f"{model_type.upper()} prediction output: {result.stdout}")
                return True
            else:
                logger.error(f"{model_type.upper()} prediction failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running {model_type} prediction: {e}")
            return False
    
    def merge_predictions(self, prediction_files: Dict[str, str]) -> str:
        """Merge predictions from multiple models into a single file"""
        logger.info("Merging predictions from all models...")
        
        merged_predictions = []
        model_confidences = {'hgt': 0.4, 'gbt': 0.3, 'causal': 0.3}  # Weights for ensemble
        
        # Load all predictions
        all_predictions = {}
        for model_type, pred_file in prediction_files.items():
            try:
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                all_predictions[model_type] = data
                logger.info(f"Loaded {len(data) if isinstance(data, list) else 'grouped'} predictions from {model_type}")
            except Exception as e:
                logger.error(f"Error loading predictions from {pred_file}: {e}")
                continue
        
        # Merge predictions with ensemble approach
        merged_data = self._ensemble_predictions(all_predictions, model_confidences)
        
        # Save merged predictions
        merged_file = self.predictions_dir / "merged_predictions.json"
        with open(merged_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        logger.info(f"Merged predictions saved to: {merged_file}")
        return str(merged_file)
    
    def _ensemble_predictions(self, all_predictions: Dict[str, any], 
                            model_weights: Dict[str, float]) -> List[Dict]:
        """Combine predictions from multiple models using ensemble approach"""
        # For now, implement a simple approach: use highest confidence predictions
        # In a more sophisticated implementation, we could:
        # 1. Vote on annotation types
        # 2. Combine confidences
        # 3. Filter by consensus
        
        merged = []
        
        # Start with predictions from the first available model
        base_model = next(iter(all_predictions.keys()))
        base_predictions = all_predictions[base_model]
        
        if isinstance(base_predictions, list):
            # Direct list format
            for pred in base_predictions:
                enhanced_pred = pred.copy()
                enhanced_pred['model_type'] = base_model
                enhanced_pred['ensemble_confidence'] = pred.get('confidence', 0.5) * model_weights.get(base_model, 0.33)
                merged.append(enhanced_pred)
        elif isinstance(base_predictions, dict):
            # File-grouped format
            for file_path, file_preds in base_predictions.items():
                for pred in file_preds:
                    enhanced_pred = pred.copy()
                    enhanced_pred['file_path'] = file_path
                    enhanced_pred['model_type'] = base_model
                    enhanced_pred['ensemble_confidence'] = pred.get('confidence', 0.5) * model_weights.get(base_model, 0.33)
                    merged.append(enhanced_pred)
        
        # TODO: Implement more sophisticated ensemble logic
        # For now, return the base predictions enhanced with model info
        
        logger.info(f"Created ensemble with {len(merged)} predictions")
        return merged
    
    def place_annotations(self, predictions_file: str, validate: bool = True) -> Dict[str, any]:
        """Place annotations based on predictions"""
        logger.info("Placing annotations...")
        
        # Copy project to annotations directory for modification
        annotated_project_dir = self.annotations_dir / "annotated_source"
        
        if annotated_project_dir.exists():
            import shutil
            shutil.rmtree(annotated_project_dir)
        
        import shutil
        shutil.copytree(self.project_root, annotated_project_dir)
        
        # Initialize annotation placer with perfect placement (default)
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(annotated_project_dir),
            output_dir=str(self.annotations_dir),
            backup=True,
            perfect_placement=True  # Use perfect placement by default
        )
        
        # Load and process predictions
        predictions = placer.load_predictions(predictions_file)
        
        # Place annotations
        stats = placer.process_predictions(predictions)
        
        # Validate if requested
        validation_results = None
        if validate:
            validation_results = placer.validate_annotations([
                CheckerType.NULLNESS, 
                CheckerType.INDEX
            ])
        
        # Generate report
        report_path = placer.generate_report(stats, validation_results)
        
        return {
            'stats': stats,
            'validation_results': validation_results,
            'report_path': report_path,
            'annotated_project_dir': str(annotated_project_dir)
        }
    
    def run_complete_pipeline(self, models: List[str] = None, slicer: str = 'cf',
                            use_augmented_slices: bool = True, dataflow_cfgs: bool = True,
                            validate_annotations: bool = True) -> Dict[str, any]:
        """Run the complete prediction and annotation pipeline"""
        logger.info("Starting complete prediction and annotation pipeline")
        
        try:
            # Step 1: Run predictions
            prediction_files = self.run_predictions(
                models=models,
                slicer=slicer,
                use_augmented_slices=use_augmented_slices,
                dataflow_cfgs=dataflow_cfgs
            )
            
            if not prediction_files:
                raise RuntimeError("No predictions were generated")
            
            # Step 2: Merge predictions from multiple models
            merged_predictions_file = self.merge_predictions(prediction_files)
            
            # Step 3: Place annotations
            annotation_results = self.place_annotations(
                merged_predictions_file,
                validate=validate_annotations
            )
            
            # Step 4: Create final summary
            summary = {
                'pipeline_status': 'completed',
                'prediction_files': prediction_files,
                'merged_predictions': merged_predictions_file,
                'annotation_results': annotation_results,
                'output_directory': str(self.output_dir)
            }
            
            # Save summary
            summary_file = self.output_dir / "pipeline_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Complete pipeline finished successfully!")
            logger.info(f"Annotated project: {annotation_results['annotated_project_dir']}")
            logger.info(f"Report: {annotation_results['report_path']}")
            logger.info(f"Summary: {summary_file}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description='Integrated prediction and annotation pipeline with Lower Bound Checker support'
    )
    
    parser.add_argument('--project_root', required=True,
                       help='Root directory of the Java project to analyze')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for all results')
    parser.add_argument('--models_dir', default=os.environ.get('MODELS_DIR', 'models'),
                       help='Directory containing trained models')
    parser.add_argument('--models', nargs='*', choices=['hgt', 'gbt', 'causal'],
                       default=['hgt', 'gbt', 'causal'],
                       help='Models to use for prediction')
    parser.add_argument('--slicer', choices=['specimin', 'wala', 'cf'], default='cf',
                       help='Slicer to use (default: cf for Checker Framework)')
    parser.add_argument('--use_augmented_slices', action='store_true', default=True,
                       help='Use augmented slices (default: True)')
    parser.add_argument('--use_original_slices', action='store_true', default=False,
                       help='Use original slices instead of augmented')
    parser.add_argument('--dataflow_cfgs', action='store_true', default=True,
                       help='Use dataflow-augmented CFGs (default: True)')
    parser.add_argument('--validate_annotations', action='store_true', default=True,
                       help='Validate placed annotations (default: True)')
    parser.add_argument('--skip_validation', action='store_true', default=False,
                       help='Skip annotation validation')
    
    args = parser.parse_args()
    
    # Determine slice preference
    use_augmented_slices = args.use_augmented_slices and not args.use_original_slices
    validate_annotations = args.validate_annotations and not args.skip_validation
    
    try:
        # Initialize pipeline
        pipeline = IntegratedPipeline(
            project_root=args.project_root,
            output_dir=args.output_dir,
            models_dir=args.models_dir
        )
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            models=args.models,
            slicer=args.slicer,
            use_augmented_slices=use_augmented_slices,
            dataflow_cfgs=args.dataflow_cfgs,
            validate_annotations=validate_annotations
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Annotated project: {results['annotation_results']['annotated_project_dir']}")
        print(f"Report: {results['annotation_results']['report_path']}")
        print(f"Summary: {results['output_directory']}/pipeline_summary.json")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
