#!/usr/bin/env python3
"""
Main Optimized Pipeline Entry Point

This is the new main entry point for the GenDATA pipeline that uses the
optimized performance pipeline by default, providing significant improvements
in annotation type prediction accuracy.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_performance_pipeline import OptimizedPerformancePipeline, create_optimized_pipeline
from pipeline_config import AUGMENTATION_POLICY_CONFIG, DEFAULT_PROJECT_ROOT, DEFAULT_WARNINGS_FILE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainOptimizedPipeline:
    """
    Main entry point for the optimized GenDATA pipeline.
    Uses performance-optimized augmentation by default.
    """
    
    def __init__(self, config_path: Optional[str] = None, device: str = 'auto'):
        self.config_path = config_path
        self.device = device
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize optimized pipeline
        self.pipeline = create_optimized_pipeline(config_dict=self.config, device=device)
        
        logger.info("Initialized MainOptimizedPipeline with performance-focused configuration")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        # Start with optimized defaults
        config = AUGMENTATION_POLICY_CONFIG.copy()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
        
        return config
    
    def train_annotation_type(
        self,
        annotation_type: str,
        model_type: Optional[str] = None,
        warnings_file: Optional[str] = None,
        project_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Train an annotation type model with optimized augmentation.
        
        Args:
            annotation_type: The annotation type to train ('positive', 'nonnegative', 'gtenegativeone')
            model_type: Model type to use (auto-selected if None)
            warnings_file: Path to warnings file (uses default if None)
            project_root: Project root directory (uses default if None)
            output_dir: Output directory for results
            use_optimization: Whether to use optimized augmentation (default: True)
        
        Returns:
            Dictionary with training results and performance metrics
        """
        
        # Use defaults if not provided
        warnings_file = warnings_file or DEFAULT_WARNINGS_FILE
        project_root = project_root or DEFAULT_PROJECT_ROOT
        output_dir = output_dir or f"results_optimized_{annotation_type}"
        
        logger.info(f"Training {annotation_type} with {'optimized' if use_optimization else 'baseline'} augmentation")
        
        if use_optimization:
            # Use optimized pipeline
            result = self.pipeline.train_annotation_type_with_optimized_augmentation(
                annotation_type=annotation_type,
                model_type=model_type,
                warnings_file=warnings_file,
                project_root=project_root,
                output_dir=output_dir
            )
        else:
            # Use baseline pipeline
            from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
            
            baseline_pipeline = SimpleAnnotationTypePipeline(
                project_root=project_root,
                warnings_file=warnings_file,
                cfwr_root=self.config.get('cfwr_root', '/home/ubuntu/GenDATA')
            )
            
            result = baseline_pipeline.train_annotation_type(
                annotation_type=annotation_type,
                model_type=model_type or 'gcn',
                output_dir=output_dir
            )
        
        return result
    
    def predict_annotation_type(
        self,
        annotation_type: str,
        model_type: Optional[str] = None,
        warnings_file: Optional[str] = None,
        project_root: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict annotation types using trained models.
        
        Args:
            annotation_type: The annotation type to predict
            model_type: Model type to use (auto-selected if None)
            warnings_file: Path to warnings file (uses default if None)
            project_root: Project root directory (uses default if None)
            output_dir: Output directory for predictions
        
        Returns:
            Dictionary with prediction results
        """
        
        # Use defaults if not provided
        warnings_file = warnings_file or DEFAULT_WARNINGS_FILE
        project_root = project_root or DEFAULT_PROJECT_ROOT
        output_dir = output_dir or f"predictions_optimized_{annotation_type}"
        
        # For prediction, we use the baseline pipeline since it has the prediction logic
        from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
        
        baseline_pipeline = SimpleAnnotationTypePipeline(
            project_root=project_root,
            warnings_file=warnings_file,
            cfwr_root=self.config.get('cfwr_root', '/home/ubuntu/GenDATA'),
            mode='predict'
        )
        
        result = baseline_pipeline.predict_annotation_type(
            annotation_type=annotation_type,
            model_type=model_type or 'gcn',
            output_dir=output_dir
        )
        
        return result
    
    def predict_with_enhanced_pipeline(
        self,
        project_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        java_files: List[str] = None,
        use_lower_bound_checker: bool = True
    ) -> Dict[str, Any]:
        """
        Run prediction using the enhanced pipeline with Lower Bound Checker integration.
        This is now the default prediction behavior.
        
        Args:
            project_root: Root directory of the target project
            output_dir: Directory to save prediction results
            models_dir: Directory containing trained models
            java_files: List of specific Java files to process (if None, processes all files)
            use_lower_bound_checker: Whether to run Lower Bound Checker (default: True)
            
        Returns:
            Dictionary with prediction results and performance metrics
        """
        
        # Use defaults if not provided
        project_root = project_root or DEFAULT_PROJECT_ROOT
        output_dir = output_dir or "predictions_enhanced"
        models_dir = models_dir or "models_annotation_types"
        
        logger.info(f"🚀 Running Enhanced Prediction Pipeline on {project_root}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🔍 Lower Bound Checker: {'enabled' if use_lower_bound_checker else 'disabled'}")
        
        # Run the enhanced prediction pipeline
        success = self.pipeline.predict_with_enhanced_pipeline(
            project_root=project_root,
            output_dir=output_dir,
            models_dir=models_dir,
            java_files=java_files,
            use_lower_bound_checker=use_lower_bound_checker
        )
        
        if success:
            result = {
                'success': True,
                'project_root': project_root,
                'output_dir': output_dir,
                'models_dir': models_dir,
                'lower_bound_checker_used': use_lower_bound_checker,
                'message': 'Enhanced prediction pipeline completed successfully'
            }
            
            logger.info("✅ Enhanced prediction completed successfully")
            return result
        else:
            result = {
                'success': False,
                'project_root': project_root,
                'output_dir': output_dir,
                'models_dir': models_dir,
                'lower_bound_checker_used': use_lower_bound_checker,
                'message': 'Enhanced prediction pipeline failed'
            }
            
            logger.error("❌ Enhanced prediction failed")
            return result
    
    def train_all_annotation_types(
        self,
        warnings_file: Optional[str] = None,
        project_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Train all annotation types with optimized augmentation.
        
        Args:
            warnings_file: Path to warnings file (uses default if None)
            project_root: Project root directory (uses default if None)
            output_dir: Base output directory for results
            use_optimization: Whether to use optimized augmentation (default: True)
        
        Returns:
            Dictionary with results for all annotation types
        """
        
        annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
        results = {}
        
        logger.info(f"Training all annotation types with {'optimized' if use_optimization else 'baseline'} augmentation")
        
        for ann_type in annotation_types:
            logger.info(f"Training {ann_type}...")
            
            type_output_dir = f"{output_dir or 'results_optimized_all'}_{ann_type}"
            
            result = self.train_annotation_type(
                annotation_type=ann_type,
                warnings_file=warnings_file,
                project_root=project_root,
                output_dir=type_output_dir,
                use_optimization=use_optimization
            )
            
            results[ann_type] = result
            
            logger.info(f"Completed {ann_type}: {result.get('improvement_percentage', 0):.2f}% improvement")
        
        # Calculate overall statistics
        improvements = [r.get('improvement_percentage', 0) for r in results.values()]
        overall_stats = {
            'total_annotation_types': len(annotation_types),
            'average_improvement': sum(improvements) / len(improvements) if improvements else 0,
            'max_improvement': max(improvements) if improvements else 0,
            'min_improvement': min(improvements) if improvements else 0,
            'successful_trainings': sum(1 for r in results.values() if r.get('success', False))
        }
        
        results['_overall_stats'] = overall_stats
        
        logger.info(f"Training completed. Average improvement: {overall_stats['average_improvement']:.2f}%")
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from the optimized pipeline"""
        return self.pipeline.get_performance_summary()
    
    def get_optimization_recommendations(self, annotation_type: str) -> Dict[str, Any]:
        """Get optimization recommendations for a specific annotation type"""
        return self.pipeline.optimize_for_annotation_type(annotation_type)

def main():
    """Main entry point for the optimized pipeline"""
    
    parser = argparse.ArgumentParser(
        description="GenDATA Optimized Pipeline - Performance-focused annotation type training and prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all annotation types with optimization (default)
  python main_optimized_pipeline.py --train-all

  # Train specific annotation type with best model
  python main_optimized_pipeline.py --train nonnegative --model gcn

  # Train with custom configuration
  python main_optimized_pipeline.py --train positive --config custom_config.json

  # Predict using trained models (legacy mode)
  python main_optimized_pipeline.py --predict nonnegative

  # Run enhanced prediction with Lower Bound Checker (default)
  python main_optimized_pipeline.py --predict-enhanced --project-root /path/to/project

  # Enhanced prediction on specific files
  python main_optimized_pipeline.py --predict-enhanced --java-files File1.java File2.java

  # Compare optimized vs baseline performance
  python main_optimized_pipeline.py --train-all --compare-baseline
        """
    )
    
    # Main actions
    parser.add_argument('--train', type=str, choices=['positive', 'nonnegative', 'gtenegativeone'],
                        help='Train a specific annotation type')
    parser.add_argument('--train-all', action='store_true',
                        help='Train all annotation types with optimization')
    parser.add_argument('--predict', type=str, choices=['positive', 'nonnegative', 'gtenegativeone'],
                        help='Predict using trained models (legacy mode)')
    parser.add_argument('--predict-enhanced', action='store_true',
                        help='Run enhanced prediction with Lower Bound Checker integration (default behavior)')
    
    # Configuration options
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['gcn', 'gbt', 'causal'],
                        help='Model type to use (auto-selected if not specified)')
    parser.add_argument('--warnings-file', type=str,
                        help='Path to warnings file')
    parser.add_argument('--project-root', type=str,
                        help='Project root directory')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for results')
    
    # Optimization options
    parser.add_argument('--no-optimization', action='store_true',
                        help='Use baseline augmentation instead of optimized')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Compare optimized vs baseline performance')
    parser.add_argument('--performance-summary', action='store_true',
                        help='Show performance summary')
    
    # Enhanced prediction options
    parser.add_argument('--java-files', nargs='*',
                        help='Specific Java files to process (if not provided, processes all files)')
    parser.add_argument('--no-lower-bound-checker', action='store_true',
                        help='Disable Lower Bound Checker execution (use legacy mode)')
    parser.add_argument('--models-dir', type=str,
                        help='Directory containing trained models')
    
    # System options
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    try:
        pipeline = MainOptimizedPipeline(config_path=args.config, device=args.device)
        logger.info("Optimized pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    try:
        # Handle different actions
        if args.train:
            logger.info(f"Training {args.train} with optimization")
            result = pipeline.train_annotation_type(
                annotation_type=args.train,
                model_type=args.model,
                warnings_file=args.warnings_file,
                project_root=args.project_root,
                output_dir=args.output_dir,
                use_optimization=not args.no_optimization
            )
            
            print(f"\nTraining Result for {args.train}:")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Improvement: {result.get('improvement_percentage', 0):.2f}%")
            print(f"  Training Time: {result.get('training_time', 0):.3f}s")
            
        elif args.train_all:
            logger.info("Training all annotation types")
            results = pipeline.train_all_annotation_types(
                warnings_file=args.warnings_file,
                project_root=args.project_root,
                output_dir=args.output_dir,
                use_optimization=not args.no_optimization
            )
            
            print(f"\nTraining Results for All Annotation Types:")
            for ann_type, result in results.items():
                if ann_type != '_overall_stats':
                    print(f"  {ann_type}: {result.get('improvement_percentage', 0):.2f}% improvement")
            
            if '_overall_stats' in results:
                stats = results['_overall_stats']
                print(f"\nOverall Statistics:")
                print(f"  Average Improvement: {stats['average_improvement']:.2f}%")
                print(f"  Max Improvement: {stats['max_improvement']:.2f}%")
                print(f"  Successful Trainings: {stats['successful_trainings']}/{stats['total_annotation_types']}")
        
        elif args.predict:
            logger.info(f"Predicting {args.predict}")
            result = pipeline.predict_annotation_type(
                annotation_type=args.predict,
                model_type=args.model,
                warnings_file=args.warnings_file,
                project_root=args.project_root,
                output_dir=args.output_dir
            )
            
            print(f"\nPrediction Result for {args.predict}:")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Predictions Generated: {result.get('predictions_count', 0)}")
        
        elif args.predict_enhanced:
            logger.info("Running enhanced prediction with Lower Bound Checker integration")
            result = pipeline.predict_with_enhanced_pipeline(
                project_root=args.project_root,
                output_dir=args.output_dir,
                models_dir=args.models_dir,
                java_files=args.java_files,
                use_lower_bound_checker=not args.no_lower_bound_checker
            )
            
            print(f"\nEnhanced Prediction Result:")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Project Root: {result.get('project_root', 'N/A')}")
            print(f"  Output Directory: {result.get('output_dir', 'N/A')}")
            print(f"  Lower Bound Checker Used: {result.get('lower_bound_checker_used', False)}")
            print(f"  Message: {result.get('message', 'N/A')}")
        
        elif args.performance_summary:
            summary = pipeline.get_performance_summary()
            print(f"\nPerformance Summary:")
            print(json.dumps(summary, indent=2))
        
        else:
            parser.print_help()
            return 1
        
        # Compare with baseline if requested
        if args.compare_baseline and (args.train or args.train_all):
            logger.info("Comparing optimized vs baseline performance")
            
            if args.train:
                baseline_result = pipeline.train_annotation_type(
                    annotation_type=args.train,
                    model_type=args.model,
                    warnings_file=args.warnings_file,
                    project_root=args.project_root,
                    output_dir=f"{args.output_dir or 'results'}_baseline",
                    use_optimization=False
                )
                
                print(f"\nBaseline vs Optimized Comparison for {args.train}:")
                print(f"  Baseline: {baseline_result.get('improvement_percentage', 0):.2f}%")
                print(f"  Optimized: {result.get('improvement_percentage', 0):.2f}%")
                print(f"  Difference: {result.get('improvement_percentage', 0) - baseline_result.get('improvement_percentage', 0):.2f}%")
        
        logger.info("Pipeline execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
