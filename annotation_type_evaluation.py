#!/usr/bin/env python3
"""
Annotation Type Evaluation System

This module provides comprehensive evaluation for annotation type prediction,
addressing the critical gap where only binary classification was evaluated
instead of multi-class annotation type accuracy.
"""

import os
import json
import time
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, multilabel_confusion_matrix
)
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from annotation_type_prediction import (
    LowerBoundAnnotationType, AnnotationTypeGBTModel, AnnotationTypeHGTModel
)
from annotation_type_causal_model import AnnotationTypeCausalModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnnotationTypeEvaluationResult:
    """Results from annotation type evaluation"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int
    training_time: float
    prediction_time: float
    confusion_matrix: List[List[int]]
    classification_report: Dict
    class_accuracies: Dict[str, float]
    difficult_cases: List[Dict]

class AnnotationTypeEvaluator:
    """Comprehensive evaluator for annotation type prediction"""
    
    def __init__(self, train_dir: str = "test_results/statistical_dataset/train/cfg_output",
                 test_dir: str = "test_results/statistical_dataset/test/cfg_output"):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.results = {}
    
    def load_cfg_files(self, directory: str) -> List[Dict[str, Any]]:
        """Load CFG files from directory"""
        cfg_files = []
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return cfg_files
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                cfg_path = os.path.join(directory, filename)
                try:
                    with open(cfg_path, 'r') as f:
                        cfg_data = json.load(f)
                        cfg_files.append({
                            'file': cfg_path,
                            'method': cfg_data.get('method_name', filename.replace('.json', '')),
                            'data': cfg_data
                        })
                except Exception as e:
                    logger.warning(f"Error loading CFG {filename}: {e}")
        
        logger.info(f"Loaded {len(cfg_files)} CFG files from {directory}")
        return cfg_files
    
    def create_ground_truth_annotation_types(self, cfg_data: Dict[str, Any]) -> List[str]:
        """Create ground truth annotation types for each node"""
        from annotation_type_prediction import AnnotationTypeClassifier
        
        classifier = AnnotationTypeClassifier()
        ground_truth = []
        
        for node in cfg_data.get('nodes', []):
            from node_level_models import NodeClassifier
            if NodeClassifier.is_annotation_target(node):
                features = classifier.extract_features(node, cfg_data)
                annotation_type = classifier.determine_annotation_type(features)
                ground_truth.append(annotation_type.value)
            else:
                ground_truth.append(LowerBoundAnnotationType.NO_ANNOTATION.value)
        
        return ground_truth
    
    def extract_difficult_cases(self, cfg_data: Dict[str, Any], predictions: List[str], 
                               ground_truth: List[str]) -> List[Dict[str, Any]]:
        """Extract cases where annotation type prediction was incorrect"""
        difficult_cases = []
        nodes = cfg_data.get('nodes', [])
        
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            if pred != true and i < len(nodes):
                node = nodes[i]
                case = {
                    'node_id': i,
                    'predicted_type': pred,
                    'actual_type': true,
                    'label': node.get('label', ''),
                    'line': node.get('line', 0),
                    'node_type': node.get('node_type', ''),
                    'error_category': self._categorize_error(pred, true)
                }
                difficult_cases.append(case)
        
        return difficult_cases
    
    def _categorize_error(self, predicted: str, actual: str) -> str:
        """Categorize the type of prediction error"""
        if actual == LowerBoundAnnotationType.NO_ANNOTATION.value:
            return "false_positive"
        elif predicted == LowerBoundAnnotationType.NO_ANNOTATION.value:
            return "false_negative"
        else:
            # Check if it's related annotation types
            positive_types = ["@Positive", "@NonNegative", "@GTENegativeOne"]
            length_types = ["@MinLen", "@ArrayLen", "@LengthOf", "@LTLengthOf", "@GTLengthOf"]
            index_types = ["@IndexFor", "@SearchIndexFor", "@SearchIndexBottom"]
            
            if predicted in positive_types and actual in positive_types:
                return "positive_type_confusion"
            elif predicted in length_types and actual in length_types:
                return "length_type_confusion"
            elif predicted in index_types and actual in index_types:
                return "index_type_confusion"
            else:
                return "category_mismatch"
    
    def evaluate_gbt_model(self, train_cfgs: List[Dict[str, Any]], 
                          test_cfgs: List[Dict[str, Any]]) -> AnnotationTypeEvaluationResult:
        """Evaluate GBT model for annotation type prediction"""
        logger.info("Evaluating Annotation Type GBT Model...")
        
        try:
            # Initialize model
            model = AnnotationTypeGBTModel()
            
            # Train model
            start_time = time.time()
            training_result = model.train_model(train_cfgs)
            training_time = time.time() - start_time
            
            if not training_result.get('success', True):
                logger.error(f"GBT training failed: {training_result.get('error', 'Unknown error')}")
                return AnnotationTypeEvaluationResult(
                    "AnnotationTypeGBT", 0.0, 0.0, 0.0, 0.0, 0, training_time, 0.0, [[0]], {}, {}, []
                )
            
            # Evaluate on test data
            start_time = time.time()
            all_predictions = []
            all_ground_truth = []
            all_difficult_cases = []
            
            for cfg_file in test_cfgs:
                cfg_data = cfg_file['data']
                
                # Get predictions
                predictions = model.predict_annotation_types(cfg_data)
                pred_types = [LowerBoundAnnotationType.NO_ANNOTATION.value] * len(cfg_data.get('nodes', []))
                
                # Map predictions to nodes
                for pred in predictions:
                    node_id = pred.get('node_id', 0)
                    if node_id < len(pred_types):
                        pred_types[node_id] = pred['annotation_type']
                
                # Get ground truth
                ground_truth = self.create_ground_truth_annotation_types(cfg_data)
                
                # Only evaluate annotation targets
                from node_level_models import NodeClassifier
                for i, node in enumerate(cfg_data.get('nodes', [])):
                    if NodeClassifier.is_annotation_target(node) and i < len(pred_types) and i < len(ground_truth):
                        all_predictions.append(pred_types[i])
                        all_ground_truth.append(ground_truth[i])
                
                # Extract difficult cases
                difficult_cases = self.extract_difficult_cases(cfg_data, pred_types, ground_truth)
                all_difficult_cases.extend(difficult_cases)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            if len(all_predictions) == 0 or len(all_ground_truth) == 0:
                logger.warning("No annotation type predictions available for evaluation")
                return AnnotationTypeEvaluationResult(
                    "AnnotationTypeGBT", 0.0, 0.0, 0.0, 0.0, 0, training_time, prediction_time, [[0]], {}, {}, []
                )
            
            accuracy = accuracy_score(all_ground_truth, all_predictions)
            precision = precision_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            
            # Classification report
            report = classification_report(all_ground_truth, all_predictions, output_dict=True, zero_division=0)
            
            # Confusion matrix
            labels = sorted(list(set(all_ground_truth + all_predictions)))
            cm = confusion_matrix(all_ground_truth, all_predictions, labels=labels)
            
            # Per-class accuracies
            class_accuracies = {}
            for label in labels:
                class_mask = np.array(all_ground_truth) == label
                if np.sum(class_mask) > 0:
                    class_pred = np.array(all_predictions)[class_mask]
                    class_acc = np.sum(class_pred == label) / len(class_pred)
                    class_accuracies[label] = class_acc
            
            logger.info(f"GBT Annotation Type Evaluation - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            return AnnotationTypeEvaluationResult(
                model_name="AnnotationTypeGBT",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=len(all_predictions),
                training_time=training_time,
                prediction_time=prediction_time,
                confusion_matrix=cm.tolist(),
                classification_report=report,
                class_accuracies=class_accuracies,
                difficult_cases=all_difficult_cases[:10]  # Top 10 difficult cases
            )
            
        except Exception as e:
            logger.error(f"Error evaluating annotation type GBT model: {e}")
            return AnnotationTypeEvaluationResult(
                "AnnotationTypeGBT", 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, [[0]], {}, {}, []
            )
    
    def evaluate_hgt_model(self, train_cfgs: List[Dict[str, Any]], 
                          test_cfgs: List[Dict[str, Any]]) -> AnnotationTypeEvaluationResult:
        """Evaluate HGT model for annotation type prediction"""
        logger.info("Evaluating Annotation Type HGT Model...")
        
        try:
            # Initialize model
            model = AnnotationTypeHGTModel()
            
            # Train model
            start_time = time.time()
            training_result = model.train_model(train_cfgs, epochs=100)
            training_time = time.time() - start_time
            
            if not training_result.get('success', True):
                logger.error(f"HGT training failed: {training_result.get('error', 'Unknown error')}")
                return AnnotationTypeEvaluationResult(
                    "AnnotationTypeHGT", 0.0, 0.0, 0.0, 0.0, 0, training_time, 0.0, [[0]], {}, {}, []
                )
            
            # Evaluate on test data
            start_time = time.time()
            all_predictions = []
            all_ground_truth = []
            all_difficult_cases = []
            
            for cfg_file in test_cfgs:
                cfg_data = cfg_file['data']
                
                # Get predictions
                predictions = model.predict_annotation_types(cfg_data)
                pred_types = [LowerBoundAnnotationType.NO_ANNOTATION.value] * len(cfg_data.get('nodes', []))
                
                # Map predictions to nodes
                for pred in predictions:
                    node_id = pred.get('node_id', 0)
                    if node_id < len(pred_types):
                        pred_types[node_id] = pred['annotation_type']
                
                # Get ground truth
                ground_truth = self.create_ground_truth_annotation_types(cfg_data)
                
                # Only evaluate annotation targets
                from node_level_models import NodeClassifier
                for i, node in enumerate(cfg_data.get('nodes', [])):
                    if NodeClassifier.is_annotation_target(node) and i < len(pred_types) and i < len(ground_truth):
                        all_predictions.append(pred_types[i])
                        all_ground_truth.append(ground_truth[i])
                
                # Extract difficult cases
                difficult_cases = self.extract_difficult_cases(cfg_data, pred_types, ground_truth)
                all_difficult_cases.extend(difficult_cases)
            
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            if len(all_predictions) == 0 or len(all_ground_truth) == 0:
                logger.warning("No annotation type predictions available for evaluation")
                return AnnotationTypeEvaluationResult(
                    "AnnotationTypeHGT", 0.0, 0.0, 0.0, 0.0, 0, training_time, prediction_time, [[0]], {}, {}, []
                )
            
            accuracy = accuracy_score(all_ground_truth, all_predictions)
            precision = precision_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            
            # Classification report
            report = classification_report(all_ground_truth, all_predictions, output_dict=True, zero_division=0)
            
            # Confusion matrix
            labels = sorted(list(set(all_ground_truth + all_predictions)))
            cm = confusion_matrix(all_ground_truth, all_predictions, labels=labels)
            
            # Per-class accuracies
            class_accuracies = {}
            for label in labels:
                class_mask = np.array(all_ground_truth) == label
                if np.sum(class_mask) > 0:
                    class_pred = np.array(all_predictions)[class_mask]
                    class_acc = np.sum(class_pred == label) / len(class_pred)
                    class_accuracies[label] = class_acc
            
            logger.info(f"HGT Annotation Type Evaluation - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            return AnnotationTypeEvaluationResult(
                model_name="AnnotationTypeHGT",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=len(all_predictions),
                training_time=training_time,
                prediction_time=prediction_time,
                confusion_matrix=cm.tolist(),
                classification_report=report,
                class_accuracies=class_accuracies,
                difficult_cases=all_difficult_cases[:10]  # Top 10 difficult cases
            )
            
        except Exception as e:
            logger.error(f"Error evaluating annotation type HGT model: {e}")
            return AnnotationTypeEvaluationResult(
                "AnnotationTypeHGT", 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, [[0]], {}, {}, []
            )
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive annotation type evaluation"""
        logger.info("🔬 Starting Comprehensive Annotation Type Evaluation")
        logger.info("="*80)
        
        # Load data
        train_cfgs = self.load_cfg_files(self.train_dir)
        test_cfgs = self.load_cfg_files(self.test_dir)
        
        if not train_cfgs or not test_cfgs:
            logger.error("Failed to load training or test data")
            return {}
        
        logger.info(f"📊 Dataset: {len(train_cfgs)} train + {len(test_cfgs)} test CFGs")
        logger.info(f"🎯 Evaluation Type: Multi-Class Annotation Type Prediction")
        logger.info("="*80)
        
        results = {}
        
        # Evaluate GBT model
        logger.info("Evaluating Annotation Type GBT Model...")
        gbt_result = self.evaluate_gbt_model(train_cfgs, test_cfgs)
        results['GBT'] = gbt_result
        
        # Evaluate HGT model
        logger.info("Evaluating Annotation Type HGT Model...")
        hgt_result = self.evaluate_hgt_model(train_cfgs, test_cfgs)
        results['HGT'] = hgt_result
        
        # Evaluate Causal model
        logger.info("Evaluating Annotation Type Causal Model...")
        causal_result = self.evaluate_causal_model(train_cfgs, test_cfgs)
        results['Causal'] = causal_result
        
        # Print summary
        self._print_evaluation_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _print_evaluation_summary(self, results: Dict[str, AnnotationTypeEvaluationResult]):
        """Print evaluation summary"""
        logger.info("")
        logger.info("="*80)
        logger.info("ANNOTATION TYPE PREDICTION EVALUATION RESULTS")
        logger.info("="*80)
        
        logger.info("")
        logger.info("📈 MULTI-CLASS ANNOTATION TYPE PERFORMANCE:")
        logger.info("-" * 80)
        logger.info(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}")
        logger.info("-" * 80)
        
        for model_name, result in results.items():
            logger.info(f"{result.model_name:<20} {result.accuracy:<10.3f} {result.precision:<10.3f} "
                       f"{result.recall:<10.3f} {result.f1_score:<10.3f} {result.support:<10}")
        
        logger.info("-" * 80)
        
        # Find best performing model
        best_model = max(results.values(), key=lambda x: x.f1_score)
        logger.info(f"🏆 Best F1 Score: {best_model.model_name} ({best_model.f1_score:.3f})")
        logger.info(f"🎯 Best Accuracy: {best_model.model_name} ({best_model.accuracy:.3f})")
        
        # Show per-class performance for best model
        logger.info("")
        logger.info(f"🔍 PER-CLASS PERFORMANCE ({best_model.model_name}):")
        logger.info("-" * 60)
        for class_name, accuracy in best_model.class_accuracies.items():
            logger.info(f"  • {class_name}: {accuracy:.3f}")
        
        # Show difficult cases
        if best_model.difficult_cases:
            logger.info("")
            logger.info(f"🔍 DIFFICULT CASES ({best_model.model_name}):")
            logger.info("-" * 60)
            for case in best_model.difficult_cases[:5]:  # Show top 5
                logger.info(f"  • Line {case['line']}: Predicted {case['predicted_type']}, "
                           f"Actual {case['actual_type']} ({case['error_category']})")
        
        logger.info("="*80)
        logger.info("🎯 Annotation type evaluation complete!")
    
    def _save_results(self, results: Dict[str, AnnotationTypeEvaluationResult]):
        """Save evaluation results to file"""
        output_dir = "test_results/annotation_type_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'model_name': result.model_name,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'support': result.support,
                'training_time': result.training_time,
                'prediction_time': result.prediction_time,
                'confusion_matrix': result.confusion_matrix,
                'classification_report': result.classification_report,
                'class_accuracies': result.class_accuracies,
                'difficult_cases': result.difficult_cases
            }
        
        # Save to JSON
        output_file = os.path.join(output_dir, "annotation_type_evaluation_results.json")
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {output_file}")

def main():
    """Run annotation type evaluation"""
    evaluator = AnnotationTypeEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        logger.info("✅ Annotation type evaluation completed successfully!")
    else:
        logger.error("❌ Annotation type evaluation failed!")

if __name__ == '__main__':
    main()
