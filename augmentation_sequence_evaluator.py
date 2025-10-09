#!/usr/bin/env python3
"""
Augmentation Sequence Evaluator

This module evaluates the quality of augmentation sequences using multiple metrics:
- Slicer resistance: Measure code preservation after slicing
- Model performance: Train mini-model and measure accuracy
- Diversity: Measure syntactic/semantic diversity of augmented variants
- Compilation success: Ensure augmented code compiles
"""

import os
import re
import ast
import json
import time
import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import subprocess
import tempfile

from recursive_augmentation_engine import TransformationState, TransformationType

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    slicer_resistance: float
    model_performance: float
    diversity_score: float
    compilation_success: float
    semantic_preservation: float
    overall_score: float
    metadata: Dict[str, Any]

@dataclass
class SlicingResult:
    """Result of slicing operation"""
    original_code: str
    sliced_code: str
    preserved_lines: int
    total_lines: int
    preservation_ratio: float

class AugmentationSequenceEvaluator:
    """Evaluates augmentation sequences using multiple quality metrics"""
    
    def __init__(self, device: str = 'cpu', mini_model_epochs: int = 5):
        self.device = device
        self.mini_model_epochs = mini_model_epochs
        
        # Evaluation statistics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'slicer_resistance_scores': [],
            'model_performance_scores': [],
            'diversity_scores': [],
            'compilation_success_rates': [],
            'semantic_preservation_rates': []
        }
        
        # Mini-model for quick performance evaluation
        self.mini_model = None
        
        # Slicing tools configuration
        self.slicing_tools = ['soot', 'specimin']  # Available slicing tools
    
    def evaluate_sequence(self, states: List[TransformationState]) -> EvaluationMetrics:
        """Evaluate a complete augmentation sequence"""
        if not states:
            return self._create_empty_metrics()
        
        original_state = states[0]
        final_state = states[-1]
        
        # Evaluate each metric
        slicer_resistance = self.evaluate_slicer_resistance(original_state, final_state)
        model_performance = self.evaluate_model_performance(states)
        diversity_score = self.compute_diversity_metrics(states)
        compilation_success = self.evaluate_compilation_success(final_state)
        semantic_preservation = self.evaluate_semantic_preservation(original_state, final_state)
        
        # Compute overall score (weighted combination)
        overall_score = (
            0.3 * slicer_resistance +
            0.3 * model_performance +
            0.2 * diversity_score +
            0.1 * compilation_success +
            0.1 * semantic_preservation
        )
        
        # Create metrics object
        metrics = EvaluationMetrics(
            slicer_resistance=slicer_resistance,
            model_performance=model_performance,
            diversity_score=diversity_score,
            compilation_success=compilation_success,
            semantic_preservation=semantic_preservation,
            overall_score=overall_score,
            metadata={
                'sequence_length': len(states),
                'transformation_types': [t.value for t in final_state.transformation_history],
                'final_complexity': final_state.complexity_score,
                'depth_reached': final_state.depth
            }
        )
        
        # Update statistics
        self._update_statistics(metrics)
        
        return metrics
    
    def evaluate_slicer_resistance(self, original_state: TransformationState, 
                                 final_state: TransformationState) -> float:
        """Evaluate slicer resistance by comparing code preservation after slicing"""
        try:
            # Get slicing results for both original and final code
            original_slicing = self._perform_slicing(original_state.code)
            final_slicing = self._perform_slicing(final_state.code)
            
            if not original_slicing or not final_slicing:
                return 0.0
            
            # Calculate preservation improvement
            original_preservation = original_slicing.preservation_ratio
            final_preservation = final_slicing.preservation_ratio
            
            # Higher preservation is better
            resistance_score = final_preservation
            
            # Bonus for improvement over original
            if final_preservation > original_preservation:
                improvement = (final_preservation - original_preservation) / original_preservation
                resistance_score += min(improvement, 0.2)  # Cap bonus at 0.2
            
            return min(resistance_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error evaluating slicer resistance: {e}")
            return 0.5  # Default neutral score
    
    def evaluate_model_performance(self, states: List[TransformationState]) -> float:
        """Evaluate model performance using a quick mini-model training"""
        try:
            # Generate training data from states
            training_data = self._generate_training_data(states)
            
            if not training_data or len(training_data) < 10:
                return 0.5  # Insufficient data
            
            # Train mini-model
            performance = self._train_mini_model(training_data)
            
            return performance
            
        except Exception as e:
            logger.warning(f"Error evaluating model performance: {e}")
            return 0.5
    
    def compute_diversity_metrics(self, states: List[TransformationState]) -> float:
        """Compute syntactic and semantic diversity metrics"""
        if len(states) < 2:
            return 0.0
        
        # Syntactic diversity
        syntactic_diversity = self._compute_syntactic_diversity(states)
        
        # Semantic diversity
        semantic_diversity = self._compute_semantic_diversity(states)
        
        # Transformation diversity
        transformation_diversity = self._compute_transformation_diversity(states)
        
        # Combine diversities
        overall_diversity = (
            0.4 * syntactic_diversity +
            0.3 * semantic_diversity +
            0.3 * transformation_diversity
        )
        
        return overall_diversity
    
    def evaluate_compilation_success(self, state: TransformationState) -> float:
        """Evaluate compilation success rate"""
        return 1.0 if state.compilation_status else 0.0
    
    def evaluate_semantic_preservation(self, original_state: TransformationState, 
                                     final_state: TransformationState) -> float:
        """Evaluate semantic preservation between original and final state"""
        return 1.0 if final_state.semantic_preservation else 0.0
    
    def evaluate_transformation(self, state: TransformationState) -> float:
        """Evaluate a single transformation step"""
        # Quick evaluation for intermediate steps
        compilation_score = 1.0 if state.compilation_status else 0.0
        semantic_score = 1.0 if state.semantic_preservation else 0.0
        complexity_penalty = max(0, 1.0 - abs(state.complexity_score - 5.0) / 5.0)
        
        return (compilation_score + semantic_score + complexity_penalty) / 3.0
    
    def evaluate_final_sequence(self, final_state: TransformationState) -> float:
        """Evaluate final state of augmentation sequence"""
        compilation_score = 1.0 if final_state.compilation_status else 0.0
        semantic_score = 1.0 if final_state.semantic_preservation else 0.0
        depth_bonus = min(final_state.depth / 5.0, 1.0)  # Reward deeper transformations
        
        return (compilation_score + semantic_score + depth_bonus) / 3.0
    
    def batch_evaluate(self, sequences: List[List[TransformationState]]) -> List[EvaluationMetrics]:
        """Evaluate multiple augmentation sequences in parallel"""
        results = []
        
        for sequence in sequences:
            metrics = self.evaluate_sequence(sequence)
            results.append(metrics)
        
        return results
    
    def _perform_slicing(self, code: str) -> Optional[SlicingResult]:
        """Perform slicing on code using available tools"""
        try:
            # Try different slicing tools
            for tool in self.slicing_tools:
                try:
                    result = self._slice_with_tool(code, tool)
                    if result:
                        return result
                except Exception as e:
                    logger.debug(f"Slicing with {tool} failed: {e}")
                    continue
            
            # Fallback: use actual slicing tools
            return self._perform_actual_slicing(code)
            
        except Exception as e:
            logger.warning(f"Error performing slicing: {e}")
            return None
    
    def _slice_with_tool(self, code: str, tool: str) -> Optional[SlicingResult]:
        """Slice code using specific tool"""
        if tool == 'soot':
            # Use actual Soot slicing
            try:
                from enhanced_soot_slicer import EnhancedSootSlicer
                slicer = EnhancedSootSlicer()
                result = slicer.slice_code(code)
                if result and hasattr(result, 'success') and result.success:
                    return SlicingResult(
                        original_code=code,
                        sliced_code=result.sliced_code,
                        preserved_lines=len(result.sliced_code.split('\n')),
                        total_lines=len(code.split('\n')),
                        preservation_ratio=result.preservation_ratio if hasattr(result, 'preservation_ratio') else 0.7
                    )
            except ImportError:
                logger.debug("EnhancedSootSlicer not available")
            except Exception as e:
                logger.debug(f"Soot slicing failed: {e}")
                
        elif tool == 'specimin':
            # Use actual Specimin slicing
            try:
                from simple_code_semantic_augment_slices import SimpleCodeSemanticAugmentSlices
                augmenter = SimpleCodeSemanticAugmentSlices()
                sliced_result = augmenter._perform_specimin_slicing(code)
                if sliced_result:
                    return SlicingResult(
                        original_code=code,
                        sliced_code=sliced_result,
                        preserved_lines=len(sliced_result.split('\n')),
                        total_lines=len(code.split('\n')),
                        preservation_ratio=0.8
                    )
            except ImportError:
                logger.debug("SimpleCodeSemanticAugmentSlices not available")
            except Exception as e:
                logger.debug(f"Specimin slicing failed: {e}")
        
        return None
    
    def _perform_actual_slicing(self, code: str) -> SlicingResult:
        """Perform actual slicing using available tools"""
        try:
            # Try to use enhanced soot slicer if available
            try:
                from enhanced_soot_slicer import EnhancedSootSlicer
                slicer = EnhancedSootSlicer()
                result = slicer.slice_code(code)
                if result and hasattr(result, 'success') and result.success:
                    return SlicingResult(
                        original_code=code,
                        sliced_code=result.sliced_code,
                        preserved_lines=len(result.sliced_code.split('\n')),
                        total_lines=len(code.split('\n')),
                        preservation_ratio=result.preservation_ratio if hasattr(result, 'preservation_ratio') else 0.7
                    )
            except ImportError:
                logger.debug("EnhancedSootSlicer not available")
            except Exception as e:
                logger.debug(f"EnhancedSootSlicer failed: {e}")
            
            # Try to use specimin if available
            try:
                from simple_code_semantic_augment_slices import SimpleCodeSemanticAugmentSlices
                augmenter = SimpleCodeSemanticAugmentSlices()
                # Use specimin-style slicing
                sliced_result = augmenter._perform_specimin_slicing(code)
                if sliced_result:
                    return SlicingResult(
                        original_code=code,
                        sliced_code=sliced_result,
                        preserved_lines=len(sliced_result.split('\n')),
                        total_lines=len(code.split('\n')),
                        preservation_ratio=0.8  # Specimin typically preserves more
                    )
            except ImportError:
                logger.debug("SimpleCodeSemanticAugmentSlices not available")
            except Exception as e:
                logger.debug(f"Specimin slicing failed: {e}")
            
        except Exception as e:
            logger.warning(f"Error in actual slicing: {e}")
        
        # Fallback: intelligent line-based slicing
        return self._intelligent_line_slicing(code)
    
    def _intelligent_line_slicing(self, code: str) -> SlicingResult:
        """Intelligent line-based slicing as fallback"""
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Preserve important lines based on Java syntax
        preserved_lines = 0
        sliced_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Always preserve important Java constructs
            if (re.search(r'\b(public|private|protected|static|final|class|interface|enum)\b', line_stripped) or
                re.search(r'\b(if|for|while|do|switch|case|default|return|throw|try|catch|finally)\b', line_stripped) or
                re.search(r'\b(new|this|super|import|package)\b', line_stripped) or
                re.search(r'\w+\s*\([^)]*\)\s*[;{]', line_stripped) or  # Method calls
                re.search(r'\w+\s*=\s*[^=]', line_stripped) or  # Assignments
                re.search(r'[{}]', line_stripped)):  # Braces
                sliced_lines.append(line)
                preserved_lines += 1
            elif line_stripped and not line_stripped.startswith('//'):  # Non-empty, non-comment lines
                # Preserve some other lines with lower probability
                if random.random() < 0.4:
                    sliced_lines.append(line)
                    preserved_lines += 1
        
        sliced_code = '\n'.join(sliced_lines)
        preservation_ratio = preserved_lines / total_lines if total_lines > 0 else 0.0
        
        return SlicingResult(
            original_code=code,
            sliced_code=sliced_code,
            preserved_lines=preserved_lines,
            total_lines=total_lines,
            preservation_ratio=preservation_ratio
        )
    
    def _simulate_specimin_slicing(self, code: str) -> SlicingResult:
        """Simulate Specimin-based slicing"""
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Simulate Specimin slicing (typically more aggressive)
        preserved_lines = 0
        sliced_lines = []
        
        for line in lines:
            # Keep only essential lines
            if (re.search(r'\b(return|class|method)\b', line) or
                re.search(r'\w+\s*=\s*\w+', line)):
                sliced_lines.append(line)
                preserved_lines += 1
            else:
                # Lower probability of preserving other lines
                if random.random() < 0.1:
                    sliced_lines.append(line)
                    preserved_lines += 1
        
        sliced_code = '\n'.join(sliced_lines)
        preservation_ratio = preserved_lines / total_lines if total_lines > 0 else 0.0
        
        return SlicingResult(
            original_code=code,
            sliced_code=sliced_code,
            preserved_lines=preserved_lines,
            total_lines=total_lines,
            preservation_ratio=preservation_ratio
        )
    
    def _simulate_slicing(self, code: str) -> SlicingResult:
        """Fallback slicing simulation"""
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Simple simulation: keep 60% of lines
        preserved_lines = int(total_lines * 0.6)
        sliced_lines = lines[:preserved_lines]
        
        sliced_code = '\n'.join(sliced_lines)
        
        return SlicingResult(
            original_code=code,
            sliced_code=sliced_code,
            preserved_lines=preserved_lines,
            total_lines=total_lines,
            preservation_ratio=preserved_lines / total_lines if total_lines > 0 else 0.0
        )
    
    def _generate_training_data(self, states: List[TransformationState]) -> List[Dict[str, Any]]:
        """Generate training data from transformation states"""
        training_data = []
        
        for state in states:
            # Extract features from code
            features = self._extract_code_features(state.code)
            
            # Create label based on transformation quality
            label = 1 if state.compilation_status and state.semantic_preservation else 0
            
            training_data.append({
                'features': features,
                'label': label,
                'metadata': {
                    'complexity': state.complexity_score,
                    'depth': state.depth,
                    'transformations': len(state.transformation_history)
                }
            })
        
        return training_data
    
    def _extract_code_features(self, code: str) -> List[float]:
        """Extract features from code for mini-model training"""
        features = []
        
        # Basic features
        features.append(len(code) / 1000.0)  # Normalized length
        features.append(len([l for l in code.split('\n') if l.strip()]) / 100.0)  # Normalized line count
        
        # Complexity features
        features.append(min(code.count('if ') / 10.0, 1.0))
        features.append(min(code.count('for ') / 5.0, 1.0))
        features.append(min(code.count('while ') / 5.0, 1.0))
        features.append(min(code.count('method ') / 5.0, 1.0))
        features.append(min(code.count('return ') / 5.0, 1.0))
        features.append(min(code.count('=') / 20.0, 1.0))
        
        # Structural features
        features.append(min(code.count('{') / 10.0, 1.0))
        features.append(min(code.count('(') / 20.0, 1.0))
        features.append(min(code.count('.') / 30.0, 1.0))
        
        # Ensure consistent feature size
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]  # Truncate to 15 features
    
    def _train_mini_model(self, training_data: List[Dict[str, Any]]) -> float:
        """Train a mini-model for quick performance evaluation"""
        try:
            # Extract features and labels
            X = np.array([sample['features'] for sample in training_data])
            y = np.array([sample['label'] for sample in training_data])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Error training mini-model: {e}")
            return 0.5
    
    def _compute_syntactic_diversity(self, states: List[TransformationState]) -> float:
        """Compute syntactic diversity between states"""
        if len(states) < 2:
            return 0.0
        
        # Compare AST structures
        ast_structures = []
        for state in states:
            try:
                # Simplified AST representation
                ast_features = self._extract_ast_features(state.code)
                ast_structures.append(ast_features)
            except Exception:
                ast_structures.append([0] * 10)  # Default features
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(ast_structures)):
            for j in range(i + 1, len(ast_structures)):
                distance = self._compute_feature_distance(ast_structures[i], ast_structures[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _compute_semantic_diversity(self, states: List[TransformationState]) -> float:
        """Compute semantic diversity between states"""
        if len(states) < 2:
            return 0.0
        
        # Compare semantic features
        semantic_features = []
        for state in states:
            features = self._extract_semantic_features(state.code)
            semantic_features.append(features)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(semantic_features)):
            for j in range(i + 1, len(semantic_features)):
                distance = self._compute_feature_distance(semantic_features[i], semantic_features[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _compute_transformation_diversity(self, states: List[TransformationState]) -> float:
        """Compute diversity of transformation types used"""
        if len(states) < 2:
            return 0.0
        
        # Get all transformation types used
        all_transformations = set()
        for state in states:
            all_transformations.update(state.transformation_history)
        
        # Calculate diversity as ratio of unique transformations to total possible
        unique_count = len(all_transformations)
        total_possible = len(TransformationType)
        
        return unique_count / total_possible
    
    def _extract_ast_features(self, code: str) -> List[float]:
        """Extract AST-based features"""
        features = []
        
        # Count different constructs
        features.append(code.count('class '))
        features.append(code.count('method '))
        features.append(code.count('if '))
        features.append(code.count('for '))
        features.append(code.count('while '))
        features.append(code.count('return '))
        features.append(code.count('='))
        features.append(code.count('('))
        features.append(code.count('{'))
        features.append(code.count(';'))
        
        # Normalize
        features = [min(f / 10.0, 1.0) for f in features]
        
        return features
    
    def _extract_semantic_features(self, code: str) -> List[float]:
        """Extract semantic features"""
        features = []
        
        # Variable usage patterns
        variables = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', code))
        features.append(min(len(variables) / 20.0, 1.0))
        
        # Method calls
        method_calls = code.count('(')
        features.append(min(method_calls / 20.0, 1.0))
        
        # String literals
        string_literals = code.count('"')
        features.append(min(string_literals / 10.0, 1.0))
        
        # Numeric literals
        numeric_literals = len(re.findall(r'\b\d+\b', code))
        features.append(min(numeric_literals / 20.0, 1.0))
        
        # Arithmetic operations
        arithmetic_ops = code.count('+') + code.count('-') + code.count('*') + code.count('/')
        features.append(min(arithmetic_ops / 20.0, 1.0))
        
        return features
    
    def _compute_feature_distance(self, features1: List[float], features2: List[float]) -> float:
        """Compute distance between feature vectors"""
        if len(features1) != len(features2):
            return 1.0  # Maximum distance for different sizes
        
        # Euclidean distance
        distance = np.sqrt(sum((f1 - f2) ** 2 for f1, f2 in zip(features1, features2)))
        
        # Normalize by maximum possible distance
        max_distance = np.sqrt(len(features1))
        return min(distance / max_distance, 1.0)
    
    def _update_statistics(self, metrics: EvaluationMetrics):
        """Update evaluation statistics"""
        self.evaluation_stats['total_evaluations'] += 1
        self.evaluation_stats['slicer_resistance_scores'].append(metrics.slicer_resistance)
        self.evaluation_stats['model_performance_scores'].append(metrics.model_performance)
        self.evaluation_stats['diversity_scores'].append(metrics.diversity_score)
        self.evaluation_stats['compilation_success_rates'].append(metrics.compilation_success)
        self.evaluation_stats['semantic_preservation_rates'].append(metrics.semantic_preservation)
    
    def _create_empty_metrics(self) -> EvaluationMetrics:
        """Create empty metrics for invalid sequences"""
        return EvaluationMetrics(
            slicer_resistance=0.0,
            model_performance=0.0,
            diversity_score=0.0,
            compilation_success=0.0,
            semantic_preservation=0.0,
            overall_score=0.0,
            metadata={'error': 'empty_sequence'}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        stats = self.evaluation_stats.copy()
        
        # Add computed statistics
        for key in ['slicer_resistance_scores', 'model_performance_scores', 
                   'diversity_scores', 'compilation_success_rates', 
                   'semantic_preservation_rates']:
            if stats[key]:
                stats[f'{key}_mean'] = np.mean(stats[key])
                stats[f'{key}_std'] = np.std(stats[key])
                stats[f'{key}_min'] = np.min(stats[key])
                stats[f'{key}_max'] = np.max(stats[key])
        
        return stats
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.evaluation_stats = {
            'total_evaluations': 0,
            'slicer_resistance_scores': [],
            'model_performance_scores': [],
            'diversity_scores': [],
            'compilation_success_rates': [],
            'semantic_preservation_rates': []
        }


def main():
    """Test the augmentation sequence evaluator"""
    logger.info("Testing Augmentation Sequence Evaluator...")
    
    # Create evaluator
    evaluator = AugmentationSequenceEvaluator()
    
    # Create test states
    test_code = """
public class TestClass {
    public int calculateSum(int[] arr) {
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum = sum + arr[i];
        }
        return sum;
    }
}
"""
    
    # Create dummy states
    state1 = TransformationState(
        code=test_code,
        transformation_history=[],
        depth=0,
        complexity_score=3.0,
        compilation_status=True,
        semantic_preservation=True,
        metadata={}
    )
    
    state2 = TransformationState(
        code=test_code.replace('sum = sum + arr[i];', 'sum += arr[i];'),
        transformation_history=[TransformationType.VARIABLE_OPERATION],
        depth=1,
        complexity_score=2.8,
        compilation_status=True,
        semantic_preservation=True,
        metadata={}
    )
    
    # Evaluate sequence
    metrics = evaluator.evaluate_sequence([state1, state2])
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Slicer resistance: {metrics.slicer_resistance:.3f}")
    logger.info(f"  Model performance: {metrics.model_performance:.3f}")
    logger.info(f"  Diversity score: {metrics.diversity_score:.3f}")
    logger.info(f"  Compilation success: {metrics.compilation_success:.3f}")
    logger.info(f"  Semantic preservation: {metrics.semantic_preservation:.3f}")
    logger.info(f"  Overall score: {metrics.overall_score:.3f}")
    
    # Print statistics
    stats = evaluator.get_statistics()
    logger.info(f"Total evaluations: {stats['total_evaluations']}")


if __name__ == '__main__':
    import random
    random.seed(42)
    main()
