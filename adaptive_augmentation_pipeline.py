#!/usr/bin/env python3
"""
Adaptive Augmentation Pipeline Integration

This module integrates learned augmentation policies into the main pipeline,
providing fallback to default augmentation and continuous learning capabilities.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import time

from recursive_augmentation_engine import (
    RecursiveAugmentationEngine, TransformationType, TransformationState
)
from augmentation_policy_learner import (
    ReinforcementLearningPolicy, MCTSAugmentationSearch, EvolutionaryAugmentationOptimizer
)
from transformation_policy_gnn import TransformationPolicyGNN, RandomWalkAgent
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator, EvaluationMetrics

logger = logging.getLogger(__name__)

@dataclass
class AugmentationPolicy:
    """Represents a learned augmentation policy"""
    name: str
    method: str  # 'rl', 'mcts', 'evolutionary', 'gnn', 'random'
    model_path: Optional[str]
    performance_score: float
    metadata: Dict[str, Any]

@dataclass
class AugmentationResult:
    """Result of augmentation process"""
    original_code: str
    augmented_variants: List[str]
    transformation_sequences: List[List[TransformationType]]
    evaluation_metrics: List[EvaluationMetrics]
    policy_used: str
    processing_time: float
    metadata: Dict[str, Any]

class AdaptiveAugmentationPipeline:
    """Main pipeline for adaptive augmentation with learned policies"""
    
    def __init__(self, config_path: Optional[str] = None, device: str = 'auto', config_dict: Optional[Dict[str, Any]] = None):
        self.device = self._setup_device(device)
        
        # Load configuration
        if config_dict:
            self.config = self._load_config_from_dict(config_dict)
        else:
            self.config = self._load_config(config_path)
        
        # Initialize components
        self.engine = RecursiveAugmentationEngine(seed=self.config.get('seed', 42))
        self.evaluator = AugmentationSequenceEvaluator(device=self.device)
        
        # Initialize policy learners
        self.policy_learners = self._initialize_policy_learners()
        
        # Load or create policies
        self.policies = self._load_policies()
        
        # Current active policy
        self.active_policy = None
        
        # Performance tracking
        self.performance_history = []
        self.ab_testing_results = {}
        
        # Statistics
        self.stats = {
            'total_augmentations': 0,
            'policy_usage': {},
            'average_performance': {},
            'fallback_usage': 0,
            'online_learning_updates': 0
        }
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'method': 'rl',  # 'rl', 'mcts', 'evolutionary', 'gnn', 'random'
            'max_recursion_depth': 3,
            'num_variants': 10,
            'enable_online_learning': True,
            'exploration_rate': 0.1,
            'reward_weights': {
                'accuracy': 0.4,
                'slicer_resistance': 0.3,
                'diversity': 0.2,
                'compilation': 0.1
            },
            'policy_models_dir': 'models/augmentation_policies',
            'ab_testing_enabled': True,
            'fallback_threshold': 0.5,
            'seed': 42
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
        
        return default_config
    
    def _load_config_from_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from dictionary"""
        # Start with default config
        config = self._load_config(None)  # Get default config
        
        # Update with provided config
        config.update(config_dict)
        
        return config
    
    def _initialize_policy_learners(self) -> Dict[str, Any]:
        """Initialize policy learners"""
        learners = {}
        
        try:
            # RL Policy Learner
            learners['rl'] = ReinforcementLearningPolicy(
                device=self.device,
                learning_rate=self.config.get('rl_learning_rate', 3e-4)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize RL learner: {e}")
        
        try:
            # MCTS Search
            learners['mcts'] = MCTSAugmentationSearch(
                exploration_constant=self.config.get('mcts_exploration', 1.414),
                max_iterations=self.config.get('mcts_iterations', 1000)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize MCTS: {e}")
        
        try:
            # Evolutionary Optimizer
            learners['evolutionary'] = EvolutionaryAugmentationOptimizer(
                population_size=self.config.get('evo_population_size', 50),
                mutation_rate=self.config.get('evo_mutation_rate', 0.1)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Evolutionary optimizer: {e}")
        
        try:
            # GNN Policy
            learners['gnn'] = TransformationPolicyGNN(
                hidden_dim=self.config.get('gnn_hidden_dim', 256)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize GNN policy: {e}")
        
        return learners
    
    def _load_policies(self) -> Dict[str, AugmentationPolicy]:
        """Load existing policies"""
        policies = {}
        models_dir = self.config.get('policy_models_dir', 'models/augmentation_policies')
        
        if os.path.exists(models_dir):
            for method in ['rl', 'mcts', 'evolutionary', 'gnn']:
                model_path = os.path.join(models_dir, f'{method}_policy.pth')
                if os.path.exists(model_path):
                    try:
                        policy = AugmentationPolicy(
                            name=f'{method}_policy',
                            method=method,
                            model_path=model_path,
                            performance_score=self._load_policy_performance(model_path),
                            metadata={'loaded_from': model_path}
                        )
                        policies[method] = policy
                        logger.info(f"Loaded {method} policy from {model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load {method} policy: {e}")
        
        # Add random policy as fallback
        policies['random'] = AugmentationPolicy(
            name='random_policy',
            method='random',
            model_path=None,
            performance_score=0.5,  # Baseline performance
            metadata={'fallback': True}
        )
        
        return policies
    
    def _load_policy_performance(self, model_path: str) -> float:
        """Load performance score for a policy"""
        try:
            # Try to load performance metadata
            metadata_path = model_path.replace('.pth', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('performance_score', 0.5)
        except Exception:
            pass
        
        return 0.5  # Default performance
    
    def generate_augmented_variants(self, code: str, num_variants: Optional[int] = None,
                                  policy_method: Optional[str] = None) -> AugmentationResult:
        """Generate augmented variants using learned policy"""
        start_time = time.time()
        
        if num_variants is None:
            num_variants = self.config.get('num_variants', 10)
        
        if policy_method is None:
            policy_method = self.config.get('method', 'rl')
        
        # Select policy
        policy = self._select_policy(policy_method)
        
        # Generate variants
        augmented_variants = []
        transformation_sequences = []
        evaluation_metrics = []
        
        for i in range(num_variants):
            try:
                # Generate variant using selected policy
                variant_result = self._generate_single_variant(code, policy, i)
                
                if variant_result:
                    augmented_variants.append(variant_result['code'])
                    transformation_sequences.append(variant_result['sequence'])
                    evaluation_metrics.append(variant_result['metrics'])
                
            except Exception as e:
                logger.warning(f"Error generating variant {i}: {e}")
                continue
        
        # Fallback if no variants generated
        if not augmented_variants:
            logger.warning("No variants generated, using fallback")
            augmented_variants = self._generate_fallback_variants(code, num_variants)
            transformation_sequences = [[] for _ in augmented_variants]
            evaluation_metrics = [self.evaluator._create_empty_metrics() for _ in augmented_variants]
            self.stats['fallback_usage'] += 1
        
        processing_time = time.time() - start_time
        
        # Create result
        result = AugmentationResult(
            original_code=code,
            augmented_variants=augmented_variants,
            transformation_sequences=transformation_sequences,
            evaluation_metrics=evaluation_metrics,
            policy_used=policy.name,
            processing_time=processing_time,
            metadata={
                'num_variants_requested': num_variants,
                'num_variants_generated': len(augmented_variants),
                'success_rate': len(augmented_variants) / num_variants if num_variants > 0 else 0
            }
        )
        
        # Update statistics
        self._update_statistics(result, policy)
        
        # Online learning update
        if self.config.get('enable_online_learning', True):
            self._update_policy_online(result, policy)
        
        return result
    
    def _select_policy(self, method: str) -> AugmentationPolicy:
        """Select augmentation policy"""
        if method in self.policies:
            policy = self.policies[method]
            
            # Check if policy meets performance threshold
            threshold = self.config.get('fallback_threshold', 0.5)
            if policy.performance_score >= threshold:
                self.active_policy = policy
                return policy
            else:
                logger.warning(f"Policy {method} below threshold ({policy.performance_score} < {threshold})")
        
        # Fallback to random policy
        logger.info("Using fallback random policy")
        self.active_policy = self.policies['random']
        return self.policies['random']
    
    def _generate_single_variant(self, code: str, policy: AugmentationPolicy, 
                               variant_idx: int) -> Optional[Dict[str, Any]]:
        """Generate a single augmented variant using policy"""
        try:
            if policy.method == 'rl':
                return self._generate_with_rl_policy(code, policy, variant_idx)
            elif policy.method == 'mcts':
                return self._generate_with_mcts_policy(code, policy, variant_idx)
            elif policy.method == 'evolutionary':
                return self._generate_with_evolutionary_policy(code, policy, variant_idx)
            elif policy.method == 'gnn':
                return self._generate_with_gnn_policy(code, policy, variant_idx)
            elif policy.method == 'random':
                return self._generate_with_random_policy(code, policy, variant_idx)
            else:
                logger.warning(f"Unknown policy method: {policy.method}")
                return None
                
        except Exception as e:
            logger.warning(f"Error generating variant with {policy.method}: {e}")
            return None
    
    def _generate_with_rl_policy(self, code: str, policy: AugmentationPolicy, 
                               variant_idx: int) -> Optional[Dict[str, Any]]:
        """Generate variant using RL policy"""
        # Load RL model if available
        if policy.model_path and os.path.exists(policy.model_path):
            try:
                learner = self.policy_learners.get('rl')
                if learner:
                    # Load trained model
                    learner.policy_net.load_state_dict(torch.load(policy.model_path, map_location=self.device))
                    learner.policy_net.eval()
                    
                    # Generate sequence using RL policy
                    sequence = self._generate_sequence_with_rl(code, learner)
                else:
                    sequence = self._generate_random_sequence()
            except Exception as e:
                logger.warning(f"Error loading RL model: {e}")
                sequence = self._generate_random_sequence()
        else:
            sequence = self._generate_random_sequence()
        
        # Apply sequence
        states = self.engine.apply_recursive_transformation(
            code, max_depth=len(sequence), transformation_sequence=sequence
        )
        
        if len(states) > 1:
            final_state = states[-1]
            metrics = self.evaluator.evaluate_sequence(states)
            
            return {
                'code': final_state.code,
                'sequence': sequence,
                'metrics': metrics
            }
        
        return None
    
    def _generate_with_mcts_policy(self, code: str, policy: AugmentationPolicy, 
                                 variant_idx: int) -> Optional[Dict[str, Any]]:
        """Generate variant using MCTS policy"""
        try:
            mcts = self.policy_learners.get('mcts')
            if mcts:
                # Create initial state
                initial_state = TransformationState(
                    code=code,
                    transformation_history=[],
                    depth=0,
                    complexity_score=self.engine._compute_complexity_score(code),
                    compilation_status=True,
                    semantic_preservation=True,
                    metadata={}
                )
                
                # Run MCTS search
                sequence = mcts.search(initial_state, self.engine, self.evaluator)
                
                # Apply sequence
                states = self.engine.apply_recursive_transformation(
                    code, max_depth=len(sequence), transformation_sequence=sequence
                )
                
                if len(states) > 1:
                    final_state = states[-1]
                    metrics = self.evaluator.evaluate_sequence(states)
                    
                    return {
                        'code': final_state.code,
                        'sequence': sequence,
                        'metrics': metrics
                    }
        except Exception as e:
            logger.warning(f"MCTS generation failed: {e}")
        
        return None
    
    def _generate_with_evolutionary_policy(self, code: str, policy: AugmentationPolicy, 
                                         variant_idx: int) -> Optional[Dict[str, Any]]:
        """Generate variant using evolutionary policy"""
        try:
            evo = self.policy_learners.get('evolutionary')
            if evo:
                # Run evolutionary optimization
                best_genome = evo.optimize(code, self.engine, self.evaluator, max_generations=10)
                
                # Apply best sequence
                states = self.engine.apply_recursive_transformation(
                    code, 
                    max_depth=len(best_genome.sequence),
                    transformation_sequence=best_genome.sequence
                )
                
                if len(states) > 1:
                    final_state = states[-1]
                    metrics = self.evaluator.evaluate_sequence(states)
                    
                    return {
                        'code': final_state.code,
                        'sequence': best_genome.sequence,
                        'metrics': metrics
                    }
        except Exception as e:
            logger.warning(f"Evolutionary generation failed: {e}")
        
        return None
    
    def _generate_with_gnn_policy(self, code: str, policy: AugmentationPolicy, 
                                variant_idx: int) -> Optional[Dict[str, Any]]:
        """Generate variant using GNN policy"""
        try:
            gnn = self.policy_learners.get('gnn')
            if gnn and policy.model_path and os.path.exists(policy.model_path):
                # Load trained GNN model
                gnn.load_state_dict(torch.load(policy.model_path, map_location=self.device))
                gnn.eval()
                
                # Create random walk agent
                agent = RandomWalkAgent(gnn, exploration_rate=self.config.get('exploration_rate', 0.1))
                
                # Generate sequence using GNN agent
                sequence = self._generate_sequence_with_gnn(code, agent)
                
                # Apply sequence
                states = self.engine.apply_recursive_transformation(
                    code, max_depth=len(sequence), transformation_sequence=sequence
                )
                
                if len(states) > 1:
                    final_state = states[-1]
                    metrics = self.evaluator.evaluate_sequence(states)
                    
                    return {
                        'code': final_state.code,
                        'sequence': sequence,
                        'metrics': metrics
                    }
        except Exception as e:
            logger.warning(f"GNN generation failed: {e}")
        
        return None
    
    def _generate_with_random_policy(self, code: str, policy: AugmentationPolicy, 
                                   variant_idx: int) -> Optional[Dict[str, Any]]:
        """Generate variant using random policy"""
        sequence = self._generate_random_sequence()
        
        # Apply sequence
        states = self.engine.apply_recursive_transformation(
            code, max_depth=len(sequence), transformation_sequence=sequence
        )
        
        if len(states) > 1:
            final_state = states[-1]
            metrics = self.evaluator.evaluate_sequence(states)
            
            return {
                'code': final_state.code,
                'sequence': sequence,
                'metrics': metrics
            }
        
        return None
    
    def _generate_sequence_with_rl(self, code: str, learner) -> List[TransformationType]:
        """Generate transformation sequence using RL policy"""
        sequence = []
        current_code = code
        max_depth = self.config.get('max_recursion_depth', 3)
        
        for depth in range(max_depth):
            # Create current state
            current_state = TransformationState(
                code=current_code,
                transformation_history=sequence,
                depth=depth,
                complexity_score=self.engine._compute_complexity_score(current_code),
                compilation_status=True,
                semantic_preservation=True,
                metadata={}
            )
            
            # Get valid actions
            valid_actions = self.engine.get_valid_next_transformations(current_state)
            if not valid_actions:
                break
            
            # Select action using RL policy
            action = learner.select_action(current_state, valid_actions)
            sequence.append(action)
            
            # Apply action
            states = self.engine.apply_recursive_transformation(
                current_code, max_depth=1, transformation_sequence=[action]
            )
            
            if len(states) > 1:
                current_code = states[1].code
            else:
                break
        
        return sequence
    
    def _generate_sequence_with_gnn(self, code: str, agent) -> List[TransformationType]:
        """Generate transformation sequence using GNN agent"""
        sequence = []
        current_code = code
        max_depth = self.config.get('max_recursion_depth', 3)
        
        for depth in range(max_depth):
            # Create current state
            current_state = TransformationState(
                code=current_code,
                transformation_history=sequence,
                depth=depth,
                complexity_score=self.engine._compute_complexity_score(current_code),
                compilation_status=True,
                semantic_preservation=True,
                metadata={}
            )
            
            # Get valid actions
            valid_actions = self.engine.get_valid_next_transformations(current_state)
            if not valid_actions:
                break
            
            # Select action using GNN agent
            action = agent.select_action(current_state, valid_actions)
            sequence.append(action)
            
            # Apply action
            states = self.engine.apply_recursive_transformation(
                current_code, max_depth=1, transformation_sequence=[action]
            )
            
            if len(states) > 1:
                current_code = states[1].code
            else:
                break
        
        return sequence
    
    def _generate_random_sequence(self) -> List[TransformationType]:
        """Generate random transformation sequence"""
        max_length = min(self.config.get('max_recursion_depth', 3), 5)
        sequence_length = np.random.randint(2, max_length + 1)
        
        sequence = []
        for _ in range(sequence_length):
            action = np.random.choice(list(TransformationType))
            sequence.append(action)
        
        return sequence
    
    def _generate_fallback_variants(self, code: str, num_variants: int) -> List[str]:
        """Generate fallback variants using basic transformations"""
        variants = []
        
        for i in range(num_variants):
            # Apply simple random transformations
            states = self.engine.apply_recursive_transformation(
                code, max_depth=2, transformation_sequence=None
            )
            
            if len(states) > 1:
                variants.append(states[-1].code)
            else:
                variants.append(code)  # Fallback to original
        
        return variants
    
    def _update_statistics(self, result: AugmentationResult, policy: AugmentationPolicy):
        """Update pipeline statistics"""
        self.stats['total_augmentations'] += 1
        
        # Update policy usage
        if policy.name not in self.stats['policy_usage']:
            self.stats['policy_usage'][policy.name] = 0
        self.stats['policy_usage'][policy.name] += 1
        
        # Update average performance
        if result.evaluation_metrics:
            avg_score = np.mean([m.overall_score for m in result.evaluation_metrics])
            if policy.name not in self.stats['average_performance']:
                self.stats['average_performance'][policy.name] = []
            self.stats['average_performance'][policy.name].append(avg_score)
    
    def _update_policy_online(self, result: AugmentationResult, policy: AugmentationPolicy):
        """Update policy using online learning"""
        if not self.config.get('enable_online_learning', True):
            return
        
        try:
            # Create training episode from result
            if result.evaluation_metrics and result.transformation_sequences:
                # Simple online learning update
                # In practice, you'd accumulate more data before updating
                performance_scores = [m.overall_score for m in result.evaluation_metrics]
                avg_performance = np.mean(performance_scores)
                
                # Update policy performance score
                policy.performance_score = 0.9 * policy.performance_score + 0.1 * avg_performance
                
                self.stats['online_learning_updates'] += 1
                
                logger.debug(f"Updated {policy.name} performance: {policy.performance_score:.3f}")
                
        except Exception as e:
            logger.warning(f"Error in online learning update: {e}")
    
    def compare_with_baseline(self, test_code: str, num_trials: int = 10) -> Dict[str, Any]:
        """Compare learned policies with baseline random policy"""
        if not self.config.get('ab_testing_enabled', True):
            return {}
        
        results = {}
        
        for method in ['rl', 'mcts', 'evolutionary', 'gnn']:
            if method in self.policies:
                scores = []
                times = []
                
                for trial in range(num_trials):
                    start_time = time.time()
                    result = self.generate_augmented_variants(test_code, num_variants=5, policy_method=method)
                    trial_time = time.time() - start_time
                    
                    if result.evaluation_metrics:
                        avg_score = np.mean([m.overall_score for m in result.evaluation_metrics])
                        scores.append(avg_score)
                        times.append(trial_time)
                
                results[method] = {
                    'average_score': np.mean(scores) if scores else 0.0,
                    'std_score': np.std(scores) if scores else 0.0,
                    'average_time': np.mean(times) if times else 0.0,
                    'num_trials': len(scores)
                }
        
        # Compare with random baseline
        random_scores = []
        random_times = []
        
        for trial in range(num_trials):
            start_time = time.time()
            result = self.generate_augmented_variants(test_code, num_variants=5, policy_method='random')
            trial_time = time.time() - start_time
            
            if result.evaluation_metrics:
                avg_score = np.mean([m.overall_score for m in result.evaluation_metrics])
                random_scores.append(avg_score)
                random_times.append(trial_time)
        
        results['random'] = {
            'average_score': np.mean(random_scores) if random_scores else 0.0,
            'std_score': np.std(random_scores) if random_scores else 0.0,
            'average_time': np.mean(random_times) if random_times else 0.0,
            'num_trials': len(random_scores)
        }
        
        self.ab_testing_results = results
        return results
    
    def save_policy(self, policy_name: str, model_path: Optional[str] = None):
        """Save current policy"""
        if not model_path:
            models_dir = self.config.get('policy_models_dir', 'models/augmentation_policies')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f'{policy_name}_policy.pth')
        
        try:
            if policy_name in self.policies:
                policy = self.policies[policy_name]
                
                # Save model if applicable
                if policy.model_path and os.path.exists(policy.model_path):
                    # Copy existing model
                    import shutil
                    shutil.copy2(policy.model_path, model_path)
                elif policy_name in self.policy_learners:
                    # Save current learner state
                    learner = self.policy_learners[policy_name]
                    if hasattr(learner, 'policy_net'):
                        torch.save(learner.policy_net.state_dict(), model_path)
                    elif hasattr(learner, 'state_dict'):
                        torch.save(learner.state_dict(), model_path)
                
                # Save metadata
                metadata_path = model_path.replace('.pth', '_metadata.json')
                metadata = {
                    'name': policy.name,
                    'method': policy.method,
                    'performance_score': policy.performance_score,
                    'metadata': policy.metadata,
                    'saved_at': time.time()
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Saved policy {policy_name} to {model_path}")
                
        except Exception as e:
            logger.error(f"Error saving policy {policy_name}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        
        # Add computed statistics
        for policy_name, scores in stats['average_performance'].items():
            if scores:
                stats[f'{policy_name}_performance_mean'] = np.mean(scores)
                stats[f'{policy_name}_performance_std'] = np.std(scores)
        
        # Add A/B testing results
        if self.ab_testing_results:
            stats['ab_testing_results'] = self.ab_testing_results
        
        return stats


def main():
    """Test the adaptive augmentation pipeline"""
    logger.info("Testing Adaptive Augmentation Pipeline...")
    
    # Test code
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
    
    # Create pipeline
    pipeline = AdaptiveAugmentationPipeline(device='cpu')
    
    # Test random policy
    logger.info("Testing random policy...")
    result = pipeline.generate_augmented_variants(test_code, num_variants=3, policy_method='random')
    
    logger.info(f"Generated {len(result.augmented_variants)} variants")
    logger.info(f"Policy used: {result.policy_used}")
    logger.info(f"Processing time: {result.processing_time:.3f}s")
    
    if result.evaluation_metrics:
        avg_score = np.mean([m.overall_score for m in result.evaluation_metrics])
        logger.info(f"Average evaluation score: {avg_score:.3f}")
    
    # Test A/B comparison
    logger.info("Running A/B test...")
    ab_results = pipeline.compare_with_baseline(test_code, num_trials=3)
    
    for method, results in ab_results.items():
        logger.info(f"{method}: score={results['average_score']:.3f}±{results['std_score']:.3f}, "
                   f"time={results['average_time']:.3f}s")
    
    # Print statistics
    stats = pipeline.get_statistics()
    logger.info(f"Total augmentations: {stats['total_augmentations']}")
    logger.info(f"Policy usage: {stats['policy_usage']}")


if __name__ == '__main__':
    import random
    random.seed(42)
    main()
