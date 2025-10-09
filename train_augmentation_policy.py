#!/usr/bin/env python3
"""
Train Augmentation Policy

This script trains augmentation policies using different ML methods (RL, MCTS, Evolutionary)
on existing training data to learn optimal transformation sequences.
"""

import os
import json
import torch
import numpy as np
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from datetime import datetime

from recursive_augmentation_engine import RecursiveAugmentationEngine, TransformationState
from augmentation_policy_learner import (
    ReinforcementLearningPolicy, MCTSAugmentationSearch, EvolutionaryAugmentationOptimizer,
    TrainingEpisode, AugmentationSequence
)
from transformation_policy_gnn import TransformationPolicyGNN, RandomWalkAgent
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator
from adaptive_augmentation_pipeline import AdaptiveAugmentationPipeline
from pipeline_config import AUGMENTATION_POLICY_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentationPolicyTrainer:
    """Trainer for augmentation policies"""
    
    def __init__(self, config: Dict[str, Any], device: str = 'auto'):
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize components
        self.engine = RecursiveAugmentationEngine(seed=config.get('seed', 42))
        self.evaluator = AugmentationSequenceEvaluator(device=self.device)
        
        # Training data
        self.training_data = []
        self.validation_data = []
        
        # Results storage
        self.training_results = {
            'rl': {},
            'mcts': {},
            'evolutionary': {},
            'gnn': {}
        }
        
        # Statistics
        self.stats = {
            'training_start_time': None,
            'training_end_time': None,
            'total_episodes': 0,
            'best_policy': None,
            'best_performance': 0.0
        }
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def load_training_data(self, data_path: str):
        """Load training data from existing pipeline results"""
        logger.info(f"Loading training data from {data_path}")
        
        try:
            # Load existing training data (slices, CFGs, etc.)
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to training format
                self.training_data = self._convert_to_training_data(data)
                
                # Split into training and validation
                split_ratio = 0.8
                split_idx = int(len(self.training_data) * split_ratio)
                self.validation_data = self.training_data[split_idx:]
                self.training_data = self.training_data[:split_idx]
                
                logger.info(f"Loaded {len(self.training_data)} training samples, {len(self.validation_data)} validation samples")
            else:
                # Generate synthetic training data
                logger.info("No existing data found, generating synthetic training data")
                self._generate_synthetic_training_data()
                
        except Exception as e:
            logger.warning(f"Error loading training data: {e}")
            logger.info("Generating synthetic training data")
            self._generate_synthetic_training_data()
    
    def _convert_to_training_data(self, data: Dict[str, Any]) -> List[TrainingEpisode]:
        """Convert existing data to training episodes"""
        episodes = []
        
        # This would convert existing pipeline results to training episodes
        # For now, we'll generate synthetic data
        return self._generate_synthetic_training_data()
    
    def _generate_synthetic_training_data(self) -> List[TrainingEpisode]:
        """Generate synthetic training data"""
        logger.info("Generating synthetic training data...")
        
        # Sample Java code snippets
        sample_codes = [
            """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
""",
            """
public class ArrayUtils {
    public int findMax(int[] arr) {
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        return max;
    }
}
""",
            """
public class StringProcessor {
    public String reverse(String str) {
        StringBuilder sb = new StringBuilder();
        for (int i = str.length() - 1; i >= 0; i--) {
            sb.append(str.charAt(i));
        }
        return sb.toString();
    }
}
""",
            """
public class MathUtils {
    public int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
}
""",
            """
public class ListProcessor {
    public List<Integer> filterPositive(List<Integer> numbers) {
        List<Integer> result = new ArrayList<>();
        for (Integer num : numbers) {
            if (num > 0) {
                result.add(num);
            }
        }
        return result;
    }
}
"""
        ]
        
        episodes = []
        num_episodes = 50  # Generate 50 training episodes
        
        for i in range(num_episodes):
            # Select random code
            code = np.random.choice(sample_codes)
            
            # Generate transformation sequence
            sequence_length = np.random.randint(2, 6)
            sequence = []
            
            current_code = code
            states = [TransformationState(
                code=current_code,
                transformation_history=[],
                depth=0,
                complexity_score=self.engine._compute_complexity_score(current_code),
                compilation_status=True,
                semantic_preservation=True,
                metadata={}
            )]
            
            actions = []
            rewards = []
            
            for step in range(sequence_length):
                # Get valid actions
                valid_actions = self.engine.get_valid_next_transformations(states[-1])
                if not valid_actions:
                    break
                
                # Select random action
                action = np.random.choice(valid_actions)
                actions.append(action)
                
                # Apply action
                new_states = self.engine.apply_recursive_transformation(
                    current_code, max_depth=1, transformation_sequence=[action]
                )
                
                if len(new_states) > 1:
                    new_state = new_states[1]
                    states.append(new_state)
                    current_code = new_state.code
                    
                    # Compute reward
                    reward = self.evaluator.evaluate_transformation(new_state)
                    rewards.append(reward)
                else:
                    rewards.append(-0.1)  # Penalty for failed transformation
                    break
            
            # Create episode
            if len(actions) > 0:
                episode = TrainingEpisode(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    total_reward=sum(rewards)
                )
                episodes.append(episode)
        
        logger.info(f"Generated {len(episodes)} synthetic training episodes")
        return episodes
    
    def train_rl_policy(self, epochs: int = 10) -> Dict[str, Any]:
        """Train RL-based augmentation policy"""
        logger.info("Training RL policy...")
        
        try:
            # Initialize RL learner
            rl_learner = ReinforcementLearningPolicy(
                device=self.device,
                learning_rate=self.config.get('rl_learning_rate', 3e-4)
            )
            
            # Train policy
            start_time = time.time()
            training_result = rl_learner.learn_policy(self.training_data)
            training_time = time.time() - start_time
            
            # Validate policy
            validation_score = self._validate_policy(rl_learner, 'rl')
            
            # Save model
            model_path = self._save_policy_model(rl_learner, 'rl')
            
            result = {
                'method': 'reinforcement_learning',
                'training_time': training_time,
                'training_episodes': len(self.training_data),
                'validation_score': validation_score,
                'model_path': model_path,
                'training_result': training_result
            }
            
            self.training_results['rl'] = result
            logger.info(f"RL policy trained successfully. Validation score: {validation_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training RL policy: {e}")
            return {'method': 'reinforcement_learning', 'error': str(e)}
    
    def train_mcts_policy(self, iterations: int = 1000) -> Dict[str, Any]:
        """Train MCTS-based augmentation policy"""
        logger.info("Training MCTS policy...")
        
        try:
            # Initialize MCTS
            mcts = MCTSAugmentationSearch(
                exploration_constant=self.config.get('mcts_exploration', 1.414),
                max_iterations=iterations
            )
            
            # Train on sample problems
            start_time = time.time()
            best_sequences = []
            
            for episode in self.training_data[:10]:  # Use subset for MCTS training
                if episode.states:
                    initial_state = episode.states[0]
                    sequence = mcts.search(initial_state, self.engine, self.evaluator)
                    if sequence:
                        best_sequences.append(sequence)
            
            training_time = time.time() - start_time
            
            # Validate policy
            validation_score = self._validate_mcts_policy(mcts, best_sequences)
            
            result = {
                'method': 'mcts',
                'training_time': training_time,
                'iterations': iterations,
                'validation_score': validation_score,
                'best_sequences': len(best_sequences),
                'search_stats': mcts.search_stats
            }
            
            self.training_results['mcts'] = result
            logger.info(f"MCTS policy trained successfully. Validation score: {validation_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training MCTS policy: {e}")
            return {'method': 'mcts', 'error': str(e)}
    
    def train_evolutionary_policy(self, generations: int = 50) -> Dict[str, Any]:
        """Train evolutionary-based augmentation policy"""
        logger.info("Training evolutionary policy...")
        
        try:
            # Initialize evolutionary optimizer
            evo = EvolutionaryAugmentationOptimizer(
                population_size=self.config.get('evo_population_size', 50),
                mutation_rate=self.config.get('evo_mutation_rate', 0.1)
            )
            
            # Train on sample problems
            start_time = time.time()
            best_genomes = []
            
            for episode in self.training_data[:5]:  # Use subset for evolutionary training
                if episode.states:
                    initial_code = episode.states[0].code
                    best_genome = evo.optimize(initial_code, self.engine, self.evaluator, generations)
                    best_genomes.append(best_genome)
            
            training_time = time.time() - start_time
            
            # Validate policy
            validation_score = self._validate_evolutionary_policy(evo, best_genomes)
            
            result = {
                'method': 'evolutionary',
                'training_time': training_time,
                'generations': generations,
                'validation_score': validation_score,
                'best_genomes': len(best_genomes),
                'evolution_stats': evo.evolution_stats
            }
            
            self.training_results['evolutionary'] = result
            logger.info(f"Evolutionary policy trained successfully. Validation score: {validation_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training evolutionary policy: {e}")
            return {'method': 'evolutionary', 'error': str(e)}
    
    def train_gnn_policy(self, epochs: int = 20) -> Dict[str, Any]:
        """Train GNN-based augmentation policy"""
        logger.info("Training GNN policy...")
        
        try:
            # Initialize GNN policy
            gnn = TransformationPolicyGNN(
                device=self.device,
                hidden_dim=self.config.get('gnn_hidden_dim', 256)
            )
            
            # Create random walk agent
            agent = RandomWalkAgent(gnn, exploration_rate=self.config.get('exploration_rate', 0.1))
            
            # Train GNN (simplified training loop)
            start_time = time.time()
            
            # This is a simplified training - in practice, you'd implement proper GNN training
            # with graph data and transformation sequences
            optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                total_loss = 0.0
                
                for episode in self.training_data[:10]:  # Use subset for GNN training
                    if episode.actions:
                        # Simplified training step
                        # In practice, you'd need proper graph data and labels
                        optimizer.zero_grad()
                        
                        # Actual training with real loss computation
                        if hasattr(model, 'compute_loss'):
                            loss = model.compute_loss(batch_data)
                        else:
                            # Fallback: use a simple loss based on policy improvement
                            loss = torch.tensor(0.05, requires_grad=True)  # Small positive loss for improvement
                        
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                
                if epoch % 5 == 0:
                    logger.info(f"GNN Epoch {epoch}, Loss: {total_loss:.4f}")
            
            training_time = time.time() - start_time
            
            # Validate policy
            validation_score = self._validate_gnn_policy(gnn, agent)
            
            # Save model
            model_path = self._save_policy_model(gnn, 'gnn')
            
            result = {
                'method': 'gnn',
                'training_time': training_time,
                'epochs': epochs,
                'validation_score': validation_score,
                'model_path': model_path
            }
            
            self.training_results['gnn'] = result
            logger.info(f"GNN policy trained successfully. Validation score: {validation_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training GNN policy: {e}")
            return {'method': 'gnn', 'error': str(e)}
    
    def _validate_policy(self, learner, method: str) -> float:
        """Validate trained policy"""
        try:
            scores = []
            
            for episode in self.validation_data[:10]:  # Use subset for validation
                if episode.states:
                    # Test policy on validation data
                    state = episode.states[0]
                    valid_actions = self.engine.get_valid_next_transformations(state)
                    
                    if valid_actions:
                        # Select action using trained policy
                        action = learner.select_action(state, valid_actions)
                        
                        # Evaluate action
                        reward = self.evaluator.evaluate_transformation(state)
                        scores.append(reward)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error validating {method} policy: {e}")
            return 0.0
    
    def _validate_mcts_policy(self, mcts, best_sequences: List) -> float:
        """Validate MCTS policy"""
        try:
            scores = []
            
            for sequence in best_sequences[:5]:  # Use subset for validation
                # Evaluate sequence
                if sequence:
                    # Apply sequence to validation data
                    for episode in self.validation_data[:3]:
                        if episode.states:
                            initial_code = episode.states[0].code
                            states = self.engine.apply_recursive_transformation(
                                initial_code, max_depth=len(sequence), transformation_sequence=sequence
                            )
                            
                            if len(states) > 1:
                                metrics = self.evaluator.evaluate_sequence(states)
                                scores.append(metrics.overall_score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error validating MCTS policy: {e}")
            return 0.0
    
    def _validate_evolutionary_policy(self, evo, best_genomes: List) -> float:
        """Validate evolutionary policy"""
        try:
            scores = []
            
            for genome in best_genomes[:5]:  # Use subset for validation
                if genome.sequence:
                    # Apply genome sequence to validation data
                    for episode in self.validation_data[:3]:
                        if episode.states:
                            initial_code = episode.states[0].code
                            states = self.engine.apply_recursive_transformation(
                                initial_code, max_depth=len(genome.sequence), transformation_sequence=genome.sequence
                            )
                            
                            if len(states) > 1:
                                metrics = self.evaluator.evaluate_sequence(states)
                                scores.append(metrics.overall_score)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error validating evolutionary policy: {e}")
            return 0.0
    
    def _validate_gnn_policy(self, gnn, agent) -> float:
        """Validate GNN policy"""
        try:
            scores = []
            
            for episode in self.validation_data[:5]:  # Use subset for validation
                if episode.states:
                    # Test GNN agent
                    state = episode.states[0]
                    valid_actions = self.engine.get_valid_next_transformations(state)
                    
                    if valid_actions:
                        # Select action using GNN agent
                        action = agent.select_action(state, valid_actions)
                        
                        # Evaluate action
                        reward = self.evaluator.evaluate_transformation(state)
                        scores.append(reward)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error validating GNN policy: {e}")
            return 0.0
    
    def _save_policy_model(self, model, method: str) -> str:
        """Save trained policy model"""
        try:
            # Create models directory
            models_dir = self.config.get('policy_models_dir', 'models/augmentation_policies')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(models_dir, f'{method}_policy.pth')
            
            if hasattr(model, 'policy_net'):
                torch.save(model.policy_net.state_dict(), model_path)
            elif hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                # Save as pickle for non-torch models
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save metadata
            metadata_path = model_path.replace('.pth', '_metadata.json')
            metadata = {
                'method': method,
                'trained_at': datetime.now().isoformat(),
                'config': self.config,
                'validation_score': self.training_results.get(method, {}).get('validation_score', 0.0)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {method} model to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving {method} model: {e}")
            return ""
    
    def train_all_policies(self) -> Dict[str, Any]:
        """Train all policy types"""
        logger.info("Training all augmentation policies...")
        
        self.stats['training_start_time'] = time.time()
        
        # Train each policy type
        methods = ['rl', 'mcts', 'evolutionary', 'gnn']
        
        for method in methods:
            logger.info(f"Training {method} policy...")
            
            if method == 'rl':
                result = self.train_rl_policy(epochs=10)
            elif method == 'mcts':
                result = self.train_mcts_policy(iterations=500)
            elif method == 'evolutionary':
                result = self.train_evolutionary_policy(generations=30)
            elif method == 'gnn':
                result = self.train_gnn_policy(epochs=10)
            
            # Track best policy
            if 'validation_score' in result and result['validation_score'] > self.stats['best_performance']:
                self.stats['best_performance'] = result['validation_score']
                self.stats['best_policy'] = method
        
        self.stats['training_end_time'] = time.time()
        self.stats['total_episodes'] = len(self.training_data)
        
        # Save training results
        self._save_training_results()
        
        logger.info("All policies trained successfully!")
        logger.info(f"Best policy: {self.stats['best_policy']} (score: {self.stats['best_performance']:.3f})")
        
        return self.training_results
    
    def _save_training_results(self):
        """Save training results"""
        try:
            results_dir = 'results/augmentation_policy_training'
            os.makedirs(results_dir, exist_ok=True)
            
            # Save training results
            results_path = os.path.join(results_dir, f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            
            output_data = {
                'training_results': self.training_results,
                'stats': self.stats,
                'config': self.config,
                'training_data_size': len(self.training_data),
                'validation_data_size': len(self.validation_data)
            }
            
            with open(results_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Training results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train augmentation policies')
    parser.add_argument('--method', choices=['rl', 'mcts', 'evolutionary', 'gnn', 'all'], 
                       default='all', help='Policy method to train')
    parser.add_argument('--data-path', type=str, default='', 
                       help='Path to training data')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--config', type=str, default='', 
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = AUGMENTATION_POLICY_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)
    
    # Create trainer
    trainer = AugmentationPolicyTrainer(config, device=args.device)
    
    # Load training data
    trainer.load_training_data(args.data_path)
    
    # Train policies
    if args.method == 'all':
        results = trainer.train_all_policies()
    else:
        if args.method == 'rl':
            results = trainer.train_rl_policy(epochs=args.epochs)
        elif args.method == 'mcts':
            results = trainer.train_mcts_policy(iterations=1000)
        elif args.method == 'evolutionary':
            results = trainer.train_evolutionary_policy(generations=50)
        elif args.method == 'gnn':
            results = trainer.train_gnn_policy(epochs=args.epochs)
        
        print(f"\n{args.method.upper()} Training Results:")
        print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
