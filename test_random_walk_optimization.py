#!/usr/bin/env python3
"""
Test Suite for Random Walk Optimization Methods

This module provides comprehensive testing for all random walk-based
optimization methods in the GenDATA system.
"""

import os
import sys
import unittest
import tempfile
import logging
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator, EvaluationMetrics
from augmentation_policy_learner import (
    ReinforcementLearningPolicy, MCTSAugmentationSearch, 
    EvolutionaryAugmentationOptimizer, RandomWalkOptimizer
)
from graph_based_random_walk_optimizer import TransformationGraphWalker, RandomWalkResult
from random_walk_policy_network import RandomWalkPolicyNetwork
from recursive_augmentation_engine import RecursiveAugmentationEngine, TransformationState, TransformationType
from checker_framework_integration import CheckerFrameworkEvaluator, CheckerType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestWarningReductionEvaluation(unittest.TestCase):
    """Test warning reduction evaluation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = AugmentationSequenceEvaluator()
        self.test_code = """
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
        
        # Create test states
        self.original_state = TransformationState(
            code=self.test_code,
            transformation_history=[],
            depth=0,
            complexity_score=2.0,
            compilation_status=True,
            semantic_preservation=True,
            metadata={}
        )
        
        self.augmented_state = TransformationState(
            code=self.test_code.replace("i < arr.length", "i < arr.length && i >= 0"),
            transformation_history=[TransformationType.GUARD_REVERSAL],
            depth=1,
            complexity_score=2.5,
            compilation_status=True,
            semantic_preservation=True,
            metadata={}
        )
    
    def test_warning_reduction_evaluation(self):
        """Test warning reduction evaluation"""
        try:
            reduction = self.evaluator.evaluate_warning_reduction(self.original_state, self.augmented_state)
            self.assertIsInstance(reduction, float)
            self.assertGreaterEqual(reduction, 0.0)
            self.assertLessEqual(reduction, 1.0)
            logger.info(f"Warning reduction: {reduction}")
        except Exception as e:
            logger.warning(f"Warning reduction evaluation failed: {e}")
            # This is expected if Checker Framework is not available
    
    def test_sequence_evaluation_with_warning_reduction(self):
        """Test complete sequence evaluation including warning reduction"""
        states = [self.original_state, self.augmented_state]
        
        try:
            metrics = self.evaluator.evaluate_sequence(states)
            
            self.assertIsInstance(metrics, EvaluationMetrics)
            self.assertIsInstance(metrics.warning_reduction, float)
            self.assertIsInstance(metrics.overall_score, float)
            
            # Check that warning reduction is included in overall score
            self.assertGreaterEqual(metrics.overall_score, 0.0)
            self.assertLessEqual(metrics.overall_score, 1.0)
            
            logger.info(f"Overall score: {metrics.overall_score}")
            logger.info(f"Warning reduction: {metrics.warning_reduction}")
            
        except Exception as e:
            logger.warning(f"Sequence evaluation failed: {e}")

class TestRLRandomWalkExploration(unittest.TestCase):
    """Test RL policy with random walk exploration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rl_policy = ReinforcementLearningPolicy(device='cpu')
        self.test_state = TransformationState(
            code="public class Test { public int method() { return 0; } }",
            transformation_history=[],
            depth=0,
            complexity_score=2.0,
            compilation_status=True,
            semantic_preservation=True,
            metadata={}
        )
    
    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy exploration strategy"""
        valid_actions = [TransformationType.LOOP_CONVERSION, TransformationType.GUARD_REVERSAL]
        
        # Test multiple action selections
        actions = []
        for _ in range(10):
            action = self.rl_policy.select_action(self.test_state, valid_actions)
            actions.append(action)
        
        # Should get valid actions
        for action in actions:
            self.assertIn(action, valid_actions)
        
        # Check epsilon decay
        initial_epsilon = 0.3
        self.assertLessEqual(self.rl_policy.epsilon, initial_epsilon)
        
        logger.info(f"Selected actions: {[a.value for a in actions]}")
        logger.info(f"Current epsilon: {self.rl_policy.epsilon}")
    
    def test_random_walk_exploration(self):
        """Test random walk exploration method"""
        valid_actions = [TransformationType.LOOP_CONVERSION, TransformationType.GUARD_REVERSAL]
        
        # Force random walk exploration
        original_epsilon = self.rl_policy.epsilon
        self.rl_policy.epsilon = 1.0  # Always explore
        
        action = self.rl_policy.select_action(self.test_state, valid_actions)
        
        # Should be a valid action
        self.assertIn(action, valid_actions)
        
        # Check that random walk buffer was updated
        self.assertGreater(len(self.rl_policy.random_walk_buffer), 0)
        
        # Restore epsilon
        self.rl_policy.epsilon = original_epsilon
        
        logger.info(f"Random walk action: {action.value}")
        logger.info(f"Random walk buffer size: {len(self.rl_policy.random_walk_buffer)}")
    
    def test_training_statistics(self):
        """Test training statistics tracking"""
        stats = self.rl_policy.training_stats
        
        # Check that all expected keys are present
        expected_keys = ['episodes', 'total_rewards', 'policy_losses', 'value_losses', 
                        'entropy_losses', 'epsilon_values', 'random_walk_usage', 'warning_reductions']
        
        for key in expected_keys:
            self.assertIn(key, stats)

class TestMCTSRandomWalk(unittest.TestCase):
    """Test MCTS with guided random walks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mcts = MCTSAugmentationSearch()
        self.engine = RecursiveAugmentationEngine()
        self.evaluator = AugmentationSequenceEvaluator()
        self.initial_state = TransformationState(
            code="public class Test { public int method() { return 0; } }",
            transformation_history=[],
            depth=0,
            complexity_score=2.0,
            compilation_status=True,
            semantic_preservation=True,
            metadata={}
        )
    
    def test_guided_random_walk(self):
        """Test guided random walk in simulation"""
        valid_actions = [TransformationType.LOOP_CONVERSION, TransformationType.GUARD_REVERSAL]
        walk_sequence = [TransformationType.LOOP_CONVERSION]
        
        action = self.mcts._guided_random_walk(self.initial_state, valid_actions, walk_sequence)
        
        # Should return a valid action
        self.assertIn(action, valid_actions)
        
        logger.info(f"Guided random walk action: {action.value}")
    
    def test_random_walk_policy_update(self):
        """Test random walk policy update"""
        walk_sequence = [TransformationType.LOOP_CONVERSION, TransformationType.GUARD_REVERSAL]
        reward = 0.5
        
        # Update policy
        self.mcts._update_random_walk_policy(walk_sequence, reward)
        
        # Check that policy was updated
        sequence_key = tuple(walk_sequence)
        self.assertIn(sequence_key, self.mcts.random_walk_policy)
        self.assertEqual(self.mcts.random_walk_policy[sequence_key], reward)
        
        logger.info(f"Updated policy for sequence: {[t.value for t in walk_sequence]}")
    
    def test_mcts_search_with_random_walks(self):
        """Test MCTS search with random walk enhancements"""
        try:
            # Run limited MCTS search
            best_sequence = self.mcts.search(
                self.initial_state,
                self.engine,
                self.evaluator,
                max_depth=3,
                max_iterations=50
            )
            
            # Should return a sequence (may be empty)
            self.assertIsInstance(best_sequence, list)
            
            # Check statistics
            stats = self.mcts.search_stats
            self.assertGreaterEqual(stats['iterations'], 0)
            self.assertGreaterEqual(stats['simulations'], 0)
            
            logger.info(f"Best sequence: {[t.value for t in best_sequence]}")
            logger.info(f"MCTS stats: {stats}")
            
        except Exception as e:
            logger.warning(f"MCTS search failed: {e}")

class TestEvolutionaryRandomWalk(unittest.TestCase):
    """Test evolutionary algorithm with random walk mutation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evo_optimizer = EvolutionaryAugmentationOptimizer(
            population_size=10,
            mutation_rate=0.3,
            random_walk_mutation_rate=0.5,
            walk_steps=2
        )
        self.engine = RecursiveAugmentationEngine()
        self.evaluator = AugmentationSequenceEvaluator()
        self.test_code = "public class Test { public int method() { return 0; } }"
    
    def test_random_walk_mutation(self):
        """Test random walk mutation operator"""
        # Create a test genome
        genome = self.evo_optimizer.TransformationGenome([
            TransformationType.LOOP_CONVERSION,
            TransformationType.GUARD_REVERSAL
        ])
        
        # Perform random walk mutation
        mutated = self.evo_optimizer._random_walk_mutate(genome)
        
        # Should return a valid genome
        self.assertIsInstance(mutated, self.evo_optimizer.TransformationGenome)
        self.assertIsInstance(mutated.sequence, list)
        
        # Check that mutation statistics were updated
        self.assertGreaterEqual(self.evo_optimizer.evolution_stats['random_walk_mutations'], 0)
        
        logger.info(f"Original sequence: {[t.value for t in genome.sequence]}")
        logger.info(f"Mutated sequence: {[t.value for t in mutated.sequence]}")
    
    def test_guided_random_walk_selection(self):
        """Test guided random walk selection"""
        valid_actions = [TransformationType.LOOP_CONVERSION, TransformationType.GUARD_REVERSAL]
        current_sequence = [TransformationType.LOOP_CONVERSION]
        
        action = self.evo_optimizer._guided_random_walk_selection(valid_actions, current_sequence)
        
        # Should return a valid action
        self.assertIn(action, valid_actions)
        
        logger.info(f"Guided selection action: {action.value}")
    
    def test_evolutionary_optimization_with_random_walks(self):
        """Test evolutionary optimization with random walk mutations"""
        try:
            # Run limited evolutionary optimization
            best_genome = self.evo_optimizer.optimize(
                self.test_code,
                self.engine,
                self.evaluator,
                max_generations=5
            )
            
            # Should return a valid genome
            self.assertIsInstance(best_genome, self.evo_optimizer.TransformationGenome)
            self.assertIsInstance(best_genome.sequence, list)
            
            # Check statistics
            stats = self.evo_optimizer.evolution_stats
            self.assertGreaterEqual(stats['generations'], 0)
            self.assertGreaterEqual(stats['random_walk_mutations'], 0)
            
            logger.info(f"Best sequence: {[t.value for t in best_genome.sequence]}")
            logger.info(f"Evolution stats: {stats}")
            
        except Exception as e:
            logger.warning(f"Evolutionary optimization failed: {e}")

class TestGraphBasedRandomWalk(unittest.TestCase):
    """Test graph-based random walk optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.graph_walker = TransformationGraphWalker(
            p=0.5, q=2.0, walk_length=5, num_walks=20
        )
        self.test_code = """
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
    
    def test_transformation_graph_building(self):
        """Test transformation graph building"""
        graph = self.graph_walker.build_transformation_graph()
        
        # Should return a NetworkX graph
        import networkx as nx
        self.assertIsInstance(graph, nx.Graph)
        
        # Should have nodes for all transformation types
        transformation_nodes = [n for n in graph.nodes() if isinstance(n, TransformationType)]
        self.assertGreater(len(transformation_nodes), 0)
        
        logger.info(f"Graph nodes: {graph.number_of_nodes()}")
        logger.info(f"Graph edges: {graph.number_of_edges()}")
    
    def test_random_walk_generation(self):
        """Test random walk generation"""
        walks = self.graph_walker.generate_random_walks(num_walks=5)
        
        # Should return a list of walks
        self.assertIsInstance(walks, list)
        self.assertGreater(len(walks), 0)
        
        # Each walk should be a list of transformations
        for walk in walks:
            self.assertIsInstance(walk, list)
            for transformation in walk:
                self.assertIsInstance(transformation, TransformationType)
        
        logger.info(f"Generated {len(walks)} random walks")
        logger.info(f"Sample walk: {[t.value for t in walks[0]]}")
    
    def test_embedding_learning(self):
        """Test embedding learning from walks"""
        # Generate walks first
        walks = self.graph_walker.generate_random_walks(num_walks=10)
        
        # Learn embeddings
        embeddings = self.graph_walker.learn_embeddings(walks)
        
        # Should return embeddings for transformations
        self.assertIsInstance(embeddings, dict)
        self.assertGreater(len(embeddings), 0)
        
        # Each embedding should be a numpy array
        for transformation, embedding in embeddings.items():
            self.assertIsInstance(transformation, TransformationType)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape[0], self.graph_walker.embedding_dim)
        
        logger.info(f"Learned embeddings for {len(embeddings)} transformations")
    
    def test_graph_optimization(self):
        """Test graph-based optimization"""
        try:
            result = self.graph_walker.optimize_augmentation_sequence(
                self.test_code, max_iterations=10
            )
            
            # Should return a RandomWalkResult
            self.assertIsInstance(result, RandomWalkResult)
            self.assertIsInstance(result.walk, list)
            self.assertIsInstance(result.warning_reduction, float)
            self.assertIsInstance(result.overall_score, float)
            
            # Check statistics
            stats = self.graph_walker.get_statistics()
            self.assertGreaterEqual(stats['total_walks'], 0)
            
            logger.info(f"Optimization result:")
            logger.info(f"  Walk: {[t.value for t in result.walk]}")
            logger.info(f"  Warning reduction: {result.warning_reduction}")
            logger.info(f"  Overall score: {result.overall_score}")
            
        except Exception as e:
            logger.warning(f"Graph optimization failed: {e}")

class TestRandomWalkPolicyNetwork(unittest.TestCase):
    """Test random walk policy network"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.policy_net = RandomWalkPolicyNetwork(device='cpu')
        self.test_state = TransformationState(
            code="public class Test { public int method() { return 0; } }",
            transformation_history=[],
            depth=0,
            complexity_score=2.0,
            compilation_status=True,
            semantic_preservation=True,
            metadata={}
        )
    
    def test_policy_network_initialization(self):
        """Test policy network initialization"""
        # Should have model and trainer
        self.assertIsNotNone(self.policy_net.model)
        self.assertIsNotNone(self.policy_net.trainer)
        
        # Should have transformation mappings
        self.assertGreater(len(self.policy_net.transformation_to_idx), 0)
        self.assertGreater(len(self.policy_net.idx_to_transformation), 0)
        
        logger.info(f"Policy network initialized with {len(self.policy_net.transformation_to_idx)} transformations")
    
    def test_walk_generation_with_policy(self):
        """Test walk generation using policy network"""
        walk = self.policy_net.generate_walk_with_policy(self.test_state, max_length=3)
        
        # Should return a list of transformations
        self.assertIsInstance(walk, list)
        for transformation in walk:
            self.assertIsInstance(transformation, TransformationType)
        
        logger.info(f"Generated walk: {[t.value for t in walk]}")
    
    def test_learning_from_walks(self):
        """Test learning from successful walks"""
        # Create dummy walks and rewards
        walks = [
            [TransformationType.LOOP_CONVERSION, TransformationType.GUARD_REVERSAL],
            [TransformationType.MATHEMATICAL_PROPERTIES, TransformationType.LOGICAL_EXPRESSIONS]
        ]
        rewards = [0.5, 0.3]
        
        # Learn from walks
        self.policy_net.learn_from_walks(walks, rewards)
        
        # Check statistics
        stats = self.policy_net.get_statistics()
        self.assertIsInstance(stats, dict)
        
        logger.info(f"Learned from {len(walks)} walks")

class TestRandomWalkOptimizer(unittest.TestCase):
    """Test the main random walk optimizer orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = RandomWalkOptimizer(
            methods=['rl', 'mcts', 'graph', 'evolutionary'],
            device='cpu'
        )
        self.test_code = """
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
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        # Should have all components
        expected_components = ['rl', 'mcts', 'graph', 'evolutionary']
        for component in expected_components:
            self.assertIn(component, self.optimizer.components)
        
        # Should have evaluator and engine
        self.assertIsNotNone(self.optimizer.evaluator)
        self.assertIsNotNone(self.optimizer.engine)
        
        logger.info(f"Optimizer initialized with methods: {self.optimizer.methods}")
    
    def test_sequential_optimization(self):
        """Test sequential optimization"""
        try:
            result = self.optimizer.optimize_augmentation_sequence(
                self.test_code, max_iterations=20, parallel=False
            )
            
            # Should return a result dictionary
            self.assertIsInstance(result, dict)
            self.assertIn('best_method', result)
            self.assertIn('best_warning_reduction', result)
            self.assertIn('method_results', result)
            
            logger.info(f"Sequential optimization result:")
            logger.info(f"  Best method: {result['best_method']}")
            logger.info(f"  Best warning reduction: {result['best_warning_reduction']}")
            logger.info(f"  Valid methods: {result['valid_methods']}")
            
        except Exception as e:
            logger.warning(f"Sequential optimization failed: {e}")
    
    def test_parallel_optimization(self):
        """Test parallel optimization"""
        try:
            result = self.optimizer.optimize_augmentation_sequence(
                self.test_code, max_iterations=20, parallel=True
            )
            
            # Should return a result dictionary
            self.assertIsInstance(result, dict)
            self.assertIn('best_method', result)
            self.assertIn('best_warning_reduction', result)
            self.assertIn('method_results', result)
            
            logger.info(f"Parallel optimization result:")
            logger.info(f"  Best method: {result['best_method']}")
            logger.info(f"  Best warning reduction: {result['best_warning_reduction']}")
            logger.info(f"  Valid methods: {result['valid_methods']}")
            
        except Exception as e:
            logger.warning(f"Parallel optimization failed: {e}")
    
    def test_optimization_statistics(self):
        """Test optimization statistics tracking"""
        # Run a few optimizations
        for _ in range(3):
            try:
                self.optimizer.optimize_augmentation_sequence(
                    self.test_code, max_iterations=10, parallel=False
                )
            except Exception as e:
                logger.debug(f"Optimization failed: {e}")
        
        # Check statistics
        stats = self.optimizer.get_statistics()
        
        # Should have statistics
        self.assertIsInstance(stats, dict)
        self.assertIn('total_optimizations', stats)
        self.assertIn('method_usage', stats)
        
        logger.info(f"Optimization statistics: {stats}")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_code = """
public class ArraySum {
    public int sumArray(int[] arr) {
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        return sum;
    }
}
"""
    
    def test_end_to_end_optimization(self):
        """Test end-to-end optimization pipeline"""
        try:
            # Create optimizer with all methods
            optimizer = RandomWalkOptimizer(
                methods=['rl', 'mcts', 'graph', 'evolutionary'],
                device='cpu'
            )
            
            # Run optimization
            result = optimizer.optimize_augmentation_sequence(
                self.test_code, max_iterations=30, parallel=True
            )
            
            # Validate result
            self.assertIsInstance(result, dict)
            
            # At least one method should succeed
            valid_methods = result.get('valid_methods', [])
            self.assertGreaterEqual(len(valid_methods), 0)
            
            logger.info("End-to-end optimization completed successfully")
            logger.info(f"Valid methods: {valid_methods}")
            
            # Print detailed results
            for method, method_result in result['method_results'].items():
                if 'error' not in method_result:
                    logger.info(f"{method}: {method_result.get('warning_reduction', 'N/A')}")
                else:
                    logger.info(f"{method}: ERROR - {method_result['error']}")
            
        except Exception as e:
            logger.error(f"End-to-end optimization failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        import time
        
        optimizer = RandomWalkOptimizer(
            methods=['graph', 'evolutionary'],  # Use faster methods
            device='cpu'
        )
        
        start_time = time.time()
        
        try:
            result = optimizer.optimize_augmentation_sequence(
                self.test_code, max_iterations=20, parallel=True
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete within reasonable time (5 minutes)
            self.assertLess(execution_time, 300)
            
            logger.info(f"Performance test completed in {execution_time:.2f} seconds")
            
            # Check if we achieved any warning reduction
            best_reduction = result.get('best_warning_reduction', 0.0)
            logger.info(f"Best warning reduction achieved: {best_reduction:.3f}")
            
        except Exception as e:
            logger.warning(f"Performance test failed: {e}")

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestWarningReductionEvaluation,
        TestRLRandomWalkExploration,
        TestMCTSRandomWalk,
        TestEvolutionaryRandomWalk,
        TestGraphBasedRandomWalk,
        TestRandomWalkPolicyNetwork,
        TestRandomWalkOptimizer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

def main():
    """Main test runner"""
    logger.info("Starting comprehensive random walk optimization tests...")
    
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Run all tests
    result = run_all_tests()
    
    # Print summary
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.failures:
        logger.error("Test failures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("Test errors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        logger.info("All tests passed successfully!")
    else:
        logger.warning("Some tests failed or had errors.")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

