#!/usr/bin/env python3
"""
Graph-Based Random Walk Optimizer

This module implements Node2Vec-style biased random walks on the transformation
dependency graph to discover optimal augmentation sequences that minimize
Checker Framework warnings.
"""

import os
import random
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

# Try to import gensim for Word2Vec, fall back to sklearn if not available
try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

from recursive_augmentation_engine import (
    RecursiveAugmentationEngine, TransformationType, TransformationState
)
from augmentation_sequence_evaluator import AugmentationSequenceEvaluator

logger = logging.getLogger(__name__)

@dataclass
class RandomWalkResult:
    """Result of a random walk sequence"""
    walk: List[TransformationType]
    warning_reduction: float
    overall_score: float
    metadata: Dict[str, Any]

class TransformationGraphWalker:
    """Random walk on augmentation transformation graph using Node2Vec approach"""
    
    def __init__(self, p: float = 0.5, q: float = 2.0, walk_length: int = 10,
                 num_walks: int = 100, embedding_dim: int = 128):
        """
        Initialize the transformation graph walker
        
        Args:
            p: Return parameter (likelihood of returning to previous node)
            q: In-out parameter (explore vs exploit)
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            embedding_dim: Dimension of learned embeddings
        """
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        
        # Graph and embeddings
        self.graph = None
        self.embeddings = None
        self.transformation_to_idx = {}
        self.idx_to_transformation = {}
        
        # Random walk statistics
        self.walk_stats = {
            'total_walks': 0,
            'successful_walks': 0,
            'average_warning_reduction': 0.0,
            'best_warning_reduction': 0.0
        }
        
        # Historical success tracking
        self.successful_walks = []  # Store successful walk sequences
        self.transformation_success_rates = defaultdict(float)
        
        # Initialize augmentation engine and evaluator
        self.engine = RecursiveAugmentationEngine()
        self.evaluator = AugmentationSequenceEvaluator()
    
    def build_transformation_graph(self) -> nx.Graph:
        """Build NetworkX graph from transformation dependencies"""
        # Get dependency graph from recursive augmentation engine
        dependency_graph = self.engine._build_dependency_graph()
        
        # Create NetworkX graph
        graph = nx.Graph()
        
        # Add all transformation types as nodes
        for transformation in TransformationType:
            graph.add_node(transformation, type='transformation')
        
        # Add edges based on dependencies
        for source, dependencies in dependency_graph.items():
            for dep in dependencies:
                target = dep.target
                weight = dep.weight
                
                # Add edge with weight
                if graph.has_edge(source, target):
                    # Update weight if edge exists (take maximum)
                    current_weight = graph[source][target]['weight']
                    graph[source][target]['weight'] = max(current_weight, weight)
                else:
                    graph.add_edge(source, target, weight=weight)
        
        # Add self-loops for all nodes (can stay in same transformation)
        for node in graph.nodes():
            graph.add_edge(node, node, weight=0.5)
        
        self.graph = graph
        logger.info(f"Built transformation graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        return graph
    
    def generate_random_walks(self, num_walks: Optional[int] = None) -> List[List[TransformationType]]:
        """Generate biased random walks through transformation graph"""
        if self.graph is None:
            self.build_transformation_graph()
        
        if num_walks is None:
            num_walks = self.num_walks
        
        walks = []
        nodes = list(self.graph.nodes())
        
        for _ in range(num_walks):
            # Start from random node
            start_node = random.choice(nodes)
            walk = self._biased_random_walk(start_node)
            if walk:
                walks.append(walk)
        
        logger.info(f"Generated {len(walks)} random walks")
        return walks
    
    def _biased_random_walk(self, start_node: TransformationType) -> List[TransformationType]:
        """Perform biased random walk starting from given node"""
        walk = [start_node]
        current = start_node
        previous = None
        
        for _ in range(self.walk_length - 1):
            # Get neighbors
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            
            # Calculate transition probabilities
            probs = []
            for neighbor in neighbors:
                prob = self._calculate_transition_probability(previous, current, neighbor)
                probs.append(prob)
            
            # Normalize probabilities
            total_prob = sum(probs)
            if total_prob == 0:
                # Fall back to uniform random selection
                next_node = random.choice(neighbors)
            else:
                probs = [p / total_prob for p in probs]
                next_node = np.random.choice(neighbors, p=probs)
            
            walk.append(next_node)
            previous = current
            current = next_node
        
        return walk
    
    def _calculate_transition_probability(self, previous: Optional[TransformationType], 
                                        current: TransformationType, 
                                        next_node: TransformationType) -> float:
        """Calculate transition probability based on Node2Vec bias"""
        if previous is None:
            # First step - use edge weight
            return self.graph[current][next_node]['weight']
        
        # Calculate bias based on distance to previous node
        if next_node == previous:
            # Return to previous node - bias by 1/p
            base_weight = self.graph[current][next_node]['weight']
            return base_weight / self.p
        elif self.graph.has_edge(previous, next_node):
            # In-out parameter - bias by 1/q
            base_weight = self.graph[current][next_node]['weight']
            return base_weight / self.q
        else:
            # Normal transition
            return self.graph[current][next_node]['weight']
    
    def learn_embeddings(self, walks: List[List[TransformationType]]) -> Dict[TransformationType, np.ndarray]:
        """Learn transformation embeddings using Word2Vec-style training"""
        if not walks:
            logger.warning("No walks provided for embedding learning")
            return {}
        
        # Convert transformations to strings for Word2Vec
        walk_strings = [[t.value for t in walk] for walk in walks]
        
        if HAS_GENSIM:
            # Use gensim Word2Vec
            model = Word2Vec(
                sentences=walk_strings,
                vector_size=self.embedding_dim,
                window=5,
                min_count=1,
                workers=4,
                sg=1,  # Skip-gram
                epochs=10
            )
            
            # Extract embeddings
            embeddings = {}
            for transformation in TransformationType:
                if transformation.value in model.wv:
                    embeddings[transformation] = model.wv[transformation.value]
                else:
                    # Random embedding for unseen transformations
                    embeddings[transformation] = np.random.normal(0, 0.1, self.embedding_dim)
        else:
            # Fallback to TF-IDF + SVD
            logger.info("Using TF-IDF + SVD fallback for embeddings (gensim not available)")
            
            # Create vocabulary
            all_transformations = set()
            for walk in walk_strings:
                all_transformations.update(walk)
            
            # Create corpus (each walk as a document)
            corpus = [' '.join(walk) for walk in walk_strings]
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(vocabulary=list(all_transformations))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Dimensionality reduction
            svd = TruncatedSVD(n_components=self.embedding_dim)
            reduced_matrix = svd.fit_transform(tfidf_matrix)
            
            # Map transformations to embeddings
            embeddings = {}
            vocab = vectorizer.vocabulary_
            for transformation in TransformationType:
                if transformation.value in vocab:
                    idx = vocab[transformation.value]
                    # Average embedding across all walks containing this transformation
                    walk_indices = [i for i, walk in enumerate(walk_strings) if transformation.value in walk]
                    if walk_indices:
                        avg_embedding = np.mean(reduced_matrix[walk_indices], axis=0)
                        embeddings[transformation] = avg_embedding
                    else:
                        embeddings[transformation] = np.random.normal(0, 0.1, self.embedding_dim)
                else:
                    embeddings[transformation] = np.random.normal(0, 0.1, self.embedding_dim)
        
        self.embeddings = embeddings
        logger.info(f"Learned embeddings for {len(embeddings)} transformations")
        
        return embeddings
    
    def predict_next_transformation(self, current_state: TransformationState, 
                                  valid_transformations: List[TransformationType]) -> TransformationType:
        """Predict next transformation using learned embeddings and historical success"""
        if not self.embeddings:
            # No embeddings learned yet - use random selection
            return random.choice(valid_transformations)
        
        if not valid_transformations:
            return random.choice(list(TransformationType))
        
        # Get current transformation (last in history)
        current_transformation = current_state.transformation_history[-1] if current_state.transformation_history else None
        
        if current_transformation is None:
            # No history - use success rate based selection
            success_rates = [self.transformation_success_rates[t] for t in valid_transformations]
            if sum(success_rates) > 0:
                # Weighted selection based on success rates
                probs = [rate / sum(success_rates) for rate in success_rates]
                return np.random.choice(valid_transformations, p=probs)
            else:
                return random.choice(valid_transformations)
        
        # Calculate similarity-based probabilities
        current_embedding = self.embeddings.get(current_transformation, np.zeros(self.embedding_dim))
        similarities = []
        
        for transformation in valid_transformations:
            # Embedding similarity
            if transformation in self.embeddings:
                embedding = self.embeddings[transformation]
                similarity = np.dot(current_embedding, embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(embedding) + 1e-8
                )
            else:
                similarity = 0.0
            
            # Combine with success rate
            success_rate = self.transformation_success_rates[transformation]
            combined_score = 0.7 * similarity + 0.3 * success_rate
            similarities.append(combined_score)
        
        # Normalize and select
        total_similarity = sum(similarities)
        if total_similarity > 0:
            probs = [sim / total_similarity for sim in similarities]
            return np.random.choice(valid_transformations, p=probs)
        else:
            return random.choice(valid_transformations)
    
    def evaluate_walk_sequence(self, walk: List[TransformationType], 
                             initial_code: str) -> RandomWalkResult:
        """Evaluate a random walk sequence using the augmentation engine"""
        try:
            # Apply transformation sequence
            states = self.engine.apply_recursive_transformation(
                initial_code,
                max_depth=len(walk),
                transformation_sequence=walk
            )
            
            if len(states) <= 1:
                # No successful transformations
                return RandomWalkResult(
                    walk=walk,
                    warning_reduction=0.0,
                    overall_score=0.0,
                    metadata={'error': 'no_transformations_applied'}
                )
            
            # Evaluate the sequence
            metrics = self.evaluator.evaluate_sequence(states)
            
            # Update statistics
            self.walk_stats['total_walks'] += 1
            if metrics.warning_reduction > 0:
                self.walk_stats['successful_walks'] += 1
                self.walk_stats['average_warning_reduction'] = (
                    (self.walk_stats['average_warning_reduction'] * (self.walk_stats['successful_walks'] - 1) +
                     metrics.warning_reduction) / self.walk_stats['successful_walks']
                )
                self.walk_stats['best_warning_reduction'] = max(
                    self.walk_stats['best_warning_reduction'], metrics.warning_reduction
                )
                
                # Store successful walk
                self.successful_walks.append(walk)
                
                # Update transformation success rates
                for transformation in walk:
                    self.transformation_success_rates[transformation] += 1
                
                # Normalize success rates
                for transformation in self.transformation_success_rates:
                    if transformation in walk:
                        self.transformation_success_rates[transformation] /= len(walk)
            
            return RandomWalkResult(
                walk=walk,
                warning_reduction=metrics.warning_reduction,
                overall_score=metrics.overall_score,
                metadata=metrics.metadata
            )
            
        except Exception as e:
            logger.warning(f"Error evaluating walk sequence: {e}")
            return RandomWalkResult(
                walk=walk,
                warning_reduction=0.0,
                overall_score=0.0,
                metadata={'error': str(e)}
            )
    
    def optimize_augmentation_sequence(self, initial_code: str, 
                                     max_iterations: int = 100) -> RandomWalkResult:
        """Find optimal augmentation sequence using random walks"""
        logger.info(f"Starting random walk optimization with {max_iterations} iterations")
        
        # Build graph and learn initial embeddings
        self.build_transformation_graph()
        initial_walks = self.generate_random_walks(num_walks=50)
        self.learn_embeddings(initial_walks)
        
        best_result = None
        best_warning_reduction = -1.0
        
        for iteration in range(max_iterations):
            # Generate new walks using current embeddings
            walks = self.generate_random_walks(num_walks=10)
            
            # Evaluate walks
            for walk in walks:
                result = self.evaluate_walk_sequence(walk, initial_code)
                
                if result.warning_reduction > best_warning_reduction:
                    best_warning_reduction = result.warning_reduction
                    best_result = result
                    logger.info(f"Iteration {iteration}: New best warning reduction: {best_warning_reduction:.3f}")
                
                # Early stopping if we achieve good warning reduction
                if best_warning_reduction >= 0.8:
                    logger.info(f"Early stopping: achieved {best_warning_reduction:.3f} warning reduction")
                    break
            
            # Relearn embeddings with successful walks
            if self.successful_walks and iteration % 10 == 0:
                self.learn_embeddings(self.successful_walks[-50:])  # Use last 50 successful walks
        
        if best_result is None:
            # Fallback: return random walk
            fallback_walk = [random.choice(list(TransformationType)) for _ in range(3)]
            best_result = RandomWalkResult(
                walk=fallback_walk,
                warning_reduction=0.0,
                overall_score=0.0,
                metadata={'method': 'fallback'}
            )
        
        logger.info(f"Optimization complete. Best warning reduction: {best_result.warning_reduction:.3f}")
        return best_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get random walk statistics"""
        stats = self.walk_stats.copy()
        
        if self.walk_stats['total_walks'] > 0:
            stats['success_rate'] = self.walk_stats['successful_walks'] / self.walk_stats['total_walks']
        else:
            stats['success_rate'] = 0.0
        
        stats['total_successful_walks_stored'] = len(self.successful_walks)
        stats['transformation_success_rates'] = dict(self.transformation_success_rates)
        
        return stats


def main():
    """Test the transformation graph walker"""
    logger.info("Testing Transformation Graph Walker...")
    
    # Create test code
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
    
    # Initialize walker
    walker = TransformationGraphWalker(p=0.5, q=2.0, walk_length=5)
    
    # Test optimization
    result = walker.optimize_augmentation_sequence(test_code, max_iterations=20)
    
    logger.info(f"Optimization result:")
    logger.info(f"  Walk: {[t.value for t in result.walk]}")
    logger.info(f"  Warning reduction: {result.warning_reduction:.3f}")
    logger.info(f"  Overall score: {result.overall_score:.3f}")
    
    # Print statistics
    stats = walker.get_statistics()
    logger.info(f"Statistics: {stats}")


if __name__ == '__main__':
    import random
    random.seed(42)
    main()

