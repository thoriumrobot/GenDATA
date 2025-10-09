#!/usr/bin/env python3
"""
Augmentation Policy Learning

This module implements multiple machine learning methods for discovering optimal
augmentation sequences using reinforcement learning, Monte Carlo Tree Search,
and evolutionary algorithms.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import json
import logging
from abc import ABC, abstractmethod
import math

from recursive_augmentation_engine import (
    RecursiveAugmentationEngine, TransformationType, TransformationState
)
from unified_augmentation_registry import UnifiedAugmentationRegistry
from code_location_analyzer import CodeLocation, CodeLocationAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class AugmentationSequence:
    """Represents a sequence of augmentation transformations"""
    transformations: List[TransformationType]
    performance_score: float
    metadata: Dict[str, Any]

@dataclass
class TrainingEpisode:
    """Represents a training episode for RL"""
    states: List[TransformationState]
    actions: List[TransformationType]
    rewards: List[float]
    total_reward: float

class AugmentationPolicyNetwork(nn.Module):
    """Neural network for predicting next transformation"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 output_dim: int = len(TransformationType), dropout: float = 0.3):
        super().__init__()
        
        # State encoder (for code representation)
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # History encoder (for transformation history)
        self.history_encoder = nn.LSTM(
            input_size=len(TransformationType),
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Value head (for advantage estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_features: torch.Tensor, 
                history_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Encode state
        state_encoded = self.state_encoder(state_features)
        
        # Encode history
        history_output, _ = self.history_encoder(history_sequence)
        history_encoded = history_output[:, -1, :]  # Take last output
        
        # Combine state and history
        combined = torch.cat([state_encoded, history_encoded], dim=-1)
        
        # Policy and value
        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return policy_logits, value

class AugmentationPolicyLearner(ABC):
    """Abstract base class for augmentation policy learners"""
    
    def __init__(self, registry: Optional[UnifiedAugmentationRegistry] = None):
        self.registry = registry or UnifiedAugmentationRegistry()
        self.location_analyzer = CodeLocationAnalyzer()
    
    @abstractmethod
    def learn_policy(self, training_data: List[TrainingEpisode]) -> Dict[str, Any]:
        """Learn augmentation policy from training data"""
        pass
    
    @abstractmethod
    def select_action(self, state: TransformationState, 
                     valid_actions: List[TransformationType],
                     code_location: Optional[CodeLocation] = None) -> TransformationType:
        """Select next transformation given current state and optional location"""
        pass

class ReinforcementLearningPolicy(AugmentationPolicyLearner):
    """Reinforcement Learning based policy learner using PPO"""
    
    def __init__(self, device: str = 'cpu', learning_rate: float = 3e-4,
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, epsilon: float = 0.3,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.05,
                 registry: Optional[UnifiedAugmentationRegistry] = None):
        super().__init__(registry)
        self.device = device
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Random walk exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize policy network
        self.policy_net = AugmentationPolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        self.random_walk_buffer = deque(maxlen=5000)  # Store random walk experiences
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'total_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'epsilon_values': [],
            'random_walk_usage': [],
            'warning_reductions': []
        }
    
    def learn_policy(self, training_data: List[TrainingEpisode]) -> Dict[str, Any]:
        """Learn policy using PPO"""
        logger.info(f"Learning RL policy from {len(training_data)} episodes")
        
        # Collect all experiences
        all_states = []
        all_actions = []
        all_rewards = []
        all_old_log_probs = []
        all_values = []
        
        for episode in training_data:
            for i, (state, action, reward) in enumerate(zip(episode.states, episode.actions, episode.rewards)):
                # Get state features and history
                state_features = self._encode_state(state)
                history_sequence = self._encode_history(episode.states[:i+1])
                
                # Get action logits and value
                with torch.no_grad():
                    action_logits, value = self.policy_net(state_features, history_sequence)
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_idx = self._action_to_index(action)
                    old_log_prob = torch.log(action_probs[0, action_idx] + 1e-8)
                
                # Enhance reward with warning reduction (if available)
                enhanced_reward = self._enhance_reward_with_warning_reduction(reward, state, episode.states)
                
                all_states.append((state_features, history_sequence))
                all_actions.append(action_idx)
                all_rewards.append(enhanced_reward)
                all_old_log_probs.append(old_log_prob)
                all_values.append(value[0, 0])
        
        # Convert to tensors
        all_states = [(torch.stack([s[0]]), torch.stack([s[1]])) for s in all_states]
        all_actions = torch.tensor(all_actions, device=self.device)
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
        all_old_log_probs = torch.stack(all_old_log_probs)
        all_values = torch.stack(all_values)
        
        # Compute advantages
        advantages = self._compute_advantages(all_rewards, all_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training
        for epoch in range(10):  # Multiple epochs
            # Get new action logits and values
            new_log_probs = []
            new_values = []
            
            for state_features, history_sequence in all_states:
                action_logits, value = self.policy_net(state_features, history_sequence)
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Get log probabilities for selected actions
                selected_log_probs = torch.log(action_probs[0, all_actions] + 1e-8)
                new_log_probs.append(selected_log_probs)
                new_values.append(value[0, 0])
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values)
            
            # Compute ratios
            ratio = torch.exp(new_log_probs - all_old_log_probs)
            
            # Compute clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, all_rewards)
            
            # Entropy loss
            entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            # Update training statistics
            self.training_stats['policy_losses'].append(policy_loss.item())
            self.training_stats['value_losses'].append(value_loss.item())
            self.training_stats['entropy_losses'].append(entropy_loss.item())
        
        return {
            'method': 'reinforcement_learning',
            'training_episodes': len(training_data),
            'final_policy_loss': self.training_stats['policy_losses'][-1] if self.training_stats['policy_losses'] else 0,
            'final_value_loss': self.training_stats['value_losses'][-1] if self.training_stats['value_losses'] else 0,
            'final_entropy_loss': self.training_stats['entropy_losses'][-1] if self.training_stats['entropy_losses'] else 0
        }
    
    def select_action(self, state: TransformationState, 
                     valid_actions: List[TransformationType],
                     code_location: Optional[CodeLocation] = None) -> TransformationType:
        """Select action using learned policy with random walk exploration and location awareness"""
        # Filter valid actions by location if provided
        if code_location:
            valid_actions = self.registry.get_valid_transformations(state.code, code_location)
            # Intersect with provided valid_actions
            valid_actions = [action for action in valid_actions if action in valid_actions]
        
        if not valid_actions:
            logger.warning("No valid actions available")
            return random.choice(list(TransformationType))
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random walk exploration
            action = self._random_walk_exploration(state, valid_actions, code_location)
            self.training_stats['random_walk_usage'].append(1)
        else:
            # Exploitation using learned policy
            action = self._policy_exploitation(state, valid_actions, code_location)
            self.training_stats['random_walk_usage'].append(0)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_stats['epsilon_values'].append(self.epsilon)
        
        return action
    
    def _policy_exploitation(self, state: TransformationState, 
                           valid_actions: List[TransformationType],
                           code_location: Optional[CodeLocation] = None) -> TransformationType:
        """Select action using learned policy"""
        with torch.no_grad():
            state_features = self._encode_state(state)
            history_sequence = self._encode_history([state])
            
            action_logits, _ = self.policy_net(state_features, history_sequence)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Mask invalid actions
            action_mask = torch.zeros(len(TransformationType), device=self.device)
            for action in valid_actions:
                action_idx = self._action_to_index(action)
                action_mask[action_idx] = 1.0
            
            masked_probs = action_probs * action_mask
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
                action_idx = torch.multinomial(masked_probs, 1).item()
            else:
                # Fall back to random selection
                action_idx = random.randint(0, len(valid_actions) - 1)
                action_idx = self._action_to_index(valid_actions[action_idx])
            
            return self._index_to_action(action_idx)
    
    def _random_walk_exploration(self, state: TransformationState, 
                               valid_actions: List[TransformationType],
                               code_location: Optional[CodeLocation] = None) -> TransformationType:
        """Perform random walk exploration through transformation space"""
        try:
            # Import here to avoid circular imports
            from augmentation_sequence_evaluator import AugmentationSequenceEvaluator
            from recursive_augmentation_engine import RecursiveAugmentationEngine
            
            # Initialize components
            evaluator = AugmentationSequenceEvaluator()
            engine = RecursiveAugmentationEngine()
            
            # Perform k-step random walk
            walk_length = random.randint(1, 3)  # Short random walks
            current_state = state
            best_action = None
            best_warning_reduction = -1.0
            
            for step in range(walk_length):
                # Get valid next transformations
                valid_next = engine.get_valid_next_transformations(current_state)
                if not valid_next:
                    break
                
                # Randomly select next transformation
                action = random.choice(valid_next)
                
                # Apply transformation
                new_states = engine.apply_recursive_transformation(
                    current_state.code,
                    max_depth=1,
                    transformation_sequence=[action]
                )
                
                if len(new_states) > 1:
                    new_state = new_states[1]
                    
                    # Evaluate warning reduction
                    warning_reduction = evaluator.evaluate_warning_reduction(state, new_state)
                    
                    # Store this random walk experience
                    self.random_walk_buffer.append({
                        'state': current_state,
                        'action': action,
                        'reward': warning_reduction,
                        'next_state': new_state
                    })
                    
                    # Track best action
                    if warning_reduction > best_warning_reduction:
                        best_warning_reduction = warning_reduction
                        best_action = action
                    
                    current_state = new_state
            
            # Update statistics
            self.training_stats['warning_reductions'].append(best_warning_reduction)
            
            # Return best action found during random walk, or random if none found
            return best_action if best_action is not None else random.choice(valid_actions)
            
        except Exception as e:
            logger.warning(f"Error in random walk exploration: {e}")
            # Fall back to random selection
            return random.choice(valid_actions)
    
    def _encode_state(self, state: TransformationState) -> torch.Tensor:
        """Encode transformation state to features"""
        # Simple feature encoding - in practice, you'd use more sophisticated methods
        features = torch.zeros(512, device=self.device)
        
        # Basic features
        features[0] = state.depth
        features[1] = state.complexity_score
        features[2] = 1.0 if state.compilation_status else 0.0
        features[3] = 1.0 if state.semantic_preservation else 0.0
        
        # Code length and complexity features
        code_length = len(state.code)
        features[4] = min(code_length / 1000.0, 1.0)  # Normalized code length
        
        # Count various constructs
        features[5] = min(state.code.count('for ') / 10.0, 1.0)
        features[6] = min(state.code.count('if ') / 10.0, 1.0)
        features[7] = min(state.code.count('method ') / 5.0, 1.0)
        features[8] = min(state.code.count('return ') / 5.0, 1.0)
        
        return features.unsqueeze(0)
    
    def _encode_history(self, states: List[TransformationState]) -> torch.Tensor:
        """Encode transformation history"""
        max_history = 10
        history_encoded = torch.zeros(1, max_history, len(TransformationType), device=self.device)
        
        # Get recent transformations
        recent_transformations = []
        for state in states[-max_history:]:
            if state.transformation_history:
                recent_transformations.extend(state.transformation_history[-1:])  # Last transformation only
        
        # Encode transformations as one-hot
        for i, transformation in enumerate(recent_transformations[-max_history:]):
            action_idx = self._action_to_index(transformation)
            history_encoded[0, i, action_idx] = 1.0
        
        return history_encoded
    
    def _action_to_index(self, action: TransformationType) -> int:
        """Convert action to index"""
        return list(TransformationType).index(action)
    
    def _index_to_action(self, index: int) -> TransformationType:
        """Convert index to action"""
        return list(TransformationType)[index]
    
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute advantages using GAE"""
        gamma = 0.99
        lam = 0.95
        
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (values[t+1] if t+1 < len(values) else 0) - values[t]
            advantage = delta + gamma * lam * advantage
            advantages.insert(0, advantage)
        
        return torch.stack(advantages)
    
    def _enhance_reward_with_warning_reduction(self, original_reward: float, 
                                            current_state: TransformationState,
                                            episode_states: List[TransformationState]) -> float:
        """Enhance reward with warning reduction evaluation"""
        try:
            # If we have multiple states in episode, evaluate warning reduction
            if len(episode_states) > 1:
                from augmentation_sequence_evaluator import AugmentationSequenceEvaluator
                evaluator = AugmentationSequenceEvaluator()
                
                # Compare first and current state
                original_state = episode_states[0]
                warning_reduction = evaluator.evaluate_warning_reduction(original_state, current_state)
                
                # Weight the reward: 60% warning reduction + 40% original reward
                enhanced_reward = 0.6 * warning_reduction + 0.4 * original_reward
                
                return enhanced_reward
            else:
                return original_reward
                
        except Exception as e:
            logger.debug(f"Error enhancing reward: {e}")
            return original_reward

class MCTSAugmentationSearch(AugmentationPolicyLearner):
    """Monte Carlo Tree Search for augmentation sequence discovery"""
    
    def __init__(self, exploration_constant: float = 2.0, max_iterations: int = 1000,
                 registry: Optional[UnifiedAugmentationRegistry] = None):
        super().__init__(registry)
        self.exploration_constant = exploration_constant  # Higher for more random exploration
        self.max_iterations = max_iterations
        
        # Random walk policy for guided simulation
        self.random_walk_policy = defaultdict(float)  # Track successful walk patterns
        self.historical_success = defaultdict(list)  # Store successful walk sequences
        
        # Statistics
        self.search_stats = {
            'iterations': 0,
            'nodes_created': 0,
            'simulations': 0,
            'best_sequences': [],
            'random_walk_simulations': 0,
            'guided_simulations': 0
        }
    
    def learn_policy(self, training_data: List[TrainingEpisode]) -> Dict[str, Any]:
        """Learn policy from training data (not used in MCTS)"""
        return {'method': 'mcts', 'note': 'MCTS does not learn from training data'}
    
    def select_action(self, state: TransformationState, 
                     valid_actions: List[TransformationType],
                     code_location: Optional[CodeLocation] = None) -> TransformationType:
        """Select action using MCTS"""
        if code_location:
            valid_actions = self.registry.get_valid_transformations(state.code, code_location)
        
        if not valid_actions:
            return random.choice(list(TransformationType))
        
        # Simple selection for now - could be enhanced with tree search
        return random.choice(valid_actions)
    
    class MCTSNode:
        """Node in MCTS tree"""
        
        def __init__(self, state: TransformationState, parent=None, action=None):
            self.state = state
            self.parent = parent
            self.action = action
            self.children = []
            self.visits = 0
            self.total_reward = 0.0
            self.untried_actions = []
            self.is_terminal = False
        
        def is_fully_expanded(self) -> bool:
            return len(self.untried_actions) == 0
        
        def ucb_score(self, exploration_constant: float) -> float:
            """Calculate UCB1 score"""
            if self.visits == 0:
                return float('inf')
            
            exploitation = self.total_reward / self.visits
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
            
            return exploitation + exploration
        
        def select_child(self, exploration_constant: float):
            """Select best child using UCB1"""
            return max(self.children, key=lambda child: child.ucb_score(exploration_constant))
    
    def search(self, initial_state: TransformationState, 
               engine: RecursiveAugmentationEngine,
               evaluator: 'AugmentationSequenceEvaluator',
               max_depth: int = 5) -> List[TransformationType]:
        """Perform MCTS search for optimal augmentation sequence"""
        
        root = self.MCTSNode(initial_state)
        root.untried_actions = engine.get_valid_next_transformations(initial_state)
        
        for iteration in range(self.max_iterations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal and not node.is_fully_expanded():
                node = self._expand(node, engine)
            
            # Simulation
            reward = self._simulate(node, engine, evaluator, max_depth)
            
            # Backpropagation
            self._backpropagate(node, reward)
            
            self.search_stats['iterations'] += 1
        
        # Return best sequence
        if root.children:
            best_child = max(root.children, key=lambda child: child.total_reward / max(child.visits, 1))
            return self._extract_sequence(best_child)
        
        return []
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase - traverse tree using UCB1"""
        while not node.is_terminal and node.is_fully_expanded():
            node = node.select_child(self.exploration_constant)
        return node
    
    def _expand(self, node: MCTSNode, engine: RecursiveAugmentationEngine) -> MCTSNode:
        """Expansion phase - add new child node"""
        if not node.untried_actions:
            return node
        
        action = node.untried_actions.pop()
        
        # Apply action to create new state
        new_states = engine.apply_recursive_transformation(
            node.state.code, 
            max_depth=1, 
            transformation_sequence=[action]
        )
        
        if len(new_states) > 1:
            new_state = new_states[1]  # Get transformed state
            child = self.MCTSNode(new_state, parent=node, action=action)
            child.untried_actions = engine.get_valid_next_transformations(new_state)
            
            # Check if terminal
            if child.state.depth >= 5 or not child.untried_actions:
                child.is_terminal = True
            
            node.children.append(child)
            self.search_stats['nodes_created'] += 1
            
            return child
        
        return node
    
    def _simulate(self, node: MCTSNode, engine: RecursiveAugmentationEngine,
                  evaluator: 'AugmentationSequenceEvaluator', max_depth: int) -> float:
        """Simulation phase - guided random walk rollout"""
        current_state = node.state
        depth = current_state.depth
        sequence_reward = 0.0
        walk_sequence = []
        
        while depth < max_depth:
            valid_actions = engine.get_valid_next_transformations(current_state)
            if not valid_actions:
                break
            
            # Use guided random walk for action selection
            action = self._guided_random_walk(current_state, valid_actions, walk_sequence)
            walk_sequence.append(action)
            
            # Apply action
            new_states = engine.apply_recursive_transformation(
                current_state.code,
                max_depth=1,
                transformation_sequence=[action]
            )
            
            if len(new_states) > 1:
                current_state = new_states[1]
                depth += 1
                
                # Evaluate intermediate reward (prioritize warning reduction)
                reward = self._evaluate_transformation_with_warning_reduction(
                    evaluator, current_state, node.state
                )
                sequence_reward += reward
            else:
                break
        
        # Final evaluation with warning reduction bonus
        final_reward = self._evaluate_final_sequence_with_warning_reduction(
            evaluator, current_state, node.state
        )
        sequence_reward += final_reward
        
        # Update random walk policy based on success
        self._update_random_walk_policy(walk_sequence, sequence_reward)
        
        self.search_stats['simulations'] += 1
        if len(walk_sequence) > 0:
            self.search_stats['guided_simulations'] += 1
        else:
            self.search_stats['random_walk_simulations'] += 1
            
        return sequence_reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase - update node values"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _extract_sequence(self, node: MCTSNode) -> List[TransformationType]:
        """Extract transformation sequence from root to node"""
        sequence = []
        current = node
        
        while current.parent is not None:
            sequence.insert(0, current.action)
            current = current.parent
        
        return sequence
    
    def _guided_random_walk(self, current_state: TransformationState, 
                          valid_actions: List[TransformationType],
                          walk_sequence: List[TransformationType]) -> TransformationType:
        """Use historical data to bias random walk selection"""
        if not self.random_walk_policy or not walk_sequence:
            # No historical data - use random selection
            return random.choice(valid_actions)
        
        # Calculate probabilities based on historical success
        action_probs = []
        for action in valid_actions:
            # Base probability
            base_prob = 1.0 / len(valid_actions)
            
            # Historical success bonus
            action_key = tuple(walk_sequence + [action])
            historical_bonus = self.random_walk_policy.get(action_key, 0.0)
            
            # Combined probability
            combined_prob = base_prob + 0.3 * historical_bonus
            action_probs.append(combined_prob)
        
        # Normalize probabilities
        total_prob = sum(action_probs)
        if total_prob > 0:
            action_probs = [p / total_prob for p in action_probs]
            return np.random.choice(valid_actions, p=action_probs)
        else:
            return random.choice(valid_actions)
    
    def _update_random_walk_policy(self, walk_sequence: List[TransformationType], 
                                 reward: float):
        """Update random walk policy based on simulation success"""
        if not walk_sequence or reward <= 0:
            return
        
        # Store successful walk sequence
        sequence_key = tuple(walk_sequence)
        self.historical_success[sequence_key].append(reward)
        
        # Update policy weights
        avg_reward = np.mean(self.historical_success[sequence_key])
        self.random_walk_policy[sequence_key] = avg_reward
        
        # Keep only recent successes (limit memory)
        if len(self.historical_success[sequence_key]) > 10:
            self.historical_success[sequence_key] = self.historical_success[sequence_key][-10:]
    
    def _evaluate_transformation_with_warning_reduction(self, evaluator, 
                                                      current_state: TransformationState,
                                                      original_state: TransformationState) -> float:
        """Evaluate transformation with warning reduction priority"""
        try:
            # Get warning reduction score
            warning_reduction = evaluator.evaluate_warning_reduction(original_state, current_state)
            
            # Get other metrics
            compilation_score = 1.0 if current_state.compilation_status else 0.0
            semantic_score = 1.0 if current_state.semantic_preservation else 0.0
            
            # Weighted combination: 60% warning reduction + 20% compilation + 20% semantic
            return 0.6 * warning_reduction + 0.2 * compilation_score + 0.2 * semantic_score
            
        except Exception as e:
            logger.debug(f"Error in warning reduction evaluation: {e}")
            # Fallback to basic evaluation
            return evaluator.evaluate_transformation(current_state)
    
    def _evaluate_final_sequence_with_warning_reduction(self, evaluator,
                                                      current_state: TransformationState,
                                                      original_state: TransformationState) -> float:
        """Evaluate final sequence with warning reduction priority"""
        try:
            # Get warning reduction score
            warning_reduction = evaluator.evaluate_warning_reduction(original_state, current_state)
            
            # Get other metrics
            compilation_score = 1.0 if current_state.compilation_status else 0.0
            semantic_score = 1.0 if current_state.semantic_preservation else 0.0
            depth_bonus = min(current_state.depth / 5.0, 1.0)
            
            # Weighted combination: 60% warning reduction + 15% compilation + 15% semantic + 10% depth
            return 0.6 * warning_reduction + 0.15 * compilation_score + 0.15 * semantic_score + 0.1 * depth_bonus
            
        except Exception as e:
            logger.debug(f"Error in final sequence evaluation: {e}")
            # Fallback to basic evaluation
            return evaluator.evaluate_final_sequence(current_state)

class EvolutionaryAugmentationOptimizer(AugmentationPolicyLearner):
    """Evolutionary algorithm for augmentation sequence optimization"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, tournament_size: int = 5,
                 elitism_ratio: float = 0.1, random_walk_mutation_rate: float = 0.25,
                 walk_steps: int = 3, registry: Optional[UnifiedAugmentationRegistry] = None):
        super().__init__(registry)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = int(population_size * elitism_ratio)
        
        # Random walk mutation parameters
        self.random_walk_mutation_rate = random_walk_mutation_rate
        self.walk_steps = walk_steps
        
        # Random walk success tracking
        self.random_walk_success_patterns = defaultdict(list)
        
        # Statistics
        self.evolution_stats = {
            'generations': 0,
            'best_fitness': [],
            'average_fitness': [],
            'population_diversity': [],
            'random_walk_mutations': 0,
            'successful_random_walks': 0
        }
    
    class TransformationGenome:
        """Genome representing a transformation sequence"""
        
        def __init__(self, sequence: List[TransformationType]):
            self.sequence = sequence
            self.fitness = 0.0
            self.metadata = {}
        
        def __len__(self):
            return len(self.sequence)
        
        def copy(self):
            return self.__class__(self.sequence.copy())
    
    def learn_policy(self, training_data: List[TrainingEpisode]) -> Dict[str, Any]:
        """Learn policy from training data (not used in Evolutionary)"""
        return {'method': 'evolutionary', 'note': 'Evolutionary does not learn from training data'}
    
    def select_action(self, state: TransformationState, 
                     valid_actions: List[TransformationType],
                     code_location: Optional[CodeLocation] = None) -> TransformationType:
        """Select action using evolutionary approach"""
        if code_location:
            valid_actions = self.registry.get_valid_transformations(state.code, code_location)
        
        if not valid_actions:
            return random.choice(list(TransformationType))
        
        # Simple selection for now - could be enhanced with population-based selection
        return random.choice(valid_actions)
    
    def optimize(self, initial_code: str, engine: RecursiveAugmentationEngine,
                 evaluator: 'AugmentationSequenceEvaluator',
                 max_generations: int = 100) -> TransformationGenome:
        """Run evolutionary optimization"""
        
        # Initialize population
        population = self._initialize_population(initial_code, engine)
        
        for generation in range(max_generations):
            # Evaluate fitness
            for genome in population:
                genome.fitness = self._evaluate_fitness(genome, initial_code, engine, evaluator)
            
            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)
            
            # Update statistics
            best_fitness = population[0].fitness
            avg_fitness = sum(g.fitness for g in population) / len(population)
            diversity = self._calculate_diversity(population)
            
            self.evolution_stats['generations'] = generation + 1
            self.evolution_stats['best_fitness'].append(best_fitness)
            self.evolution_stats['average_fitness'].append(avg_fitness)
            self.evolution_stats['population_diversity'].append(diversity)
            
            logger.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.3f}, "
                       f"Average = {avg_fitness:.3f}, Diversity = {diversity:.3f}")
            
            # Check convergence
            if self._check_convergence(population):
                logger.info("Convergence reached")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism - keep best individuals
            new_population.extend(population[:self.elitism_count])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = self._mutate(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Replace population
            population = new_population[:self.population_size]
        
        # Return best genome
        population.sort(key=lambda g: g.fitness, reverse=True)
        return population[0]
    
    def _initialize_population(self, initial_code: str, 
                             engine: RecursiveAugmentationEngine) -> List[TransformationGenome]:
        """Initialize population with random transformation sequences"""
        population = []
        
        for _ in range(self.population_size):
            # Generate random sequence
            sequence_length = random.randint(2, 6)
            sequence = []
            
            current_code = initial_code
            for _ in range(sequence_length):
                valid_actions = engine.get_valid_next_transformations(
                    TransformationState(
                        code=current_code,
                        transformation_history=sequence,
                        depth=len(sequence),
                        complexity_score=0.0,
                        compilation_status=True,
                        semantic_preservation=True,
                        metadata={}
                    )
                )
                
                if valid_actions:
                    action = random.choice(valid_actions)
                    sequence.append(action)
                    
                    # Apply action to update current code
                    states = engine.apply_recursive_transformation(
                        current_code, max_depth=1, transformation_sequence=[action]
                    )
                    if len(states) > 1:
                        current_code = states[1].code
                else:
                    break
            
            genome = self.TransformationGenome(sequence)
            population.append(genome)
        
        return population
    
    def _evaluate_fitness(self, genome: TransformationGenome, initial_code: str,
                         engine: RecursiveAugmentationEngine,
                         evaluator: 'AugmentationSequenceEvaluator') -> float:
        """Evaluate fitness of a genome"""
        try:
            # Apply transformation sequence
            states = engine.apply_recursive_transformation(
                initial_code,
                max_depth=len(genome.sequence),
                transformation_sequence=genome.sequence
            )
            
            if len(states) <= 1:
                return 0.0
            
            final_state = states[-1]
            
            # Evaluate sequence
            fitness = evaluator.evaluate_sequence(states)
            
            # Add penalties for invalid transformations
            if not final_state.compilation_status:
                fitness *= 0.5
            if not final_state.semantic_preservation:
                fitness *= 0.7
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Error evaluating genome: {e}")
            return 0.0
    
    def _tournament_selection(self, population: List[TransformationGenome]) -> TransformationGenome:
        """Tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda g: g.fitness)
    
    def _crossover(self, parent1: TransformationGenome, 
                  parent2: TransformationGenome) -> Tuple[TransformationGenome, TransformationGenome]:
        """Crossover two genomes"""
        # Uniform crossover
        sequence1 = []
        sequence2 = []
        
        max_length = max(len(parent1), len(parent2))
        
        for i in range(max_length):
            if i < len(parent1) and i < len(parent2):
                if random.random() < 0.5:
                    sequence1.append(parent1.sequence[i])
                    sequence2.append(parent2.sequence[i])
                else:
                    sequence1.append(parent2.sequence[i])
                    sequence2.append(parent1.sequence[i])
            elif i < len(parent1):
                sequence1.append(parent1.sequence[i])
                sequence2.append(parent1.sequence[i])
            else:
                sequence1.append(parent2.sequence[i])
                sequence2.append(parent2.sequence[i])
        
        return (self.TransformationGenome(sequence1), 
                self.TransformationGenome(sequence2))
    
    def _mutate(self, genome: TransformationGenome) -> TransformationGenome:
        """Mutate a genome with random walk mutation option"""
        mutated = genome.copy()
        
        # Decide mutation type with random walk option
        mutation_types = ['insert', 'delete', 'replace', 'random_walk']
        mutation_weights = [0.25, 0.25, 0.25, self.random_walk_mutation_rate]
        
        mutation_type = np.random.choice(mutation_types, p=mutation_weights)
        
        if mutation_type == 'insert' and len(mutated.sequence) < 8:
            # Insert random transformation
            position = random.randint(0, len(mutated.sequence))
            transformation = random.choice(list(TransformationType))
            mutated.sequence.insert(position, transformation)
        
        elif mutation_type == 'delete' and len(mutated.sequence) > 1:
            # Delete random transformation
            position = random.randint(0, len(mutated.sequence) - 1)
            mutated.sequence.pop(position)
        
        elif mutation_type == 'replace':
            # Replace random transformation
            if mutated.sequence:
                position = random.randint(0, len(mutated.sequence) - 1)
                transformation = random.choice(list(TransformationType))
                mutated.sequence[position] = transformation
        
        elif mutation_type == 'random_walk':
            # Random walk mutation
            mutated = self._random_walk_mutate(mutated)
            self.evolution_stats['random_walk_mutations'] += 1
        
        return mutated
    
    def _calculate_diversity(self, population: List[TransformationGenome]) -> float:
        """Calculate population diversity"""
        if len(population) <= 1:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._sequence_distance(population[i].sequence, population[j].sequence)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _sequence_distance(self, seq1: List[TransformationType], 
                          seq2: List[TransformationType]) -> float:
        """Calculate distance between two sequences"""
        # Jaccard distance
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 1.0
        
        return 1.0 - (intersection / union)
    
    def _check_convergence(self, population: List[TransformationGenome], 
                          threshold: float = 0.95) -> bool:
        """Check if population has converged"""
        if len(population) < 2:
            return False
        
        # Check fitness convergence
        fitnesses = [g.fitness for g in population]
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        if max_fitness > 0:
            convergence_ratio = avg_fitness / max_fitness
            return convergence_ratio > threshold
        
        return False
    
    def _random_walk_mutate(self, genome: TransformationGenome) -> TransformationGenome:
        """Perform random walk mutation on genome"""
        try:
            # Import here to avoid circular imports
            from augmentation_sequence_evaluator import AugmentationSequenceEvaluator
            from recursive_augmentation_engine import RecursiveAugmentationEngine
            
            # Initialize components
            evaluator = AugmentationSequenceEvaluator()
            engine = RecursiveAugmentationEngine()
            
            # Create dummy initial state for the walk
            initial_state = TransformationState(
                code="public class Test { public int method() { return 0; } }",
                transformation_history=[],
                depth=0,
                complexity_score=2.0,
                compilation_status=True,
                semantic_preservation=True,
                metadata={}
            )
            
            # Perform k-step random walk from current sequence
            current_sequence = genome.sequence.copy()
            best_sequence = current_sequence.copy()
            best_fitness = genome.fitness
            
            for step in range(self.walk_steps):
                # Get valid next transformations
                current_state = TransformationState(
                    code=initial_state.code,
                    transformation_history=current_sequence,
                    depth=len(current_sequence),
                    complexity_score=initial_state.complexity_score,
                    compilation_status=initial_state.compilation_status,
                    semantic_preservation=initial_state.semantic_preservation,
                    metadata=initial_state.metadata
                )
                
                valid_actions = engine.get_valid_next_transformations(current_state)
                if not valid_actions:
                    break
                
                # Use guided selection based on historical success
                action = self._guided_random_walk_selection(valid_actions, current_sequence)
                current_sequence.append(action)
                
                # Evaluate fitness of new sequence
                try:
                    states = engine.apply_recursive_transformation(
                        initial_state.code,
                        max_depth=len(current_sequence),
                        transformation_sequence=current_sequence
                    )
                    
                    if len(states) > 1:
                        # Evaluate warning reduction
                        warning_reduction = evaluator.evaluate_warning_reduction(initial_state, states[-1])
                        
                        # Track successful random walk patterns
                        sequence_key = tuple(current_sequence)
                        self.random_walk_success_patterns[sequence_key].append(warning_reduction)
                        
                        # Update best sequence if improvement found
                        if warning_reduction > best_fitness:
                            best_fitness = warning_reduction
                            best_sequence = current_sequence.copy()
                            self.evolution_stats['successful_random_walks'] += 1
                    
                except Exception as e:
                    logger.debug(f"Error evaluating random walk sequence: {e}")
                    continue
            
            # Return genome with best sequence found during random walk
            return self.TransformationGenome(best_sequence)
            
        except Exception as e:
            logger.warning(f"Error in random walk mutation: {e}")
            return genome  # Return original genome on error
    
    def _guided_random_walk_selection(self, valid_actions: List[TransformationType],
                                    current_sequence: List[TransformationType]) -> TransformationType:
        """Select next transformation using historical success patterns"""
        if not self.random_walk_success_patterns or not current_sequence:
            return random.choice(valid_actions)
        
        # Calculate probabilities based on historical success
        action_probs = []
        for action in valid_actions:
            # Base probability
            base_prob = 1.0 / len(valid_actions)
            
            # Historical success bonus
            test_sequence = current_sequence + [action]
            sequence_key = tuple(test_sequence)
            
            if sequence_key in self.random_walk_success_patterns:
                historical_success = np.mean(self.random_walk_success_patterns[sequence_key])
                success_bonus = min(historical_success * 0.5, 0.3)  # Cap bonus
            else:
                success_bonus = 0.0
            
            # Combined probability
            combined_prob = base_prob + success_bonus
            action_probs.append(combined_prob)
        
        # Normalize probabilities
        total_prob = sum(action_probs)
        if total_prob > 0:
            action_probs = [p / total_prob for p in action_probs]
            return np.random.choice(valid_actions, p=action_probs)
        else:
            return random.choice(valid_actions)


class RandomWalkOptimizer(AugmentationPolicyLearner):
    """Orchestrates all random walk-based optimization methods"""
    
    def __init__(self, methods: List[str] = None, device: str = 'cpu',
                 registry: Optional[UnifiedAugmentationRegistry] = None):
        super().__init__(registry)
        if methods is None:
            methods = ['rl', 'mcts', 'graph', 'evolutionary']
        
        self.methods = methods
        self.device = device
        
        # Initialize components
        self.components = {}
        
        if 'rl' in methods:
            self.components['rl'] = ReinforcementLearningPolicy(device=device, registry=self.registry)
            logger.info("Initialized RL policy with random walk exploration")
        
        if 'mcts' in methods:
            self.components['mcts'] = MCTSAugmentationSearch(registry=self.registry)
            logger.info("Initialized MCTS with guided random walks")
        
        if 'graph' in methods:
            from graph_based_random_walk_optimizer import TransformationGraphWalker
            self.components['graph'] = TransformationGraphWalker()
            logger.info("Initialized graph-based random walk optimizer")
        
        if 'evolutionary' in methods:
            self.components['evolutionary'] = EvolutionaryAugmentationOptimizer(registry=self.registry)
            logger.info("Initialized evolutionary optimizer with random walk mutation")
        
        # Initialize evaluator
        from augmentation_sequence_evaluator import AugmentationSequenceEvaluator
        self.evaluator = AugmentationSequenceEvaluator(device=device)
        
        # Initialize engine
        from recursive_augmentation_engine import RecursiveAugmentationEngine
        self.engine = RecursiveAugmentationEngine()
        
        # Statistics
        self.optimization_stats = {
            'total_optimizations': 0,
            'method_results': {},
            'best_warning_reductions': [],
            'average_warning_reductions': [],
            'method_usage': defaultdict(int)
        }
    
    def learn_policy(self, training_data: List[TrainingEpisode]) -> Dict[str, Any]:
        """Learn policy from training data using ensemble methods"""
        results = {}
        for method_name, component in self.components.items():
            if hasattr(component, 'learn_policy'):
                results[method_name] = component.learn_policy(training_data)
        return {'method': 'ensemble', 'results': results}
    
    def select_action(self, state: TransformationState, 
                     valid_actions: List[TransformationType],
                     code_location: Optional[CodeLocation] = None) -> TransformationType:
        """Select action using ensemble of methods"""
        if code_location:
            valid_actions = self.registry.get_valid_transformations(state.code, code_location)
        
        if not valid_actions:
            return random.choice(list(TransformationType))
        
        # Use the best performing method (RL by default)
        if 'rl' in self.components:
            return self.components['rl'].select_action(state, valid_actions, code_location)
        else:
            # Fallback to random selection
            return random.choice(valid_actions)
    
    def optimize_augmentation_sequence(self, initial_code: str, 
                                     max_iterations: int = 100,
                                     parallel: bool = True) -> Dict[str, Any]:
        """Find optimal sequence using ensemble of random walk methods"""
        logger.info(f"Starting random walk optimization with methods: {self.methods}")
        
        self.optimization_stats['total_optimizations'] += 1
        
        if parallel and len(self.methods) > 1:
            return self._parallel_optimization(initial_code, max_iterations)
        else:
            return self._sequential_optimization(initial_code, max_iterations)
    
    def _parallel_optimization(self, initial_code: str, max_iterations: int) -> Dict[str, Any]:
        """Run optimization methods in parallel"""
        import concurrent.futures
        import threading
        
        results = {}
        threads = []
        
        def run_method(method_name):
            try:
                if method_name == 'rl':
                    result = self._run_rl_optimization(initial_code, max_iterations)
                elif method_name == 'mcts':
                    result = self._run_mcts_optimization(initial_code, max_iterations)
                elif method_name == 'graph':
                    result = self._run_graph_optimization(initial_code, max_iterations)
                elif method_name == 'evolutionary':
                    result = self._run_evolutionary_optimization(initial_code, max_iterations)
                else:
                    result = {'error': f'Unknown method: {method_name}'}
                
                results[method_name] = result
                self.optimization_stats['method_usage'][method_name] += 1
                
            except Exception as e:
                logger.error(f"Error in {method_name} optimization: {e}")
                results[method_name] = {'error': str(e)}
        
        # Run methods in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.methods)) as executor:
            futures = {executor.submit(run_method, method): method for method in self.methods}
            
            for future in concurrent.futures.as_completed(futures):
                method = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Exception in {method}: {e}")
        
        # Combine results
        return self._combine_results(results)
    
    def _sequential_optimization(self, initial_code: str, max_iterations: int) -> Dict[str, Any]:
        """Run optimization methods sequentially"""
        results = {}
        
        for method in self.methods:
            try:
                if method == 'rl':
                    result = self._run_rl_optimization(initial_code, max_iterations)
                elif method == 'mcts':
                    result = self._run_mcts_optimization(initial_code, max_iterations)
                elif method == 'graph':
                    result = self._run_graph_optimization(initial_code, max_iterations)
                elif method == 'evolutionary':
                    result = self._run_evolutionary_optimization(initial_code, max_iterations)
                else:
                    result = {'error': f'Unknown method: {method}'}
                
                results[method] = result
                self.optimization_stats['method_usage'][method] += 1
                
            except Exception as e:
                logger.error(f"Error in {method} optimization: {e}")
                results[method] = {'error': str(e)}
        
        return self._combine_results(results)
    
    def _run_rl_optimization(self, initial_code: str, max_iterations: int) -> Dict[str, Any]:
        """Run RL optimization"""
        try:
            rl_policy = self.components['rl']
            
            # Create initial state
            initial_state = TransformationState(
                code=initial_code,
                transformation_history=[],
                depth=0,
                complexity_score=3.0,
                compilation_status=True,
                semantic_preservation=True,
                metadata={}
            )
            
            # Generate episodes using RL policy
            episodes = []
            for _ in range(min(max_iterations, 20)):  # Limit episodes for RL
                episode = self._generate_episode_with_policy(rl_policy, initial_state)
                if episode:
                    episodes.append(episode)
            
            # Train policy
            if episodes:
                training_result = rl_policy.learn_policy(episodes)
                
                # Generate final sequence
                final_sequence = []
                current_state = initial_state
                
                for _ in range(5):  # Max sequence length
                    valid_actions = self.engine.get_valid_next_transformations(current_state)
                    if not valid_actions:
                        break
                    
                    action = rl_policy.select_action(current_state, valid_actions)
                    final_sequence.append(action)
                    
                    # Apply transformation
                    new_states = self.engine.apply_recursive_transformation(
                        current_state.code,
                        max_depth=1,
                        transformation_sequence=[action]
                    )
                    
                    if len(new_states) > 1:
                        current_state = new_states[1]
                    else:
                        break
                
                # Evaluate final sequence
                warning_reduction = self.evaluator.evaluate_warning_reduction(initial_state, current_state)
                
                return {
                    'method': 'rl',
                    'sequence': [t.value for t in final_sequence],
                    'warning_reduction': warning_reduction,
                    'episodes_generated': len(episodes),
                    'training_result': training_result
                }
            else:
                return {'method': 'rl', 'error': 'No episodes generated'}
                
        except Exception as e:
            return {'method': 'rl', 'error': str(e)}
    
    def _run_mcts_optimization(self, initial_code: str, max_iterations: int) -> Dict[str, Any]:
        """Run MCTS optimization"""
        try:
            mcts_search = self.components['mcts']
            
            # Create initial state
            initial_state = TransformationState(
                code=initial_code,
                transformation_history=[],
                depth=0,
                complexity_score=3.0,
                compilation_status=True,
                semantic_preservation=True,
                metadata={}
            )
            
            # Run MCTS search
            best_sequence = mcts_search.search(
                initial_state,
                self.engine,
                self.evaluator,
                max_depth=min(max_iterations // 10, 10)
            )
            
            # Apply sequence and evaluate
            if best_sequence:
                states = self.engine.apply_recursive_transformation(
                    initial_code,
                    max_depth=len(best_sequence),
                    transformation_sequence=best_sequence
                )
                
                if len(states) > 1:
                    warning_reduction = self.evaluator.evaluate_warning_reduction(initial_state, states[-1])
                else:
                    warning_reduction = 0.0
            else:
                warning_reduction = 0.0
            
            return {
                'method': 'mcts',
                'sequence': [t.value for t in best_sequence],
                'warning_reduction': warning_reduction,
                'iterations': mcts_search.search_stats['iterations'],
                'search_stats': mcts_search.search_stats
            }
            
        except Exception as e:
            return {'method': 'mcts', 'error': str(e)}
    
    def _run_graph_optimization(self, initial_code: str, max_iterations: int) -> Dict[str, Any]:
        """Run graph-based optimization"""
        try:
            graph_walker = self.components['graph']
            
            # Run graph-based optimization
            result = graph_walker.optimize_augmentation_sequence(initial_code, max_iterations)
            
            return {
                'method': 'graph',
                'sequence': [t.value for t in result.walk],
                'warning_reduction': result.warning_reduction,
                'overall_score': result.overall_score,
                'metadata': result.metadata,
                'walker_stats': graph_walker.get_statistics()
            }
            
        except Exception as e:
            return {'method': 'graph', 'error': str(e)}
    
    def _run_evolutionary_optimization(self, initial_code: str, max_iterations: int) -> Dict[str, Any]:
        """Run evolutionary optimization"""
        try:
            evo_optimizer = self.components['evolutionary']
            
            # Run evolutionary optimization
            best_genome = evo_optimizer.optimize(
                initial_code,
                self.engine,
                self.evaluator,
                max_generations=min(max_iterations // 10, 50)
            )
            
            # Apply best sequence and evaluate
            if best_genome.sequence:
                states = self.engine.apply_recursive_transformation(
                    initial_code,
                    max_depth=len(best_genome.sequence),
                    transformation_sequence=best_genome.sequence
                )
                
                if len(states) > 1:
                    initial_state = states[0]
                    warning_reduction = self.evaluator.evaluate_warning_reduction(initial_state, states[-1])
                else:
                    warning_reduction = 0.0
            else:
                warning_reduction = 0.0
            
            return {
                'method': 'evolutionary',
                'sequence': [t.value for t in best_genome.sequence],
                'warning_reduction': warning_reduction,
                'fitness': best_genome.fitness,
                'evolution_stats': evo_optimizer.evolution_stats
            }
            
        except Exception as e:
            return {'method': 'evolutionary', 'error': str(e)}
    
    def _generate_episode_with_policy(self, policy, initial_state: TransformationState) -> Optional[TrainingEpisode]:
        """Generate training episode using policy"""
        try:
            states = [initial_state]
            actions = []
            rewards = []
            
            current_state = initial_state
            
            for step in range(5):  # Max episode length
                valid_actions = self.engine.get_valid_next_transformations(current_state)
                if not valid_actions:
                    break
                
                action = policy.select_action(current_state, valid_actions)
                actions.append(action)
                
                # Apply transformation
                new_states = self.engine.apply_recursive_transformation(
                    current_state.code,
                    max_depth=1,
                    transformation_sequence=[action]
                )
                
                if len(new_states) > 1:
                    new_state = new_states[1]
                    states.append(new_state)
                    
                    # Calculate reward (warning reduction)
                    reward = self.evaluator.evaluate_warning_reduction(initial_state, new_state)
                    rewards.append(reward)
                    
                    current_state = new_state
                else:
                    break
            
            if actions:
                total_reward = sum(rewards)
                return TrainingEpisode(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    total_reward=total_reward
                )
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error generating episode: {e}")
            return None
    
    def _combine_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from all methods"""
        # Find best result based on warning reduction
        best_method = None
        best_warning_reduction = -1.0
        
        valid_results = {}
        for method, result in results.items():
            if 'error' not in result and 'warning_reduction' in result:
                valid_results[method] = result
                if result['warning_reduction'] > best_warning_reduction:
                    best_warning_reduction = result['warning_reduction']
                    best_method = method
        
        # Calculate average warning reduction
        if valid_results:
            avg_warning_reduction = np.mean([r['warning_reduction'] for r in valid_results.values()])
        else:
            avg_warning_reduction = 0.0
        
        # Update statistics
        self.optimization_stats['best_warning_reductions'].append(best_warning_reduction)
        self.optimization_stats['average_warning_reductions'].append(avg_warning_reduction)
        self.optimization_stats['method_results'] = results
        
        return {
            'best_method': best_method,
            'best_warning_reduction': best_warning_reduction,
            'average_warning_reduction': avg_warning_reduction,
            'method_results': results,
            'valid_methods': list(valid_results.keys()),
            'optimization_stats': self.optimization_stats
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = self.optimization_stats.copy()
        
        # Add computed statistics
        if stats['best_warning_reductions']:
            stats['best_warning_reduction_avg'] = np.mean(stats['best_warning_reductions'])
            stats['best_warning_reduction_std'] = np.std(stats['best_warning_reductions'])
            stats['best_warning_reduction_max'] = np.max(stats['best_warning_reductions'])
        
        if stats['average_warning_reductions']:
            stats['average_warning_reduction_avg'] = np.mean(stats['average_warning_reductions'])
            stats['average_warning_reduction_std'] = np.std(stats['average_warning_reductions'])
        
        return stats


def main():
    """Test the augmentation policy learners"""
    # This would be called with actual training data in practice
    logger.info("Augmentation policy learners implemented successfully")
    
    # Test RL policy network
    rl_learner = ReinforcementLearningPolicy(device='cpu')
    logger.info("RL policy learner initialized")
    
    # Test MCTS
    mcts_search = MCTSAugmentationSearch()
    logger.info("MCTS search initialized")
    
    # Test Evolutionary algorithm
    evo_optimizer = EvolutionaryAugmentationOptimizer()
    logger.info("Evolutionary optimizer initialized")
    
    # Test Random Walk Optimizer
    rw_optimizer = RandomWalkOptimizer(device='cpu')
    logger.info("Random Walk Optimizer initialized with all methods")


if __name__ == '__main__':
    main()
