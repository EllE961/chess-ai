"""
Monte Carlo Tree Search implementation for the chess AI.

This module implements the MCTS algorithm used to search for the best move in a chess position,
guided by a neural network for position evaluation.
"""

import chess
import math
import numpy as np
import torch
import logging
import random
from typing import Dict, List, Tuple, Optional, Any, Union

from .chess_environment import ChessEnvironment
from .neural_network import ChessNetwork

logger = logging.getLogger(__name__)

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    
    Each node represents a game state and stores statistics from the search process.
    """
    
    def __init__(self, prior: float = 0.0, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        """
        Initialize a new MCTS node.
        
        Args:
            prior: Prior probability assigned to this node by the policy network.
            parent: Parent node.
            move: Chess move that led to this node from the parent.
        """
        # Search statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        
        # Tree structure
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, MCTSNode] = {}
        
        # Virtual loss for parallel MCTS
        self.virtual_loss = 0
        
    def expanded(self) -> bool:
        """
        Check if the node has been expanded.
        
        Returns:
            True if the node has children, False otherwise.
        """
        return len(self.children) > 0
        
    def value(self) -> float:
        """
        Calculate the mean value of this node.
        
        Returns:
            The mean value, or 0 if the node has not been visited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def add_virtual_loss(self, virtual_loss: int) -> None:
        """
        Add virtual loss to this node for parallel MCTS.
        
        Virtual loss discourages multiple threads from exploring the same path.
        
        Args:
            virtual_loss: Amount of virtual loss to add.
        """
        self.virtual_loss += virtual_loss
        
    def remove_virtual_loss(self, virtual_loss: int) -> None:
        """
        Remove virtual loss from this node.
        
        Args:
            virtual_loss: Amount of virtual loss to remove.
        """
        self.virtual_loss -= virtual_loss
        
    def get_ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """
        Calculate the UCB score for this node using the PUCT algorithm.
        
        The score balances exploitation (node value) with exploration (prior probability
        and visit counts).
        
        Args:
            parent_visit_count: Visit count of the parent node.
            c_puct: Exploration constant.
            
        Returns:
            The UCB score.
        """
        # Q value (exploitation)
        q_value = self.value()
        
        # U value (exploration)
        u_value = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        # Subtract virtual loss to discourage parallel exploration of the same path
        visit_adjustment = self.virtual_loss / (1 + self.visit_count) if self.visit_count > 0 else 0
        
        return q_value + u_value - visit_adjustment
        
    def select_child(self, c_puct: float) -> Tuple[chess.Move, 'MCTSNode']:
        """
        Select the child with the highest UCB score.
        
        Args:
            c_puct: Exploration constant.
            
        Returns:
            Tuple of (move, child_node) for the selected child.
            
        Raises:
            ValueError: If the node has no children.
        """
        if not self.expanded():
            raise ValueError("Cannot select child of unexpanded node")
            
        # Calculate UCB scores for all children
        parent_visit_count = self.visit_count
        
        # Select the child with the highest score
        return max(
            self.children.items(),
            key=lambda item: item[1].get_ucb_score(parent_visit_count, c_puct)
        )
        
    def expand(self, policy: Dict[chess.Move, float]) -> None:
        """
        Expand the node with the given policy.
        
        Args:
            policy: Dictionary mapping moves to their prior probabilities.
        """
        for move, prob in policy.items():
            if move not in self.children:
                self.children[move] = MCTSNode(prior=prob, parent=self, move=move)


class MCTS:
    """
    Monte Carlo Tree Search implementation.
    
    Uses a neural network to guide the search process by providing position
    evaluations and move probabilities.
    """
    
    def __init__(self, model: ChessNetwork, config: Dict[str, Any]):
        """
        Initialize the MCTS algorithm.
        
        Args:
            model: Neural network model for position evaluation.
            config: Configuration parameters.
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Extract MCTS configuration
        mcts_config = config.get('mcts', {})
        self.c_puct = mcts_config.get('c_puct', 1.5)
        self.num_simulations = mcts_config.get('num_simulations', 800)
        self.dirichlet_alpha = mcts_config.get('dirichlet_alpha', 0.3)
        self.dirichlet_epsilon = mcts_config.get('dirichlet_epsilon', 0.25)
        self.virtual_loss = mcts_config.get('virtual_loss', 3)
        
        # Temperature parameters for move selection
        self.temperature_init = mcts_config.get('temperature_init', 1.0)
        self.temperature_final = mcts_config.get('temperature_final', 0.1)
        self.move_temp_threshold = mcts_config.get('move_temp_threshold', 30)
        
        logger.info(f"Initialized MCTS with {self.num_simulations} simulations per move")
        
    def search(self, env: ChessEnvironment) -> Dict[chess.Move, float]:
        """
        Run MCTS search on the given position and return move probabilities.
        
        Args:
            env: Chess environment representing the current position.
            
        Returns:
            Dictionary mapping moves to their probabilities.
        """
        logger.debug("Starting MCTS search")
        
        # Create root node
        root = MCTSNode()
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Create a copy of the environment for simulation
            sim_env = env.copy()
            
            # Select and expand a leaf node
            leaf, sim_path = self._select_and_expand(root, sim_env)
            
            # Evaluate the leaf position
            value = self._evaluate(leaf, sim_env)
            
            # Backpropagate the value
            self._backpropagate(leaf, value, sim_path)
            
        # Get the improved policy based on visit counts
        policy = self._get_improved_policy(root, env.board.fullmove_number)
        
        logger.debug(f"MCTS search completed with {self.num_simulations} simulations")
        return policy
        
    def _select_and_expand(self, root: MCTSNode, env: ChessEnvironment) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        Select a path through the tree and expand a leaf node if necessary.
        
        Args:
            root: Root node of the search tree.
            env: Chess environment for the simulation.
            
        Returns:
            Tuple of (leaf_node, search_path):
                - leaf_node: The selected leaf node.
                - search_path: List of nodes traversed, including the leaf.
        """
        search_path = [root]
        node = root
        
        # Add Dirichlet noise to the root node for exploration
        if not root.expanded():
            # First, evaluate the position to get a prior policy
            prior_policy = self._get_policy(env)
            
            # Then add Dirichlet noise to the root node children's priors
            if self.dirichlet_epsilon > 0:
                self._add_dirichlet_noise(prior_policy)
                
            # Expand the root with the noisy policy
            root.expand(prior_policy)
        
        # Selection phase - descend the tree until we reach a leaf node
        while node.expanded() and not env.is_game_over():
            # Select the best child according to UCB
            move, node = node.select_child(self.c_puct)
            
            # Add virtual loss to this node and update the environment
            node.add_virtual_loss(self.virtual_loss)
            env.make_move(move)
            search_path.append(node)
            
        # Expansion phase - if the game is not over, expand the leaf node
        if not env.is_game_over() and not node.expanded():
            prior_policy = self._get_policy(env)
            node.expand(prior_policy)
            
        return node, search_path
        
    def _get_policy(self, env: ChessEnvironment) -> Dict[chess.Move, float]:
        """
        Get the prior policy from the neural network.
        
        Args:
            env: Chess environment representing the current position.
            
        Returns:
            Dictionary mapping legal moves to their prior probabilities.
        """
        # Get board encoding
        encoded_state = env.encode_board()
        encoded_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get policy from neural network
        with torch.no_grad():
            policy_tensor, _ = self.model(encoded_tensor)
            policy_tensor = policy_tensor.squeeze(0).cpu().numpy()
            
        # Convert to dictionary of legal moves only
        policy = {}
        legal_moves = env.get_legal_moves()
        
        policy_sum = 0.0
        for move in legal_moves:
            try:
                move_idx = env.move_to_index(move)
                policy[move] = policy_tensor[move_idx]
                policy_sum += policy[move]
            except (IndexError, ValueError):
                # If move_to_index fails or index is out of bounds, assign small probability
                policy[move] = 1e-6
                policy_sum += 1e-6
                
        # Normalize to ensure probabilities sum to 1
        if policy_sum > 0:
            for move in policy:
                policy[move] /= policy_sum
                
        return policy
        
    def _add_dirichlet_noise(self, policy: Dict[chess.Move, float]) -> None:
        """
        Add Dirichlet noise to the policy for exploration.
        
        Args:
            policy: Dictionary of moves and their prior probabilities.
        """
        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
        
        # Mix the noise with the policy
        moves = list(policy.keys())
        for i, move in enumerate(moves):
            policy[move] = (1 - self.dirichlet_epsilon) * policy[move] + self.dirichlet_epsilon * noise[i]
            
    def _evaluate(self, node: MCTSNode, env: ChessEnvironment) -> float:
        """
        Evaluate the position at the given node.
        
        Args:
            node: The node to evaluate.
            env: Chess environment representing the position.
            
        Returns:
            Value of the position (-1 to 1) from the current player's perspective.
        """
        # If the game is over, use the game result
        if env.is_game_over():
            return env.get_result()
            
        # Otherwise, use the neural network to evaluate the position
        encoded_state = env.encode_board()
        encoded_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            _, value_tensor = self.model(encoded_tensor)
            value = value_tensor.item()
            
        return value
        
    def _backpropagate(self, node: MCTSNode, value: float, search_path: List[MCTSNode]) -> None:
        """
        Backpropagate the value through the search path.
        
        Args:
            node: Leaf node that was evaluated.
            value: Value to backpropagate.
            search_path: Path of nodes from root to leaf.
        """
        # Start with the player at the leaf node
        current_player = len(search_path) % 2
        
        # Traverse the path backwards
        for node in reversed(search_path):
            # Remove virtual loss
            node.remove_virtual_loss(self.virtual_loss)
            
            # Update statistics
            node.visit_count += 1
            
            # Flip the value for alternating players
            if current_player == 1:  # Opponent's perspective
                value = -value
                
            node.value_sum += value
            
            # Switch to the other player
            current_player = 1 - current_player
            
    def _get_improved_policy(self, root: MCTSNode, move_number: int) -> Dict[chess.Move, float]:
        """
        Get the improved policy based on the visit counts of the root's children.
        
        Args:
            root: Root node of the search tree.
            move_number: Current move number for temperature scheduling.
            
        Returns:
            Dictionary mapping moves to their probabilities.
        """
        # Get the temperature parameter based on the move number
        if move_number < self.move_temp_threshold:
            temperature = self.temperature_init
        else:
            temperature = self.temperature_final
            
        # Get visit counts
        visits = {move: child.visit_count for move, child in root.children.items()}
        total_visits = sum(visits.values())
        
        # Calculate the policy based on temperature
        policy = {}
        
        if temperature > 0:
            # Apply temperature scaling
            scaled_visits = {move: count ** (1 / temperature) for move, count in visits.items()}
            total_scaled = sum(scaled_visits.values())
            
            # Normalize to get probabilities
            policy = {move: count / total_scaled for move, count in scaled_visits.items()}
        else:
            # Zero temperature: deterministic policy selecting the most visited child
            best_move = max(visits.items(), key=lambda x: x[1])[0]
            policy = {move: 1.0 if move == best_move else 0.0 for move in visits}
            
        return policy
        
    def select_move(self, env: ChessEnvironment, deterministic: bool = False) -> chess.Move:
        """
        Select a move based on the MCTS search.
        
        Args:
            env: Chess environment representing the current position.
            deterministic: If True, always select the best move. If False, sample according to policy.
            
        Returns:
            The selected chess move.
        """
        # Get the policy from MCTS search
        policy = self.search(env)
        
        if deterministic:
            # Select the move with the highest probability
            return max(policy.items(), key=lambda x: x[1])[0]
        else:
            # Sample a move based on the probabilities
            moves = list(policy.keys())
            probabilities = list(policy.values())
            
            try:
                chosen_move = random.choices(moves, weights=probabilities, k=1)[0]
                return chosen_move
            except ValueError:
                # Fallback if there's an issue with probabilities
                logger.warning("Error sampling move, using deterministic selection")
                return max(policy.items(), key=lambda x: x[1])[0]