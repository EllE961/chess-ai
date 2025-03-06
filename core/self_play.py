"""
Self-play game generation for reinforcement learning.

This module manages the generation of self-play games for training data
using the current neural network model and MCTS.
"""

import chess
import numpy as np
import torch
import logging
import time
import os
from typing import List, Tuple, Dict, Any, Optional

from .chess_environment import ChessEnvironment
from .mcts import MCTS
from .neural_network import ChessNetwork

logger = logging.getLogger(__name__)

# Define a type for training examples
TrainingExample = Tuple[np.ndarray, np.ndarray, float]

class SelfPlay:
    """
    Manages self-play game generation for neural network training.
    
    Generates games by having the AI play against itself, saving states,
    policies, and outcomes for training.
    """
    
    def __init__(self, model: ChessNetwork, config: Dict[str, Any]):
        """
        Initialize the self-play manager.
        
        Args:
            model: Neural network model to use for self-play.
            config: Configuration parameters.
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Create MCTS with the current model
        self.mcts = MCTS(model, config)
        
        # Extract self-play configuration
        training_config = config.get('training', {})
        self.num_games = training_config.get('num_self_play_games', 25)
        
        # Temperature parameters
        mcts_config = config.get('mcts', {})
        self.temperature_init = mcts_config.get('temperature_init', 1.0)
        self.temperature_final = mcts_config.get('temperature_final', 0.1)
        self.move_temp_threshold = mcts_config.get('move_temp_threshold', 30)
        
        # Game recording
        self.game_dir = os.path.join(config.get('data_dir', './data'), 'game_records')
        os.makedirs(self.game_dir, exist_ok=True)
        
        logger.info(f"Initialized self-play with {self.num_games} games per iteration")
        
    def generate_games(self) -> List[Tuple[List[TrainingExample], Dict[str, Any]]]:
        """
        Generate self-play games for training.
        
        Returns:
            List of tuples, each containing:
                - List of training examples from a game
                - Game metadata (moves, result, etc.)
        """
        all_game_data = []
        
        for game_num in range(self.num_games):
            logger.info(f"Starting self-play game {game_num+1}/{self.num_games}")
            start_time = time.time()
            
            # Play a single game
            examples, metadata = self.execute_episode()
            
            # Record game data
            all_game_data.append((examples, metadata))
            
            elapsed = time.time() - start_time
            logger.info(f"Game {game_num+1} completed in {elapsed:.1f}s: "
                        f"{len(metadata['moves'])} moves, result: {metadata['result']}")
            
            # Save game record
            self._save_game_record(metadata, game_num)
            
        return all_game_data
        
    def execute_episode(self) -> Tuple[List[TrainingExample], Dict[str, Any]]:
        """
        Play a full self-play game and generate training examples.
        
        Returns:
            Tuple of:
                - List of training examples (state, policy, value)
                - Game metadata dictionary
        """
        # Initialize environment
        env = ChessEnvironment()
        
        # Initialize data collections
        training_examples = []
        game_history = []
        
        # Track the number of moves without capture or pawn movement (for draw detection)
        consecutive_quiet_moves = 0
        
        # Play until the game is over
        while not env.is_game_over():
            # Get the current board state
            current_state = env.encode_board()
            
            # Run MCTS to get improved policy
            policy = self.mcts.search(env)
            
            # Store the current state and policy
            game_history.append((current_state, policy, env.get_fen()))
            
            # Convert policy to tensor format for training
            policy_tensor = self._policy_to_tensor(policy, env)
            
            # Select a move based on the policy
            temperature = self._get_temperature(env.board.fullmove_number)
            move = self._select_move(policy, temperature)
            
            # Check if move is a capture or pawn move
            is_quiet_move = not env.board.is_capture(move) and env.board.piece_at(move.from_square).piece_type != chess.PAWN
            if is_quiet_move:
                consecutive_quiet_moves += 1
            else:
                consecutive_quiet_moves = 0
                
            # Make the move
            env.make_move(move)
            
            # Early stopping for very long games or likely draws
            if (env.board.fullmove_number > 200 or consecutive_quiet_moves >= 50):
                logger.info(f"Game stopped early after {env.board.fullmove_number} moves")
                break
                
        # Game result
        if env.is_game_over():
            result = env.get_result()
            result_string = "1-0" if result > 0 else ("0-1" if result < 0 else "1/2-1/2")
        else:
            # Draw if game was stopped early
            result = 0
            result_string = "1/2-1/2"
            
        # Create training examples with correct value targets
        for state, policy, _ in game_history:
            # The value target is the final result (from the perspective of the player who made the move)
            value_target = result
            result = -result  # Flip result for the opponent's perspective
            
            # Add to training examples
            training_examples.append((state, policy_tensor, value_target))
            
        # Create metadata for the game record
        metadata = {
            'moves': [m.uci() for m in env.move_history],
            'result': result_string,
            'termination': self._get_termination_reason(env),
            'final_fen': env.get_fen()
        }
        
        return training_examples, metadata
        
    def _policy_to_tensor(self, policy: Dict[chess.Move, float], env: ChessEnvironment) -> np.ndarray:
        """
        Convert policy dictionary to a tensor format for training.
        
        Args:
            policy: Dictionary mapping moves to their probabilities.
            env: Chess environment for move indexing.
            
        Returns:
            NumPy array of shape (policy_output_size,) representing the policy.
        """
        policy_size = self.config.get('neural_network', {}).get('policy_output_size', 4672)
        policy_tensor = np.zeros(policy_size, dtype=np.float32)
        
        for move, prob in policy.items():
            try:
                move_idx = env.move_to_index(move)
                policy_tensor[move_idx] = prob
            except (IndexError, ValueError):
                # Skip if move cannot be properly indexed
                logger.warning(f"Could not index move {move} in policy tensor")
                continue
                
        return policy_tensor
        
    def _get_temperature(self, move_number: int) -> float:
        """
        Get the temperature parameter based on the move number.
        
        Args:
            move_number: Current move number.
            
        Returns:
            Temperature value for move selection.
        """
        if move_number < self.move_temp_threshold:
            return self.temperature_init
        else:
            return self.temperature_final
            
    def _select_move(self, policy: Dict[chess.Move, float], temperature: float) -> chess.Move:
        """
        Select a move from the policy.
        
        Args:
            policy: Dictionary mapping moves to their probabilities.
            temperature: Temperature parameter for move selection.
            
        Returns:
            Selected chess move.
        """
        moves = list(policy.keys())
        probabilities = list(policy.values())
        
        if temperature <= 0.01:
            # Deterministic selection
            return moves[np.argmax(probabilities)]
        else:
            # Apply temperature
            probabilities = np.array(probabilities) ** (1 / temperature)
            probabilities = probabilities / np.sum(probabilities)
            
            # Sample move
            idx = np.random.choice(len(moves), p=probabilities)
            return moves[idx]
            
    def _get_termination_reason(self, env: ChessEnvironment) -> str:
        """
        Get a string describing how the game ended.
        
        Args:
            env: Chess environment at the end of the game.
            
        Returns:
            String describing the termination reason.
        """
        board = env.board
        
        if board.is_checkmate():
            return "checkmate"
        elif board.is_stalemate():
            return "stalemate"
        elif board.is_insufficient_material():
            return "insufficient material"
        elif board.is_fifty_moves():
            return "fifty-move rule"
        elif board.is_repetition():
            return "threefold repetition"
        elif board.fullmove_number > 200:
            return "move limit exceeded"
        else:
            return "unknown"
            
    def _save_game_record(self, metadata: Dict[str, Any], game_num: int) -> None:
        """
        Save the game record to disk.
        
        Args:
            metadata: Game metadata dictionary.
            game_num: Game number for filename.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"game_{timestamp}_{game_num}.pgn"
        filepath = os.path.join(self.game_dir, filename)
        
        # Create PGN string
        pgn = []
        pgn.append("[Event \"Self-play Game\"]")
        pgn.append("[Site \"Local Machine\"]")
        pgn.append(f"[Date \"{time.strftime('%Y.%m.%d')}\"]")
        pgn.append(f"[Round \"{game_num+1}\"]")
        pgn.append("[White \"ChessAI\"]")
        pgn.append("[Black \"ChessAI\"]")
        pgn.append(f"[Result \"{metadata['result']}\"]")
        pgn.append(f"[Termination \"{metadata['termination']}\"]")
        pgn.append(f"[FinalFEN \"{metadata['final_fen']}\"]")
        pgn.append("")
        
        # Add moves
        move_lines = []
        for i, uci in enumerate(metadata['moves']):
            move_num = (i // 2) + 1
            if i % 2 == 0:
                move_lines.append(f"{move_num}. {uci}")
            else:
                move_lines.append(f"{uci}")
                
        # Format the moves nicely with line breaks
        pgn.append(" ".join(move_lines))
        
        # Add the result
        pgn.append(f"{metadata['result']}")
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write("\n".join(pgn))
            
        logger.debug(f"Game record saved to {filepath}")