"""
Model evaluation for the chess AI.

This module provides functionality for evaluating a chess model by playing
games against another model and calculating win rates.
"""

import os
import chess
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

from ..core.neural_network import ChessNetwork
from ..core.mcts import MCTS
from ..core.chess_environment import ChessEnvironment

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates chess models by playing games between them.
    
    Determines if a new model is better than the previous best model by
    playing a series of games and calculating the win rate.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        eval_config = config.get('evaluation', {})
        
        self.num_games = eval_config.get('eval_games', 40)
        self.game_cap = eval_config.get('eval_game_cap', 200)
        self.temperature = eval_config.get('eval_temperature', 0.2)
        
        # Device for model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized model evaluator with {self.num_games} games per evaluation")
        
    def evaluate(self, model1_path: str, model2_path: str) -> float:
        """
        Evaluate two models against each other.
        
        Args:
            model1_path: Path to the first model (typically the new model).
            model2_path: Path to the second model (typically the previous best).
            
        Returns:
            Performance score for model1 (0 to 1, where >0.5 means model1 is better).
        """
        logger.info(f"Evaluating {os.path.basename(model1_path)} vs {os.path.basename(model2_path)}")
        
        # Load models
        model1, config1 = self._load_model(model1_path)
        model2, config2 = self._load_model(model2_path)
        
        # Configure MCTS for each model
        mcts1 = MCTS(model1, self.config)
        mcts2 = MCTS(model2, self.config)
        
        # Track results
        model1_wins = 0
        model2_wins = 0
        draws = 0
        
        # Play evaluation games
        logger.info(f"Playing {self.num_games} evaluation games...")
        for game_idx in tqdm(range(self.num_games)):
            # Alternate which model plays white (to ensure fairness)
            if game_idx % 2 == 0:
                white_model, black_model = mcts1, mcts2
                white_name, black_name = "model1", "model2"
            else:
                white_model, black_model = mcts2, mcts1
                white_name, black_name = "model2", "model1"
                
            # Play the game
            result = self._play_game(white_model, black_model)
            
            # Record the result from model1's perspective
            if result == 1:  # White wins
                if white_name == "model1":
                    model1_wins += 1
                else:
                    model2_wins += 1
            elif result == -1:  # Black wins
                if black_name == "model1":
                    model1_wins += 1
                else:
                    model2_wins += 1
            else:  # Draw
                draws += 1
                
            # Log the result
            logger.debug(f"Game {game_idx+1}: {white_name} (White) vs {black_name} (Black) - "
                        f"Result: {self._result_to_string(result)}")
                
        # Calculate score for model1
        total_games = model1_wins + model2_wins + draws
        model1_score = (model1_wins + 0.5 * draws) / total_games if total_games > 0 else 0.5
        
        # Log results
        logger.info(f"Evaluation results:")
        logger.info(f"  Model1 wins: {model1_wins}")
        logger.info(f"  Model2 wins: {model2_wins}")
        logger.info(f"  Draws: {draws}")
        logger.info(f"  Model1 score: {model1_score:.4f}")
        
        return model1_score
        
    def _load_model(self, model_path: str) -> Tuple[ChessNetwork, Dict[str, Any]]:
        """
        Load a model from a checkpoint file.
        
        Args:
            model_path: Path to the model checkpoint.
            
        Returns:
            Tuple of (model, config).
        """
        try:
            model, checkpoint = ChessNetwork.load_checkpoint(model_path, self.device)
            model.eval()  # Set to evaluation mode
            return model, checkpoint.get('config', self.config)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
            
    def _play_game(self, white_mcts: MCTS, black_mcts: MCTS) -> int:
        """
        Play a game between two MCTS instances.
        
        Args:
            white_mcts: MCTS for the white player.
            black_mcts: MCTS for the black player.
            
        Returns:
            1 if white wins, -1 if black wins, 0 for draw.
        """
        env = ChessEnvironment()
        move_count = 0
        
        # Play until the game is over or move cap is reached
        while not env.is_game_over() and move_count < self.game_cap:
            # Select which MCTS to use based on current player
            current_mcts = white_mcts if env.board.turn == chess.WHITE else black_mcts
            
            # Get policy from MCTS
            policy = current_mcts.search(env)
            
            # Select best move (with some temperature for exploration)
            if self.temperature <= 0.01:
                # Deterministic selection
                best_move = max(policy.items(), key=lambda x: x[1])[0]
            else:
                # Sample with temperature
                moves = list(policy.keys())
                probs = np.array(list(policy.values())) ** (1 / self.temperature)
                probs = probs / np.sum(probs)
                best_move = np.random.choice(moves, p=probs)
            
            # Make the move
            env.make_move(best_move)
            move_count += 1
            
        # Determine game result
        if env.board.is_checkmate():
            return -1 if env.board.turn == chess.WHITE else 1
        else:
            return 0  # Draw or move cap reached
            
    def _result_to_string(self, result: int) -> str:
        """
        Convert a numeric result to a string.
        
        Args:
            result: Game result (1 for white win, -1 for black win, 0 for draw).
            
        Returns:
            String representation of the result.
        """
        if result == 1:
            return "White Win"
        elif result == -1:
            return "Black Win"
        else:
            return "Draw"
            
    def evaluate_against_standard(self, model_path: str) -> Dict[str, float]:
        """
        Evaluate a model against standard benchmarks.
        
        This can be used to track progress over time against fixed opponents.
        
        Args:
            model_path: Path to the model to evaluate.
            
        Returns:
            Dictionary with evaluation results.
        """
        # This is a placeholder for more sophisticated evaluation
        # In a real implementation, you might have several fixed-strength
        # opponents or puzzle suites to evaluate against
        
        # Load the model
        model, _ = self._load_model(model_path)
        model_mcts = MCTS(model, self.config)
        
        # Results will be stored here
        results = {}
        
        # Example: evaluate solving tactical puzzles
        # puzzle_score = self._evaluate_puzzles(model_mcts)
        # results['puzzle_score'] = puzzle_score
        
        # Example: evaluate against different fixed-strength opponents
        # for strength in [1200, 1600, 2000, 2400]:
        #     opponent = self._create_fixed_strength_opponent(strength)
        #     score = self._evaluate_against_opponent(model_mcts, opponent)
        #     results[f'elo_{strength}_score'] = score
        
        # For now, just return a placeholder
        results['baseline_score'] = 0.5
        
        return results