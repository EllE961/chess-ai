"""
Main entry point for the autonomous chess AI system.

This script provides the main application that integrates all components
and provides a command-line interface for controlling the system.
"""

import os
import sys
import time
import argparse
import chess
import yaml
import logging
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import load_config
from core.neural_network import ChessNetwork
from core.mcts import MCTS
from core.chess_environment import ChessEnvironment
from vision.board_detector import BoardDetector
from vision.piece_classifier import PieceClassifier
from vision.position_extractor import PositionExtractor
from automation.move_executor import MoveExecutor
from automation.platform_adapter import PlatformAdapter
from automation.calibrator import Calibrator
from utils.logger import setup_logger, log_system_info, log_config
from utils.visualization import visualize_board, visualize_move, visualize_policy

class ChessAI:
    """
    Main class for the autonomous chess AI system.
    
    Integrates all components: neural network, MCTS, vision, and automation
    to create a complete system that can play chess on digital platforms.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the chess AI system.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set up logging
        self.logger = setup_logger(self.config)
        log_system_info()
        log_config(self.config)
        
        # Set up device
        import torch
        use_gpu = self.config.get('system', {}).get('use_gpu', True)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._init_components()
        
        # Game state
        self.current_board = chess.Board()
        self.our_color = None  # Will be set when game starts
        self.game_active = False
        
    def _init_components(self):
        """Initialize all system components."""
        self.logger.info("Initializing system components...")
        
        # Neural network and MCTS
        self.logger.info("Loading neural network model...")
        self.model = self._load_model()
        self.mcts = MCTS(self.model, self.config)
        
        # Vision components
        self.logger.info("Initializing vision system...")
        self.board_detector = BoardDetector(self.config)
        self.piece_classifier = PieceClassifier(self.config)
        self.position_extractor = PositionExtractor()
        
        # Automation components
        self.logger.info("Initializing automation system...")
        self.move_executor = MoveExecutor(self.config)
        self.platform_adapter = PlatformAdapter(self.config)
        self.calibrator = Calibrator(self.config)
        
        # Visualization directory
        self.viz_dir = os.path.join(self.config.get('log_dir', './logs'), 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.logger.info("System initialization complete")
        
    def _load_model(self):
        """
        Load the neural network model.
        
        Returns:
            Loaded ChessNetwork model
        """
        model_path = os.path.join(self.config.get('model_dir', './models'), 'best_model.pt')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            self.logger.warning(f"No model found at {model_path}, initializing new model")
            model = ChessNetwork(self.config)
            model.to(self.device)
            return model
            
        # Load model from checkpoint
        try:
            self.logger.info(f"Loading model from {model_path}")
            model, _ = ChessNetwork.load_checkpoint(model_path, self.device)
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.warning("Initializing new model")
            model = ChessNetwork(self.config)
            model.to(self.device)
            return model
            
    def calibrate(self):
        """
        Calibrate the system by detecting the board and configuring mouse coordinates.
        
        Returns:
            True if calibration was successful, False otherwise
        """
        self.logger.info("Starting system calibration...")
        
        try:
            # Run calibration to detect board and perspective
            board_coords, white_perspective = self.calibrator.calibrate(self.board_detector)
            
            # Set board info for move executor
            self.move_executor.set_board_info(board_coords, white_perspective)
            
            self.logger.info(f"Calibration complete. Board detected with {'white' if white_perspective else 'black'}'s perspective")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False
            
    def detect_game_state(self):
        """
        Detect the current state of the game from screen.
        
        Returns:
            True if it's our turn to move, False otherwise
        """
        # Check if a game is active and whose turn it is
        self.game_active, is_our_turn = self.platform_adapter.detect_game_state()
        
        if not self.game_active:
            self.logger.info("No active game detected")
            return False
            
        # Get current board position
        board_position = self.detect_board_position()
        if board_position is None:
            self.logger.warning("Could not detect board position")
            return False
            
        # Update our internal board state
        self.current_board = board_position
        
        # Determine our color if not already set
        if self.our_color is None:
            # If it's our turn on the first move, we're likely white
            if is_our_turn and self.current_board.fullmove_number == 1:
                self.our_color = chess.WHITE
                self.logger.info("Detected we are playing as white")
            # If it's not our turn but a move has been made, we're likely black
            elif not is_our_turn and self.current_board.fullmove_number > 1:
                self.our_color = chess.BLACK
                self.logger.info("Detected we are playing as black")
                
        # Return True if it's our turn to move
        return is_our_turn and self.current_board.turn == self.our_color
        
    def detect_board_position(self):
        """
        Detect the current chess position from the screen.
        
        Returns:
            chess.Board object with the detected position, or None if detection failed
        """
        try:
            # Detect the board
            board_coords, warped_board = self.board_detector.detect_board()
            
            # Extract individual squares
            squares = self.board_detector.extract_squares(warped_board)
            
            # Classify pieces on each square
            board_state = self.piece_classifier.classify_board(squares)
            
            # Convert to FEN notation
            white_perspective = self.move_executor.white_perspective
            
            # Determine additional FEN parameters based on current state
            turn = chess.WHITE  # Default to white's turn
            castling_rights = "KQkq"  # Default to all castling rights
            en_passant = None
            halfmove_clock = 0
            fullmove_number = 1
            
            # Use current board state if available
            if hasattr(self, 'current_board') and self.current_board:
                turn = self.current_board.turn
                castling_rights = "".join([
                    "K" if self.current_board.has_kingside_castling_rights(chess.WHITE) else "",
                    "Q" if self.current_board.has_queenside_castling_rights(chess.WHITE) else "",
                    "k" if self.current_board.has_kingside_castling_rights(chess.BLACK) else "",
                    "q" if self.current_board.has_queenside_castling_rights(chess.BLACK) else ""
                ]) or "-"
                en_passant = chess.square_name(self.current_board.ep_square) if self.current_board.ep_square is not None else None
                halfmove_clock = self.current_board.halfmove_clock
                fullmove_number = self.current_board.fullmove_number
                
            # Convert to FEN
            fen = self.position_extractor.board_state_to_fen(
                board_state, white_perspective, turn, castling_rights,
                en_passant, halfmove_clock, fullmove_number
            )
            
            # Validate and clean up FEN if needed
            if not self.position_extractor.validate_fen(fen):
                self.logger.warning(f"Invalid FEN detected: {fen}")
                # Try to find the most similar valid position
                return self.position_extractor.get_most_similar_position(fen, self.current_board)
            
            # Create and return chess.Board object
            try:
                board = chess.Board(fen)
                self.logger.info(f"Detected position: {fen}")
                return board
            except ValueError as e:
                self.logger.error(f"Error creating board from FEN: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error detecting board position: {e}")
            return None
            
    def calculate_best_move(self):
        """
        Calculate the best move for the current position using MCTS and neural network.
        
        Returns:
            chess.Move object representing the best move, or None if no move could be calculated
        """
        self.logger.info("Calculating best move...")
        
        try:
            # Convert the chess.Board to a ChessEnvironment
            env = ChessEnvironment(self.current_board.fen())
            
            # Run MCTS to get move probabilities
            policy = self.mcts.search(env)
            
            # Log the top moves
            sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
            top_moves = sorted_moves[:5]
            
            self.logger.info("Top moves:")
            for move, prob in top_moves:
                self.logger.info(f"  {move.uci()}: {prob:.4f}")
                
            # Visualize the policy if in debug mode
            if self.config.get('system', {}).get('debug_mode', False):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                viz_path = os.path.join(self.viz_dir, f"policy_{timestamp}.png")
                visualize_policy(self.current_board, policy, output_path=viz_path)
                
            # Get the move with highest probability
            best_move = max(policy.items(), key=lambda x: x[1])[0]
            
            self.logger.info(f"Best move: {best_move.uci()}")
            return best_move
            
        except Exception as e:
            self.logger.error(f"Error calculating best move: {e}")
            return None
            
    def execute_move(self, move):
        """
        Execute a chess move on the screen.
        
        Args:
            move: chess.Move object representing the move to execute
            
        Returns:
            True if the move was executed successfully, False otherwise
        """
        try:
            self.logger.info(f"Executing move: {move.uci()}")
            
            # Visualize the move if in debug mode
            if self.config.get('system', {}).get('debug_mode', False):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                viz_path = os.path.join(self.viz_dir, f"move_{timestamp}.png")
                visualize_move(self.current_board, move, output_path=viz_path)
                
            # Update our internal board state
            self.current_board.push(move)
            
            # Use the move executor to perform the move
            success = self.move_executor.execute_move(move)
            
            # Wait for the move animation to complete
            time.sleep(1)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing move: {e}")
            return False
            
    def play_game(self):
        """
        Main game playing loop.
        
        Returns:
            True if the game was completed, False if interrupted
        """
        self.logger.info("Starting game play. Press Ctrl+C to stop.")
        
        try:
            while True:
                # Check if it's our turn
                if self.detect_game_state():
                    self.logger.info("It's our turn!")
                    
                    # Calculate best move
                    best_move = self.calculate_best_move()
                    
                    if best_move is None:
                        self.logger.error("Could not calculate a move")
                        time.sleep(5)
                        continue
                        
                    # Execute the move
                    success = self.execute_move(best_move)
                    
                    if success:
                        self.logger.info(f"Move executed: {best_move.uci()}")
                    else:
                        self.logger.error(f"Failed to execute move: {best_move.uci()}")
                        
                    self.logger.info(f"Current position: {self.current_board.fen()}")
                    
                else:
                    self.logger.info("Waiting for opponent's move...")
                    
                # Wait before checking again
                polling_interval = self.config.get('automation', {}).get('polling_interval', 1.0)
                time.sleep(polling_interval)
                
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Game stopped by user")
            return False
        except Exception as e:
            self.logger.error(f"Error during gameplay: {e}")
            return False
            
    def start_new_game(self, color='white'):
        """
        Start a new game with the specified color preference.
        
        Args:
            color: Color to play as ('white', 'black', or 'random')
            
        Returns:
            True if the game was started successfully, False otherwise
        """
        # Map color string to chess.COLOR
        if color.lower() == 'white':
            self.our_color = chess.WHITE
        elif color.lower() == 'black':
            self.our_color = chess.BLACK
        else:  # 'random'
            self.our_color = None  # Will be determined during the game
            
        # Reset board state
        self.current_board = chess.Board()
        
        self.logger.info(f"Starting new game as {color}")
        
        # Try to start a new game on the platform
        success = self.platform_adapter.start_new_game(color)
        
        if not success:
            self.logger.warning("Failed to automatically start a new game")
            self.logger.info("Please start a new game manually on the platform")
            
        # Wait for the game to start
        self.logger.info("Waiting for game to start...")
        time.sleep(5)
        
        # Detect board and calibrate
        if not self.calibrate():
            self.logger.error("Calibration failed, cannot start game")
            return False
            
        # Start playing
        return self.play_game()
        
    def resign_game(self):
        """
        Resign the current game.
        
        Returns:
            True if the resignation was successful, False otherwise
        """
        if not self.game_active:
            self.logger.warning("No active game to resign")
            return False
            
        self.logger.info("Resigning game")
        return self.platform_adapter.resign_game()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous Chess AI")
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["play", "train", "calibrate"],
                       default="play", help="Operation mode")
    parser.add_argument("--color", type=str, choices=["white", "black", "random"],
                       default="white", help="Color to play as")
    
    args = parser.parse_args()
    
    # Create the AI
    chess_ai = ChessAI(args.config)
    
    if args.mode == "calibrate":
        # Just run calibration
        chess_ai.calibrate()
    elif args.mode == "train":
        # For training, we would call a different script
        from train import train_model
        train_model(args.config)
    else:  # "play"
        # For playing, initiate a new game
        chess_ai.start_new_game(args.color)

if __name__ == "__main__":
    main()