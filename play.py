"""
Play script for the chess AI system.

This script provides a simplified entry point for playing games with the AI.
"""

import os
import sys
import argparse
import logging
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import load_config
from utils.logger import setup_logger
from main import ChessAI

def play_game(config_path, color='white', auto_start=True, max_games=1):
    """
    Play chess games with the AI.
    
    Args:
        config_path: Path to the configuration file
        color: Color to play as ('white', 'black', or 'random')
        auto_start: Whether to automatically start a new game
        max_games: Maximum number of games to play
        
    Returns:
        Number of games played
    """
    # Create the AI
    chess_ai = ChessAI(config_path)
    
    # First, calibrate the system
    if not chess_ai.calibrate():
        print("Calibration failed. Please ensure the chess board is clearly visible.")
        return 0
        
    games_played = 0
    
    try:
        while games_played < max_games:
            if auto_start:
                # Start a new game
                chess_ai.start_new_game(color)
            else:
                # Just play the current game
                chess_ai.play_game()
                
            games_played += 1
            
            if games_played < max_games:
                print(f"Game {games_played} completed. Starting next game in 10 seconds...")
                time.sleep(10)
                
    except KeyboardInterrupt:
        print("\nPlay session stopped by user")
        
    print(f"Play session completed. Played {games_played} games.")
    return games_played

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play Chess with AI")
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--color", type=str, choices=["white", "black", "random"],
                       default="white", help="Color to play as")
    parser.add_argument("--manual", action="store_true",
                       help="Don't automatically start a new game (use existing game)")
    parser.add_argument("--games", type=int, default=1,
                       help="Number of games to play")
    
    args = parser.parse_args()
    
    # Play the game(s)
    play_game(args.config, args.color, not args.manual, args.games)

if __name__ == "__main__":
    main()