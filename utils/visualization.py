"""
Visualization utilities for the chess AI system.

This module provides functions for visualizing training progress, chess positions,
move predictions, and other aspects of the system.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import chess
import chess.svg
from cairosvg import svg2png
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

def plot_training_metrics(metrics: Dict[str, List[float]], 
                        output_dir: Optional[str] = None,
                        show_plot: bool = True) -> None:
    """
    Plot training metrics over time.
    
    Args:
        metrics: Dictionary of metric names to lists of values
        output_dir: Directory to save the plot, if provided
        show_plot: Whether to display the plot interactively
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots based on metric categories
    num_metrics = len(metrics)
    rows = max(1, (num_metrics + 1) // 2)
    
    # Group related metrics
    metric_groups = {
        'Loss': ['loss', 'policy_loss', 'value_loss'],
        'Accuracy': ['value_accuracy', 'policy_accuracy'],
        'Performance': ['performance', 'win_rate', 'draw_rate'],
        'Learning': ['learning_rate', 'buffer_size']
    }
    
    # Plot each group
    subplot_idx = 1
    for group_name, group_metrics in metric_groups.items():
        # Check if any metrics in this group exist
        if not any(m in metrics for m in group_metrics):
            continue
            
        plt.subplot(rows, 2, subplot_idx)
        
        for metric_name in group_metrics:
            if metric_name in metrics:
                values = metrics[metric_name]
                iterations = list(range(1, len(values) + 1))
                plt.plot(iterations, values, label=metric_name, marker='o', markersize=3)
                
        plt.xlabel('Iteration')
        plt.title(group_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        subplot_idx += 1
        
    plt.tight_layout()
    plt.suptitle('Training Progress', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Training metrics plot saved to {plot_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_board(board: chess.Board, 
                  last_move: Optional[chess.Move] = None,
                  output_path: Optional[str] = None,
                  size: int = 600) -> Optional[str]:
    """
    Visualize a chess board position.
    
    Args:
        board: Chess board to visualize
        last_move: Optional last move to highlight
        output_path: Path to save the visualization, if provided
        size: Size of the board image in pixels
        
    Returns:
        Path to the saved image if output_path is provided, None otherwise
    """
    # Create SVG representation of the board
    board_svg = chess.svg.board(
        board=board,
        size=size,
        lastmove=last_move,
        check=board.king(board.turn) if board.is_check() else None
    )
    
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert SVG to PNG and save
        svg2png(bytestring=board_svg.encode('utf-8'), write_to=output_path, output_width=size, output_height=size)
        logger.debug(f"Board visualization saved to {output_path}")
        return output_path
    else:
        # Display using matplotlib
        svg_handle = plt.imread(board_svg)
        plt.figure(figsize=(10, 10))
        plt.imshow(svg_handle)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return None

def visualize_move(board: chess.Board, move: chess.Move, 
                 output_path: Optional[str] = None,
                 original_path: Optional[str] = None,
                 size: int = 600) -> Tuple[Optional[str], Optional[str]]:
    """
    Visualize a chess move by showing before and after positions.
    
    Args:
        board: Chess board before the move
        move: Move to visualize
        output_path: Base path to save the visualization (will add _before/_after suffixes)
        original_path: Optional path to an already saved 'before' image
        size: Size of the board image in pixels
        
    Returns:
        Tuple of (before_path, after_path) if output_path is provided, (None, None) otherwise
    """
    # Save or display the original position
    before_path = None
    if output_path:
        before_path = original_path or f"{os.path.splitext(output_path)[0]}_before.png"
        visualize_board(board, output_path=before_path, size=size)
    else:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Before move")
        visualize_board(board)
        
    # Make the move on a copy of the board
    after_board = board.copy()
    after_board.push(move)
    
    # Save or display the position after the move
    after_path = None
    if output_path:
        after_path = f"{os.path.splitext(output_path)[0]}_after.png"
        visualize_board(after_board, last_move=move, output_path=after_path, size=size)
    else:
        plt.subplot(1, 2, 2)
        plt.title(f"After move: {move.uci()}")
        visualize_board(after_board, last_move=move)
        plt.tight_layout()
        plt.show()
        
    return before_path, after_path

def visualize_policy(board: chess.Board, policy: Dict[chess.Move, float],
                    num_moves: int = 5,
                    output_path: Optional[str] = None,
                    size: int = 800) -> Optional[str]:
    """
    Visualize the policy (move probabilities) on a chess board.
    
    Args:
        board: Chess board
        policy: Dictionary mapping moves to their probabilities
        num_moves: Number of top moves to visualize
        output_path: Path to save the visualization, if provided
        size: Size of the board image in pixels
        
    Returns:
        Path to the saved image if output_path is provided, None otherwise
    """
    # Sort moves by probability (descending)
    sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top N moves
    top_moves = sorted_moves[:num_moves]
    
    # Create a list of arrows for the top moves
    arrows = []
    for move, prob in top_moves:
        # Determine arrow width based on probability
        # Normalize between 1 and 10
        width = 1 + 9 * prob
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, width=width))
        
    # Create SVG representation of the board with arrows
    board_svg = chess.svg.board(
        board=board,
        size=size,
        arrows=arrows,
        check=board.king(board.turn) if board.is_check() else None
    )
    
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert SVG to PNG and save
        svg2png(bytestring=board_svg.encode('utf-8'), write_to=output_path, output_width=size, output_height=size)
        logger.debug(f"Policy visualization saved to {output_path}")
        return output_path
    else:
        # Display using matplotlib
        svg_handle = plt.imread(board_svg)
        plt.figure(figsize=(10, 10))
        plt.imshow(svg_handle)
        plt.axis('off')
        
        # Add a legend for the top moves
        for i, (move, prob) in enumerate(top_moves):
            plt.text(
                size + 10, 50 + i * 30,
                f"{move.uci()}: {prob:.2%}",
                fontsize=12, va='center'
            )
            
        plt.tight_layout()
        plt.show()
        return None

def plot_policy_distribution(policy: Dict[chess.Move, float],
                           num_moves: int = 10,
                           output_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
    """
    Plot the probability distribution of moves in the policy.
    
    Args:
        policy: Dictionary mapping moves to their probabilities
        num_moves: Number of top moves to show
        output_path: Path to save the plot, if provided
        show_plot: Whether to display the plot interactively
    """
    # Sort moves by probability (descending)
    sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top N moves
    top_moves = sorted_moves[:num_moves]
    
    # Extract moves and probabilities
    moves = [move.uci() for move, _ in top_moves]
    probs = [prob for _, prob in top_moves]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(moves)), probs, align='center')
    plt.xticks(range(len(moves)), moves, rotation=45)
    plt.xlabel('Move')
    plt.ylabel('Probability')
    plt.title('Top Move Probabilities')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.debug(f"Policy distribution plot saved to {output_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_evaluation_results(results: Dict[str, Any],
                               output_dir: Optional[str] = None,
                               show_plot: bool = True) -> None:
    """
    Visualize evaluation results from model comparison.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the visualizations, if provided
        show_plot: Whether to display the plots interactively
    """
    # Extract results
    model1_wins = results.get('model1_wins', 0)
    model2_wins = results.get('model2_wins', 0)
    draws = results.get('draws', 0)
    total_games = model1_wins + model2_wins + draws
    
    if total_games == 0:
        logger.warning("No games to visualize")
        return
        
    # Calculate percentages
    model1_pct = model1_wins / total_games * 100
    model2_pct = model2_wins / total_games * 100
    draws_pct = draws / total_games * 100
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Bar chart
    labels = ['Model 1 Wins', 'Draws', 'Model 2 Wins']
    values = [model1_wins, draws, model2_wins]
    percentages = [model1_pct, draws_pct, model2_pct]
    colors = ['#2ca02c', '#d3d3d3', '#d62728']
    
    plt.bar(labels, values, color=colors)
    
    # Add labels on the bars
    for i, (value, pct) in enumerate(zip(values, percentages)):
        plt.text(i, value + 0.5, f"{value} ({pct:.1f}%)", ha='center')
        
    plt.ylabel('Number of Games')
    plt.title('Evaluation Results')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add model score at the bottom
    model1_score = (model1_wins + 0.5 * draws) / total_games
    plt.figtext(0.5, 0.01, f"Model 1 Score: {model1_score:.4f}", ha='center', fontsize=12, bbox={'facecolor': 'lightgray', 'alpha': 0.5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'evaluation_results.png')
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Evaluation results plot saved to {plot_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close()