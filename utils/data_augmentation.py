"""
Data augmentation for chess positions.

This module provides functions to augment chess position data for training,
using techniques like flipping, rotating, and transforming positions.
"""

import numpy as np
import chess
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

def augment_position(state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate augmented versions of a position through symmetry transformations.
    
    Args:
        state: Board state representation as numpy array of shape (19, 8, 8)
        policy: Policy vector of shape (policy_size,)
        
    Returns:
        List of tuples (augmented_state, augmented_policy)
    """
    augmentations = []
    
    # Add original position
    augmentations.append((state, policy))
    
    # Add horizontal flip
    flipped_state, flipped_policy = flip_position(state, policy, 'horizontal')
    augmentations.append((flipped_state, flipped_policy))
    
    logger.debug(f"Generated {len(augmentations)} augmentations for position")
    
    return augmentations

def flip_position(state: np.ndarray, policy: np.ndarray, 
                 flip_type: str = 'horizontal') -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip a position horizontally or vertically.
    
    Args:
        state: Board state representation as numpy array of shape (19, 8, 8)
        policy: Policy vector of shape (policy_size,)
        flip_type: Type of flip ('horizontal' or 'vertical')
        
    Returns:
        Tuple of (flipped_state, flipped_policy)
    """
    # Create copy to avoid modifying original
    flipped_state = state.copy()
    
    # Transform the board state
    if flip_type == 'horizontal':
        # Flip horizontally (a-h becomes h-a)
        for i in range(state.shape[0]):
            flipped_state[i] = np.fliplr(state[i])
            
    elif flip_type == 'vertical':
        # Flip vertically (1-8 becomes 8-1)
        for i in range(state.shape[0]):
            flipped_state[i] = np.flipud(state[i])
            
    else:
        logger.warning(f"Unknown flip type: {flip_type}. Returning original.")
        return state, policy
        
    # Transform the policy
    # This is a simplified example - a real implementation would need
    # to convert policy indices correctly based on the specific policy representation
    flipped_policy = _transform_policy(policy, transform_type=flip_type)
    
    return flipped_state, flipped_policy

def rotate_position(state: np.ndarray, policy: np.ndarray, 
                   degrees: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a position by the specified angle.
    
    Args:
        state: Board state representation as numpy array of shape (19, 8, 8)
        policy: Policy vector of shape (policy_size,)
        degrees: Rotation angle in degrees (90, 180, or 270)
        
    Returns:
        Tuple of (rotated_state, rotated_policy)
    """
    if degrees not in [90, 180, 270]:
        logger.warning(f"Unsupported rotation angle: {degrees}. Returning original.")
        return state, policy
        
    # Create copy to avoid modifying original
    rotated_state = state.copy()
    
    # Rotate each channel of the state
    k = degrees // 90  # Number of 90-degree rotations
    for i in range(state.shape[0]):
        rotated_state[i] = np.rot90(state[i], k=k)
        
    # Transform the policy based on rotation
    transform_type = f'rotate{degrees}'
    rotated_policy = _transform_policy(policy, transform_type=transform_type)
    
    return rotated_state, rotated_policy

def _transform_policy(policy: np.ndarray, transform_type: str) -> np.ndarray:
    """
    Transform a policy vector based on board transformation.
    
    Args:
        policy: Policy vector of shape (policy_size,)
        transform_type: Type of transformation ('horizontal', 'vertical', 'rotate90', etc.)
        
    Returns:
        Transformed policy vector
    """
    # This is a placeholder for a more complex implementation
    # In a real system, the policy transformation depends on how moves are encoded
    
    # Create a copy to avoid modifying the original
    transformed_policy = np.zeros_like(policy)
    
    # Example implementation for a simplified 64x64 policy (from_square * 64 + to_square)
    if len(policy) == 64*64:
        for from_square in range(64):
            from_rank = from_square // 8
            from_file = from_square % 8
            
            for to_square in range(64):
                to_rank = to_square // 8
                to_file = to_square % 8
                
                # Original policy index
                orig_idx = from_square * 64 + to_square
                
                # Calculate new coordinates based on transformation
                if transform_type == 'horizontal':
                    new_from_file = 7 - from_file
                    new_from_rank = from_rank
                    new_to_file = 7 - to_file
                    new_to_rank = to_rank
                elif transform_type == 'vertical':
                    new_from_file = from_file
                    new_from_rank = 7 - from_rank
                    new_to_file = to_file
                    new_to_rank = 7 - to_rank
                elif transform_type == 'rotate90':
                    new_from_file = from_rank
                    new_from_rank = 7 - from_file
                    new_to_file = to_rank
                    new_to_rank = 7 - to_file
                elif transform_type == 'rotate180':
                    new_from_file = 7 - from_file
                    new_from_rank = 7 - from_rank
                    new_to_file = 7 - to_file
                    new_to_rank = 7 - to_rank
                elif transform_type == 'rotate270':
                    new_from_file = 7 - from_rank
                    new_from_rank = from_file
                    new_to_file = 7 - to_rank
                    new_to_rank = to_file
                else:
                    # Unknown transformation, return original policy
                    return policy
                    
                # Calculate new policy index
                new_from_square = new_from_rank * 8 + new_from_file
                new_to_square = new_to_rank * 8 + new_to_file
                new_idx = new_from_square * 64 + new_to_square
                
                # Copy probability
                transformed_policy[new_idx] = policy[orig_idx]
    else:
        logger.warning(f"Policy transformation not implemented for size {len(policy)}")
        transformed_policy = policy  # Return original if not implemented
        
    return transformed_policy

def augment_self_play_data(examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Augment a batch of self-play training examples.
    
    Args:
        examples: List of (state, policy, value) tuples
        
    Returns:
        Augmented list of examples
    """
    augmented_examples = []
    
    for state, policy, value in examples:
        # Generate augmentations for this position
        augmentations = augment_position(state, policy)
        
        # Add each augmentation with the same value
        for aug_state, aug_policy in augmentations:
            augmented_examples.append((aug_state, aug_policy, value))
            
    logger.debug(f"Augmented {len(examples)} examples to {len(augmented_examples)}")
    
    return augmented_examples

def position_from_fen(fen: str) -> np.ndarray:
    """
    Create a state representation from a FEN string.
    
    Args:
        fen: FEN string representing a chess position
        
    Returns:
        NumPy array of shape (19, 8, 8) representing the position
    """
    # Try to create a board from the FEN
    try:
        board = chess.Board(fen)
    except ValueError as e:
        logger.error(f"Invalid FEN: {e}")
        # Return standard position if FEN is invalid
        board = chess.Board()
        
    # Initialize state representation
    state = np.zeros((19, 8, 8), dtype=np.float32)
    
    # Piece planes (12 planes: 6 piece types * 2 colors)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                  chess.ROOK, chess.QUEEN, chess.KING]
    
    for color in [chess.WHITE, chess.BLACK]:
        for i, piece_type in enumerate(piece_types):
            # Determine plane index based on piece type and color
            plane_idx = i + (0 if color == chess.WHITE else 6)
            
            # Set 1s for squares containing this piece
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type and piece.color == color:
                    # Convert to (rank, file) coordinates
                    rank = 7 - (square // 8)  # 0 is 8th rank, 7 is 1st rank
                    file = square % 8         # 0 is a-file, 7 is h-file
                    state[plane_idx][rank][file] = 1.0
    
    # Current player color (1 plane)
    if board.turn == chess.WHITE:
        state[12].fill(1.0)
    
    # Castling rights (4 planes)
    if board.has_kingside_castling_rights(chess.WHITE):
        state[13].fill(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        state[14].fill(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        state[15].fill(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        state[16].fill(1.0)
        
    # En passant (1 plane)
    if board.ep_square:
        # Convert to (rank, file) coordinates
        rank = 7 - (board.ep_square // 8)
        file = board.ep_square % 8
        state[17][rank][file] = 1.0
        
    # Halfmove clock for 50-move rule (1 plane)
    # Normalize to [0, 1] range
    halfmove_value = min(1.0, board.halfmove_clock / 100.0)
    state[18].fill(halfmove_value)
    
    return state