"""
Chess environment module for the AI system.

This module provides a wrapper around the python-chess library to represent the
chess game state, rules, and mechanics.
"""

import chess
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class ChessEnvironment:
    """
    Chess environment wrapper providing game state representation and manipulation.
    
    This class encapsulates the python-chess Board object and provides additional
    functionality for encoding the board state for neural network input and
    handling game rules.
    """
    
    def __init__(self, fen: Optional[str] = None):
        """
        Initialize the chess environment.
        
        Args:
            fen: Optional FEN string to initialize the board state.
                 If None, the standard chess starting position is used.
        """
        if fen:
            try:
                self.board = chess.Board(fen)
            except ValueError as e:
                logger.error(f"Invalid FEN: {e}")
                logger.info("Using standard starting position")
                self.board = chess.Board()
        else:
            self.board = chess.Board()
            
        self.move_history = []
        
    def reset(self) -> None:
        """Reset the board to the initial position."""
        self.board.reset()
        self.move_history = []
        
    def copy(self) -> 'ChessEnvironment':
        """
        Create a deep copy of the current environment.
        
        Returns:
            A new ChessEnvironment instance with the same state.
        """
        env_copy = ChessEnvironment()
        env_copy.board = self.board.copy()
        env_copy.move_history = self.move_history.copy()
        return env_copy
        
    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move on the board.
        
        Args:
            move: The chess move to make.
            
        Returns:
            True if the move was legal and made, False otherwise.
        """
        if move in self.board.legal_moves:
            self.move_history.append(move)
            self.board.push(move)
            return True
        else:
            logger.warning(f"Illegal move attempted: {move}")
            return False
            
    def make_move_from_uci(self, uci: str) -> bool:
        """
        Make a move specified in UCI format.
        
        Args:
            uci: Move in UCI format (e.g., "e2e4").
            
        Returns:
            True if the move was legal and made, False otherwise.
        """
        try:
            move = chess.Move.from_uci(uci)
            return self.make_move(move)
        except ValueError:
            logger.error(f"Invalid UCI move: {uci}")
            return False
            
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if the game is over (checkmate, stalemate, etc.), False otherwise.
        """
        return self.board.is_game_over()
        
    def get_result(self) -> float:
        """
        Get the game result from the perspective of the current player.
        
        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw.
        """
        if not self.board.is_game_over():
            return 0.0
            
        outcome = self.board.outcome()
        if outcome.winner is None:  # Draw
            return 0.0
            
        return 1.0 if outcome.winner == self.board.turn else -1.0
        
    def get_legal_moves(self) -> List[chess.Move]:
        """
        Get all legal moves in the current position.
        
        Returns:
            List of legal chess moves.
        """
        return list(self.board.legal_moves)
        
    def encode_board(self) -> np.ndarray:
        """
        Encode the board state into a format suitable for neural network input.
        
        This encoding uses a 19-plane representation:
        - 12 planes for pieces (6 piece types * 2 colors)
        - 1 plane for player color (1 for white, 0 for black)
        - 4 planes for castling rights
        - 1 plane for en passant
        - 1 plane for halfmove clock (50-move rule)
        
        Returns:
            Numpy array of shape (19, 8, 8) representing the board state.
        """
        # Initialize the encoded state with zeros
        encoded = np.zeros((19, 8, 8), dtype=np.float32)
        
        # Piece planes (12 planes)
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                      chess.ROOK, chess.QUEEN, chess.KING]
        
        for color in [chess.WHITE, chess.BLACK]:
            for i, piece_type in enumerate(piece_types):
                # Determine plane index based on piece type and color
                plane_idx = i + (0 if color == chess.WHITE else 6)
                
                # Set 1s for squares containing this piece
                for square in chess.SQUARES:
                    piece = self.board.piece_at(square)
                    if piece and piece.piece_type == piece_type and piece.color == color:
                        # Convert to (rank, file) coordinates
                        rank = 7 - (square // 8)  # 0 is 8th rank, 7 is 1st rank
                        file = square % 8         # 0 is a-file, 7 is h-file
                        encoded[plane_idx][rank][file] = 1.0
        
        # Current player color (1 plane)
        if self.board.turn == chess.WHITE:
            encoded[12].fill(1.0)
        
        # Castling rights (4 planes)
        if self.board.has_kingside_castling_rights(chess.WHITE):
            encoded[13].fill(1.0)
        if self.board.has_queenside_castling_rights(chess.WHITE):
            encoded[14].fill(1.0)
        if self.board.has_kingside_castling_rights(chess.BLACK):
            encoded[15].fill(1.0)
        if self.board.has_queenside_castling_rights(chess.BLACK):
            encoded[16].fill(1.0)
            
        # En passant (1 plane)
        if self.board.ep_square:
            # Convert to (rank, file) coordinates
            rank = 7 - (self.board.ep_square // 8)
            file = self.board.ep_square % 8
            encoded[17][rank][file] = 1.0
            
        # Halfmove clock for 50-move rule (1 plane)
        # Normalize to [0, 1] range
        halfmove_value = min(1.0, self.board.halfmove_clock / 100.0)
        encoded[18].fill(halfmove_value)
            
        return encoded
        
    def move_to_index(self, move: chess.Move) -> int:
        """
        Convert a chess move to an index in the policy vector.
        
        The policy vector has shape (4672,) representing:
        - Queen moves: 56 squares * 8 directions * 7 distances = 3136
        - Knight moves: 56 squares * 8 directions = 448
        - Underpromotions: 24 pawn squares * 3 piece types * 2 directions = 144
        - Total: 3136 + 448 + 144 = 3728
        
        Note: The actual implementation will depend on how moves are represented
        in the policy output. This is a simplified example.
        
        Args:
            move: The chess move to convert.
            
        Returns:
            Integer index in the policy vector.
        """
        # This is a simplified mapping. A complete implementation would need
        # to handle all move types including promotions.
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        
        # Simple encoding: 64 * 64 possible from-to combinations
        # Plus additional indices for promotions
        if promotion:
            # Basic mapping for promotions
            promotion_offset = 64 * 64
            piece_offset = {
                chess.QUEEN: 0,
                chess.ROOK: 1,
                chess.BISHOP: 2,
                chess.KNIGHT: 3
            }[promotion]
            return promotion_offset + (piece_offset * 64) + to_square
        else:
            return from_square * 64 + to_square
            
    def index_to_move(self, index: int) -> chess.Move:
        """
        Convert a policy index to a chess move.
        
        Args:
            index: Index in the policy vector.
            
        Returns:
            Chess move corresponding to the index.
            
        Raises:
            ValueError: If the index doesn't correspond to a valid move.
        """
        # Handle promotion moves
        if index >= 64 * 64:
            promotion_index = index - (64 * 64)
            piece_type = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promotion_index // 64]
            to_square = promotion_index % 64
            
            # Find a valid from_square that can promote to to_square
            for from_square in range(64):
                try:
                    move = chess.Move(from_square, to_square, promotion=piece_type)
                    if move in self.board.legal_moves:
                        return move
                except ValueError:
                    continue
        else:
            from_square = index // 64
            to_square = index % 64
            
            # Try to create a move
            try:
                move = chess.Move(from_square, to_square)
                if move in self.board.legal_moves:
                    return move
            except ValueError:
                pass
                
        # If we get here, the index didn't correspond to a legal move
        raise ValueError(f"Index {index} does not correspond to a legal move")
        
    def get_fen(self) -> str:
        """
        Get the current position as a FEN string.
        
        Returns:
            FEN string representation of the board.
        """
        return self.board.fen()
        
    def get_board_state(self) -> Dict[str, Any]:
        """
        Get a dictionary containing information about the current board state.
        
        Returns:
            Dictionary with board information.
        """
        return {
            'fen': self.board.fen(),
            'turn': 'white' if self.board.turn == chess.WHITE else 'black',
            'fullmove_number': self.board.fullmove_number,
            'halfmove_clock': self.board.halfmove_clock,
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_insufficient_material': self.board.is_insufficient_material(),
            'is_game_over': self.board.is_game_over(),
            'legal_moves': [move.uci() for move in self.board.legal_moves]
        }