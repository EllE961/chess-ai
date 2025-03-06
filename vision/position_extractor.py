"""
Extract chess position from classified squares.

This module converts the classified chess pieces to a FEN (Forsyth-Edwards Notation)
string representing the board position, which can be used by the chess engine.
"""

import chess
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PositionExtractor:
    """
    Extracts a chess position from classified squares.
    
    Converts the output of the piece classifier to a FEN string
    representing the current board position.
    """
    
    def __init__(self):
        """Initialize the position extractor."""
        # Mapping from piece class names to FEN characters
        self.piece_to_fen = {
            "white_pawn": "P",
            "white_knight": "N",
            "white_bishop": "B",
            "white_rook": "R",
            "white_queen": "Q",
            "white_king": "K",
            "black_pawn": "p",
            "black_knight": "n",
            "black_bishop": "b",
            "black_rook": "r",
            "black_queen": "q",
            "black_king": "k",
            "empty": ""
        }
        
    def board_state_to_fen(self, board_state: List[List[Tuple[str, float]]],
                          white_perspective: bool = True,
                          turn: chess.Color = chess.WHITE,
                          castling_rights: str = "KQkq",
                          en_passant: Optional[str] = None,
                          halfmove_clock: int = 0,
                          fullmove_number: int = 1) -> str:
        """
        Convert classified board state to FEN notation.
        
        Args:
            board_state: 2D list of (piece_class, confidence) tuples
            white_perspective: Whether the board is viewed from white's perspective
            turn: Whose turn it is (chess.WHITE or chess.BLACK)
            castling_rights: String representing castling rights (e.g., "KQkq")
            en_passant: En passant square in algebraic notation (e.g., "e3") or None
            halfmove_clock: Number of half-moves since last capture or pawn advance
            fullmove_number: Current move number
            
        Returns:
            FEN string representing the position
        """
        # If viewing from black's perspective, flip the board
        if not white_perspective:
            board_state = self._flip_board(board_state)
            
        # Build the piece placement part of FEN
        fen_rows = []
        for rank in range(8):
            empty_count = 0
            fen_row = ""
            
            for file in range(8):
                piece_class, _ = board_state[rank][file]
                
                if piece_class == "empty":
                    empty_count += 1
                else:
                    # If we had empty squares before this piece, add the count
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                        
                    # Add the piece
                    fen_row += self.piece_to_fen[piece_class]
                    
            # Add any remaining empty squares at the end of the row
            if empty_count > 0:
                fen_row += str(empty_count)
                
            fen_rows.append(fen_row)
            
        # Join rows with '/' separator
        piece_placement = "/".join(fen_rows)
        
        # Build the complete FEN string
        fen_parts = [
            piece_placement,
            "w" if turn == chess.WHITE else "b",
            castling_rights if castling_rights else "-",
            en_passant if en_passant else "-",
            str(halfmove_clock),
            str(fullmove_number)
        ]
        
        fen = " ".join(fen_parts)
        
        logger.debug(f"Generated FEN: {fen}")
        return fen
        
    def _flip_board(self, board_state: List[List[Tuple[str, float]]]) -> List[List[Tuple[str, float]]]:
        """
        Flip the board orientation (for when viewed from black's perspective).
        
        Args:
            board_state: 2D list of (piece_class, confidence) tuples
            
        Returns:
            Flipped board state
        """
        # Reverse the rows (ranks) and columns (files)
        return [row[::-1] for row in board_state[::-1]]
        
    def validate_fen(self, fen: str) -> bool:
        """
        Validate a FEN string by creating a chess.Board object.
        
        Args:
            fen: FEN string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            chess.Board(fen)
            return True
        except ValueError as e:
            logger.warning(f"Invalid FEN: {e}")
            return False
            
    def extract_position(self, board_state: List[List[Tuple[str, float]]], white_perspective: bool,
                        previous_position: Optional[chess.Board] = None) -> chess.Board:
        """
        Extract a chess position from classified board state.
        
        This method does additional validation and correction to ensure the
        position is legal and makes sense in the context of the game.
        
        Args:
            board_state: 2D list of (piece_class, confidence) tuples
            white_perspective: Whether the board is viewed from white's perspective
            previous_position: Previous board position for context
            
        Returns:
            Chess board object representing the position
        """
        # Generate basic FEN from board state
        fen = self.board_state_to_fen(board_state, white_perspective)
        
        # Try to create a board from the FEN
        try:
            board = chess.Board(fen)
            logger.debug("Valid position detected")
            return board
        except ValueError:
            logger.warning(f"Invalid position detected. Attempting to correct.")
            
            # Try to correct common issues
            corrected_fen = self._correct_fen(fen)
            
            try:
                board = chess.Board(corrected_fen)
                logger.debug(f"Position corrected: {corrected_fen}")
                return board
            except ValueError:
                logger.error("Could not correct position")
                
                # If we have a previous position, use it
                if previous_position:
                    logger.warning("Using previous position")
                    return previous_position
                else:
                    # Last resort: return the initial position
                    logger.warning("Using initial position")
                    return chess.Board()
                    
    def _correct_fen(self, fen: str) -> str:
        """
        Attempt to correct a broken FEN string.
        
        Args:
            fen: FEN string to correct
            
        Returns:
            Corrected FEN string
        """
        # Split into components
        parts = fen.split()
        piece_placement = parts[0]
        
        # Common issue: ranks don't add up to 8 squares
        ranks = piece_placement.split('/')
        corrected_ranks = []
        
        for rank in ranks:
            # Count the squares in this rank
            square_count = 0
            for char in rank:
                if char.isdigit():
                    square_count += int(char)
                else:
                    square_count += 1
                    
            # If not 8 squares, adjust
            if square_count < 8:
                # Add empty squares at the end
                rank += str(8 - square_count)
            elif square_count > 8:
                # Try to correct by modifying numbers
                new_rank = ""
                current_count = 0
                
                for char in rank:
                    if current_count >= 8:
                        break
                        
                    if char.isdigit():
                        # Limit this number to what we can add
                        num = min(int(char), 8 - current_count)
                        new_rank += str(num)
                        current_count += num
                    else:
                        new_rank += char
                        current_count += 1
                        
                # If still not 8, add or truncate
                if current_count < 8:
                    new_rank += str(8 - current_count)
                    
                rank = new_rank
                
            corrected_ranks.append(rank)
            
        # Ensure we have 8 ranks
        while len(corrected_ranks) < 8:
            corrected_ranks.append("8")  # Add empty ranks
            
        # Truncate if more than 8
        corrected_ranks = corrected_ranks[:8]
        
        # Combine back into FEN
        corrected_piece_placement = "/".join(corrected_ranks)
        
        # Reconstruct the FEN with corrected piece placement
        if len(parts) >= 6:
            return f"{corrected_piece_placement} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}"
        else:
            # If missing components, add defaults
            return f"{corrected_piece_placement} w KQkq - 0 1"
            
    def get_most_similar_position(self, fen: str, previous_board: Optional[chess.Board],
                                 max_diff: int = 10) -> chess.Board:
        """
        Find the most similar legal position to the detected one.
        
        This is useful when the detected position might have errors but is close
        to a legal position.
        
        Args:
            fen: Detected FEN string
            previous_board: Previous chess board for context
            max_diff: Maximum number of piece changes allowed
            
        Returns:
            Chess board object with the most similar legal position
        """
        # First, try to use the exact detected position
        try:
            board = chess.Board(fen)
            return board
        except ValueError:
            logger.warning(f"Invalid FEN: {fen}")
            
        # If that fails and we have a previous position, try to find a similar legal position
        if previous_board:
            # Filter legal moves based on similarity
            best_match = None
            min_diff = float('inf')
            
            # Consider all legal moves from the previous position
            for move in previous_board.legal_moves:
                # Apply the move
                test_board = previous_board.copy()
                test_board.push(move)
                
                # Calculate the difference
                diff = self._board_difference(fen, test_board.fen())
                
                if diff < min_diff and diff <= max_diff:
                    min_diff = diff
                    best_match = test_board.copy()
                    
            if best_match:
                logger.info(f"Found similar position with difference {min_diff}")
                return best_match
                
        # If all else fails, return a new board in the starting position
        logger.warning("Could not find similar position, using initial position")
        return chess.Board()
        
    def _board_difference(self, fen1: str, fen2: str) -> int:
        """
        Calculate the number of different pieces between two positions.
        
        Args:
            fen1: First FEN string
            fen2: Second FEN string
            
        Returns:
            Number of differing squares
        """
        # Extract just the piece placement part
        placement1 = fen1.split(" ")[0]
        placement2 = fen2.split(" ")[0]
        
        # Convert to expanded representation (no numbers for empty squares)
        expanded1 = self._expand_fen(placement1)
        expanded2 = self._expand_fen(placement2)
        
        # Count differences
        diff_count = sum(1 for a, b in zip(expanded1, expanded2) if a != b)
        
        return diff_count
        
    def _expand_fen(self, placement: str) -> str:
        """
        Expand FEN placement by replacing numbers with empty squares.
        
        Args:
            placement: Piece placement part of FEN
            
        Returns:
            Expanded representation with no numbers
        """
        expanded = ""
        
        for char in placement:
            if char.isdigit():
                # Replace digit with that many empty squares
                expanded += "." * int(char)
            elif char != "/":
                # Keep pieces and ignore rank separators
                expanded += char
                
        return expanded