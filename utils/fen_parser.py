"""
FEN (Forsyth-Edwards Notation) parsing utilities.

This module provides utilities for parsing and manipulating FEN strings,
which represent chess positions.
"""

import chess
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set

logger = logging.getLogger(__name__)

def parse_fen(fen: str) -> Dict[str, Any]:
    """
    Parse a FEN string into its components.
    
    Args:
        fen: FEN string to parse
        
    Returns:
        Dictionary containing the parsed components:
            - piece_placement: Piece placement part
            - active_color: Active color ('w' or 'b')
            - castling: Castling availability
            - en_passant: En passant target square
            - halfmove_clock: Halfmove clock
            - fullmove_number: Fullmove number
            
    Raises:
        ValueError: If the FEN string is invalid
    """
    # Split FEN into its components
    components = fen.split()
    
    # Check if the FEN has the correct number of components
    if len(components) != 6:
        raise ValueError(f"Invalid FEN: Expected 6 components, got {len(components)}")
        
    # Extract components
    piece_placement = components[0]
    active_color = components[1]
    castling = components[2]
    en_passant = components[3]
    halfmove_clock = components[4]
    fullmove_number = components[5]
    
    # Validate piece placement
    ranks = piece_placement.split('/')
    if len(ranks) != 8:
        raise ValueError(f"Invalid piece placement: Expected 8 ranks, got {len(ranks)}")
        
    # Validate each rank
    for rank in ranks:
        total_squares = 0
        for char in rank:
            if char.isdigit():
                total_squares += int(char)
            else:
                total_squares += 1
                
        if total_squares != 8:
            raise ValueError(f"Invalid rank '{rank}': Does not represent 8 squares")
            
    # Validate active color
    if active_color not in ['w', 'b']:
        raise ValueError(f"Invalid active color: '{active_color}'. Must be 'w' or 'b'")
        
    # Validate castling
    if not re.match(r'^(K?Q?k?q?|-)+$', castling):
        raise ValueError(f"Invalid castling availability: '{castling}'")
        
    # Validate en passant
    if en_passant != '-' and not re.match(r'^[a-h][36]$', en_passant):
        raise ValueError(f"Invalid en passant target square: '{en_passant}'")
        
    # Validate halfmove clock
    try:
        half_move = int(halfmove_clock)
        if half_move < 0:
            raise ValueError(f"Invalid halfmove clock: '{halfmove_clock}'. Must be non-negative")
    except ValueError:
        raise ValueError(f"Invalid halfmove clock: '{halfmove_clock}'. Must be an integer")
        
    # Validate fullmove number
    try:
        full_move = int(fullmove_number)
        if full_move < 1:
            raise ValueError(f"Invalid fullmove number: '{fullmove_number}'. Must be positive")
    except ValueError:
        raise ValueError(f"Invalid fullmove number: '{fullmove_number}'. Must be an integer")
        
    # Return the parsed components
    return {
        'piece_placement': piece_placement,
        'active_color': active_color,
        'castling': castling,
        'en_passant': en_passant,
        'halfmove_clock': int(halfmove_clock),
        'fullmove_number': int(fullmove_number)
    }

def validate_fen(fen: str) -> bool:
    """
    Validate a FEN string.
    
    Args:
        fen: FEN string to validate
        
    Returns:
        True if the FEN is valid, False otherwise
    """
    try:
        # Try to create a chess.Board with the FEN
        chess.Board(fen)
        
        # Try to parse the FEN
        parse_fen(fen)
        
        return True
    except (ValueError, chess.InvalidBaseBoard) as e:
        logger.warning(f"Invalid FEN: {e}")
        return False

def get_pieces_from_fen(fen: str) -> Dict[chess.Square, chess.Piece]:
    """
    Get a dictionary of pieces from a FEN string.
    
    Args:
        fen: FEN string
        
    Returns:
        Dictionary mapping squares to pieces
        
    Raises:
        ValueError: If the FEN string is invalid
    """
    try:
        board = chess.Board(fen)
        return {square: piece for square, piece in board.piece_map().items()}
    except ValueError as e:
        logger.error(f"Error parsing FEN: {e}")
        raise

def fen_to_piece_counts(fen: str) -> Dict[str, int]:
    """
    Count the pieces in a FEN string.
    
    Args:
        fen: FEN string
        
    Returns:
        Dictionary mapping piece symbols to counts
        
    Raises:
        ValueError: If the FEN string is invalid
    """
    piece_placement = fen.split()[0]
    piece_counts = {
        'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,  # White pieces
        'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0   # Black pieces
    }
    
    for char in piece_placement:
        if char in piece_counts:
            piece_counts[char] += 1
            
    return piece_counts

def is_probably_chess_position(fen: str) -> bool:
    """
    Check if a FEN string likely represents a valid chess position.
    
    This performs additional checks beyond basic FEN validation, such as:
    - Each side has exactly one king
    - Pawns are not on the first or last rank
    - Not too many pieces of any type
    
    Args:
        fen: FEN string to check
        
    Returns:
        True if the FEN likely represents a valid chess position, False otherwise
    """
    if not validate_fen(fen):
        return False
        
    # Count pieces
    piece_counts = fen_to_piece_counts(fen)
    
    # Each side must have exactly one king
    if piece_counts['K'] != 1 or piece_counts['k'] != 1:
        logger.warning("Invalid position: Each side must have exactly one king")
        return False
        
    # Check for too many pieces
    max_counts = {
        'P': 8, 'N': 10, 'B': 10, 'R': 10, 'Q': 9,  # White pieces (allowing for promotions)
        'p': 8, 'n': 10, 'b': 10, 'r': 10, 'q': 9   # Black pieces
    }
    
    for piece, count in piece_counts.items():
        if piece in ['K', 'k']:  # Kings already checked
            continue
            
        if count > max_counts[piece]:
            logger.warning(f"Invalid position: Too many {piece} pieces ({count})")
            return False
            
    # Check for pawns on the first or last rank
    ranks = fen.split()[0].split('/')
    
    # First rank (from black's perspective)
    if any(c in 'Pp' for c in ranks[0]):
        logger.warning("Invalid position: Pawns on the first rank")
        return False
        
    # Last rank (from black's perspective)
    if any(c in 'Pp' for c in ranks[7]):
        logger.warning("Invalid position: Pawns on the last rank")
        return False
        
    return True

def enhance_fen(fen: str) -> Dict[str, Any]:
    """
    Extract additional information from a FEN string.
    
    Args:
        fen: FEN string
        
    Returns:
        Dictionary with enhanced information about the position
    """
    try:
        board = chess.Board(fen)
        
        # Get piece counts
        piece_counts = {
            'white': {
                'pawns': len(board.pieces(chess.PAWN, chess.WHITE)),
                'knights': len(board.pieces(chess.KNIGHT, chess.WHITE)),
                'bishops': len(board.pieces(chess.BISHOP, chess.WHITE)),
                'rooks': len(board.pieces(chess.ROOK, chess.WHITE)),
                'queens': len(board.pieces(chess.QUEEN, chess.WHITE)),
                'total': len(board.pieces(chess.PAWN, chess.WHITE)) +
                         len(board.pieces(chess.KNIGHT, chess.WHITE)) +
                         len(board.pieces(chess.BISHOP, chess.WHITE)) +
                         len(board.pieces(chess.ROOK, chess.WHITE)) +
                         len(board.pieces(chess.QUEEN, chess.WHITE)) +
                         len(board.pieces(chess.KING, chess.WHITE))
            },
            'black': {
                'pawns': len(board.pieces(chess.PAWN, chess.BLACK)),
                'knights': len(board.pieces(chess.KNIGHT, chess.BLACK)),
                'bishops': len(board.pieces(chess.BISHOP, chess.BLACK)),
                'rooks': len(board.pieces(chess.ROOK, chess.BLACK)),
                'queens': len(board.pieces(chess.QUEEN, chess.BLACK)),
                'total': len(board.pieces(chess.PAWN, chess.BLACK)) +
                         len(board.pieces(chess.KNIGHT, chess.BLACK)) +
                         len(board.pieces(chess.BISHOP, chess.BLACK)) +
                         len(board.pieces(chess.ROOK, chess.BLACK)) +
                         len(board.pieces(chess.QUEEN, chess.BLACK)) +
                         len(board.pieces(chess.KING, chess.BLACK))
            }
        }
        
        # Material difference (approximate value: P=1, N=B=3, R=5, Q=9)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Not counting king in material
        }
        
        white_material = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in piece_values.items()
        )
        
        black_material = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in piece_values.items()
        )
        
        material_difference = white_material - black_material
        
        # Phase of the game (based on remaining material)
        total_material = white_material + black_material
        max_material = 2 * (8 + 2*3 + 2*3 + 2*5 + 9)  # Max material at start
        game_phase = total_material / max_material
        
        # Game phase categories
        phase_category = "opening"
        if game_phase < 0.7:
            phase_category = "middlegame"
        if game_phase < 0.3:
            phase_category = "endgame"
            
        # Check and checkmate status
        status = {
            'check': board.is_check(),
            'checkmate': board.is_checkmate(),
            'stalemate': board.is_stalemate(),
            'insufficient_material': board.is_insufficient_material(),
            'game_over': board.is_game_over()
        }
        
        return {
            'fen': fen,
            'piece_counts': piece_counts,
            'material': {
                'white': white_material,
                'black': black_material,
                'difference': material_difference
            },
            'game_phase': {
                'value': game_phase,
                'category': phase_category
            },
            'status': status,
            'turn': 'white' if board.turn == chess.WHITE else 'black',
            'legal_moves_count': len(list(board.legal_moves))
        }
        
    except ValueError as e:
        logger.error(f"Error enhancing FEN: {e}")
        return {'error': str(e)}

def fen_difference(fen1: str, fen2: str) -> Dict[str, Any]:
    """
    Calculate the difference between two FEN positions.
    
    Args:
        fen1: First FEN string
        fen2: Second FEN string
        
    Returns:
        Dictionary describing the differences
        
    Raises:
        ValueError: If either FEN string is invalid
    """
    # Validate FENs
    if not validate_fen(fen1) or not validate_fen(fen2):
        raise ValueError("Invalid FEN string(s)")
        
    # Create boards
    board1 = chess.Board(fen1)
    board2 = chess.Board(fen2)
    
    # Get piece maps
    pieces1 = board1.piece_map()
    pieces2 = board2.piece_map()
    
    # Find differences
    removed_pieces = {}
    added_pieces = {}
    moved_pieces = {}
    
    # Track pieces by type and color
    pieces1_by_type = {}
    pieces2_by_type = {}
    
    for square, piece in pieces1.items():
        key = (piece.piece_type, piece.color)
        if key not in pieces1_by_type:
            pieces1_by_type[key] = []
        pieces1_by_type[key].append(square)
        
    for square, piece in pieces2.items():
        key = (piece.piece_type, piece.color)
        if key not in pieces2_by_type:
            pieces2_by_type[key] = []
        pieces2_by_type[key].append(square)
        
    # Check for removed or moved pieces
    for square, piece in pieces1.items():
        if square not in pieces2 or pieces2[square] != piece:
            # This piece was removed or moved
            piece_str = piece.symbol()
            square_name = chess.square_name(square)
            
            # Check if it might have moved
            key = (piece.piece_type, piece.color)
            if key in pieces2_by_type and len(pieces1_by_type.get(key, [])) <= len(pieces2_by_type.get(key, [])):
                # Might have moved - add to moved pieces for now
                if piece_str not in moved_pieces:
                    moved_pieces[piece_str] = []
                moved_pieces[piece_str].append(square_name)
            else:
                # Definitely removed
                if piece_str not in removed_pieces:
                    removed_pieces[piece_str] = []
                removed_pieces[piece_str].append(square_name)
                
    # Check for added pieces
    for square, piece in pieces2.items():
        if square not in pieces1:
            # This piece was added or is the destination of a move
            piece_str = piece.symbol()
            square_name = chess.square_name(square)
            
            key = (piece.piece_type, piece.color)
            if key in pieces1_by_type and len(pieces1_by_type.get(key, [])) >= len(pieces2_by_type.get(key, [])):
                # Likely the destination of a move - already handled above
                pass
            else:
                # Added piece
                if piece_str not in added_pieces:
                    added_pieces[piece_str] = []
                added_pieces[piece_str].append(square_name)
                
    # Other differences
    turn_changed = board1.turn != board2.turn
    castling_changed = board1.castling_rights != board2.castling_rights
    en_passant_changed = board1.ep_square != board2.ep_square
    
    # Try to determine the move that was made
    possible_moves = []
    
    if len(moved_pieces) == 1 and len(added_pieces) == 0 and len(removed_pieces) == 0:
        # Simple move (piece moved from one square to another)
        piece_type, source_squares = list(moved_pieces.items())[0]
        for source in source_squares:
            for square, piece in pieces2.items():
                if piece.symbol() == piece_type and chess.square_name(square) not in pieces1:
                    possible_moves.append(f"{source}{chess.square_name(square)}")
                    
    elif len(moved_pieces) == 1 and len(added_pieces) == 1 and len(removed_pieces) == 0:
        # Possible promotion
        piece_type, source_squares = list(moved_pieces.items())[0]
        if piece_type in "Pp":  # Pawn
            promoted_piece, dest_squares = list(added_pieces.items())[0]
            for source in source_squares:
                for dest in dest_squares:
                    # Check if it's a valid promotion square
                    dest_rank = dest[1]
                    if (piece_type == "P" and dest_rank == "8") or (piece_type == "p" and dest_rank == "1"):
                        possible_moves.append(f"{source}{dest}{promoted_piece.lower()}")
                        
    elif len(moved_pieces) == 2 and len(added_pieces) == 0 and len(removed_pieces) == 0:
        # Possible castling
        pieces = list(moved_pieces.keys())
        if "K" in pieces and "R" in pieces:
            # White castling
            if "e1" in moved_pieces["K"]:
                if "h1" in moved_pieces["R"]:
                    possible_moves.append("e1g1")  # Kingside
                elif "a1" in moved_pieces["R"]:
                    possible_moves.append("e1c1")  # Queenside
        elif "k" in pieces and "r" in pieces:
            # Black castling
            if "e8" in moved_pieces["k"]:
                if "h8" in moved_pieces["r"]:
                    possible_moves.append("e8g8")  # Kingside
                elif "a8" in moved_pieces["r"]:
                    possible_moves.append("e8c8")  # Queenside
                    
    return {
        'removed_pieces': removed_pieces,
        'added_pieces': added_pieces,
        'moved_pieces': moved_pieces,
        'turn_changed': turn_changed,
        'castling_changed': castling_changed,
        'en_passant_changed': en_passant_changed,
        'possible_moves': possible_moves
    }