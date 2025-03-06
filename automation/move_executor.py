"""
Chess move execution through mouse automation.

This module handles the translation of chess moves to mouse actions,
enabling the AI to play on digital chess platforms by controlling the mouse.
"""

import time
import random
import logging
import pyautogui
import chess
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

# Prevent PyAutoGUI from raising exceptions when mouse hits screen edge
pyautogui.FAILSAFE = False

class MoveExecutor:
    """
    Executes chess moves by controlling the mouse.
    
    Translates chess moves (e.g., e2e4) into mouse movements and clicks
    to interact with chess platforms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the move executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        automation_config = config.get('automation', {})
        
        # Configuration parameters
        self.mouse_move_delay = automation_config.get('mouse_move_delay', 0.1)
        self.move_execution_delay = automation_config.get('move_execution_delay', 0.5)
        self.click_randomness = automation_config.get('click_randomness', 5)
        
        # Apply mouse move delay to PyAutoGUI
        pyautogui.PAUSE = self.mouse_move_delay
        
        # Board information (will be set by calibration)
        self.board_coords = None
        self.square_size = None
        self.white_perspective = True
        
        logger.info("Move executor initialized")
        
    def set_board_info(self, board_coords: np.ndarray, white_perspective: bool = True) -> None:
        """
        Set the board coordinates and perspective.
        
        This must be called before executing moves.
        
        Args:
            board_coords: NumPy array of shape (4, 2) with corner coordinates
            white_perspective: Whether the board is viewed from white's perspective
        """
        self.board_coords = board_coords
        self.white_perspective = white_perspective
        
        # Calculate square size based on board width
        # Distance between top-left and top-right divided by 8
        width = np.linalg.norm(board_coords[1] - board_coords[0])
        self.square_size = width / 8
        
        logger.info(f"Board info set: square size = {self.square_size:.1f} pixels, "
                   f"perspective = {'white' if white_perspective else 'black'}")
        
    def execute_move(self, move: Union[chess.Move, str]) -> bool:
        """
        Execute a chess move by controlling the mouse.
        
        Args:
            move: Chess move to execute (either chess.Move object or UCI string)
            
        Returns:
            True if the move was executed successfully, False otherwise
            
        Raises:
            ValueError: If board information has not been set
        """
        if not self.board_coords or not self.square_size:
            raise ValueError("Board information not set. Call set_board_info first.")
            
        # Convert string move to chess.Move if needed
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                logger.error(f"Invalid UCI move: {move}")
                return False
                
        logger.info(f"Executing move: {move.uci()}")
        
        try:
            # Get the coordinates of the from and to squares
            from_coords = self._square_to_screen_coords(move.from_square)
            to_coords = self._square_to_screen_coords(move.to_square)
            
            # Execute the move with the mouse
            self._execute_mouse_move(from_coords, to_coords)
            
            # Handle promotion if needed
            if move.promotion:
                self._handle_promotion(move.promotion, to_coords)
                
            logger.info(f"Move {move.uci()} executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error executing move {move.uci()}: {e}")
            return False
            
    def _square_to_screen_coords(self, square: int) -> Tuple[int, int]:
        """
        Convert a chess square index to screen coordinates.
        
        Args:
            square: Chess square index (0-63, where 0=a1, 63=h8)
            
        Returns:
            Tuple of (x, y) screen coordinates for the center of the square
        """
        # Get file and rank (0-7)
        file_idx = chess.square_file(square)  # 0=a, 7=h
        rank_idx = chess.square_rank(square)  # 0=1, 7=8
        
        # Adjust for board perspective
        if not self.white_perspective:
            file_idx = 7 - file_idx
            rank_idx = 7 - rank_idx
            
        # Calculate interpolation factors
        x_factor = file_idx / 7.0  # Horizontal factor (0 to 1)
        y_factor = 1.0 - (rank_idx / 7.0)  # Vertical factor (0 to 1), flipped for UI coordinates
        
        # Get the four corners of the board
        tl = self.board_coords[0]  # Top-left
        tr = self.board_coords[1]  # Top-right
        br = self.board_coords[2]  # Bottom-right
        bl = self.board_coords[3]  # Bottom-left
        
        # Interpolate to get the square position
        top_edge = tl + x_factor * (tr - tl)
        bottom_edge = bl + x_factor * (br - bl)
        square_pos = top_edge + (1 - y_factor) * (bottom_edge - top_edge)
        
        # Convert to integer coordinates
        return tuple(map(int, square_pos))
        
    def _add_click_randomness(self, coords: Tuple[int, int]) -> Tuple[int, int]:
        """
        Add random offset to click coordinates for more human-like behavior.
        
        Args:
            coords: Original (x, y) coordinates
            
        Returns:
            Modified coordinates with random offset
        """
        # Add random offset within square bounds
        offset_range = min(self.square_size / 3, self.click_randomness)
        x_offset = random.uniform(-offset_range, offset_range)
        y_offset = random.uniform(-offset_range, offset_range)
        
        return (int(coords[0] + x_offset), int(coords[1] + y_offset))
        
    def _execute_mouse_move(self, from_coords: Tuple[int, int], to_coords: Tuple[int, int]) -> None:
        """
        Execute a chess move using mouse controls.
        
        Args:
            from_coords: Starting square coordinates
            to_coords: Destination square coordinates
        """
        # Add randomness to both coordinates
        from_coords = self._add_click_randomness(from_coords)
        to_coords = self._add_click_randomness(to_coords)
        
        # Move mouse to the starting square and click
        self._move_mouse_humanlike(from_coords)
        pyautogui.click()
        
        # Small random delay between clicks
        time.sleep(random.uniform(0.3, 0.7) * self.move_execution_delay)
        
        # Move mouse to the destination square and click
        self._move_mouse_humanlike(to_coords)
        pyautogui.click()
        
        # Wait after move completion
        time.sleep(random.uniform(0.5, 1.0) * self.move_execution_delay)
        
    def _move_mouse_humanlike(self, coords: Tuple[int, int], duration: Optional[float] = None) -> None:
        """
        Move the mouse in a human-like manner.
        
        Args:
            coords: Target (x, y) coordinates
            duration: Optional duration for the movement (overrides config)
        """
        if duration is None:
            # Randomize movement duration based on distance
            base_duration = self.move_execution_delay
            current_pos = pyautogui.position()
            distance = ((current_pos[0] - coords[0]) ** 2 + (current_pos[1] - coords[1]) ** 2) ** 0.5
            
            # Longer movements take more time (with randomness)
            duration = base_duration * (0.5 + 0.5 * min(1.0, distance / 500)) * random.uniform(0.8, 1.2)
            
        # Use pyautogui's easing function for smoother movement
        pyautogui.moveTo(
            coords[0], coords[1],
            duration=duration,
            tween=pyautogui.easeOutQuad
        )
        
    def _handle_promotion(self, promotion_piece: int, to_coords: Tuple[int, int]) -> None:
        """
        Handle piece promotion by clicking the appropriate piece option.
        
        Args:
            promotion_piece: The piece type to promote to (chess.QUEEN, etc.)
            to_coords: Coordinates of the promotion square
        """
        logger.debug(f"Handling promotion to {chess.piece_name(promotion_piece)}")
        
        # Wait for promotion dialog to appear
        time.sleep(0.5)
        
        # Different platforms have different promotion UI layouts
        # This is a simplified implementation that assumes a vertical list
        # of promotion options, with queen at the top
        
        # Mapping from piece type to position in the promotion menu
        piece_to_index = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }
        
        # Get the index for this piece type
        index = piece_to_index.get(promotion_piece, 0)  # Default to queen
        
        # Calculate offset based on piece position in the menu
        # This will need to be calibrated for specific chess platforms
        x_offset = 0
        y_offset = index * 50  # Typical spacing between promotion options
        
        # Click on the promotion piece
        promotion_coords = (to_coords[0] + x_offset, to_coords[1] + y_offset)
        self._move_mouse_humanlike(promotion_coords, duration=0.2)
        pyautogui.click()
        
        # Wait after promotion
        time.sleep(0.5)
        
    def drag_and_drop(self, from_square: int, to_square: int) -> bool:
        """
        Execute a move using drag-and-drop instead of clicks.
        
        Some chess platforms work better with drag-and-drop.
        
        Args:
            from_square: Starting square index (0-63)
            to_square: Destination square index (0-63)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.board_coords or not self.square_size:
            raise ValueError("Board information not set. Call set_board_info first.")
            
        try:
            # Get the coordinates of the from and to squares
            from_coords = self._square_to_screen_coords(from_square)
            to_coords = self._square_to_screen_coords(to_square)
            
            # Add randomness to both coordinates
            from_coords = self._add_click_randomness(from_coords)
            to_coords = self._add_click_randomness(to_coords)
            
            # Move mouse to the starting square
            self._move_mouse_humanlike(from_coords)
            
            # Press and hold the mouse button
            pyautogui.mouseDown()
            
            # Small delay before dragging
            time.sleep(random.uniform(0.1, 0.2))
            
            # Move to the destination
            self._move_mouse_humanlike(to_coords, duration=random.uniform(0.3, 0.5))
            
            # Small delay before releasing
            time.sleep(random.uniform(0.1, 0.2))
            
            # Release the mouse button
            pyautogui.mouseUp()
            
            # Wait after move completion
            time.sleep(random.uniform(0.5, 1.0) * self.move_execution_delay)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during drag-and-drop: {e}")
            # Try to ensure mouse button is released in case of error
            pyautogui.mouseUp()
            return False