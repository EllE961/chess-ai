"""
Adapter for different chess platform interfaces.

This module provides adapters for interacting with different chess platforms,
handling platform-specific UI elements and behaviors.
"""

import os
import time
import logging
import enum
import pyautogui
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from ..vision.board_detector import BoardDetector

logger = logging.getLogger(__name__)

class ChessPlatform(enum.Enum):
    """Enumeration of supported chess platforms."""
    LICHESS = "lichess"
    CHESS_COM = "chess.com"
    CHESS24 = "chess24"
    CUSTOM = "custom"


class PlatformAdapter:
    """
    Adapter for interacting with different chess platforms.
    
    Handles platform-specific details such as detecting game state,
    finding UI elements, and initiating games.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the platform adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        automation_config = config.get('automation', {})
        
        # Get the configured platform
        platform_name = automation_config.get('platform', 'lichess')
        try:
            self.platform = ChessPlatform(platform_name.lower())
        except ValueError:
            logger.warning(f"Unknown platform '{platform_name}', using custom")
            self.platform = ChessPlatform.CUSTOM
            
        # Game state
        self.is_playing = False
        self.is_our_turn = False
        self.played_as_white = None  # None until determined
        
        # Template image paths
        self.template_dir = config.get('template_dir', os.path.join(
            config.get('base_dir', '.'), 'templates'
        ))
        
        # Template matching parameters
        self.template_threshold = 0.8
        self.previous_screen = None
        
        logger.info(f"Platform adapter initialized for {self.platform.value}")
        
    def detect_platform(self) -> ChessPlatform:
        """
        Automatically detect which chess platform is being used.
        
        Returns:
            Detected chess platform
        """
        logger.info("Attempting to detect chess platform")
        
        # Capture current screen
        screen = np.array(pyautogui.screenshot())
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        
        # Check for platform-specific indicators
        platforms = [
            (ChessPlatform.LICHESS, "lichess_logo.png"),
            (ChessPlatform.CHESS_COM, "chess_com_logo.png"),
            (ChessPlatform.CHESS24, "chess24_logo.png")
        ]
        
        for platform, template_name in platforms:
            template_path = os.path.join(self.template_dir, template_name)
            if os.path.exists(template_path):
                template = cv2.imread(template_path)
                if template is not None:
                    # Perform template matching
                    result = cv2.matchTemplate(screen_bgr, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    if max_val > self.template_threshold:
                        logger.info(f"Detected platform: {platform.value}")
                        self.platform = platform
                        return platform
                        
        logger.warning("Could not detect platform, using configured value")
        return self.platform
        
    def detect_game_state(self) -> Tuple[bool, bool]:
        """
        Detect the current game state.
        
        Returns:
            Tuple of (is_playing, is_our_turn):
                - is_playing: True if a game is in progress
                - is_our_turn: True if it's our turn to move
        """
        # Capture current screen
        screen = np.array(pyautogui.screenshot())
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        
        # Check for screen changes (movement detection)
        change_detected = False
        if self.previous_screen is not None:
            # Convert to grayscale for simpler comparison
            current_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            previous_gray = cv2.cvtColor(self.previous_screen, cv2.COLOR_RGB2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(current_gray, previous_gray)
            
            # Apply threshold to get significant changes
            _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Count changed pixels
            changed_pixels = np.count_nonzero(thresholded)
            
            # Determine if significant change occurred
            change_detected = changed_pixels > 10000  # Arbitrary threshold
            
        # Store current screen for next comparison
        self.previous_screen = screen
        
        # Platform-specific detection
        if self.platform == ChessPlatform.LICHESS:
            return self._detect_lichess_state(screen_bgr, change_detected)
        elif self.platform == ChessPlatform.CHESS_COM:
            return self._detect_chess_com_state(screen_bgr, change_detected)
        elif self.platform == ChessPlatform.CHESS24:
            return self._detect_chess24_state(screen_bgr, change_detected)
        else:
            return self._detect_generic_state(screen_bgr, change_detected)
            
    def _detect_lichess_state(self, screen: np.ndarray, change_detected: bool) -> Tuple[bool, bool]:
        """
        Detect game state on Lichess.
        
        Args:
            screen: Screenshot of the current screen
            change_detected: Whether significant change was detected since last check
            
        Returns:
            Tuple of (is_playing, is_our_turn)
        """
        # Check for "Your turn" indicator
        your_turn_template = os.path.join(self.template_dir, "lichess_your_turn.png")
        if os.path.exists(your_turn_template):
            template = cv2.imread(your_turn_template)
            if template is not None:
                result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > self.template_threshold:
                    logger.debug("Detected 'Your turn' indicator on Lichess")
                    self.is_playing = True
                    self.is_our_turn = True
                    return True, True
                    
        # Check for game controls (indicating a game is in progress)
        game_controls_template = os.path.join(self.template_dir, "lichess_game_controls.png")
        if os.path.exists(game_controls_template):
            template = cv2.imread(game_controls_template)
            if template is not None:
                result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > self.template_threshold:
                    logger.debug("Detected game controls on Lichess")
                    self.is_playing = True
                    # If controls are visible but not "Your turn", it's likely opponent's turn
                    self.is_our_turn = False
                    return True, False
                    
        # If neither indicator is found, we're probably not in a game
        # But keep previous state if unsure
        if self.is_playing:
            logger.debug("No game indicators found, but was previously playing")
            return True, self.is_our_turn
            
        logger.debug("No game detected on Lichess")
        return False, False
        
    def _detect_chess_com_state(self, screen: np.ndarray, change_detected: bool) -> Tuple[bool, bool]:
        """
        Detect game state on Chess.com.
        
        Args:
            screen: Screenshot of the current screen
            change_detected: Whether significant change was detected since last check
            
        Returns:
            Tuple of (is_playing, is_our_turn)
        """
        # Implementation would be similar to Lichess but with Chess.com templates
        # For now, use generic detection
        return self._detect_generic_state(screen, change_detected)
        
    def _detect_chess24_state(self, screen: np.ndarray, change_detected: bool) -> Tuple[bool, bool]:
        """
        Detect game state on Chess24.
        
        Args:
            screen: Screenshot of the current screen
            change_detected: Whether significant change was detected since last check
            
        Returns:
            Tuple of (is_playing, is_our_turn)
        """
        # Implementation would be similar to Lichess but with Chess24 templates
        # For now, use generic detection
        return self._detect_generic_state(screen, change_detected)
        
    def _detect_generic_state(self, screen: np.ndarray, change_detected: bool) -> Tuple[bool, bool]:
        """
        Generic game state detection that works across platforms.
        
        Args:
            screen: Screenshot of the current screen
            change_detected: Whether significant change was detected since last check
            
        Returns:
            Tuple of (is_playing, is_our_turn)
        """
        # If we detected significant change, it might indicate a move was made
        if change_detected:
            logger.debug("Detected screen change, possible move")
            # If we were waiting for opponent, now it might be our turn
            if self.is_playing and not self.is_our_turn:
                logger.info("Screen changed while waiting for opponent - likely our turn now")
                self.is_our_turn = True
                return True, True
                
            # If it was our turn, opponent might have moved
            elif self.is_playing and self.is_our_turn:
                logger.debug("Screen changed during our turn - might be just UI update")
                return True, True
                
        # No significant change
        # Keep current state if playing
        if self.is_playing:
            return True, self.is_our_turn
            
        # Try to detect a chess board as fallback
        # This is a simplistic approach - real implementation would be more robust
        try:
            detector = BoardDetector(self.config)
            detector.detect_board(screen)
            logger.info("Chess board detected, assuming game in progress")
            self.is_playing = True
            # Can't determine turn reliably, assume it's our turn
            self.is_our_turn = True
            return True, True
        except ValueError:
            # No board detected
            logger.debug("No chess board detected")
            return False, False
            
    def find_board(self, board_detector: BoardDetector) -> Tuple[np.ndarray, bool]:
        """
        Find and analyze the chess board on screen.
        
        Args:
            board_detector: BoardDetector instance
            
        Returns:
            Tuple of (board_coords, white_perspective)
        """
        # Detect the board
        board_coords, warped_board = board_detector.detect_board()
        
        # Determine if we're playing as white or black
        # This affects how we interpret the board
        white_perspective = self._determine_board_perspective(warped_board)
        
        # If this is the first time we've determined perspective, record it
        if self.played_as_white is None:
            self.played_as_white = white_perspective
            logger.info(f"Determined we are playing as {'white' if white_perspective else 'black'}")
            
        return board_coords, white_perspective
        
    def _determine_board_perspective(self, warped_board: np.ndarray) -> bool:
        """
        Determine if the board is from white's or black's perspective.
        
        Args:
            warped_board: Warped (birds-eye view) image of the board
            
        Returns:
            True for white's perspective, False for black's
        """
        # Most chess platforms have coordinate labels around the board
        # We can use these to determine the perspective
        # For a generic approach, we'll use the color of the bottom-left square
        
        # Extract bottom-left square
        height, width = warped_board.shape[:2]
        square_size = height // 8
        bottom_left = warped_board[height - square_size:height, 0:square_size]
        
        # Convert to grayscale
        gray = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)
        
        # Compute average intensity
        avg_intensity = np.mean(gray)
        
        # In standard chess, a1 (bottom-left from white's perspective) is a dark square
        # If the bottom-left square is dark, it's likely white's perspective
        is_dark = avg_intensity < 128
        
        return is_dark
        
    def start_new_game(self, color: str = 'random') -> bool:
        """
        Start a new game on the current platform.
        
        Args:
            color: Preferred color ('white', 'black', or 'random')
            
        Returns:
            True if the operation was successful, False otherwise
        """
        logger.info(f"Starting new game as {color}")
        
        if self.platform == ChessPlatform.LICHESS:
            return self._start_new_game_lichess(color)
        elif self.platform == ChessPlatform.CHESS_COM:
            return self._start_new_game_chess_com(color)
        elif self.platform == ChessPlatform.CHESS24:
            return self._start_new_game_chess24(color)
        else:
            logger.warning("Starting new game not supported for custom platforms")
            return False
            
    def _start_new_game_lichess(self, color: str) -> bool:
        """
        Start a new game on Lichess.
        
        Args:
            color: Preferred color ('white', 'black', or 'random')
            
        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Navigate to Lichess
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.5)
            pyautogui.typewrite('https://lichess.org')
            pyautogui.press('enter')
            time.sleep(3)  # Wait for page to load
            
            # Click "Play with the computer" button
            # Coordinates would need to be calibrated for different screen resolutions
            # This is just an example
            pyautogui.click(x=800, y=400)
            time.sleep(1)
            
            # Select level (e.g., level 3)
            pyautogui.click(x=800, y=500)
            time.sleep(1)
            
            # Select color
            if color == 'white':
                pyautogui.click(x=700, y=600)
            elif color == 'black':
                pyautogui.click(x=900, y=600)
            else:  # random
                pyautogui.click(x=800, y=600)
                
            time.sleep(1)
            
            # Click "Play" button
            pyautogui.click(x=800, y=700)
            
            # Wait for game to start
            time.sleep(3)
            
            # Reset game state
            self.is_playing = True
            if color == 'white':
                self.is_our_turn = True
                self.played_as_white = True
            elif color == 'black':
                self.is_our_turn = False
                self.played_as_white = False
            else:
                # For random, we'll determine later
                self.is_our_turn = None
                self.played_as_white = None
                
            return True
            
        except Exception as e:
            logger.error(f"Error starting new game on Lichess: {e}")
            return False
            
    def _start_new_game_chess_com(self, color: str) -> bool:
        """
        Start a new game on Chess.com.
        
        Args:
            color: Preferred color ('white', 'black', or 'random')
            
        Returns:
            True if the operation was successful, False otherwise
        """
        # Implementation would be similar to Lichess but with Chess.com-specific UI
        # This would need to be customized based on the actual UI layout
        logger.warning("Chess.com new game not fully implemented")
        return False
        
    def _start_new_game_chess24(self, color: str) -> bool:
        """
        Start a new game on Chess24.
        
        Args:
            color: Preferred color ('white', 'black', or 'random')
            
        Returns:
            True if the operation was successful, False otherwise
        """
        # Implementation would be similar to Lichess but with Chess24-specific UI
        # This would need to be customized based on the actual UI layout
        logger.warning("Chess24 new game not fully implemented")
        return False
        
    def resign_game(self) -> bool:
        """
        Resign the current game.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        logger.info("Resigning game")
        
        if not self.is_playing:
            logger.warning("No active game to resign")
            return False
            
        if self.platform == ChessPlatform.LICHESS:
            return self._resign_lichess()
        elif self.platform == ChessPlatform.CHESS_COM:
            return self._resign_chess_com()
        elif self.platform == ChessPlatform.CHESS24:
            return self._resign_chess24()
        else:
            # Generic approach: press Escape and look for resign button
            try:
                pyautogui.press('escape')
                time.sleep(0.5)
                
                # Look for "resign" text on screen (would require OCR in a real implementation)
                # Simplified by clicking where a resign button might be
                pyautogui.click(x=800, y=500)
                time.sleep(0.5)
                
                # Confirm resignation (might be needed on some platforms)
                pyautogui.click(x=800, y=550)
                
                # Reset game state
                self.is_playing = False
                self.is_our_turn = False
                
                return True
                
            except Exception as e:
                logger.error(f"Error resigning game: {e}")
                return False
                
    def _resign_lichess(self) -> bool:
        """
        Resign a game on Lichess.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Press Escape to open the menu
            pyautogui.press('escape')
            time.sleep(0.5)
            
            # Click resign button
            # Coordinates would need to be calibrated
            pyautogui.click(x=800, y=500)
            time.sleep(0.5)
            
            # Confirm resignation
            pyautogui.click(x=800, y=550)
            
            # Reset game state
            self.is_playing = False
            self.is_our_turn = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error resigning game on Lichess: {e}")
            return False
            
    def _resign_chess_com(self) -> bool:
        """
        Resign a game on Chess.com.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        # Implementation would be similar to Lichess but with Chess.com-specific UI
        logger.warning("Chess.com resignation not fully implemented")
        return False
        
    def _resign_chess24(self) -> bool:
        """
        Resign a game on Chess24.
        
        Returns:
            True if the operation was successful, False otherwise
        """
        # Implementation would be similar to Lichess but with Chess24-specific UI
        logger.warning("Chess24 resignation not fully implemented")
        return False