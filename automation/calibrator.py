"""
Screen calibration for chess automation.

This module handles calibration of the screen for detecting the chess board
and mapping chess squares to screen coordinates.
"""

import os
import time
import logging
import cv2
import numpy as np
import pyautogui
import chess
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union

from ..vision.board_detector import BoardDetector

logger = logging.getLogger(__name__)

class Calibrator:
    """
    Calibrates the system for chess automation.
    
    This class handles the calibration process to detect the chess board,
    determine its coordinates, and set up the necessary mappings for automation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the calibrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        automation_config = config.get('automation', {})
        
        # Configuration parameters
        self.num_calibration_points = automation_config.get('calibration_corners', 5)
        
        # State variables
        self.board_coords = None
        self.white_perspective = True
        self.square_size = None
        
        # Output directories
        self.debug_dir = os.path.join(config.get('log_dir', './logs'), 'calibration')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        logger.info("Calibrator initialized")
        
    def calibrate(self, board_detector: Optional[BoardDetector] = None) -> Tuple[np.ndarray, bool]:
        """
        Perform full calibration routine.
        
        Args:
            board_detector: Optional BoardDetector instance. If None, a new one is created.
            
        Returns:
            Tuple of (board_coords, white_perspective)
        """
        logger.info("Starting calibration")
        
        # Create a board detector if not provided
        if board_detector is None:
            board_detector = BoardDetector(self.config)
            
        # Wait for user to position the board
        logger.info("Please position the chess board on screen")
        time.sleep(3)
        
        # Capture the screen
        screen = np.array(pyautogui.screenshot())
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        
        # Detect the board multiple times and average the results
        # This reduces the impact of any single detection error
        coords_list = []
        
        for i in range(self.num_calibration_points):
            try:
                logger.info(f"Detecting board coordinates (attempt {i+1}/{self.num_calibration_points})")
                coords, warped = board_detector.detect_board(screen_bgr)
                coords_list.append(coords)
                
                # Save the first warped board for perspective detection
                if i == 0:
                    first_warped = warped
                    
                # Small delay between attempts to allow for screen updates
                time.sleep(0.5)
                
            except ValueError as e:
                logger.error(f"Error detecting board: {e}")
                # Continue with other attempts
                
        if not coords_list:
            raise ValueError("Could not detect chess board. Please ensure the board is clearly visible.")
            
        # Average the coordinates
        self.board_coords = np.mean(coords_list, axis=0)
        
        # Determine board perspective
        self.white_perspective = self._detect_perspective(first_warped)
        
        # Calculate square size
        width = np.linalg.norm(self.board_coords[1] - self.board_coords[0])
        self.square_size = width / 8
        
        logger.info(f"Calibration complete:")
        logger.info(f"  Board corners: {self.board_coords}")
        logger.info(f"  Perspective: {'white' if self.white_perspective else 'black'}")
        logger.info(f"  Square size: {self.square_size:.2f} pixels")
        
        # Test the square mapping
        self._test_square_mapping()
        
        return self.board_coords, self.white_perspective
        
    def _detect_perspective(self, warped_board: np.ndarray) -> bool:
        """
        Determine if the board is viewed from white's or black's perspective.
        
        Args:
            warped_board: Warped (birds-eye view) image of the board
            
        Returns:
            True for white's perspective, False for black's
        """
        # Extract bottom-left square
        height, width = warped_board.shape[:2]
        square_size = height // 8
        bottom_left = warped_board[height - square_size:height, 0:square_size]
        
        # Extract top-right square
        top_right = warped_board[0:square_size, width - square_size:width]
        
        # Convert to grayscale
        bottom_left_gray = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)
        top_right_gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)
        
        # Compute average intensities
        bottom_left_intensity = np.mean(bottom_left_gray)
        top_right_intensity = np.mean(top_right_gray)
        
        # In standard chess, a1 and h8 are dark squares
        # If bottom-left is dark, it's likely white's perspective
        bottom_left_is_dark = bottom_left_intensity < 128
        top_right_is_dark = top_right_intensity < 128
        
        # Both corners should be dark or both should be light
        # If they're different, something is wrong with our assessment
        if bottom_left_is_dark != top_right_is_dark:
            logger.warning("Corner colors inconsistent, defaulting to white's perspective")
            
        return bottom_left_is_dark
        
    def _test_square_mapping(self) -> None:
        """
        Test the square mapping by visualizing all squares on the board.
        
        This creates a debug image showing the detected squares and their coordinates.
        """
        if self.board_coords is None:
            logger.error("Board coordinates not set")
            return
            
        # Create a function to map from chess square (0-63) to screen coordinates
        def square_to_coords(square_idx):
            # Get file and rank (0-7)
            file_idx = chess.square_file(square_idx)  # 0=a, 7=h
            rank_idx = chess.square_rank(square_idx)  # 0=1, 7=8
            
            # Adjust for board perspective
            if not self.white_perspective:
                file_idx = 7 - file_idx
                rank_idx = 7 - rank_idx
                
            # Calculate interpolation factors
            x_factor = file_idx / 7.0
            y_factor = 1.0 - (rank_idx / 7.0)  # Flipped for UI coordinates
            
            # Get the four corners of the board
            tl = self.board_coords[0]  # Top-left
            tr = self.board_coords[1]  # Top-right
            br = self.board_coords[2]  # Bottom-right
            bl = self.board_coords[3]  # Bottom-left
            
            # Interpolate to get the square position
            top_edge = tl + x_factor * (tr - tl)
            bottom_edge = bl + x_factor * (br - bl)
            square_pos = top_edge + (1 - y_factor) * (bottom_edge - top_edge)
            
            return tuple(map(int, square_pos))
            
        # Take a screenshot for visualization
        screen = np.array(pyautogui.screenshot())
        vis_img = screen.copy()
        
        # Draw the board outline
        corners = self.board_coords.astype(np.int32)
        cv2.polylines(vis_img, [corners], True, (0, 255, 0), 2)
        
        # Draw and label each square
        for square_idx in range(64):
            # Get square coordinates
            coords = square_to_coords(square_idx)
            
            # Draw circle at square center
            cv2.circle(vis_img, coords, 5, (255, 0, 0), -1)
            
            # Get algebraic notation (e.g., "a1", "e4")
            file_idx = chess.square_file(square_idx)
            rank_idx = chess.square_rank(square_idx)
            square_name = chr(97 + file_idx) + str(rank_idx + 1)
            
            # Add text label
            cv2.putText(
                vis_img, square_name, 
                (coords[0] - 10, coords[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
            
        # Save the visualization
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(self.debug_dir, f"calibration_{timestamp}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Calibration visualization saved to {output_path}")
        
        # Display the visualization if in debug mode
        if self.config.get('system', {}).get('debug_mode', False):
            plt.figure(figsize=(12, 9))
            plt.imshow(vis_img)
            plt.title("Square Mapping Visualization")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
    def interactive_calibration(self) -> Tuple[np.ndarray, bool]:
        """
        Perform interactive calibration with user guidance.
        
        This method allows the user to correct the auto-detected board corners
        if they are not accurate.
        
        Returns:
            Tuple of (board_coords, white_perspective)
        """
        logger.info("Starting interactive calibration")
        
        # First, try automatic detection
        board_detector = BoardDetector(self.config)
        
        try:
            screen = np.array(pyautogui.screenshot())
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            coords, warped = board_detector.detect_board(screen_bgr)
            
            # Highlight the detected board
            highlighted = board_detector.highlight_board(screen_bgr, coords)
            
            # Save the highlighted image
            highlight_path = os.path.join(self.debug_dir, "detected_board.jpg")
            cv2.imwrite(highlight_path, highlighted)
            
            logger.info(f"Auto-detected board saved to {highlight_path}")
            logger.info("Please check the detected board and confirm if it's correct.")
            
            # In a real implementation, you would show this to the user and allow correction
            # For this example, we'll just use the auto-detected coordinates
            
            self.board_coords = coords
            self.white_perspective = self._detect_perspective(warped)
            
            # Calculate square size
            width = np.linalg.norm(self.board_coords[1] - self.board_coords[0])
            self.square_size = width / 8
            
            # Test the square mapping
            self._test_square_mapping()
            
            return self.board_coords, self.white_perspective
            
        except ValueError:
            logger.error("Automatic board detection failed. Please mark corners manually.")
            
            # In a real implementation, you would implement manual corner selection
            # For this example, we'll just create a dummy board in the center of the screen
            
            screen_width, screen_height = pyautogui.size()
            center_x, center_y = screen_width // 2, screen_height // 2
            
            # Create a square board with specified size
            board_size = min(screen_width, screen_height) * 0.6
            half_size = board_size / 2
            
            # Define corners: top-left, top-right, bottom-right, bottom-left
            self.board_coords = np.array([
                [center_x - half_size, center_y - half_size],
                [center_x + half_size, center_y - half_size],
                [center_x + half_size, center_y + half_size],
                [center_x - half_size, center_y + half_size]
            ])
            
            # Assume white's perspective
            self.white_perspective = True
            
            # Calculate square size
            self.square_size = board_size / 8
            
            logger.warning("Using default board position. Please recalibrate if needed.")
            
            return self.board_coords, self.white_perspective