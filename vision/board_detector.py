"""
Chess board detection using computer vision.

This module provides functionality to detect a chess board on the screen,
extract its position and orientation, and segment it into squares.
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import Tuple, List, Dict, Any, Optional
from PIL import ImageGrab

logger = logging.getLogger(__name__)

class BoardDetector:
    """
    Detects and processes chess boards on screen.
    
    Uses computer vision techniques to find the chess board, determine its
    corners, and extract the individual squares.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the board detector.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        vision_config = config.get('vision', {})
        
        # Configuration parameters
        self.detection_threshold = vision_config.get('board_detection_threshold', 0.8)
        self.square_size = vision_config.get('square_size', 100)
        self.detection_interval = vision_config.get('detection_interval', 0.5)
        
        # State variables
        self.last_detection_time = 0
        self.last_board_coords = None
        self.last_warped_board = None
        
        # Directories
        self.debug_dir = os.path.join(config.get('log_dir', './logs'), 'vision_debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        logger.info("Board detector initialized")
        
    def capture_screen(self) -> np.ndarray:
        """
        Capture the current screen.
        
        Returns:
            NumPy array containing the screen image in BGR format.
        """
        # Capture screen using PIL and convert to numpy array
        screen = np.array(ImageGrab.grab())
        
        # Convert from RGB to BGR (OpenCV format)
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        
        return screen_bgr
        
    def detect_board(self, image: Optional[np.ndarray] = None, force_detection: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect a chess board in the given image or current screen.
        
        Args:
            image: Optional image to process. If None, captures the current screen.
            force_detection: If True, forces a new detection even if recently detected.
            
        Returns:
            Tuple of (board_coords, warped_board) where:
                - board_coords: NumPy array of shape (4, 2) with the corner coordinates
                - warped_board: Birds-eye view of the board (warped perspective)
                
        Raises:
            ValueError: If no chess board is detected.
        """
        current_time = time.time()
        
        # Use cached result if available and recent
        if (not force_detection and
            self.last_board_coords is not None and
            current_time - self.last_detection_time < self.detection_interval):
            logger.debug("Using cached board detection")
            return self.last_board_coords, self.last_warped_board
            
        # Capture screen if image not provided
        if image is None:
            image = self.capture_screen()
            
        # Convert to grayscale for better processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to get a binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size and shape
        board_contour = self._find_board_contour(contours, image.shape)
        
        if board_contour is None:
            # Try alternative approach
            board_contour = self._find_board_using_hough_lines(gray)
            
        if board_contour is None:
            # If we still can't find the board and have a previous detection,
            # use the cached coordinates
            if self.last_board_coords is not None:
                logger.warning("Could not detect board, using previous coordinates")
                return self.last_board_coords, self.last_warped_board
            else:
                logger.error("Could not detect chess board on screen")
                raise ValueError("No chess board detected")
                
        # Get corner coordinates
        board_coords = self._get_corner_coordinates(board_contour)
        
        # Create warped (birds-eye view) of the board
        warped_board = self._warp_perspective(image, board_coords)
        
        # Save the results
        self.last_board_coords = board_coords
        self.last_warped_board = warped_board
        self.last_detection_time = current_time
        
        # Save debug image if needed
        if self.config.get('system', {}).get('debug_mode', False):
            self._save_debug_image(image, board_coords, warped_board)
            
        logger.debug("Chess board detected successfully")
        return board_coords, warped_board
        
    def _find_board_contour(self, contours: List[np.ndarray], image_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Find the contour corresponding to the chess board.
        
        Args:
            contours: List of contours from findContours.
            image_shape: Shape of the original image (height, width, channels).
            
        Returns:
            The contour of the board, or None if not found.
        """
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get image dimensions
        img_height, img_width = image_shape[:2]
        img_area = img_height * img_width
        
        # Minimum and maximum area thresholds (percentage of total image)
        min_area_pct = 0.05  # Board should be at least 5% of the image
        max_area_pct = 0.95  # Board should be at most 95% of the image
        
        min_area = img_area * min_area_pct
        max_area = img_area * max_area_pct
        
        # Looking for approximately square-shaped contours with 4 corners
        for contour in contours[:10]:  # Check only the 10 largest contours
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
                
            # Approximate the contour to find corners
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # We're looking for quadrilaterals (4 corners)
            if len(approx) == 4:
                # Check if it's roughly square/rectangular
                # (all internal angles should be close to 90 degrees)
                if self._is_approximately_rectangular(approx):
                    return approx
                    
        return None
        
    def _find_board_using_hough_lines(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Alternative approach to find the board using Hough lines.
        
        Args:
            gray_image: Grayscale image.
            
        Returns:
            The contour of the board, or None if not found.
        """
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Find lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) < 8:  # Need at least 8 lines for a chess board
            return None
            
        # Process lines to find intersections
        # This is a simplified placeholder - a real implementation would be more complex
        # and would extract the four corners from the line intersections
        
        # For now, return None to fall back to previous detection
        return None
        
    def _is_approximately_rectangular(self, contour: np.ndarray) -> bool:
        """
        Check if a contour is approximately rectangular.
        
        Args:
            contour: Contour to check.
            
        Returns:
            True if the contour is approximately rectangular, False otherwise.
        """
        # Convert to a simpler format
        points = contour.reshape(-1, 2)
        
        # There should be 4 points
        if len(points) != 4:
            return False
            
        # Compute all angles
        angles = []
        for i in range(4):
            # Get three consecutive points
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            # Compute vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Compute the angle
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product == 0:
                return False
                
            cos_angle = dot_product / norm_product
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(np.degrees(angle))
            
        # Check if all angles are approximately 90 degrees (within tolerance)
        tolerance = 15  # degrees
        return all(abs(angle - 90) < tolerance for angle in angles)
        
    def _get_corner_coordinates(self, contour: np.ndarray) -> np.ndarray:
        """
        Get the corner coordinates in the order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            contour: Contour of the board.
            
        Returns:
            NumPy array of shape (4, 2) with the corner coordinates.
        """
        # Reshape to get points
        points = contour.reshape(-1, 2)
        
        # Compute the sum and difference of x and y coordinates
        # This is a common trick to order the points
        s = points.sum(axis=1)
        d = np.diff(points, axis=1)
        
        # Order points: top-left, top-right, bottom-right, bottom-left
        # Top-left has the smallest sum of coordinates
        # Top-right has the smallest difference of coordinates
        # Bottom-right has the largest sum of coordinates
        # Bottom-left has the largest difference of coordinates
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(s)]  # Top-left
        ordered[1] = points[np.argmin(d)]  # Top-right
        ordered[2] = points[np.argmax(s)]  # Bottom-right
        ordered[3] = points[np.argmax(d)]  # Bottom-left
        
        return ordered
        
    def _warp_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warp the image to get a birds-eye view of the board.
        
        Args:
            image: Source image.
            corners: Corner coordinates of the board.
            
        Returns:
            Warped image of the board.
        """
        # Size of the output image (square)
        warped_size = self.square_size * 8
        
        # Define the destination points (rectangle)
        dst_points = np.array([
            [0, 0],                          # Top-left
            [warped_size - 1, 0],            # Top-right
            [warped_size - 1, warped_size - 1],  # Bottom-right
            [0, warped_size - 1]             # Bottom-left
        ], dtype=np.float32)
        
        # Compute the perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply the transformation
        warped = cv2.warpPerspective(image, transform_matrix, (warped_size, warped_size))
        
        return warped
        
    def extract_squares(self, warped_board: np.ndarray) -> List[List[np.ndarray]]:
        """
        Extract individual squares from the warped board image.
        
        Args:
            warped_board: Warped (birds-eye view) image of the board.
            
        Returns:
            2D list of square images, where squares[rank][file] is the image
            at the given rank and file (0-7).
        """
        # Get the size of the warped board
        height, width = warped_board.shape[:2]
        
        # Each square is 1/8 of the board size
        square_size = height // 8
        
        # Extract all 64 squares
        squares = []
        for rank in range(8):
            rank_squares = []
            for file in range(8):
                # Extract the square
                square = warped_board[
                    rank * square_size:(rank + 1) * square_size,
                    file * square_size:(file + 1) * square_size
                ]
                rank_squares.append(square)
            squares.append(rank_squares)
            
        return squares
        
    def highlight_board(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw the detected board on the image for visualization.
        
        Args:
            image: Source image.
            corners: Corner coordinates of the board.
            
        Returns:
            Image with the board highlighted.
        """
        # Create a copy of the image
        highlighted = image.copy()
        
        # Draw the board outline
        cv2.polylines(
            highlighted, [corners.astype(np.int32)], 
            isClosed=True, color=(0, 255, 0), thickness=3
        )
        
        # Draw the corner points
        for i, point in enumerate(corners):
            cv2.circle(
                highlighted, tuple(point.astype(np.int32)), 
                radius=10, color=(0, 0, 255), thickness=-1
            )
            
            # Label the corners
            labels = ["TL", "TR", "BR", "BL"]
            cv2.putText(
                highlighted, labels[i], tuple(point.astype(np.int32) + np.array([10, 10])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
            
        return highlighted
        
    def _save_debug_image(self, image: np.ndarray, corners: np.ndarray, warped_board: np.ndarray) -> None:
        """
        Save debug images for inspection.
        
        Args:
            image: Original image.
            corners: Corner coordinates of the board.
            warped_board: Warped (birds-eye view) image of the board.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save highlighted original image
        highlighted = self.highlight_board(image, corners)
        cv2.imwrite(os.path.join(self.debug_dir, f"{timestamp}_detected.jpg"), highlighted)
        
        # Save warped board
        cv2.imwrite(os.path.join(self.debug_dir, f"{timestamp}_warped.jpg"), warped_board)
        
        # Extract and save individual squares for debugging
        squares = self.extract_squares(warped_board)
        square_grid = np.zeros((8 * 100, 8 * 100, 3), dtype=np.uint8)
        
        for rank in range(8):
            for file in range(8):
                # Resize square to 100x100 if needed
                square = squares[rank][file]
                if square.shape[0] != 100 or square.shape[1] != 100:
                    square = cv2.resize(square, (100, 100))
                    
                # Place in the grid
                square_grid[rank*100:(rank+1)*100, file*100:(file+1)*100] = square
                
                # Add grid lines
                cv2.rectangle(
                    square_grid,
                    (file*100, rank*100),
                    ((file+1)*100 - 1, (rank+1)*100 - 1),
                    (0, 0, 255), 1
                )
                
                # Add coordinates
                cv2.putText(
                    square_grid,
                    f"{chr(97+file)}{8-rank}",
                    (file*100 + 5, rank*100 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
                
        cv2.imwrite(os.path.join(self.debug_dir, f"{timestamp}_squares.jpg"), square_grid)