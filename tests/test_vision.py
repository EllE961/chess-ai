"""
Unit tests for the vision system components.

This module contains tests for the board detector, piece classifier,
and position extractor components.
"""

import unittest
import os
import sys
import numpy as np
import cv2
import chess
import torch
from unittest.mock import MagicMock, patch, PropertyMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.board_detector import BoardDetector
from vision.piece_classifier import PieceClassifier, PieceClassifierCNN
from vision.position_extractor import PositionExtractor


class TestBoardDetector(unittest.TestCase):
    """Tests for the chess board detector."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'vision': {
                'board_detection_threshold': 0.8,
                'detection_interval': 0.5,
                'square_size': 100
            },
            'log_dir': './logs'
        }
        self.detector = BoardDetector(self.config)
        
        # Create a test image with a chessboard
        self.size = 800
        self.test_image = self._create_test_board()
        
    def _create_test_board(self):
        """Create a simple chessboard image for testing."""
        # Create a blank image
        img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        
        # Draw a chess board
        square_size = self.size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    # Draw dark squares
                    x1, y1 = j * square_size, i * square_size
                    x2, y2 = (j + 1) * square_size, (i + 1) * square_size
                    img[y1:y2, x1:x2] = [50, 50, 50]  # Dark gray
                    
        return img
        
    def test_initialization(self):
        """Test board detector initialization."""
        self.assertEqual(self.detector.detection_threshold, 0.8)
        self.assertEqual(self.detector.square_size, 100)
        self.assertEqual(self.detector.detection_interval, 0.5)
        
    @patch('vision.board_detector.ImageGrab')
    def test_capture_screen(self, mock_grab):
        """Test screen capture."""
        # Set up mock to return our test image
        mock_grab.grab.return_value = self.test_image
        
        # Capture screen
        screen = self.detector.capture_screen()
        
        # Check results
        self.assertIsInstance(screen, np.ndarray)
        self.assertEqual(screen.shape, self.test_image.shape)
        
    def test_find_board_contour(self):
        """Test finding board contour."""
        # Create a simple binary image with a square
        binary = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.rectangle(binary, (100, 100), (700, 700), 255, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Call the method
        contour = self.detector._find_board_contour(contours, self.test_image.shape)
        
        # Check results
        self.assertIsNotNone(contour)
        self.assertEqual(len(contour), 4)  # Four corners
        
    def test_get_corner_coordinates(self):
        """Test extracting corner coordinates."""
        # Create a simple square contour
        contour = np.array([
            [[100, 100]],  # Top-left
            [[700, 100]],  # Top-right
            [[700, 700]],  # Bottom-right
            [[100, 700]]   # Bottom-left
        ], dtype=np.int32)
        
        # Get corners
        corners = self.detector._get_corner_coordinates(contour)
        
        # Check results
        self.assertEqual(corners.shape, (4, 2))
        self.assertTrue(np.array_equal(corners[0], [100, 100]))  # Top-left
        self.assertTrue(np.array_equal(corners[1], [700, 100]))  # Top-right
        self.assertTrue(np.array_equal(corners[2], [700, 700]))  # Bottom-right
        self.assertTrue(np.array_equal(corners[3], [100, 700]))  # Bottom-left
        
    def test_warp_perspective(self):
        """Test perspective warping."""
        # Create corner coordinates
        corners = np.array([
            [100, 100],  # Top-left
            [700, 100],  # Top-right
            [700, 700],  # Bottom-right
            [100, 700]   # Bottom-left
        ], dtype=np.float32)
        
        # Warp perspective
        warped = self.detector._warp_perspective(self.test_image, corners)
        
        # Check results
        self.assertEqual(warped.shape, (800, 800, 3))  # Square output
        
    def test_extract_squares(self):
        """Test extracting individual squares."""
        # Create a simple 4x4 board for testing
        board = np.zeros((400, 400, 3), dtype=np.uint8)
        square_size = 100
        
        # Draw unique value in each square for identification
        for i in range(4):
            for j in range(4):
                value = i * 4 + j + 1
                board[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = value
                
        # Extract squares
        squares = self.detector.extract_squares(board)
        
        # Check results
        self.assertEqual(len(squares), 8)
        self.assertEqual(len(squares[0]), 8)
        
        # Check that squares have the correct size
        self.assertEqual(squares[0][0].shape, (50, 50, 3))  # 400/8 = 50
        
    @patch('vision.board_detector.BoardDetector.detect_board')
    @patch('vision.board_detector.BoardDetector.highlight_board')
    @patch('cv2.imwrite')
    def test_save_debug_image(self, mock_imwrite, mock_highlight, mock_detect):
        """Test saving debug images."""
        # Set up mocks
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        warped = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_detect.return_value = (corners, warped)
        mock_highlight.return_value = self.test_image
        
        # Enable debug mode
        debug_config = self.config.copy()
        debug_config['system'] = {'debug_mode': True}
        detector = BoardDetector(debug_config)
        
        # Save debug image
        detector._save_debug_image(self.test_image, corners, warped)
        
        # Check that imwrite was called
        self.assertTrue(mock_imwrite.called)
        

class TestPieceClassifierCNN(unittest.TestCase):
    """Tests for the piece classifier CNN."""
    
    def test_initialization(self):
        """Test CNN initialization."""
        model = PieceClassifierCNN(num_classes=13)
        
        # Check architecture
        self.assertIsInstance(model.conv1, torch.nn.Conv2d)
        self.assertIsInstance(model.conv2, torch.nn.Conv2d)
        self.assertIsInstance(model.conv3, torch.nn.Conv2d)
        self.assertIsInstance(model.fc1, torch.nn.Linear)
        self.assertIsInstance(model.fc2, torch.nn.Linear)
        
    def test_forward_pass(self):
        """Test forward pass through the CNN."""
        model = PieceClassifierCNN(num_classes=13)
        
        # Create dummy input
        x = torch.randn(1, 3, 64, 64)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 13))


class TestPieceClassifier(unittest.TestCase):
    """Tests for the piece classifier."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'vision': {
                'min_piece_confidence': 0.7,
                'piece_classifier_size': 64,
                'piece_classes': [
                    "empty",
                    "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
                    "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"
                ]
            },
            'system': {
                'use_gpu': False
            },
            'model_dir': './models'
        }
        
        # Create a mock model
        mock_model = MagicMock()
        
        # Set up forward pass to return a dummy prediction
        def forward(x):
            # Return a vector with high confidence for "white_pawn"
            dummy = torch.zeros(1, 13)
            dummy[0, 1] = 10.0  # High logit for white_pawn
            return dummy
            
        mock_model.forward = forward
        mock_model.__call__ = forward
        
        # Create the classifier with the mock model
        with patch('vision.piece_classifier.PieceClassifierCNN') as mock_cnn:
            mock_cnn.return_value = mock_model
            self.classifier = PieceClassifier(self.config)
            self.classifier.model = mock_model
        
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertEqual(self.classifier.min_confidence, 0.7)
        self.assertEqual(self.classifier.input_size, 64)
        self.assertEqual(len(self.classifier.classes), 13)
        
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a dummy image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Preprocess
        tensor = self.classifier.preprocess_image(img)
        
        # Check results
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 64, 64))  # Batch, Channels, Height, Width
        self.assertTrue(0 <= tensor.min() <= tensor.max() <= 1)  # Normalized
        
    def test_classify_square(self):
        """Test square classification."""
        # Create a dummy image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Classify
        piece_class, confidence = self.classifier.classify_square(img)
        
        # Check results
        self.assertEqual(piece_class, "white_pawn")
        self.assertGreater(confidence, 0.7)
        
    def test_classify_board(self):
        """Test board classification."""
        # Create a dummy board (2x2 for simplicity)
        squares = [
            [np.ones((100, 100, 3), dtype=np.uint8) * 255, np.ones((100, 100, 3), dtype=np.uint8) * 255],
            [np.ones((100, 100, 3), dtype=np.uint8) * 255, np.ones((100, 100, 3), dtype=np.uint8) * 255]
        ]
        
        # Classify
        result = self.classifier.classify_board(squares)
        
        # Check results
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(result[0][0][0], "white_pawn")
        
    def test_batch_classify(self):
        """Test batch classification."""
        # Create a list of dummy images
        squares = [
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
            np.ones((100, 100, 3), dtype=np.uint8) * 255
        ]
        
        # Classify
        result = self.classifier.batch_classify(squares)
        
        # Check results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "white_pawn")
        self.assertEqual(result[1][0], "white_pawn")


class TestPositionExtractor(unittest.TestCase):
    """Tests for the position extractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = PositionExtractor()
        
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsInstance(self.extractor.piece_to_fen, dict)
        self.assertEqual(self.extractor.piece_to_fen["white_king"], "K")
        self.assertEqual(self.extractor.piece_to_fen["black_pawn"], "p")
        
    def test_board_state_to_fen(self):
        """Test conversion from board state to FEN."""
        # Create a simple board state
        board_state = [
            [("white_rook", 0.9), ("white_knight", 0.9), ("white_bishop", 0.9), ("white_queen", 0.9),
             ("white_king", 0.9), ("white_bishop", 0.9), ("white_knight", 0.9), ("white_rook", 0.9)],
            [("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9),
             ("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9),
             ("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9)],
            [("black_rook", 0.9), ("black_knight", 0.9), ("black_bishop", 0.9), ("black_queen", 0.9),
             ("black_king", 0.9), ("black_bishop", 0.9), ("black_knight", 0.9), ("black_rook", 0.9)]
        ]
        
        # Convert to FEN
        fen = self.extractor.board_state_to_fen(board_state)
        
        # Check results (starting position)
        self.assertIn("RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr", fen)
        
    def test_flip_board(self):
        """Test board flipping."""
        # Create a simple board state
        board_state = [
            [("white_king", 0.9), ("empty", 0.9)],
            [("empty", 0.9), ("black_king", 0.9)]
        ]
        
        # Flip the board
        flipped = self.extractor._flip_board(board_state)
        
        # Check results
        self.assertEqual(flipped[0][0], ("black_king", 0.9))
        self.assertEqual(flipped[0][1], ("empty", 0.9))
        self.assertEqual(flipped[1][0], ("empty", 0.9))
        self.assertEqual(flipped[1][1], ("white_king", 0.9))
        
    def test_validate_fen(self):
        """Test FEN validation."""
        # Valid FEN
        valid_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertTrue(self.extractor.validate_fen(valid_fen))
        
        # Invalid FEN
        invalid_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # Missing components
        self.assertFalse(self.extractor.validate_fen(invalid_fen))
        
    def test_extract_position(self):
        """Test position extraction."""
        # Create a simple board state (starting position)
        board_state = [
            [("white_rook", 0.9), ("white_knight", 0.9), ("white_bishop", 0.9), ("white_queen", 0.9),
             ("white_king", 0.9), ("white_bishop", 0.9), ("white_knight", 0.9), ("white_rook", 0.9)],
            [("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9),
             ("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9), ("white_pawn", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9),
             ("empty", 0.9), ("empty", 0.9), ("empty", 0.9), ("empty", 0.9)],
            [("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9),
             ("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9), ("black_pawn", 0.9)],
            [("black_rook", 0.9), ("black_knight", 0.9), ("black_bishop", 0.9), ("black_queen", 0.9),
             ("black_king", 0.9), ("black_bishop", 0.9), ("black_knight", 0.9), ("black_rook", 0.9)]
        ]
        
        # Extract position
        board = self.extractor.extract_position(board_state, True)
        
        # Check results
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        
    def test_get_most_similar_position(self):
        """Test finding the most similar position."""
        # Create a slightly incorrect FEN (missing a pawn)
        incorrect_fen = "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Create a correct previous board
        previous_board = chess.Board()
        
        # Find the most similar position
        board = self.extractor.get_most_similar_position(incorrect_fen, previous_board)
        
        # Check results
        self.assertIsInstance(board, chess.Board)
        
    def test_expand_fen(self):
        """Test FEN expansion."""
        # FEN with empty squares represented by numbers
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        
        # Expand
        expanded = self.extractor._expand_fen(fen)
        
        # Check results
        self.assertEqual(len(expanded), 64)  # 64 squares
        self.assertNotIn("8", expanded)  # No numbers
        self.assertIn("r", expanded)  # Pieces still present
        
    def test_board_difference(self):
        """Test calculating board differences."""
        # Two similar FENs (e2e4 move)
        fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        
        # Calculate difference
        diff = self.extractor._board_difference(fen1, fen2)
        
        # Check results
        self.assertEqual(diff, 2)  # Pawn moved from e2 to e4


if __name__ == '__main__':
    unittest.main()