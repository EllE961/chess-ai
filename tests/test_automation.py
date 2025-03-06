"""
Unit tests for the automation components.

This module contains tests for the move executor, platform adapter,
and calibrator components.
"""

import unittest
import os
import sys
import numpy as np
import chess
import time
from unittest.mock import MagicMock, patch, PropertyMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automation.move_executor import MoveExecutor
from automation.platform_adapter import PlatformAdapter, ChessPlatform
from automation.calibrator import Calibrator


class TestMoveExecutor(unittest.TestCase):
    """Tests for the move executor."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'automation': {
                'mouse_move_delay': 0.01,  # Fast for testing
                'move_execution_delay': 0.01,
                'click_randomness': 5
            }
        }
        self.executor = MoveExecutor(self.config)
        
        # Set board info
        self.board_coords = np.array([
            [100, 100],  # Top-left
            [900, 100],  # Top-right
            [900, 900],  # Bottom-right
            [100, 900]   # Bottom-left
        ], dtype=np.float32)
        
    def test_initialization(self):
        """Test move executor initialization."""
        self.assertEqual(self.executor.mouse_move_delay, 0.01)
        self.assertEqual(self.executor.move_execution_delay, 0.01)
        self.assertEqual(self.executor.click_randomness, 5)
        
    def test_set_board_info(self):
        """Test setting board information."""
        self.executor.set_board_info(self.board_coords, white_perspective=True)
        
        # Check that board info is set
        self.assertTrue(np.array_equal(self.executor.board_coords, self.board_coords))
        self.assertTrue(self.executor.white_perspective)
        
        # Square size should be calculated
        self.assertAlmostEqual(self.executor.square_size, 100.0, places=1)
        
    def test_square_to_screen_coords(self):
        """Test converting chess squares to screen coordinates."""
        # First set board info
        self.executor.set_board_info(self.board_coords, white_perspective=True)
        
        # Test conversion for e2 (white pawn)
        e2 = chess.E2  # 12
        coords = self.executor._square_to_screen_coords(e2)
        
        # E2 is in the 5th file (0-indexed)
        # And in the 2nd rank (from the bottom, 0-indexed)
        # Should be somewhere in the lower half, about middle horizontally
        self.assertTrue(400 < coords[0] < 600)  # Middle horizontally
        self.assertTrue(600 < coords[1] < 800)  # Lower half vertically
        
    def test_square_to_screen_coords_black_perspective(self):
        """Test coordinate conversion from black's perspective."""
        # Set board info with black's perspective
        self.executor.set_board_info(self.board_coords, white_perspective=False)
        
        # Test conversion for e7 (black pawn)
        e7 = chess.E7  # 52
        coords = self.executor._square_to_screen_coords(e7)
        
        # From black's perspective, e7 should be in the lower half
        self.assertTrue(400 < coords[0] < 600)  # Middle horizontally
        self.assertTrue(600 < coords[1] < 800)  # Lower half vertically
        
    def test_add_click_randomness(self):
        """Test adding randomness to click coordinates."""
        original = (500, 500)
        randomized = self.executor._add_click_randomness(original)
        
        # Check that randomized coordinates are within range
        self.assertTrue(original[0] - 5 <= randomized[0] <= original[0] + 5)
        self.assertTrue(original[1] - 5 <= randomized[1] <= original[1] + 5)
        
    @patch('pyautogui.moveTo')
    def test_move_mouse_humanlike(self, mock_move):
        """Test human-like mouse movement."""
        self.executor._move_mouse_humanlike((500, 500), duration=0.01)
        
        # Check that moveTo was called
        mock_move.assert_called_once()
        
    @patch('automation.move_executor.MoveExecutor._move_mouse_humanlike')
    @patch('pyautogui.click')
    def test_execute_mouse_move(self, mock_click, mock_move):
        """Test executing a mouse move."""
        from_coords = (200, 200)
        to_coords = (300, 300)
        
        # Execute the move
        self.executor._execute_mouse_move(from_coords, to_coords)
        
        # Check that move and click were called correctly
        self.assertEqual(mock_move.call_count, 2)
        self.assertEqual(mock_click.call_count, 2)
        
    @patch('automation.move_executor.MoveExecutor._execute_mouse_move')
    def test_execute_move(self, mock_execute):
        """Test executing a chess move."""
        # Set board info
        self.executor.set_board_info(self.board_coords, white_perspective=True)
        
        # Create a chess move
        move = chess.Move.from_uci("e2e4")
        
        # Execute the move
        self.executor.execute_move(move)
        
        # Check that execute_mouse_move was called
        mock_execute.assert_called_once()
        
    @patch('automation.move_executor.MoveExecutor._execute_mouse_move')
    def test_execute_move_from_string(self, mock_execute):
        """Test executing a move from a UCI string."""
        # Set board info
        self.executor.set_board_info(self.board_coords, white_perspective=True)
        
        # Execute the move
        self.executor.execute_move("e2e4")
        
        # Check that execute_mouse_move was called
        mock_execute.assert_called_once()
        
    @patch('automation.move_executor.MoveExecutor._execute_mouse_move')
    @patch('automation.move_executor.MoveExecutor._handle_promotion')
    def test_execute_promotion_move(self, mock_promotion, mock_execute):
        """Test executing a promotion move."""
        # Set board info
        self.executor.set_board_info(self.board_coords, white_perspective=True)
        
        # Create a promotion move
        move = chess.Move.from_uci("a7a8q")  # Promote to queen
        
        # Execute the move
        self.executor.execute_move(move)
        
        # Check that execute_mouse_move and handle_promotion were called
        mock_execute.assert_called_once()
        mock_promotion.assert_called_once_with(chess.QUEEN, mock_execute.return_value[1])
        
    @patch('pyautogui.mouseDown')
    @patch('pyautogui.mouseUp')
    @patch('automation.move_executor.MoveExecutor._move_mouse_humanlike')
    def test_drag_and_drop(self, mock_move, mock_up, mock_down):
        """Test drag and drop functionality."""
        # Set board info
        self.executor.set_board_info(self.board_coords, white_perspective=True)
        
        # Perform drag and drop
        result = self.executor.drag_and_drop(chess.E2, chess.E4)
        
        # Check that mouse functions were called correctly
        self.assertEqual(mock_move.call_count, 2)  # Move to start and end positions
        mock_down.assert_called_once()
        mock_up.assert_called_once()
        self.assertTrue(result)


class TestPlatformAdapter(unittest.TestCase):
    """Tests for the platform adapter."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'automation': {
                'platform': 'lichess',
                'polling_interval': 1.0
            },
            'template_dir': './templates'
        }
        self.adapter = PlatformAdapter(self.config)
        
    def test_initialization(self):
        """Test platform adapter initialization."""
        self.assertEqual(self.adapter.platform, ChessPlatform.LICHESS)
        self.assertFalse(self.adapter.is_playing)
        self.assertFalse(self.adapter.is_our_turn)
        
    def test_enum_values(self):
        """Test chess platform enum values."""
        self.assertEqual(ChessPlatform.LICHESS.value, "lichess")
        self.assertEqual(ChessPlatform.CHESS_COM.value, "chess.com")
        self.assertEqual(ChessPlatform.CHESS24.value, "chess24")
        self.assertEqual(ChessPlatform.CUSTOM.value, "custom")
        
    @patch('automation.platform_adapter.cv2.imread')
    @patch('automation.platform_adapter.cv2.matchTemplate')
    @patch('automation.platform_adapter.cv2.minMaxLoc')
    @patch('automation.platform_adapter.pyautogui.screenshot')
    def test_detect_platform(self, mock_screenshot, mock_minmax, mock_match, mock_imread):
        """Test platform detection."""
        # Set up mocks
        mock_screenshot.return_value = np.zeros((1000, 1000, 3), dtype=np.uint8)
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_match.return_value = np.zeros((900, 900), dtype=np.float32)
        mock_minmax.return_value = (0, 0.9, (0, 0), (0, 0))  # High match value
        
        # Patch os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            # Detect platform
            platform = self.adapter.detect_platform()
            
            # Check results
            self.assertEqual(platform, ChessPlatform.LICHESS)
            
    @patch('automation.platform_adapter.pyautogui.screenshot')
    def test_detect_game_state_generic(self, mock_screenshot):
        """Test generic game state detection."""
        # Set up mock
        mock_screenshot.return_value = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
        # Set previous screen to trigger change detection
        self.adapter.previous_screen = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        
        # Detect game state
        is_playing, is_our_turn = self.adapter._detect_generic_state(np.zeros((1000, 1000, 3), dtype=np.uint8), True)
        
        # With change detected, should indicate it's our turn
        self.assertTrue(is_playing)
        self.assertTrue(is_our_turn)
        
    @patch('automation.platform_adapter.BoardDetector')
    def test_find_board(self, mock_detector):
        """Test finding the chess board."""
        # Set up mock
        mock_board_coords = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        mock_warped = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_detector.detect_board.return_value = (mock_board_coords, mock_warped)
        
        # Draw a dark square in the bottom-left for white's perspective
        mock_warped[80:100, 0:20] = [50, 50, 50]
        
        # Find board
        board_coords, white_perspective = self.adapter.find_board(mock_detector)
        
        # Check results
        self.assertTrue(np.array_equal(board_coords, mock_board_coords))
        self.assertTrue(white_perspective)
        
    def test_determine_board_perspective(self):
        """Test determining board perspective."""
        # Create a test board image (8x8 grid)
        board = np.ones((800, 800, 3), dtype=np.uint8) * 255
        
        # Make bottom-left square dark (a1 from white's perspective)
        board[700:800, 0:100] = [50, 50, 50]
        
        # Make top-right square dark (h8 from white's perspective)
        board[0:100, 700:800] = [50, 50, 50]
        
        # Determine perspective
        perspective = self.adapter._determine_board_perspective(board)
        
        # Should detect white's perspective
        self.assertTrue(perspective)
        
    @patch('automation.platform_adapter.pyautogui.hotkey')
    @patch('automation.platform_adapter.pyautogui.typewrite')
    @patch('automation.platform_adapter.pyautogui.press')
    @patch('automation.platform_adapter.pyautogui.click')
    @patch('automation.platform_adapter.time.sleep')
    def test_start_new_game_lichess(self, mock_sleep, mock_click, mock_press, mock_type, mock_hotkey):
        """Test starting a new game on Lichess."""
        # Start a new game
        result = self.adapter._start_new_game_lichess('white')
        
        # Check that UI interactions were performed
        mock_hotkey.assert_called_once()
        self.assertGreater(mock_type.call_count, 0)
        self.assertGreater(mock_press.call_count, 0)
        self.assertGreater(mock_click.call_count, 0)
        self.assertGreater(mock_sleep.call_count, 0)
        
        # Should return True for success
        self.assertTrue(result)
        
    @patch('automation.platform_adapter.pyautogui.press')
    @patch('automation.platform_adapter.pyautogui.click')
    @patch('automation.platform_adapter.time.sleep')
    def test_resign_game(self, mock_sleep, mock_click, mock_press):
        """Test resigning a game."""
        # Set up game state
        self.adapter.is_playing = True
        
        # Resign the game
        result = self.adapter._resign_lichess()
        
        # Check that UI interactions were performed
        self.assertGreater(mock_press.call_count, 0)
        self.assertGreater(mock_click.call_count, 0)
        self.assertGreater(mock_sleep.call_count, 0)
        
        # Should return True for success and reset game state
        self.assertTrue(result)
        self.assertFalse(self.adapter.is_playing)
        self.assertFalse(self.adapter.is_our_turn)


class TestCalibrator(unittest.TestCase):
    """Tests for the calibrator."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'automation': {
                'calibration_corners': 1  # Just one attempt for testing
            },
            'log_dir': './logs'
        }
        self.calibrator = Calibrator(self.config)
        
    def test_initialization(self):
        """Test calibrator initialization."""
        self.assertEqual(self.calibrator.num_calibration_points, 1)
        
    def test_detect_perspective(self):
        """Test perspective detection."""
        # Create a test board (8x8 grid)
        board = np.ones((800, 800, 3), dtype=np.uint8) * 255
        
        # Make bottom-left square dark (a1 from white's perspective)
        board[700:800, 0:100] = [50, 50, 50]
        
        # Make top-right square dark (h8 from white's perspective)
        board[0:100, 700:800] = [50, 50, 50]
        
        # Detect perspective
        perspective = self.calibrator._detect_perspective(board)
        
        # Should detect white's perspective
        self.assertTrue(perspective)
        
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.show')
    @patch('cv2.imwrite')
    def test_test_square_mapping(self, mock_imwrite, mock_show, mock_imshow, mock_figure):
        """Test square mapping visualization."""
        # Set board coordinates
        self.calibrator.board_coords = np.array([
            [100, 100], [700, 100], [700, 700], [100, 700]
        ], dtype=np.float32)
        
        # Enable debug mode
        debug_config = self.config.copy()
        debug_config['system'] = {'debug_mode': True}
        calibrator = Calibrator(debug_config)
        calibrator.board_coords = self.calibrator.board_coords
        
        # Test with mock screenshot
        with patch('numpy.array', return_value=np.zeros((1000, 1000, 3), dtype=np.uint8)):
            calibrator._test_square_mapping()
            
        # Check that visualization functions were called
        mock_figure.assert_called_once()
        mock_imshow.assert_called_once()
        mock_show.assert_called_once()
        mock_imwrite.assert_called_once()
        
    @patch('automation.calibrator.Calibrator._test_square_mapping')
    @patch('automation.calibrator.Calibrator._detect_perspective')
    @patch('automation.calibrator.BoardDetector')
    @patch('automation.calibrator.time.sleep')
    @patch('numpy.array')
    def test_calibrate(self, mock_array, mock_sleep, mock_detector, mock_perspective, mock_test):
        """Test calibration process."""
        # Set up mocks
        mock_array.return_value = np.zeros((1000, 1000, 3), dtype=np.uint8)
        mock_board_coords = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        mock_warped = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock the board detector
        detector_instance = mock_detector.return_value
        detector_instance.detect_board.return_value = (mock_board_coords, mock_warped)
        
        # Set perspective result
        mock_perspective.return_value = True
        
        # Run calibration
        board_coords, white_perspective = self.calibrator.calibrate(detector_instance)
        
        # Check results
        self.assertTrue(np.array_equal(board_coords, mock_board_coords))
        self.assertTrue(white_perspective)
        
        # Check that methods were called
        detector_instance.detect_board.assert_called_once()
        mock_perspective.assert_called_once()
        mock_test.assert_called_once()
        
    @patch('automation.calibrator.Calibrator._test_square_mapping')
    @patch('automation.calibrator.BoardDetector')
    def test_interactive_calibration(self, mock_detector, mock_test):
        """Test interactive calibration."""
        # Set up mocks
        mock_board_coords = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        mock_warped = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock the board detector
        detector_instance = mock_detector.return_value
        detector_instance.detect_board.return_value = (mock_board_coords, mock_warped)
        detector_instance.highlight_board.return_value = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
        # Run interactive calibration
        with patch('cv2.imwrite'):
            board_coords, white_perspective = self.calibrator.interactive_calibration()
        
        # Check results
        self.assertTrue(np.array_equal(board_coords, mock_board_coords))
        self.assertIsNotNone(white_perspective)
        
        # Check that methods were called
        detector_instance.detect_board.assert_called_once()
        mock_test.assert_called_once()


if __name__ == '__main__':
    unittest.main()