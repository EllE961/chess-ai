"""
Unit tests for the core chess engine components.

This module contains tests for the neural network, MCTS, and self-play components.
"""

import unittest
import os
import sys
import torch
import chess
import numpy as np
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural_network import ChessNetwork, ResidualBlock
from core.mcts import MCTS, MCTSNode
from core.self_play import SelfPlay
from core.chess_environment import ChessEnvironment


class TestChessEnvironment(unittest.TestCase):
    """Tests for the chess environment module."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = ChessEnvironment()
        
    def test_initialization(self):
        """Test environment initialization."""
        self.assertIsInstance(self.env.board, chess.Board)
        self.assertEqual(self.env.board.fen(), chess.STARTING_FEN)
        
    def test_custom_fen_initialization(self):
        """Test initialization with custom FEN."""
        custom_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        env = ChessEnvironment(custom_fen)
        self.assertEqual(env.board.fen(), custom_fen)
        
    def test_reset(self):
        """Test reset functionality."""
        # Make a move
        self.env.make_move_from_uci("e2e4")
        self.assertNotEqual(self.env.board.fen(), chess.STARTING_FEN)
        
        # Reset
        self.env.reset()
        self.assertEqual(self.env.board.fen(), chess.STARTING_FEN)
        
    def test_make_move(self):
        """Test making moves."""
        # Make a valid move
        move = chess.Move.from_uci("e2e4")
        result = self.env.make_move(move)
        self.assertTrue(result)
        self.assertEqual(self.env.board.piece_at(chess.E4).piece_type, chess.PAWN)
        
        # Try an illegal move
        illegal_move = chess.Move.from_uci("e1e3")
        result = self.env.make_move(illegal_move)
        self.assertFalse(result)
        
    def test_make_move_from_uci(self):
        """Test making moves from UCI strings."""
        result = self.env.make_move_from_uci("e2e4")
        self.assertTrue(result)
        
        # Invalid UCI string
        result = self.env.make_move_from_uci("invalid")
        self.assertFalse(result)
        
    def test_get_legal_moves(self):
        """Test getting legal moves."""
        moves = self.env.get_legal_moves()
        self.assertEqual(len(moves), 20)  # Standard opening position
        
    def test_is_game_over(self):
        """Test game over detection."""
        self.assertFalse(self.env.is_game_over())
        
        # Set up a checkmate position
        checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        env = ChessEnvironment(checkmate_fen)
        self.assertTrue(env.is_game_over())
        
    def test_get_result(self):
        """Test getting the game result."""
        # Standard position (not over)
        self.assertEqual(self.env.get_result(), 0.0)
        
        # Checkmate position
        checkmate_fen = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        env = ChessEnvironment(checkmate_fen)
        self.assertEqual(env.get_result(), -1.0)  # Black wins
        
    def test_encode_board(self):
        """Test board encoding for neural network input."""
        encoded = self.env.encode_board()
        self.assertEqual(encoded.shape, (19, 8, 8))
        
        # Check piece planes
        # White pawns on the second rank
        self.assertEqual(np.sum(encoded[0]), 8)  # 8 white pawns
        
        # Test after a move
        self.env.make_move_from_uci("e2e4")
        encoded = self.env.encode_board()
        # There should be a white pawn at e4
        self.assertEqual(encoded[0][4][4], 1.0)
        
    def test_move_to_index(self):
        """Test conversion from move to policy index."""
        move = chess.Move.from_uci("e2e4")
        idx = self.env.move_to_index(move)
        self.assertIsInstance(idx, int)
        
        # Test that the index is within the expected range
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 64 * 64)  # Simple encoding range


class TestResidualBlock(unittest.TestCase):
    """Tests for the residual block in the neural network."""
    
    def test_initialization(self):
        """Test residual block initialization."""
        channels = 64
        block = ResidualBlock(channels)
        
        # Check components
        self.assertIsInstance(block.conv1, torch.nn.Conv2d)
        self.assertIsInstance(block.bn1, torch.nn.BatchNorm2d)
        self.assertIsInstance(block.conv2, torch.nn.Conv2d)
        self.assertIsInstance(block.bn2, torch.nn.BatchNorm2d)
        
        # Check dimensions
        self.assertEqual(block.conv1.in_channels, channels)
        self.assertEqual(block.conv1.out_channels, channels)
        self.assertEqual(block.conv1.kernel_size, (3, 3))
        
    def test_forward_pass(self):
        """Test forward pass through residual block."""
        channels = 64
        block = ResidualBlock(channels)
        
        # Create dummy input
        x = torch.randn(1, channels, 8, 8)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)


class TestChessNetwork(unittest.TestCase):
    """Tests for the chess neural network."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'neural_network': {
                'input_channels': 19,
                'num_res_blocks': 2,  # Reduced for testing
                'num_filters': 32,    # Reduced for testing
                'value_head_hidden': 32,
                'policy_output_size': 4672
            }
        }
        self.model = ChessNetwork(self.config)
        
    def test_initialization(self):
        """Test network initialization."""
        # Check basic structure
        self.assertIsInstance(self.model.conv_input, torch.nn.Conv2d)
        self.assertEqual(len(self.model.res_blocks), 2)
        
    def test_forward_pass(self):
        """Test forward pass through the network."""
        # Create dummy input
        x = torch.randn(1, 19, 8, 8)
        
        # Forward pass
        policy, value = self.model(x)
        
        # Check output shapes
        self.assertEqual(policy.shape, (1, 4672))
        self.assertEqual(value.shape, (1, 1))
        
        # Check value range
        self.assertTrue(-1.0 <= value.item() <= 1.0)
        
        # Check policy is a probability distribution
        self.assertAlmostEqual(torch.sum(policy).item(), 1.0, places=5)
        
    def test_save_and_load(self):
        """Test saving and loading model checkpoints."""
        # Create a temporary file
        temp_file = "temp_model.pt"
        
        try:
            # Save the model
            self.model.save_checkpoint(temp_file)
            
            # Check that file exists
            self.assertTrue(os.path.exists(temp_file))
            
            # Load the model
            loaded_model, _ = ChessNetwork.load_checkpoint(temp_file)
            
            # Check that it's the same model
            self.assertEqual(type(loaded_model), type(self.model))
            
            # Check that weights are the same
            for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
                
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestMCTSNode(unittest.TestCase):
    """Tests for the MCTS node class."""
    
    def test_initialization(self):
        """Test node initialization."""
        node = MCTSNode()
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(node.prior, 0.0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.move)
        self.assertEqual(len(node.children), 0)
        
    def test_expanded(self):
        """Test expanded check."""
        node = MCTSNode()
        self.assertFalse(node.expanded())
        
        # Add a child
        node.children[chess.Move.from_uci("e2e4")] = MCTSNode()
        self.assertTrue(node.expanded())
        
    def test_value(self):
        """Test value calculation."""
        node = MCTSNode()
        self.assertEqual(node.value(), 0.0)
        
        # Update values
        node.visit_count = 5
        node.value_sum = 3.0
        self.assertEqual(node.value(), 0.6)
        
    def test_get_ucb_score(self):
        """Test UCB score calculation."""
        parent = MCTSNode()
        parent.visit_count = 10
        
        child = MCTSNode(prior=0.5, parent=parent)
        
        # Test UCB score with no visits
        ucb = child.get_ucb_score(parent.visit_count, 1.5)
        self.assertGreater(ucb, 0.0)
        
        # Test after some visits
        child.visit_count = 3
        child.value_sum = 2.0
        ucb = child.get_ucb_score(parent.visit_count, 1.5)
        
        # UCB should balance exploitation (value) with exploration (prior * sqrt(N) / (1+n))
        # For our values: 2/3 + 1.5 * 0.5 * sqrt(10) / (1+3)
        expected_ucb = (2/3) + 1.5 * 0.5 * np.sqrt(10) / 4
        self.assertAlmostEqual(ucb, expected_ucb, places=5)


class TestMCTS(unittest.TestCase):
    """Tests for the MCTS algorithm."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simplified config
        self.config = {
            'neural_network': {
                'input_channels': 19,
                'num_res_blocks': 1,
                'num_filters': 16,
                'value_head_hidden': 16,
                'policy_output_size': 4672
            },
            'mcts': {
                'c_puct': 1.5,
                'num_simulations': 2,  # Very low for testing
                'temperature_init': 1.0,
                'temperature_final': 0.1,
                'move_temp_threshold': 30
            }
        }
        
        # Create a mock model
        self.model = MagicMock()
        self.model.device = torch.device('cpu')
        
        # Set up the model to return a policy and value
        policy = torch.zeros(1, 4672)
        policy[0, 0] = 1.0  # Set a default move
        value = torch.tensor([[0.0]])
        self.model.return_value = (policy, value)
        
        def side_effect(x):
            return self.model.return_value
            
        self.model.side_effect = side_effect
        self.model.__call__ = MagicMock(side_effect=side_effect)
        
        # Create the MCTS instance
        self.mcts = MCTS(self.model, self.config)
        
        # Create a chess environment
        self.env = ChessEnvironment()
        
    def test_initialization(self):
        """Test MCTS initialization."""
        self.assertEqual(self.mcts.c_puct, 1.5)
        self.assertEqual(self.mcts.num_simulations, 2)
        
    @patch('core.mcts.MCTS._get_policy')
    def test_search(self, mock_get_policy):
        """Test MCTS search."""
        # Set up the mock to return a simple policy
        e2e4 = chess.Move.from_uci("e2e4")
        d2d4 = chess.Move.from_uci("d2d4")
        mock_policy = {e2e4: 0.7, d2d4: 0.3}
        mock_get_policy.return_value = mock_policy
        
        # Run search
        policy = self.mcts.search(self.env)
        
        # Check that policy is returned
        self.assertIsInstance(policy, dict)
        self.assertIn(e2e4, policy)
        self.assertIn(d2d4, policy)
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=5)
        
    def test_select_move(self):
        """Test move selection."""
        # Create a mock policy
        e2e4 = chess.Move.from_uci("e2e4")
        d2d4 = chess.Move.from_uci("d2d4")
        policy = {e2e4: 0.7, d2d4: 0.3}
        
        # Mock the search method
        self.mcts.search = MagicMock(return_value=policy)
        
        # Test deterministic selection
        move = self.mcts.select_move(self.env, deterministic=True)
        self.assertEqual(move, e2e4)
        
        # Test probabilistic selection
        # This is non-deterministic, so we just check it returns one of the moves
        move = self.mcts.select_move(self.env, deterministic=False)
        self.assertIn(move, [e2e4, d2d4])


class TestSelfPlay(unittest.TestCase):
    """Tests for the self-play component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simplified config
        self.config = {
            'neural_network': {
                'input_channels': 19,
                'num_res_blocks': 1,
                'num_filters': 16,
                'value_head_hidden': 16,
                'policy_output_size': 4672
            },
            'mcts': {
                'c_puct': 1.5,
                'num_simulations': 2,
                'temperature_init': 1.0,
                'temperature_final': 0.1,
                'move_temp_threshold': 30
            },
            'training': {
                'num_self_play_games': 1
            },
            'data_dir': './data'
        }
        
        # Create a mock model
        self.model = MagicMock()
        self.model.device = torch.device('cpu')
        
        # Create a mock MCTS
        self.mcts = MagicMock()
        
        # Set up MCTS to return moves
        def select_move_side_effect(env, deterministic=False):
            # Just return e2e4 for white, e7e5 for black
            if env.board.turn == chess.WHITE:
                return chess.Move.from_uci("e2e4")
            else:
                return chess.Move.from_uci("e7e5")
                
        self.mcts.select_move.side_effect = select_move_side_effect
        
        # Create policy for MCTS search
        e2e4 = chess.Move.from_uci("e2e4")
        e7e5 = chess.Move.from_uci("e7e5")
        policy = {e2e4: 0.8, chess.Move.from_uci("d2d4"): 0.2}
        policy2 = {e7e5: 0.8, chess.Move.from_uci("c7c5"): 0.2}
        
        def search_side_effect(env):
            if env.board.turn == chess.WHITE:
                return policy
            else:
                return policy2
                
        self.mcts.search.side_effect = search_side_effect
        
        # Create the self-play instance
        self.self_play = SelfPlay(self.model, self.config)
        self.self_play.mcts = self.mcts
        
    @patch('os.makedirs')
    def test_execute_episode(self, mock_makedirs):
        """Test executing a self-play episode."""
        # Execute one episode
        examples, metadata = self.self_play.execute_episode()
        
        # Check that examples were generated
        self.assertGreater(len(examples), 0)
        
        # Check that metadata is correct
        self.assertIn('moves', metadata)
        self.assertIn('result', metadata)
        self.assertEqual(metadata['moves'][0], "e2e4")
        self.assertEqual(metadata['moves'][1], "e7e5")
        
    @patch('os.makedirs')
    def test_generate_games(self, mock_makedirs):
        """Test generating multiple self-play games."""
        # Generate games
        game_data = self.self_play.generate_games()
        
        # Check that game data was generated
        self.assertEqual(len(game_data), 1)  # One game
        
        # Check format of game data
        examples, metadata = game_data[0]
        self.assertGreater(len(examples), 0)
        self.assertIn('moves', metadata)


if __name__ == '__main__':
    unittest.main()