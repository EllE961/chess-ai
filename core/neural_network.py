"""
Neural network implementation for the chess AI.

This module implements a deep residual neural network with two heads (policy and value)
similar to AlphaZero's architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization, as used in AlphaZero.
    
    Contains two convolutional layers with batch normalization and a residual connection.
    """
    
    def __init__(self, channels: int):
        """
        Initialize the residual block.
        
        Args:
            channels: Number of input and output channels.
        """
        super(ResidualBlock, self).__init__()
        
        # First convolution layer with batch normalization
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolution layer with batch normalization
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Output tensor of the same shape.
        """
        residual = x
        
        # First convolution with ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Second convolution
        out = self.bn2(self.conv2(out))
        
        # Add residual connection and apply ReLU
        out += residual
        out = F.relu(out)
        
        return out


class ChessNetwork(nn.Module):
    """
    Neural network architecture for chess, inspired by AlphaZero.
    
    The network takes a board representation as input and outputs:
    1. A policy vector representing move probabilities
    2. A value scalar representing win probability for the current player
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the neural network.
        
        Args:
            config: Configuration dictionary containing network parameters.
        """
        super(ChessNetwork, self).__init__()
        
        # Extract configuration parameters
        self.config = config
        nn_config = config.get('neural_network', {})
        
        self.input_channels = nn_config.get('input_channels', 19)
        self.num_res_blocks = nn_config.get('num_res_blocks', 19)
        self.num_filters = nn_config.get('num_filters', 256)
        self.value_head_hidden = nn_config.get('value_head_hidden', 256)
        self.policy_output_size = nn_config.get('policy_output_size', 4672)
        
        # Input convolutional layer
        self.conv_input = nn.Conv2d(self.input_channels, self.num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(self.num_filters)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.num_filters) for _ in range(self.num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(self.num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, self.policy_output_size)
        
        # Value Head
        self.value_conv = nn.Conv2d(self.num_filters, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, self.value_head_hidden)
        self.value_fc2 = nn.Linear(self.value_head_hidden, 1)
        
        # Initialize weights using He initialization
        self._init_weights()
        
        logger.info(f"Initialized chess network with {self.num_res_blocks} residual blocks")
        
    def _init_weights(self) -> None:
        """Initialize the network weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, 8, 8).
            
        Returns:
            Tuple of (policy_output, value_output):
                - policy_output: Tensor of shape (batch_size, policy_output_size)
                - value_output: Tensor of shape (batch_size, 1)
        """
        # Input layer
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        # Apply softmax to get probabilities
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def save_checkpoint(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: int = 0, iteration: int = 0) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint.
            optimizer: Optional optimizer to save state.
            epoch: Current epoch number.
            iteration: Current iteration number.
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'iteration': iteration
        }
        
        if optimizer:
            checkpoint['optimizer'] = optimizer.state_dict()
            
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        
    @classmethod
    def load_checkpoint(cls, filepath: str, device: Optional[torch.device] = None) -> Tuple['ChessNetwork', Dict[str, Any]]:
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to the checkpoint file.
            device: Device to load the model to (cpu or cuda).
            
        Returns:
            Tuple of (model, checkpoint_data):
                - model: Loaded ChessNetwork instance.
                - checkpoint_data: Dictionary containing additional checkpoint data.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model from saved configuration
        config = checkpoint.get('config', {})
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {filepath}")
        
        # Return model and checkpoint data for further processing
        return model, checkpoint