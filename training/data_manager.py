"""
Data management for training the chess AI neural network.

This module provides data structures for storing, sampling, and managing
training examples generated from self-play games.
"""

import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional, Iterator

# Define the training example type
TrainingExample = Tuple[np.ndarray, np.ndarray, float]

logger = logging.getLogger(__name__)

class ChessDataset(Dataset):
    """
    Dataset for storing and retrieving chess training examples.
    
    Implements the PyTorch Dataset interface for compatibility with DataLoader.
    """
    
    def __init__(self, examples: List[TrainingExample]):
        """
        Initialize the dataset with training examples.
        
        Args:
            examples: List of training examples, each containing:
                - Board state (np.ndarray)
                - Policy vector (np.ndarray)
                - Value target (float)
        """
        self.states = []
        self.policies = []
        self.values = []
        
        for state, policy, value in examples:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)
            
        logger.debug(f"Created ChessDataset with {len(examples)} examples")
        
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.states)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training example by index.
        
        Args:
            idx: Index of the example to retrieve.
            
        Returns:
            Tuple of (state, policy, value) as PyTorch tensors.
        """
        # Convert to PyTorch tensors
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor([self.values[idx]], dtype=torch.float32)
        
        return state, policy, value


class ReplayBuffer:
    """
    Stores and manages training examples from self-play games.
    
    Implements a first-in-first-out (FIFO) buffer of fixed maximum size to
    maintain a window of recent high-quality training examples.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the replay buffer.
        
        Args:
            config: Configuration dictionary containing buffer parameters.
        """
        training_config = config.get('training', {})
        self.max_size = training_config.get('replay_buffer_size', 500000)
        self.batch_size = training_config.get('batch_size', 2048)
        
        # Buffer and metadata
        self.buffer: List[TrainingExample] = []
        self.position_count = 0
        self.game_count = 0
        
        # Path for saving/loading
        self.data_dir = config.get('data_dir', './data')
        self.buffer_path = os.path.join(self.data_dir, 'replay_buffer.pt')
        
        logger.info(f"Initialized replay buffer with max size {self.max_size}")
        
    def add_game_data(self, examples: List[TrainingExample]) -> None:
        """
        Add training examples from a completed game.
        
        Args:
            examples: List of training examples from a self-play game.
        """
        # Add all examples to the buffer
        self.buffer.extend(examples)
        
        # If buffer exceeds max size, remove oldest examples
        if len(self.buffer) > self.max_size:
            removed = len(self.buffer) - self.max_size
            self.buffer = self.buffer[-self.max_size:]
            logger.debug(f"Removed {removed} oldest examples from buffer")
            
        self.position_count += len(examples)
        self.game_count += 1
        
        logger.debug(f"Added {len(examples)} examples to replay buffer")
        
    def sample_batch(self, batch_size: Optional[int] = None) -> List[TrainingExample]:
        """
        Sample a random batch of examples.
        
        Args:
            batch_size: Number of examples to sample. If None, uses the configured batch size.
            
        Returns:
            List of sampled training examples.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Ensure we don't try to sample more than we have
        batch_size = min(batch_size, len(self.buffer))
        
        if batch_size == 0:
            logger.warning("Attempted to sample from empty buffer")
            return []
            
        # Sample random indices without replacement
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        # Return the samples
        return [self.buffer[i] for i in indices]
        
    def get_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = True, 
                      num_workers: int = 0) -> DataLoader:
        """
        Create a DataLoader for training.
        
        Args:
            batch_size: Batch size for the DataLoader. If None, uses the configured batch size.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes for data loading.
            
        Returns:
            PyTorch DataLoader for the buffer data.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        dataset = ChessDataset(self.buffer)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the replay buffer to disk.
        
        Args:
            path: Path to save the buffer. If None, uses the default path.
        """
        if path is None:
            path = self.buffer_path
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the buffer
        try:
            torch.save({
                'buffer': self.buffer,
                'position_count': self.position_count,
                'game_count': self.game_count
            }, path)
            logger.info(f"Replay buffer saved to {path}")
        except Exception as e:
            logger.error(f"Error saving replay buffer: {e}")
            
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load the replay buffer from disk.
        
        Args:
            path: Path to load the buffer from. If None, uses the default path.
            
        Returns:
            True if successful, False otherwise.
        """
        if path is None:
            path = self.buffer_path
            
        if not os.path.exists(path):
            logger.info(f"No replay buffer found at {path}")
            return False
            
        try:
            data = torch.load(path)
            self.buffer = data['buffer']
            self.position_count = data.get('position_count', len(self.buffer))
            self.game_count = data.get('game_count', 0)
            
            logger.info(f"Loaded replay buffer with {len(self.buffer)} examples")
            return True
        except Exception as e:
            logger.error(f"Error loading replay buffer: {e}")
            return False
            
    def __len__(self) -> int:
        """Get the current number of examples in the buffer."""
        return len(self.buffer)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the replay buffer.
        
        Returns:
            Dictionary with buffer statistics.
        """
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'utilization': len(self.buffer) / self.max_size if self.max_size > 0 else 0,
            'position_count': self.position_count,
            'game_count': self.game_count
        }