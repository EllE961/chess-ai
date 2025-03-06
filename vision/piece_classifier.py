"""
Chess piece classification using computer vision and deep learning.

This module provides functionality to classify chess pieces from square images
using a convolutional neural network.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

class PieceClassifierCNN(nn.Module):
    """
    Convolutional neural network for classifying chess pieces.
    
    Takes a square image as input and outputs the piece type and color,
    or empty if no piece is present.
    """
    
    def __init__(self, num_classes: int = 13):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of classes to predict (12 pieces + empty)
        """
        super(PieceClassifierCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third convolutional block
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PieceClassifier:
    """
    Classifies chess pieces from square images.
    
    Uses a pre-trained convolutional neural network to identify the
    piece type and color, or determine if a square is empty.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the piece classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        vision_config = config.get('vision', {})
        
        # Configuration parameters
        self.min_confidence = vision_config.get('min_piece_confidence', 0.7)
        self.input_size = vision_config.get('piece_classifier_size', 64)
        
        # Class information
        self.classes = vision_config.get('piece_classes', [
            "empty",
            "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
            "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"
        ])
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  config.get('system', {}).get('use_gpu', True) else 'cpu')
        
        # Load the model
        self.model = self._load_model()
        
        logger.info(f"Piece classifier initialized with {len(self.classes)} classes")
        
    def _load_model(self) -> nn.Module:
        """
        Load the piece classification model.
        
        Returns:
            PyTorch model for piece classification
        """
        model_path = os.path.join(self.config.get('model_dir', './models'), 'piece_classifier.pt')
        model = PieceClassifierCNN(num_classes=len(self.classes))
        
        # Check if the model file exists
        if os.path.exists(model_path):
            try:
                # Load the model weights
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded piece classifier model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.warning("Using untrained model. Piece classification may be inaccurate.")
        else:
            logger.warning(f"No model found at {model_path}. Using untrained model.")
            logger.warning("Piece classification will be inaccurate. Train the model first.")
            
        # Set to evaluation mode and move to the correct device
        model.eval()
        model.to(self.device)
        
        return model
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for the classifier.
        
        Args:
            image: Square image of a chess piece
            
        Returns:
            Preprocessed tensor ready for the model
        """
        # Resize to the expected input size
        if image.shape[0] != self.input_size or image.shape[1] != self.input_size:
            image = cv2.resize(image, (self.input_size, self.input_size))
            
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Transpose from (H, W, C) to (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(image, dtype=torch.float32)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    def classify_square(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single square image.
        
        Args:
            image: Square image of a chess square
            
        Returns:
            Tuple of (class_name, confidence)
        """
        # Preprocess the image
        tensor = self.preprocess_image(image)
        tensor = tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            
        # Get the most likely class
        confidence, class_idx = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        class_idx = class_idx.item()
        
        # Get the class name
        class_name = self.classes[class_idx]
        
        # If confidence is too low, default to empty
        if confidence < self.min_confidence:
            logger.debug(f"Low confidence ({confidence:.2f}) for class {class_name}, defaulting to empty")
            return "empty", confidence
            
        return class_name, confidence
        
    def classify_board(self, squares: List[List[np.ndarray]]) -> List[List[Tuple[str, float]]]:
        """
        Classify all squares on a chess board.
        
        Args:
            squares: 2D list of square images
            
        Returns:
            2D list of (class_name, confidence) tuples
        """
        result = []
        
        for rank_squares in squares:
            rank_result = []
            for square in rank_squares:
                class_name, confidence = self.classify_square(square)
                rank_result.append((class_name, confidence))
            result.append(rank_result)
            
        return result
        
    def batch_classify(self, squares: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Classify a batch of square images.
        
        This is more efficient than classifying one square at a time.
        
        Args:
            squares: List of square images
            
        Returns:
            List of (class_name, confidence) tuples
        """
        # Preprocess all images
        tensors = []
        for square in squares:
            tensor = self.preprocess_image(square)
            tensors.append(tensor)
            
        # Stack into a batch
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = F.softmax(outputs, dim=1)
            
        # Get the most likely classes
        confidences, class_indices = torch.max(probabilities, dim=1)
        
        # Convert to (class_name, confidence) tuples
        results = []
        for idx, (confidence, class_idx) in enumerate(zip(confidences, class_indices)):
            confidence = confidence.item()
            class_idx = class_idx.item()
            class_name = self.classes[class_idx]
            
            # If confidence is too low, default to empty
            if confidence < self.min_confidence:
                results.append(("empty", confidence))
            else:
                results.append((class_name, confidence))
                
        return results
        
    def train(self, train_data: List[Tuple[np.ndarray, int]], val_data: Optional[List[Tuple[np.ndarray, int]]] = None,
             num_epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train the piece classifier on a dataset of labeled squares.
        
        Args:
            train_data: List of (image, label_index) tuples for training
            val_data: Optional list of (image, label_index) tuples for validation
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Dictionary with training metrics
        """
        # For training, we need a fresh model
        model = PieceClassifierCNN(num_classes=len(self.classes))
        model.to(self.device)
        model.train()
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Create data loaders
        class SquareDataset(Dataset):
            def __init__(self, data, transform_fn):
                self.data = data
                self.transform_fn = transform_fn
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                image, label = self.data[idx]
                tensor = self.transform_fn(image)
                return tensor, label
                
        train_dataset = SquareDataset(train_data, lambda x: self.preprocess_image(x).squeeze(0))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data:
            val_dataset = SquareDataset(val_data, lambda x: self.preprocess_image(x).squeeze(0))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        
        logger.info(f"Starting training for {num_epochs} epochs with {len(train_data)} samples")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            
            # Validation phase
            if val_data:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                        
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                metrics['val_loss'].append(val_loss)
                metrics['val_acc'].append(val_acc)
                
                # Save the best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = os.path.join(self.config.get('model_dir', './models'), 'piece_classifier.pt')
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved improved model with validation accuracy {val_acc:.4f}")
                    
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                # If no validation data, save the model after each epoch
                model_path = os.path.join(self.config.get('model_dir', './models'), 'piece_classifier.pt')
                torch.save(model.state_dict(), model_path)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                
        # Update the model with the trained one
        self.model = model
        
        return metrics