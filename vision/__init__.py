"""
Computer Vision Package.

This package contains components for detecting a chess board on screen,
classifying pieces, and extracting the position information.
"""

from .board_detector import BoardDetector
from .piece_classifier import PieceClassifier
from .position_extractor import PositionExtractor

__all__ = ['BoardDetector', 'PieceClassifier', 'PositionExtractor']