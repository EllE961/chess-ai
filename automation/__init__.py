"""
Automation Package.

This package contains components for automating chess play on digital platforms,
including mouse control and chess platform integration.
"""

from .move_executor import MoveExecutor
from .platform_adapter import PlatformAdapter, ChessPlatform
from .calibrator import Calibrator

__all__ = ['MoveExecutor', 'PlatformAdapter', 'ChessPlatform', 'Calibrator']