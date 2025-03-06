"""
Logging utilities for the chess AI system.

This module provides a consistent logging setup across the application.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional

def setup_logger(config: Dict[str, Any], log_name: Optional[str] = None) -> logging.Logger:
    """
    Set up and configure the logger.
    
    Args:
        config: Configuration dictionary
        log_name: Optional name for the log file (if None, uses timestamp)
        
    Returns:
        Configured logger instance
    """
    # Get log level from config
    system_config = config.get('system', {})
    log_level_str = system_config.get('log_level', 'INFO')
    
    # Map string to log level
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Configure file handler
    log_dir = config.get('log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Get or create log file name
    if log_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_name = f"chess_ai_{timestamp}.log"
        
    log_path = os.path.join(log_dir, log_name)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level_str}")
    logger.info(f"Log file: {log_path}")
    
    return root_logger

def log_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Log the configuration parameters.
    
    Args:
        config: Configuration dictionary
        logger: Logger to use (if None, uses the module logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info("Configuration parameters:")
    
    # Log main sections
    for section, params in config.items():
        if isinstance(params, dict):
            logger.info(f"  {section}:")
            for key, value in params.items():
                # Don't log large nested objects
                if isinstance(value, dict) and len(value) > 10:
                    logger.info(f"    {key}: <dict with {len(value)} items>")
                elif isinstance(value, list) and len(value) > 10:
                    logger.info(f"    {key}: <list with {len(value)} items>")
                else:
                    logger.info(f"    {key}: {value}")
        else:
            logger.info(f"  {section}: {params}")

def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log system information.
    
    Args:
        logger: Logger to use (if None, uses the module logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    import platform
    import torch
    import numpy as np
    
    logger.info("System information:")
    logger.info(f"  Python version: {platform.python_version()}")
    logger.info(f"  System: {platform.system()} {platform.release()}")
    logger.info(f"  Machine: {platform.machine()}")
    logger.info(f"  Processor: {platform.processor()}")
    
    # PyTorch information
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        
    # NumPy information
    logger.info(f"  NumPy version: {np.__version__}")
    
    # OpenCV information
    try:
        import cv2
        logger.info(f"  OpenCV version: {cv2.__version__}")
    except ImportError:
        logger.info("  OpenCV not available")

def log_exception(e: Exception, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an exception with traceback.
    
    Args:
        e: Exception to log
        logger: Logger to use (if None, uses the module logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    import traceback
    logger.error(f"Exception: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log messages.
    
    This is useful for adding information such as game ID or iteration number
    to all log messages from a particular component.
    """
    
    def __init__(self, logger: logging.Logger, prefix: str):
        """
        Initialize the logger adapter.
        
        Args:
            logger: Logger to adapt
            prefix: Prefix to add to all messages
        """
        super().__init__(logger, {})
        self.prefix = prefix
        
    def process(self, msg, kwargs):
        """
        Process the log message by adding the prefix.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
            
        Returns:
            Tuple of (modified_message, kwargs)
        """
        return f"[{self.prefix}] {msg}", kwargs