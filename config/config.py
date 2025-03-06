"""
Configuration management for the Chess AI system.

This module handles loading, validation, and access to configuration parameters
used throughout the system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

# Create necessary directories
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, TEMPLATE_DIR]:
    os.makedirs(directory, exist_ok=True)
    
# Create subdirectories for organization
os.makedirs(os.path.join(DATA_DIR, 'game_records'), exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'training_logs'), exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'game_logs'), exist_ok=True)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file. If None, uses default hyperparameters.yaml.
        
    Returns:
        Dict containing configuration parameters.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the configuration file is not valid YAML.
    """
    if config_path is None:
        config_path = os.path.join(CONFIG_DIR, 'hyperparameters.yaml')
        
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_params = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise
        
    # Add directory paths to config
    config_params.update({
        'base_dir': BASE_DIR,
        'config_dir': CONFIG_DIR,
        'data_dir': DATA_DIR,
        'model_dir': MODEL_DIR,
        'log_dir': LOG_DIR,
        'template_dir': TEMPLATE_DIR
    })
    
    return config_params

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        True if configuration is valid, False otherwise.
    """
    # Required parameters for neural network
    required_nn_params = [
        'input_channels', 'num_res_blocks', 'num_filters'
    ]
    
    # Required parameters for MCTS
    required_mcts_params = [
        'c_puct', 'num_simulations'
    ]
    
    # Required parameters for training
    required_training_params = [
        'batch_size', 'learning_rate', 'weight_decay'
    ]
    
    # Check for required parameters
    for param in required_nn_params + required_mcts_params + required_training_params:
        if param not in config:
            logger.error(f"Missing required configuration parameter: {param}")
            return False
            
    return True

# Load the default configuration
CONFIG = load_config()

# Validate the configuration
if not validate_config(CONFIG):
    logger.warning("Configuration validation failed, using defaults")
    # You might want to set default values here or raise an exception