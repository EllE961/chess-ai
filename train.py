"""
Training script for the chess AI neural network.

This script provides a standalone entry point for training the neural network
through self-play reinforcement learning.
"""

import os
import sys
import argparse
import logging
import torch
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import load_config
from core.neural_network import ChessNetwork
from training.train_network import Trainer
from utils.logger import setup_logger, log_system_info, log_config
from utils.visualization import plot_training_metrics

def train_model(config_path, resume_iteration=0, num_iterations=None):
    """
    Train the chess AI model.
    
    Args:
        config_path: Path to the configuration file
        resume_iteration: Iteration number to resume from
        num_iterations: Number of iterations to train for (overrides config)
        
    Returns:
        Trained model
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override number of iterations if specified
    if num_iterations is not None:
        config['training']['num_iterations'] = num_iterations
        
    # Set up logging
    logger = setup_logger(config, "training")
    log_system_info()
    log_config(config)
    
    # Set up device
    use_gpu = config.get('system', {}).get('use_gpu', True)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Create model directory if it doesn't exist
    model_dir = config.get('model_dir', './models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize or load model
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    if resume_iteration > 0 and os.path.exists(best_model_path):
        logger.info(f"Loading model from {best_model_path} to resume training")
        model, checkpoint = ChessNetwork.load_checkpoint(best_model_path, device)
        logger.info(f"Loaded model from iteration {checkpoint.get('iteration', 0)}")
    else:
        logger.info("Initializing new model")
        model = ChessNetwork(config)
        model.to(device)
        
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Start timing
    start_time = time.time()
    
    # Run training
    logger.info(f"Starting training for {config['training']['num_iterations']} iterations")
    trainer.train(resume_iteration)
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Plot training metrics
    try:
        metrics = trainer.get_metrics()
        plot_dir = os.path.join(config.get('log_dir', './logs'), 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_training_metrics(metrics, output_dir=plot_dir)
        logger.info(f"Training metrics plots saved to {plot_dir}")
    except Exception as e:
        logger.error(f"Error plotting training metrics: {e}")
    
    return model

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Chess AI Model")
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from the last checkpoint")
    parser.add_argument("--resume_iteration", type=int, default=0,
                       help="Resume training from a specific iteration")
    parser.add_argument("--iterations", type=int, default=None,
                       help="Number of iterations to train for (overrides config)")
    
    args = parser.parse_args()
    
    # Determine resume iteration
    resume_iteration = args.resume_iteration
    if args.resume and resume_iteration == 0:
        # Find the latest iteration if resume is requested
        config = load_config(args.config)
        model_dir = config.get('model_dir', './models')
        
        # Find checkpoint files
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith('model_iter_')]
        if checkpoints:
            # Extract iteration numbers and find the maximum
            iter_numbers = []
            for checkpoint in checkpoints:
                try:
                    iter_num = int(checkpoint.split('_')[-1].split('.')[0])
                    iter_numbers.append(iter_num)
                except (ValueError, IndexError):
                    continue
                    
            if iter_numbers:
                resume_iteration = max(iter_numbers)
                print(f"Resuming from iteration {resume_iteration}")
    
    # Train the model
    train_model(args.config, resume_iteration, args.iterations)

if __name__ == "__main__":
    main()