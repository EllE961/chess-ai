"""
Training loop for the chess AI neural network.

This module provides the Trainer class which manages the full training process:
generating self-play games, updating the neural network, and evaluating new models.
"""

import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

from ..core.neural_network import ChessNetwork
from ..core.self_play import SelfPlay
from .data_manager import ReplayBuffer
from .loss_functions import ChessLoss
from .evaluate import ModelEvaluator

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer for the chess neural network.
    
    Manages the full training pipeline:
    1. Self-play game generation
    2. Neural network training
    3. Model evaluation
    4. Model selection and saving
    """
    
    def __init__(self, model: ChessNetwork, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model to train.
            config: Configuration dictionary.
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Extract training configuration
        training_config = config.get('training', {})
        self.num_iterations = training_config.get('num_iterations', 100)
        self.epochs_per_iteration = training_config.get('epochs_per_iteration', 10)
        self.checkpoint_freq = training_config.get('checkpoint_freq', 5)
        
        # Set up optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 0.0001)
        )
        
        # Learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=training_config.get('lr_step_size', 20),
            gamma=training_config.get('lr_gamma', 0.1)
        )
        
        # Set up components
        self.loss_fn = ChessLoss(config)
        self.replay_buffer = ReplayBuffer(config)
        self.self_play = SelfPlay(model, config)
        self.evaluator = ModelEvaluator(config)
        
        # Directories
        self.model_dir = config.get('model_dir', './models')
        self.log_dir = os.path.join(config.get('log_dir', './logs'), 'training_logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Load replay buffer if it exists
        self.replay_buffer.load()
        
        logger.info(f"Trainer initialized with {self.num_iterations} iterations, "
                   f"{self.epochs_per_iteration} epochs per iteration")
        
    def train(self, resume_iteration: int = 0) -> None:
        """
        Run the full training loop.
        
        Args:
            resume_iteration: Iteration number to resume from.
        """
        logger.info(f"Starting training from iteration {resume_iteration+1}")
        
        # Track best performance for model selection
        best_performance = float('-inf')
        current_iteration = resume_iteration
        
        # Training loop
        for iteration in range(resume_iteration, self.num_iterations):
            current_iteration = iteration
            logger.info(f"Iteration {iteration+1}/{self.num_iterations}")
            start_time = time.time()
            
            # 1. Generate self-play games
            logger.info("Generating self-play games...")
            game_data = self.self_play.generate_games()
            
            # Add game data to replay buffer
            for examples, metadata in game_data:
                self.replay_buffer.add_game_data(examples)
                
            # 2. Train neural network
            logger.info("Training neural network...")
            training_metrics = self._train_epoch(iteration)
            
            # 3. Evaluate against previous best model
            logger.info("Evaluating new model...")
            # Save current model
            current_model_path = os.path.join(self.model_dir, 'current_model.pt')
            self.model.save_checkpoint(current_model_path, self.optimizer, iteration=iteration)
            
            # Get best model path or use random baseline for first iteration
            best_model_path = os.path.join(self.model_dir, 'best_model.pt')
            if not os.path.exists(best_model_path):
                performance = 0.55  # First model automatically becomes best
                logger.info("No previous best model, using random baseline")
            else:
                # Evaluate current model against best model
                performance = self.evaluator.evaluate(current_model_path, best_model_path)
                
            # 4. Save model if improved
            eval_threshold = self.config.get('evaluation', {}).get('eval_threshold', 0.55)
            if performance >= eval_threshold:
                best_performance = performance
                self.model.save_checkpoint(best_model_path, self.optimizer, iteration=iteration)
                logger.info(f"New best model (performance: {performance:.4f})")
            else:
                logger.info(f"Model did not improve (performance: {performance:.4f})")
                
            # Always save iteration checkpoint
            if (iteration + 1) % self.checkpoint_freq == 0:
                checkpoint_path = os.path.join(self.model_dir, f'model_iter_{iteration+1}.pt')
                self.model.save_checkpoint(checkpoint_path, self.optimizer, iteration=iteration)
            
            # Save replay buffer
            self.replay_buffer.save()
            
            # Log results
            self._log_iteration_results(
                iteration, training_metrics, performance, time.time() - start_time
            )
            
        logger.info(f"Training completed after {current_iteration+1} iterations")
            
    def _train_epoch(self, iteration: int) -> Dict[str, float]:
        """
        Train the neural network for one epoch.
        
        Args:
            iteration: Current iteration number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        dataloader = self.replay_buffer.get_dataloader(
            num_workers=self.config.get('system', {}).get('num_workers', 4)
        )
        
        # Skip training if buffer is empty
        if len(dataloader) == 0:
            logger.warning("Replay buffer is empty, skipping training")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'value_accuracy': 0.0,
                'policy_accuracy': 0.0
            }
            
        # Training metrics
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        metrics_dict = {
            'value_accuracy': 0.0,
            'policy_accuracy': 0.0,
            'kl_divergence': 0.0,
            'value_mean': 0.0,
            'value_std': 0.0
        }
        
        # Train for multiple epochs per iteration
        for epoch in range(self.epochs_per_iteration):
            epoch_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_metrics = {k: 0.0 for k in metrics_dict}
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs_per_iteration}")
            
            for states, policy_targets, value_targets in pbar:
                # Move to device
                states = states.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)
                
                # Forward pass
                policy_outputs, value_outputs = self.model(states)
                
                # Calculate loss
                loss, policy_loss, value_loss = self.loss_fn(
                    policy_outputs, value_outputs, policy_targets, value_targets, self.model
                )
                
                # Calculate additional metrics
                metrics = self.loss_fn.calculate_metrics(
                    policy_outputs, value_outputs, policy_targets, value_targets
                )
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                batch_size = states.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_policy_loss += policy_loss.item() * batch_size
                epoch_value_loss += value_loss.item() * batch_size
                
                for k, v in metrics.items():
                    epoch_metrics[k] += v * batch_size
                    
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'p_loss': policy_loss.item(),
                    'v_loss': value_loss.item()
                })
                
            # Normalize epoch metrics
            total_samples = len(dataloader.dataset)
            epoch_loss /= total_samples
            epoch_policy_loss /= total_samples
            epoch_value_loss /= total_samples
            
            for k in epoch_metrics:
                epoch_metrics[k] /= total_samples
                
            # Update running metrics
            total_loss += epoch_loss
            policy_loss_sum += epoch_policy_loss
            value_loss_sum += epoch_value_loss
            
            for k, v in epoch_metrics.items():
                metrics_dict[k] += v
                
            # Log epoch metrics
            self._log_epoch_metrics(iteration, epoch, epoch_loss, epoch_policy_loss, 
                                   epoch_value_loss, epoch_metrics)
            
        # Step the learning rate scheduler once per iteration
        self.scheduler.step()
        
        # Calculate average across epochs
        avg_loss = total_loss / self.epochs_per_iteration
        avg_policy_loss = policy_loss_sum / self.epochs_per_iteration
        avg_value_loss = value_loss_sum / self.epochs_per_iteration
        
        avg_metrics = {
            k: v / self.epochs_per_iteration for k, v in metrics_dict.items()
        }
        
        # Combine all metrics
        metrics = {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            **avg_metrics,
            'lr': self.scheduler.get_last_lr()[0],
            'buffer_size': len(self.replay_buffer)
        }
        
        return metrics
        
    def _log_epoch_metrics(self, iteration: int, epoch: int, loss: float, 
                          policy_loss: float, value_loss: float, 
                          metrics: Dict[str, float]) -> None:
        """
        Log epoch metrics to TensorBoard.
        
        Args:
            iteration: Current iteration number.
            epoch: Current epoch number.
            loss: Total loss value.
            policy_loss: Policy loss value.
            value_loss: Value loss value.
            metrics: Dictionary of additional metrics.
        """
        # Calculate global step
        global_step = iteration * self.epochs_per_iteration + epoch
        
        # Log losses
        self.writer.add_scalar('epoch/loss', loss, global_step)
        self.writer.add_scalar('epoch/policy_loss', policy_loss, global_step)
        self.writer.add_scalar('epoch/value_loss', value_loss, global_step)
        
        # Log additional metrics
        for name, value in metrics.items():
            self.writer.add_scalar(f'epoch/{name}', value, global_step)
            
    def _log_iteration_results(self, iteration: int, metrics: Dict[str, float], 
                              performance: float, elapsed_time: float) -> None:
        """
        Log iteration results to TensorBoard and console.
        
        Args:
            iteration: Current iteration number.
            metrics: Dictionary of training metrics.
            performance: Model performance against previous best.
            elapsed_time: Time taken for the iteration.
        """
        # Log to TensorBoard
        self.writer.add_scalar('iteration/loss', metrics['loss'], iteration)
        self.writer.add_scalar('iteration/policy_loss', metrics['policy_loss'], iteration)
        self.writer.add_scalar('iteration/value_loss', metrics['value_loss'], iteration)
        self.writer.add_scalar('iteration/value_accuracy', metrics.get('value_accuracy', 0), iteration)
        self.writer.add_scalar('iteration/policy_accuracy', metrics.get('policy_accuracy', 0), iteration)
        self.writer.add_scalar('iteration/performance', performance, iteration)
        self.writer.add_scalar('iteration/learning_rate', metrics['lr'], iteration)
        self.writer.add_scalar('iteration/buffer_size', metrics['buffer_size'], iteration)
        self.writer.add_scalar('iteration/time', elapsed_time, iteration)
        
        # Log to console
        logger.info(f"Iteration {iteration+1} results:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        logger.info(f"  Value Loss: {metrics['value_loss']:.4f}")
        logger.info(f"  Value Accuracy: {metrics.get('value_accuracy', 0):.4f}")
        logger.info(f"  Policy Accuracy: {metrics.get('policy_accuracy', 0):.4f}")
        logger.info(f"  Performance vs Best: {performance:.4f}")
        logger.info(f"  Learning Rate: {metrics['lr']:.6f}")
        logger.info(f"  Replay Buffer Size: {metrics['buffer_size']}")
        logger.info(f"  Time: {elapsed_time:.2f}s")