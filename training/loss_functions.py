"""
Loss functions for training the chess AI neural network.

This module provides custom loss functions for the policy and value heads
of the neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

class ChessLoss:
    """
    Combined loss function for the chess neural network.
    
    Combines policy loss (cross-entropy on move probabilities) and value loss
    (mean squared error on game outcome prediction), with optional regularization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loss function.
        
        Args:
            config: Configuration dictionary containing loss parameters.
        """
        loss_config = config.get('loss', {})
        self.value_loss_weight = loss_config.get('value_loss_weight', 1.0)
        self.policy_loss_weight = loss_config.get('policy_loss_weight', 1.0)
        self.l2_weight = loss_config.get('l2_weight', 0.0001)
        
    def __call__(self, policy_output: torch.Tensor, value_output: torch.Tensor,
                policy_target: torch.Tensor, value_target: torch.Tensor,
                model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the loss.
        
        Args:
            policy_output: Predicted policy (batch_size, policy_size)
            value_output: Predicted value (batch_size, 1)
            policy_target: Target policy (batch_size, policy_size)
            value_target: Target value (batch_size, 1)
            model: The neural network model for L2 regularization
            
        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        # Value loss: Mean squared error
        value_loss = F.mse_loss(value_output, value_target)
        
        # Policy loss: Cross entropy on distributions
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        policy_loss = -torch.sum(policy_target * torch.log(policy_output + epsilon)) / policy_target.size(0)
        
        # L2 regularization
        l2_reg = torch.tensor(0.0, device=policy_output.device)
        if self.l2_weight > 0:
            for param in model.parameters():
                l2_reg += torch.norm(param)
            l2_reg *= self.l2_weight
        
        # Combine losses
        total_loss = (
            self.value_loss_weight * value_loss +
            self.policy_loss_weight * policy_loss +
            l2_reg
        )
        
        return total_loss, policy_loss, value_loss
        
    def calculate_metrics(self, policy_output: torch.Tensor, value_output: torch.Tensor,
                        policy_target: torch.Tensor, value_target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate additional metrics for monitoring training.
        
        Args:
            policy_output: Predicted policy (batch_size, policy_size)
            value_output: Predicted value (batch_size, 1)
            policy_target: Target policy (batch_size, policy_size)
            value_target: Target value (batch_size, 1)
            
        Returns:
            Dictionary of metrics
        """
        # Value accuracy: percentage of predictions within 0.2 of target
        value_accuracy = torch.mean(
            (torch.abs(value_output - value_target) < 0.2).float()
        ).item()
        
        # Policy accuracy: percentage where the highest probability move matches
        policy_top1 = torch.argmax(policy_output, dim=1)
        target_top1 = torch.argmax(policy_target, dim=1)
        policy_accuracy = torch.mean((policy_top1 == target_top1).float()).item()
        
        # KL divergence
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        kl_div = F.kl_div(
            torch.log(policy_output + epsilon),
            policy_target,
            reduction='batchmean'
        ).item()
        
        # Value mean and standard deviation
        value_mean = torch.mean(value_output).item()
        value_std = torch.std(value_output).item()
        
        return {
            'value_accuracy': value_accuracy,
            'policy_accuracy': policy_accuracy,
            'kl_divergence': kl_div,
            'value_mean': value_mean,
            'value_std': value_std
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in the policy head.
    
    This can be useful when certain moves are much more common than others,
    helping the model focus on learning the rarer but important moves.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize the focal loss.
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter (higher values increase focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the focal loss.
        
        Args:
            predictions: Predicted probabilities (batch_size, num_classes)
            targets: Target probabilities (batch_size, num_classes)
            
        Returns:
            Focal loss value
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        
        # Focal loss calculation
        ce_loss = -torch.sum(targets * torch.log(predictions + epsilon), dim=1)
        p_t = torch.sum(predictions * targets, dim=1)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return loss.mean()