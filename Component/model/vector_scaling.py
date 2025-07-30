import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

from .calibrator import Calibrator
from ..metrics import (
    BrierLoss, FocalLoss, LabelSmoothingLoss, 
    CrossEntropyLoss, MSELoss, SoftECE
)

class VectorScalingCalibrator(Calibrator):
    def __init__(self, loss_type='nll', bias=True):
        """
        Initialize the Vector Scaling calibrator.
        
        Args:
            loss_type (str): Type of loss function to use for training
            bias (bool): Whether to include bias term
        """
        super(VectorScalingCalibrator, self).__init__()
        self.loss_type = loss_type
        self.use_bias = bias
        self.weight = None
        self.bias = None
        self.num_classes = None

    def fit(self, val_logits, val_labels, lr=0.01, max_iter=50, **kwargs):
        """
        Fit the vector scaling parameters on validation set.
        
        Args:
            val_logits (torch.Tensor): Validation logits of shape [N, num_classes]
            val_labels (torch.Tensor): Validation labels of shape [N]
            lr (float): Learning rate for optimization
            max_iter (int): Maximum number of iterations
        
        Returns:
            dict: Fitted parameters
        """
        device = val_logits.device
        self.num_classes = val_logits.size(1)
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.ones(self.num_classes, device=device))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_classes, device=device))
        
        # Get loss function
        criterion = self._get_loss_function(device, self.num_classes)
        
        # Setup optimizer
        params = [self.weight]
        if self.use_bias:
            params.append(self.bias)
        optimizer = optim.LBFGS(params, lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            
            # Apply vector scaling
            scaled_logits = self.vector_scale(val_logits)
            
            # Compute loss
            if hasattr(criterion, 'forward'):
                if self.loss_type in ['ece', 'soft_ece']:
                    # ECE-based losses need probabilities
                    probs = F.softmax(scaled_logits, dim=1)
                    loss = criterion(probs, val_labels)
                else:
                    # Other losses use logits
                    loss = criterion(scaled_logits, val_labels)
            else:
                # Standard PyTorch loss functions
                loss = criterion(scaled_logits, val_labels)
            
            loss.backward()
            return loss
        
        # Optimize
        optimizer.step(closure)
        
        return {
            'weight': self.weight.detach().cpu().numpy(),
            'bias': self.bias.detach().cpu().numpy() if self.use_bias else None,
            'loss_type': self.loss_type,
            'num_classes': self.num_classes
        }

    def vector_scale(self, logits):
        """
        Apply vector scaling to logits.
        
        Args:
            logits (torch.Tensor): Input logits of shape [N, num_classes]
            
        Returns:
            torch.Tensor: Scaled logits
        """
        if self.weight is None:
            raise ValueError("Model must be fitted before scaling")
        
        # Ensure parameters are on the same device as logits
        weight = self.weight.to(logits.device)
        
        # Apply element-wise multiplication
        scaled_logits = logits * weight.unsqueeze(0)
        
        if self.use_bias and self.bias is not None:
            bias = self.bias.to(logits.device)
            scaled_logits = scaled_logits + bias.unsqueeze(0)
        
        return scaled_logits

    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate test logits using vector scaling.
        
        Args:
            test_logits (torch.Tensor): Test logits
            return_logits (bool): Whether to return logits or probabilities
            
        Returns:
            torch.Tensor: Calibrated predictions
        """
        if self.weight is None:
            raise ValueError("Model must be fitted before calibration")
        
        # Apply vector scaling
        scaled_logits = self.vector_scale(test_logits)
        
        if return_logits:
            return scaled_logits
        else:
            return F.softmax(scaled_logits, dim=1)

    def _get_loss_function(self, device, num_classes=None):
        """
        Get the appropriate loss function based on the loss_type.
        
        Args:
            device: Device to create the loss function on
            num_classes (int): Number of classes (needed for some losses)
            
        Returns:
            Loss function
        """
        if self.loss_type in ['nll', 'ce']:
            return nn.CrossEntropyLoss()
        elif self.loss_type == 'mse':
            return MSELoss()
        elif self.loss_type == 'brier':
            return BrierLoss()
        elif self.loss_type == 'focal':
            return FocalLoss(gamma=2.0)
        elif self.loss_type == 'ls':
            return LabelSmoothingLoss(alpha=0.1)
        elif self.loss_type == 'ece':
            from ..metrics.ece import ECE
            return ECE(n_bins=15)
        elif self.loss_type == 'soft_ece':
            return SoftECE(n_bins=15)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")