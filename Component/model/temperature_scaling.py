import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

from .calibrator import Calibrator
from ..metrics import (
    BrierLoss, FocalLoss, LabelSmoothingLoss, 
    CrossEntropyLoss, MSELoss, SoftECE
)

class TemperatureScalingCalibrator(Calibrator):
    def __init__(self, loss_type='nll'):
        """
        Initialize the temperature scaling calibrator.
        
        Args:
            loss_type (str): Type of loss function to use for training.
                Options: 
                - 'nll' or 'ce' (negative log likelihood/cross-entropy)
                - 'ece' (expected calibration error)
                - 'brier' (Brier score)
                - 'mse' (mean squared error)
                - 'focal' (focal loss with gamma=2.0)
                - 'ls' (label smoothing with alpha=0.1)
                - 'soft_ece' (soft expected calibration error)
        """
        super(TemperatureScalingCalibrator, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.loss_type = loss_type

    def calibrate(self, logits, return_logits=False):
        if return_logits:
            return self.temperature_scale(logits)
        else:
            return F.softmax(self.temperature_scale(logits), dim=1)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Ensure temperature is on the same device as logits
        temperature = self.temperature.to(logits.device)
        temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def _get_loss_function(self, device, num_classes=None):
        """
        Get the appropriate loss function based on the loss_type.
        
        Args:
            device (torch.device): Device to place the loss function on
            num_classes (int, optional): Number of classes, needed for some loss functions
            
        Returns:
            callable: Loss function
        """
        loss_type_lower = self.loss_type.lower()
        
        if loss_type_lower in ['nll', 'ce', 'cross_entropy', 'crossentropy']:
            return CrossEntropyLoss().to(device)
        elif loss_type_lower in ['ece', 'expected_calibration_error']:
            from ..metrics import ECE
            return ECE(n_bins=15).to(device)
        elif loss_type_lower in ['brier', 'brier_score']:
            return BrierLoss().to(device)
        elif loss_type_lower in ['mse', 'mean_squared_error']:
            return MSELoss().to(device)
        elif loss_type_lower in ['focal', 'focal_loss']:
            return FocalLoss().to(device)
        elif loss_type_lower in ['ls', 'label_smoothing']:
            return LabelSmoothingLoss().to(device)
        elif loss_type_lower in ['soft_ece', 'softece']:
            return SoftECE().to(device)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}. Options are 'nll', 'ce', 'ece', 'brier', 'mse', 'focal', 'ls', or 'soft_ece'.")

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Tune the temperature of the model using the validation set.
        
        Args:
            val_logits (torch.Tensor): Validation logits
            val_labels (torch.Tensor): Validation labels
            **kwargs: Additional arguments
                - max_iter (int): Maximum number of iterations for the optimizer
                - lr (float): Learning rate for the optimizer
                - focal_gamma (float): Gamma parameter for focal loss, defaults to 2.0
                - label_smoothing_alpha (float): Alpha parameter for label smoothing, defaults to 0.1
                
        Returns:
            float: Optimal temperature value
        """
        # Move to the same device as val_logits
        device = val_logits.device
        self.to(device)
        
        # Get number of classes from logits shape
        num_classes = val_logits.size(1)
        
        # Get loss function
        loss_fn = self._get_loss_function(device, num_classes)
        
        # Get optimizer parameters from kwargs or use defaults
        max_iter = kwargs.get('max_iter', 1000)
        lr = kwargs.get('lr', 0.01)
        
        # Optimize the temperature w.r.t. the specified loss
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(val_logits)
            
            # Use the loss function with the appropriate parameters
            loss = loss_fn(logits=scaled_logits, labels=val_labels)
            
            loss.backward()
            return loss
            
        optimizer.step(eval)

        return self.temperature.item()

    def get_temperature(self):
        return self.temperature.item()
