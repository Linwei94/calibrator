import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

from .calibrator import Calibrator

class TemperatureScalingCalibrator(Calibrator):
    def __init__(self, loss_type='nll'):
        """
        Initialize the temperature scaling calibrator.
        
        Args:
            loss_type (str): Type of loss function to use for training.
                Options: 
                - 'nll' (negative log likelihood/cross-entropy)
                - 'ece' (expected calibration error)
                - 'brier' (Brier score)
                - 'mmce' (maximum mean calibration error)
                - 'ls' (label smoothing with alpha=0.05)
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

    def _brier_loss(self, probs, labels, num_classes):
        """
        Compute Brier score loss.
        
        Args:
            probs (torch.Tensor): Predicted probabilities of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,)
            num_classes (int): Number of classes
            
        Returns:
            torch.Tensor: Brier loss
        """
        # Convert labels to one-hot encoding
        one_hot = torch.zeros(probs.size(0), num_classes, device=probs.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Compute Brier score
        return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))
    
    def _mmce_loss(self, probs, labels, num_classes):
        """
        Compute Maximum Mean Calibration Error (MMCE) loss.
        
        Args:
            probs (torch.Tensor): Predicted probabilities of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,)
            num_classes (int): Number of classes
            
        Returns:
            torch.Tensor: MMCE loss
        """
        # Convert labels to one-hot encoding
        one_hot = torch.zeros(probs.size(0), num_classes, device=probs.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Get predicted probabilities for the true classes
        pred_probs = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
        
        # Compute calibration error for each sample
        calibration_error = torch.abs(pred_probs - 1.0)
        
        # Compute MMCE (simplified version)
        return torch.mean(calibration_error)
    
    def _label_smoothing_loss(self, logits, labels, alpha=0.05):
        """
        Compute Label Smoothing loss.
        
        Args:
            logits (torch.Tensor): Logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,)
            alpha (float): Smoothing parameter
            
        Returns:
            torch.Tensor: Label smoothing loss
        """
        num_classes = logits.size(1)
        
        # Convert labels to one-hot encoding
        one_hot = torch.zeros(logits.size(0), num_classes, device=logits.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_labels = one_hot * (1 - alpha) + alpha / num_classes
        
        # Compute cross-entropy with smoothed labels
        log_probs = F.log_softmax(logits, dim=1)
        return -torch.sum(smooth_labels * log_probs, dim=1).mean()

    def _get_loss_function(self, device, num_classes=None):
        """
        Get the appropriate loss function based on the loss_type.
        
        Args:
            device (torch.device): Device to place the loss function on
            num_classes (int, optional): Number of classes, needed for some loss functions
            
        Returns:
            callable: Loss function
        """
        if self.loss_type.lower() == 'nll' or self.loss_type.lower() == 'ce':
            return nn.CrossEntropyLoss().to(device)
        elif self.loss_type.lower() == 'ece':
            from ..metrics import ECE
            return ECE(n_bins=15).to(device)
        elif self.loss_type.lower() == 'brier':
            if num_classes is None:
                raise ValueError("num_classes must be provided for Brier loss")
            return lambda probs, labels: self._brier_loss(probs, labels, num_classes)
        elif self.loss_type.lower() == 'mmce':
            if num_classes is None:
                raise ValueError("num_classes must be provided for MMCE loss")
            return lambda probs, labels: self._mmce_loss(probs, labels, num_classes)
        elif self.loss_type.lower() == 'ls':
            return lambda logits, labels: self._label_smoothing_loss(logits, labels, alpha=0.05)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}. Options are 'nll', 'ece', 'brier', 'mmce', or 'ls'.")

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Tune the temperature of the model using the validation set.
        
        Args:
            val_logits (torch.Tensor): Validation logits
            val_labels (torch.Tensor): Validation labels
            **kwargs: Additional arguments
                - max_iter (int): Maximum number of iterations for the optimizer
                - lr (float): Learning rate for the optimizer
                
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
            
            if self.loss_type.lower() in ['nll', 'ce', 'ls']:
                loss = loss_fn(scaled_logits, val_labels)
            else:  # ece, brier, mmce
                probs = F.softmax(scaled_logits, dim=1)
                if self.loss_type.lower() == 'ece':
                    loss = loss_fn(softmaxes=probs, labels=val_labels)
                else:  # brier, mmce
                    loss = loss_fn(probs, val_labels)
                
            loss.backward()
            return loss
            
        optimizer.step(eval)

        return self.temperature.item()

    def get_temperature(self):
        return self.temperature.item()
