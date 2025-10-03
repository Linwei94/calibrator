import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

from .calibrator import Calibrator
from ..metrics import (
    BrierLoss, FocalLoss, LabelSmoothingLoss, 
    CrossEntropyLoss, MSELoss, SoftECE
)
from ..metrics.WeightedSoftECE import WeightedSoftECE
from ..metrics.SmoothSoftECE import SmoothSoftECE
from ..metrics.GapIndexedSoftECE import GapIndexedSoftECE

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
                - 'weighted_soft_ece' (weighted soft expected calibration error)
                - 'smooth_soft_ece' (smooth soft expected calibration error)
                - 'gap_indexed_soft_ece' (gap indexed soft expected calibration error)
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
        elif loss_type_lower in ['weighted_soft_ece', 'weightedsoftece']:
            return WeightedSoftECE().to(device)
        elif loss_type_lower in ['smooth_soft_ece', 'smoothsoftece']:
            return SmoothSoftECE().to(device)
        elif loss_type_lower in ['gap_indexed_soft_ece', 'gapindexedsoftece']:
            return GapIndexedSoftECE().to(device)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}. Options are 'nll', 'ce', 'ece', 'brier', 'mse', 'focal', 'ls', 'soft_ece', 'weighted_soft_ece', 'smooth_soft_ece', or 'gap_indexed_soft_ece'.")

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
        max_iter = kwargs.get('max_iter', 2000)  # Increase max iterations
        lr = kwargs.get('lr', 0.1)  # Increase learning rate
        
        # Use Adam optimizer instead of LBFGS for more stable training
        optimizer = optim.Adam([self.temperature], lr=lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)  # Increase patience

        def eval():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(val_logits)
            scaled_probs = F.softmax(scaled_logits, dim=1)
            
            # Convert labels to one-hot if needed
            if len(val_labels.shape) == 1:
                one_hot = torch.zeros(val_labels.size(0), scaled_probs.size(1), device=val_labels.device)
                one_hot.scatter_(1, val_labels.unsqueeze(1), 1)
                val_labels_one_hot = one_hot
            else:
                val_labels_one_hot = val_labels
            
            # Use the loss function with the appropriate parameters
            if isinstance(loss_fn, (SoftECE, WeightedSoftECE, SmoothSoftECE, GapIndexedSoftECE)):
                loss = loss_fn(logits=scaled_logits, labels=val_labels)
            elif isinstance(loss_fn, BrierLoss):
                loss = loss_fn(softmaxes=scaled_probs, labels=val_labels_one_hot)
            elif isinstance(loss_fn, (FocalLoss, LabelSmoothingLoss)):
                loss = loss_fn(softmaxes=scaled_probs, labels=val_labels_one_hot)
            elif isinstance(loss_fn, CrossEntropyLoss):
                loss = loss_fn(logits=scaled_logits, labels=val_labels)
            elif isinstance(loss_fn, MSELoss):
                diff = (scaled_probs - val_labels_one_hot).pow(2).sum(dim=1)
                loss = diff.mean()
            else:
                # For any other loss function, pass logits and labels
                loss = loss_fn(logits=scaled_logits, labels=val_labels)
            
            loss.backward()
            return loss
            
        # Training loop
        best_loss = float('inf')
        patience = 500  # Increase patience
        patience_counter = 0
        
        print(f"Starting temperature scaling training with max_iter={max_iter}")
        for i in range(max_iter):
            loss = eval()
            optimizer.step()
            scheduler.step(loss)
            
            # Print progress every 10 iterations
            if i % 10 == 0:
                print(f"Iteration {i}/{max_iter}, Loss: {loss.item():.6f}, Temperature: {self.temperature.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {i} with best loss {best_loss:.6f}")
                    break
                    
            # Ensure temperature stays positive
            with torch.no_grad():
                self.temperature.data.clamp_(min=1e-3)

        print(f"Training completed after {i+1} iterations")
        print(f"Final temperature: {self.temperature.item():.4f}")
        print(f"Final loss: {loss.item():.6f}")

        return self.temperature.item()

    def get_temperature(self):
        return self.temperature.item()
