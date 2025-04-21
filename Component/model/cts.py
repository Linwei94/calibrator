import torch
import torch.nn as nn
import numpy as np
from .calibrator import Calibrator
from ..metrics import ECE, Accuracy
import torch.nn.functional as F
from .temperature_scaling import TemperatureScalingCalibrator

class CTSCalibrator(Calibrator):
    """
    Implements Class-based Temperature Scaling (CTS).
    
    CTS assigns a separate temperature parameter for each class, allowing for more
    flexible calibration compared to standard temperature scaling.
    """
    def __init__(self, n_class, n_iter=5, n_bins=15):
        """
        Args:
            n_class (int): Number of classes.
            n_iter (int): Number of iterations for greedy search.
            n_bins (int): Number of bins used in the ECE computation.
        """
        super(CTSCalibrator, self).__init__()
        self.n_class = n_class
        self.n_iter = n_iter
        self.n_bins = n_bins
        # Initialize the learnable temperature parameters (one per class)
        self.T = nn.Parameter(torch.ones(n_class))
        # Initialize metrics
        self.ece_metric = ECE(n_bins=n_bins)
        self.accuracy_metric = Accuracy()

    def forward(self, x):
        """
        Forward pass: scales the input logits x by dividing each column (class score) 
        by its corresponding temperature.
        
        Args:
            x (torch.Tensor): Input logits of shape (batch_size, n_class)
            
        Returns:
            torch.Tensor: Scaled logits
        """
        # Broadcasting division; x's column i is divided by self.T[i]
        return x / self.T

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Fit the calibrator using validation logits and labels.
        
        Args:
            val_logits: torch.Tensor
                Validation logits
            val_labels: torch.Tensor
                Validation labels
            **kwargs: dict
                Additional arguments:
                - grid: list of temperature values to search over
                - ts_loss: loss function type for temperature scaling initialization
        """
        device = val_logits.device
        self.to(device)
        
        # Initialize temperature parameters using traditional temperature scaling
        ts_calibrator = TemperatureScalingCalibrator(loss_type=kwargs.get('ts_loss', 'nll'))
        ts_calibrator.fit(val_logits, val_labels)
        # Initialize all class temperatures with the single temperature from TS
        self.T.data = torch.ones(self.n_class, device=device) * ts_calibrator.temperature.data
        
        # Compute initial probabilities
        val_probs = F.softmax(val_logits / self.T, dim=1)
        
        # Compute initial metrics
        ece_fn = ECE(n_bins=self.n_bins).to(device)
        initial_ece = ece_fn(softmaxes=val_probs, labels=val_labels)
        initial_acc = (val_probs.argmax(dim=1) == val_labels).float().mean().item()
        
        print(f"Initial ECE: {initial_ece:.4f}, Accuracy: {initial_acc:.4f}")
        
        # Greedy optimization loop
        grid = kwargs.get('grid', [0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        best_ece = initial_ece
        best_acc = initial_acc
        
        for iter in range(self.n_iter):
            print(f"\nIteration {iter + 1}/{self.n_iter}")
            
            # Try different temperatures for each class
            for cls in range(self.n_class):
                best_temp = self.T[cls].item()
                best_cls_ece = float('inf')
                
                # Get class-specific logits and labels
                cls_mask = (val_labels == cls)
                if not cls_mask.any():
                    continue
                    
                cls_logits = val_logits[cls_mask]
                cls_labels = val_labels[cls_mask]
                
                # Try each temperature value
                for temp in grid:
                    # Create temporary temperature vector
                    temp_tensor = self.T.data.clone()
                    temp_tensor[cls] = temp
                    
                    # Compute probabilities with current temperature
                    probs = F.softmax(cls_logits / temp_tensor[cls], dim=1)
                    
                    # Compute ECE for this class
                    cls_ece = ece_fn(softmaxes=probs, labels=cls_labels)
                    
                    if cls_ece < best_cls_ece:
                        best_cls_ece = cls_ece
                        best_temp = temp
                
                # Update temperature for this class
                self.T.data[cls] = best_temp
                print(f"Class {cls}: Best temperature = {best_temp:.2f}, ECE = {best_cls_ece:.4f}")
            
            # Compute overall metrics with updated temperatures
            val_probs = F.softmax(val_logits / self.T, dim=1)
            current_ece = ece_fn(softmaxes=val_probs, labels=val_labels)
            current_acc = (val_probs.argmax(dim=1) == val_labels).float().mean().item()
            
            print(f"Iteration {iter + 1} - ECE: {current_ece:.4f}, Accuracy: {current_acc:.4f}")
            
            # Update best metrics
            if current_ece < best_ece:
                best_ece = current_ece
                best_acc = current_acc
                print(f"New best ECE: {best_ece:.4f}, Accuracy: {best_acc:.4f}")
        
        return {
            'final_ece': best_ece,
            'final_accuracy': best_acc
        }

    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate the test logits using the optimized temperatures.
        
        Args:
            test_logits (torch.Tensor): Test logits of shape (N, n_class)
            return_logits (bool): Whether to return calibrated logits instead of probabilities
            **kwargs: Additional arguments (not used)
            
        Returns:
            If return_logits is False: Calibrated probabilities of shape (N, n_class)
            If return_logits is True: Calibrated logits of shape (N, n_class)
        """
        # Convert input to tensor if it's not already
        if not torch.is_tensor(test_logits):
            test_logits = torch.tensor(test_logits, dtype=torch.float32)
            
        # Apply temperature scaling
        calibrated_logits = self.forward(test_logits)
        
        if return_logits:
            return calibrated_logits
        
        # Apply softmax to get probabilities
        calibrated_probs = torch.nn.functional.softmax(calibrated_logits, dim=1)
        return calibrated_probs
    
    def save(self, path="./"):
        """
        Save the CTS model parameters.
        
        Args:
            path (str): Directory to save the model
        """
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, "cts_model.pth")
        torch.save(self.state_dict(), save_path)
        print("Save CTS model to:", save_path)
    
    def load(self, path="./"):
        """
        Load the CTS model parameters.
        
        Args:
            path (str): Directory to load the model from
        """
        import os
        load_path = os.path.join(path, "cts_model.pth")
        self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        print("Load CTS model from:", load_path)