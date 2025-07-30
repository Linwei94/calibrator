import torch
import numpy as np
from torch.nn import functional as F

from .calibrator import Calibrator

class HistogramBinningCalibrator(Calibrator):
    def __init__(self, n_bins=10, strategy='uniform'):
        """
        Initialize the Histogram Binning calibrator.
        
        Args:
            n_bins (int): Number of bins to use for calibration
            strategy (str): Binning strategy ('uniform' or 'quantile')
        """
        super(HistogramBinningCalibrator, self).__init__()
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_boundaries = None
        self.bin_lowers = None
        self.bin_uppers = None
        self.bin_accuracies = None

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Fit the histogram binning calibrator on validation set.
        
        Args:
            val_logits (torch.Tensor): Validation logits of shape [N, num_classes]
            val_labels (torch.Tensor): Validation labels of shape [N]
        
        Returns:
            dict: Fitted parameters
        """
        # Convert to probabilities if logits are provided
        if val_logits.dim() == 2:
            val_probs = F.softmax(val_logits, dim=1)
            # Get confidence (max probability)
            confidences, predictions = torch.max(val_probs, dim=1)
        else:
            # Already probabilities
            confidences = val_logits
            predictions = (confidences > 0.5).long()
        
        # Convert to numpy for easier processing
        confidences = confidences.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        val_labels = val_labels.detach().cpu().numpy()
        
        # Compute accuracies
        accuracies = (predictions == val_labels).astype(float)
        
        # Determine bin boundaries
        if self.strategy == 'uniform':
            # Uniform width bins
            self.bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        elif self.strategy == 'quantile':
            # Equal frequency bins
            sorted_confidences = np.sort(confidences)
            quantiles = np.linspace(0, len(sorted_confidences), self.n_bins + 1).astype(int)
            self.bin_boundaries = np.array([0] + [sorted_confidences[q-1] if q > 0 else 0 
                                                for q in quantiles[1:]])
            self.bin_boundaries[-1] = 1.0  # Ensure last boundary is 1.0
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]
        
        # Compute bin accuracies
        self.bin_accuracies = np.zeros(self.n_bins)
        
        for i in range(self.n_bins):
            # Find samples in this bin
            in_bin = (confidences > self.bin_lowers[i]) & (confidences <= self.bin_uppers[i])
            
            if np.sum(in_bin) > 0:
                # Compute empirical accuracy for this bin
                self.bin_accuracies[i] = np.mean(accuracies[in_bin])
            else:
                # If no samples in bin, use bin center
                self.bin_accuracies[i] = (self.bin_lowers[i] + self.bin_uppers[i]) / 2
        
        return {
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'bin_boundaries': self.bin_boundaries,
            'bin_accuracies': self.bin_accuracies
        }

    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate the test logits using histogram binning.
        
        Args:
            test_logits (torch.Tensor): Test logits of shape [N, num_classes]
            return_logits (bool): If True, return logits; otherwise return probabilities
        
        Returns:
            torch.Tensor: Calibrated probabilities or logits
        """
        if self.bin_boundaries is None:
            raise ValueError("Model must be fitted before calibration")
        
        device = test_logits.device
        
        # Convert to probabilities if logits are provided
        if test_logits.dim() == 2:
            test_probs = F.softmax(test_logits, dim=1)
            # Get confidence (max probability) and predictions
            confidences, predictions = torch.max(test_probs, dim=1)
        else:
            # Already probabilities
            confidences = test_logits
            predictions = (confidences > 0.5).long()
        
        # Convert to numpy for processing
        confidences_np = confidences.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        
        # Initialize calibrated confidences
        calibrated_confidences = np.zeros_like(confidences_np)
        
        # Apply histogram binning
        for i in range(self.n_bins):
            # Find samples in this bin
            in_bin = (confidences_np > self.bin_lowers[i]) & (confidences_np <= self.bin_uppers[i])
            
            if np.sum(in_bin) > 0:
                # Replace confidence with bin accuracy
                calibrated_confidences[in_bin] = self.bin_accuracies[i]
        
        # Convert back to torch tensor
        calibrated_confidences = torch.tensor(calibrated_confidences, device=device, dtype=torch.float32)
        
        if test_logits.dim() == 2:
            # Multi-class case: adjust all probabilities proportionally
            num_classes = test_logits.size(1)
            calibrated_probs = torch.zeros_like(test_probs)
            
            for i in range(len(calibrated_confidences)):
                pred_class = predictions[i]
                original_confidence = confidences[i]
                new_confidence = calibrated_confidences[i]
                
                if original_confidence > 0:
                    # Scale the predicted class probability
                    scale_factor = new_confidence / original_confidence
                    calibrated_probs[i] = test_probs[i] * scale_factor
                    
                    # Renormalize to ensure probabilities sum to 1
                    calibrated_probs[i] = calibrated_probs[i] / torch.sum(calibrated_probs[i])
                else:
                    # Uniform distribution if original confidence is 0
                    calibrated_probs[i] = torch.ones(num_classes, device=device) / num_classes
            
            if return_logits:
                # Convert back to logits with numerical stability
                calibrated_probs = torch.clamp(calibrated_probs, min=1e-7, max=1-1e-7)
                # Normalize again to ensure sum=1
                calibrated_probs = calibrated_probs / torch.sum(calibrated_probs, dim=1, keepdim=True)
                # Convert to logits using log
                logits = torch.log(calibrated_probs)
                # Replace any remaining inf/nan with reasonable values
                logits = torch.where(torch.isfinite(logits), logits, torch.tensor(-10.0, device=logits.device))
                return logits
            else:
                return calibrated_probs
        else:
            # Binary case
            if return_logits:
                # Convert to logits
                calibrated_confidences = torch.clamp(calibrated_confidences, min=1e-8, max=1-1e-8)
                return torch.log(calibrated_confidences / (1 - calibrated_confidences))
            else:
                return calibrated_confidences