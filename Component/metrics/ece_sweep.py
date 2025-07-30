import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ECESweep(nn.Module):
    '''
    Compute ECE Sweep (Monotonic Sweep Calibration Error)
    Based on "Mitigating Bias in Calibration Error Estimation" by Roelofs et al., 2022
    
    Uses equal-mass binning and chooses the number of bins to be as large as
    possible while preserving monotonicity in the calibration function.
    '''
    def __init__(self, p=1, max_bins=None):
        '''
        Args:
            p: int
                The norm to use (1 for ECE, 2 for L2 calibration error)
            max_bins: int
                Maximum number of bins to consider (defaults to sample size)
        '''
        super(ECESweep, self).__init__()
        self.p = p
        self.max_bins = max_bins

    def forward(self, logits=None, labels=None, softmaxes=None):
        '''
        args:
            logits: torch.Tensor
                The logits to calibrate, the output of the model before softmax layer
            labels: torch.Tensor
                The labels of the test data
            softmaxes: torch.Tensor
                The softmaxes of the test data, if None, compute the softmaxes from logits

        Returns:
            ece_sweep: float
                The ECE sweep value
        '''
        if softmaxes is None:
            softmaxes = F.softmax(logits, dim=1)
            
        # Convert to NumPy arrays if they're PyTorch tensors
        if torch.is_tensor(softmaxes):
            probs = softmaxes.detach().cpu().numpy()
        else:
            probs = softmaxes
            
        if torch.is_tensor(labels):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels
            
        # Get top-label confidences and predictions
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        correct = (predictions == labels_np).astype(float)
        
        n_samples = len(confidences)
        max_bins = self.max_bins if self.max_bins is not None else n_samples
        
        # Find optimal number of bins using monotonic sweep
        optimal_bins = self._find_optimal_bins(confidences, correct, max_bins)
        
        # Compute ECE with optimal number of bins using equal-mass binning
        return self._compute_ece_equal_mass(confidences, correct, optimal_bins)
    
    def _find_optimal_bins(self, confidences, correct, max_bins):
        """
        Find the largest number of bins that preserves monotonicity
        """
        n_samples = len(confidences)
        
        # Start with 2 bins and gradually increase
        for b in range(2, min(max_bins + 1, n_samples + 1)):
            if not self._is_monotonic(confidences, correct, b):
                return b - 1
        
        # If all tested bin numbers are monotonic, return the maximum
        return min(max_bins, n_samples)
    
    def _is_monotonic(self, confidences, correct, n_bins):
        """
        Check if binning with n_bins results in monotonic bin heights
        """
        if n_bins <= 1:
            return True
            
        # Create equal-mass bins
        sorted_indices = np.argsort(confidences)
        bin_heights = []
        
        samples_per_bin = len(confidences) // n_bins
        remainder = len(confidences) % n_bins
        
        start_idx = 0
        for i in range(n_bins):
            # Some bins get one extra sample if there's a remainder
            bin_size = samples_per_bin + (1 if i < remainder else 0)
            end_idx = start_idx + bin_size
            
            if start_idx >= len(sorted_indices):
                break
                
            # Get indices for this bin
            bin_indices = sorted_indices[start_idx:end_idx]
            
            # Compute average accuracy (bin height) for this bin
            if len(bin_indices) > 0:
                bin_height = np.mean(correct[bin_indices])
                bin_heights.append(bin_height)
            
            start_idx = end_idx
        
        # Check if bin heights are monotonic (non-decreasing)
        for i in range(1, len(bin_heights)):
            if bin_heights[i] < bin_heights[i-1]:
                return False
        
        return True
    
    def _compute_ece_equal_mass(self, confidences, correct, n_bins):
        """
        Compute ECE using equal-mass binning with specified number of bins
        """
        if n_bins <= 1:
            return 0.0
            
        # Sort by confidence
        sorted_indices = np.argsort(confidences)
        sorted_confidences = confidences[sorted_indices]
        sorted_correct = correct[sorted_indices]
        
        # Create equal-mass bins
        samples_per_bin = len(confidences) // n_bins
        remainder = len(confidences) % n_bins
        
        ece = 0.0
        start_idx = 0
        
        for i in range(n_bins):
            # Some bins get one extra sample if there's a remainder
            bin_size = samples_per_bin + (1 if i < remainder else 0)
            end_idx = start_idx + bin_size
            
            if start_idx >= len(sorted_indices):
                break
                
            # Get data for this bin
            bin_confidences = sorted_confidences[start_idx:end_idx]
            bin_correct = sorted_correct[start_idx:end_idx]
            
            if len(bin_confidences) > 0:
                # Compute bin statistics
                avg_confidence = np.mean(bin_confidences)
                avg_accuracy = np.mean(bin_correct)
                bin_weight = len(bin_confidences) / len(confidences)
                
                # Add to ECE
                ece += bin_weight * (abs(avg_confidence - avg_accuracy) ** self.p)
            
            start_idx = end_idx
        
        return ece ** (1.0 / self.p)