import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ECE(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        '''
        Args:
            n_bins: int
                The number of bins to use for the calibration
        '''
        super(ECE, self).__init__()
        self.n_bins = n_bins

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
            ece: float
                The ECE value
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
            
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels_np)
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece