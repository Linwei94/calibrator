# soft_ece.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftECE(nn.Module):
    """
    Soft-binned Expected Calibration Error loss (using a Gaussian kernel).
    Uses "soft binning" to compute ECE, making gradients smoother and more propagatable for confidence.
    """
    def __init__(self, n_bins=15, sigma=0.05, eps=1e-6):
        """
        Args:
            n_bins: Number of bins to divide the [0,1] interval into
            sigma: Bandwidth (standard deviation) of the Gaussian kernel
            eps: Small constant to avoid division by zero
        """
        super(SoftECE, self).__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.eps = eps
        
        # Using bin centers instead of boundaries
        # Alternative arrangements can be used if strict alignment with [0,1] endpoints is desired
        self.register_buffer('bin_centers', torch.linspace(1/(2*self.n_bins), 
                                          1 - 1/(2*self.n_bins), 
                                          self.n_bins))
    
    def forward(self, logits, labels):
        """
        Args:
            logits: Network output of shape [batch_size, num_classes]
            targets: Integer ground truth labels of shape [batch_size]
        
        Returns:
            soft ECE loss (can participate in backpropagation)
        """
        # Ensure targets is on the same device as logits
        targets = labels.to(logits.device)
        
        # 1) Convert logits to probability distribution
        probs = F.softmax(logits, dim=1)  # [B, C]
        
        # 2) Get maximum confidence for each sample
        confidences, predictions = torch.max(probs, dim=1)  # [B]
        
        # 3) Calculate if prediction is correct
        accuracies = (predictions == targets).float()       # [B]
        
        # 4) Calculate soft assignment weights for each bin (Gaussian kernel)
        #    shape: [batch_size, n_bins]
        #    weights[j, i] = exp(- (confidences[j] - bin_centers[i])^2 / (2*sigma^2))
        #    Will be normalized later
        diff = confidences.unsqueeze(1) - self.bin_centers.unsqueeze(0)  # [B, n_bins]
        weights = torch.exp(-0.5 * (diff**2) / (self.sigma**2))     # [B, n_bins]
        
        # Normalize: For each sample, ensure sum of weights across bins = 1
        # This gives each sample a distribution (soft assignment) across all bins
        weights_sum = weights.sum(dim=1, keepdim=True) + self.eps   # [B, 1]
        weights_norm = weights / weights_sum                        # [B, n_bins]
        
        # 5) Calculate "average confidence" and "average accuracy" for each bin
        #    avg_confidence_in_bin[i] = \sum_j (weights_norm[j,i] * confidences[j]) / \sum_j (weights_norm[j,i])
        #    Implemented using column-wise summation
        weighted_confidence = weights_norm * confidences.unsqueeze(1)   # [B, n_bins]
        sum_conf_in_bin = weighted_confidence.sum(dim=0)                # [n_bins]
        sum_weights_in_bin = weights_norm.sum(dim=0)                    # [n_bins]
        avg_confidence_in_bin = sum_conf_in_bin / (sum_weights_in_bin + self.eps)
        
        # Similarly calculate average accuracy
        # Ensure accuracies shape is compatible with weights_norm
        weighted_accuracy = weights_norm * accuracies.unsqueeze(1)      # [B, n_bins]
        sum_acc_in_bin = weighted_accuracy.sum(dim=0)                   # [n_bins]
        avg_accuracy_in_bin = sum_acc_in_bin / (sum_weights_in_bin + self.eps)
        
        # 6) Calculate ECE: sum of (error * bin weight proportion) for each bin
        #    prop_in_bin = \sum_j weights_norm[j, i] / batch_size
        prop_in_bin = sum_weights_in_bin / confidences.size(0)          # [n_bins]
        
        # soft ece = sum_i ( | avg_conf_in_bin[i] - avg_acc_in_bin[i] | * prop_in_bin[i] )
        ece_per_bin = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
        soft_ece = torch.sum(ece_per_bin * prop_in_bin)
        
        return soft_ece
