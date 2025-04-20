# soft_ece.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftECE(nn.Module):
    """
    Compute Soft Expected Calibration Error Loss
    
    SoftECE is a differentiable approximation of the Expected Calibration Error (ECE).
    It measures the difference between predicted confidence and accuracy in a differentiable way.
    """
    def __init__(self, num_bins=15):
        """
        Initialize SoftECE
        
        Args:
            num_bins (int): Number of bins to use for computing the soft ECE. Default is 15.
        """
        super(SoftECE, self).__init__()
        self.num_bins = num_bins

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        """
        Compute Soft Expected Calibration Error Loss
        
        Args:
            logits (torch.Tensor, optional): Raw logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,) - class indices
                or shape (batch_size, num_classes) - one-hot encoded vectors
            softmaxes (torch.Tensor, optional): Predicted probabilities of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Soft ECE loss
        """
        # Get predicted probabilities
        if logits is not None:
            outputs = F.softmax(logits, dim=1)
        elif softmaxes is not None:
            outputs = softmaxes
        else:
            raise ValueError("Either logits or softmaxes must be provided")
        
        # Get the predicted probabilities for the true classes
        if len(labels.shape) > 1:
            # If targets are one-hot encoded, get the indices
            _, target_indices = torch.max(labels, dim=1)
            # Get the predicted probabilities for the true classes
            pred_probs = torch.gather(outputs, 1, target_indices.unsqueeze(1)).squeeze(1)
        else:
            # If targets are class indices
            pred_probs = torch.gather(outputs, 1, labels.unsqueeze(1)).squeeze(1)
        
        # Get the maximum predicted probability for each sample
        max_probs, _ = torch.max(outputs, dim=1)
        
        # Compute soft ECE
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=pred_probs.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute soft bin assignments
        bin_width = 1.0 / self.num_bins
        bin_indices = torch.floor(max_probs / bin_width).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Compute soft bin counts
        bin_counts = torch.zeros(self.num_bins, device=pred_probs.device)
        bin_confidences = torch.zeros(self.num_bins, device=pred_probs.device)
        bin_accuracies = torch.zeros(self.num_bins, device=pred_probs.device)
        
        for i in range(self.num_bins):
            mask = (bin_indices == i)
            if mask.any():
                bin_counts[i] = mask.float().sum()
                bin_confidences[i] = max_probs[mask].mean()
                bin_accuracies[i] = pred_probs[mask].mean()
        
        # Compute soft ECE
        ece = torch.sum(bin_counts * torch.abs(bin_confidences - bin_accuracies)) / bin_counts.sum()
        return ece 