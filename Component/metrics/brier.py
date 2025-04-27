# brier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BrierLoss(nn.Module):
    """
    Compute Brier Score Loss
    
    The Brier score is a proper scoring function for probabilistic predictions.
    It measures the mean squared difference between predicted probabilities and actual outcomes.
    The smaller the Brier score loss, the better.
    
    For N observations labeled from C possible classes, the Brier score is defined as:
    (1/N) * sum_{i=1}^N sum_{c=1}^C (y_ic - p_ic)^2
    
    where y_ic is 1 if observation i belongs to class c, otherwise 0
    and p_ic is the predicted probability for observation i to belong to class c.
    
    The Brier score ranges between [0, 2].
    For binary classification, it is often scaled by 1/2 to range between [0, 1].
    """
    def __init__(self, scale_by_half=True):
        """
        Initialize Brier Loss
        
        Args:
            scale_by_half (bool): If True, scale the Brier score by 1/2 to lie in [0, 1]
                instead of [0, 2]. Default is True.
        """
        super(BrierLoss, self).__init__()
        self.scale_by_half = scale_by_half

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        """
        Compute Brier loss
        
        Args:
            logits (torch.Tensor, optional): Raw logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,) - class indices
                or shape (batch_size, num_classes) - one-hot encoded vectors
            softmaxes (torch.Tensor, optional): Predicted probabilities of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Brier loss
        """
        # Get predicted probabilities
        if logits is not None:
            outputs = F.softmax(logits, dim=1)
        elif softmaxes is not None:
            outputs = softmaxes
        else:
            raise ValueError("Either logits or softmaxes must be provided")
        
        # Convert labels to one-hot if they're not already
        if len(labels.shape) == 1:
            # Create one-hot encoded vectors from class indices
            one_hot = torch.zeros(labels.size(0), outputs.size(1), device=labels.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            targets = one_hot
        else:
            targets = labels
        
        # Compute Brier loss
        brier_score = torch.mean(torch.sum((outputs - targets) ** 2, dim=1))
        
        # Scale by half if requested
        if self.scale_by_half:
            brier_score *= 0.5
            
        return brier_score 