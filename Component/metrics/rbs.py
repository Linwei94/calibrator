# rbs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBS(nn.Module):
    """
    Compute Root Brier Score (RBS)
    
    The Root Brier Score is the square root of the Brier Score.
    It serves as a root calibration upper bound for evaluating model calibration.
    
    For N observations labeled from C possible classes, the Brier score is defined as:
    BS = (1/N) * sum_{i=1}^N sum_{c=1}^C (y_ic - p_ic)^2
    
    The Root Brier Score is then:
    RBS = sqrt(BS)
    
    where y_ic is 1 if observation i belongs to class c, otherwise 0
    and p_ic is the predicted probability for observation i to belong to class c.
    
    The RBS ranges between [0, sqrt(2)] ≈ [0, 1.414].
    Lower values indicate better calibration.
    
    Reference:
    Gruber and Buettner - 2024 - Better Uncertainty Calibration via Proper Scores 
    for Classification and Beyond
    """
    def __init__(self):
        """
        Initialize RBS metric
        """
        super(RBS, self).__init__()

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        """
        Compute Root Brier Score
        
        Args:
            logits (torch.Tensor, optional): Raw logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,) - class indices
                or shape (batch_size, num_classes) - one-hot encoded vectors
            softmaxes (torch.Tensor, optional): Predicted probabilities of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Root Brier Score (scalar)
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
        
        # Compute Brier Score
        brier_score = torch.mean(torch.sum((outputs - targets) ** 2, dim=1))
        
        # Take square root to get Root Brier Score
        rbs = torch.sqrt(brier_score)
            
        return rbs

