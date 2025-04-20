# brier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BrierLoss(nn.Module):
    """
    Compute Brier Score Loss
    
    The Brier score is a proper scoring function for probabilistic predictions.
    It measures the mean squared difference between predicted probabilities and actual outcomes.
    """
    def __init__(self):
        super(BrierLoss, self).__init__()

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
        return torch.mean(torch.sum((outputs - targets) ** 2, dim=1)) 