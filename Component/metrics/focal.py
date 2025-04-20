# focal.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Compute Focal Loss
    
    Focal Loss is a modification of standard cross entropy loss that focuses training on hard examples
    by down-weighting easy examples. It helps address class imbalance by reducing the relative loss
    for well-classified examples and putting more focus on hard, misclassified examples.
    """
    def __init__(self, gamma=2.0):
        """
        Initialize Focal Loss
        
        Args:
            gamma (float): Focusing parameter that reduces the relative loss for well-classified examples
                and puts more focus on hard, misclassified examples. Default is 2.0.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        """
        Compute Focal Loss
        
        Args:
            logits (torch.Tensor, optional): Raw logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,) - class indices
                or shape (batch_size, num_classes) - one-hot encoded vectors
            softmaxes (torch.Tensor, optional): Predicted probabilities of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Focal loss
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
        
        # Compute focal loss
        focal_weight = (1 - pred_probs) ** self.gamma
        return torch.mean(-focal_weight * torch.log(pred_probs + 1e-7)) 