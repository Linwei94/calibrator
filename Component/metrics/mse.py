# mse.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    """
    Compute Mean Squared Error Loss
    
    MSE Loss measures the average squared difference between predicted probabilities and actual outcomes.
    It's a common loss function for regression tasks and can also be used for classification.
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        """
        Compute Mean Squared Error Loss
        
        Args:
            logits (torch.Tensor, optional): Raw logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,) - class indices
                or shape (batch_size, num_classes) - one-hot encoded vectors
            softmaxes (torch.Tensor, optional): Predicted probabilities of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: MSE loss
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
        
        # Compute MSE loss
        return self.criterion(outputs, targets) 