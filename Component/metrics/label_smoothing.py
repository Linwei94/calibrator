# label_smoothing.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Compute Label Smoothing Loss
    
    Label smoothing is a regularization technique that prevents a model from becoming overconfident
    by smoothing the target labels. Instead of using hard 0/1 labels, it uses soft labels with a
    small probability mass distributed across all classes.
    """
    def __init__(self, alpha=0.1):
        """
        Initialize Label Smoothing Loss
        
        Args:
            alpha (float): Smoothing parameter that controls how much probability mass is distributed
                to non-target classes. Default is 0.1.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        """
        Compute Label Smoothing Loss
        
        Args:
            logits (torch.Tensor, optional): Raw logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,) - class indices
                or shape (batch_size, num_classes) - one-hot encoded vectors
            softmaxes (torch.Tensor, optional): Predicted probabilities of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Label smoothing loss
        """
        # Get predicted probabilities
        if logits is not None:
            outputs = F.softmax(logits, dim=1)
        elif softmaxes is not None:
            outputs = softmaxes
        else:
            raise ValueError("Either logits or softmaxes must be provided")
        
        num_classes = outputs.size(1)
        
        # Convert targets to one-hot if they're not already
        if len(labels.shape) == 1:
            # Create one-hot encoded vectors from class indices
            one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            targets = one_hot
        else:
            targets = labels
        
        # Apply label smoothing
        smooth_targets = targets * (1 - self.alpha) + self.alpha / num_classes
        
        # Compute cross-entropy with smoothed labels
        log_probs = torch.log(outputs + 1e-7)
        return -torch.sum(smooth_targets * log_probs, dim=1).mean() 