# cross_entropy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    Compute Cross Entropy Loss
    
    Cross Entropy Loss is a standard loss function for classification tasks.
    It measures the performance of a classification model whose output is a probability value between 0 and 1.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        """
        Compute Cross Entropy Loss
        
        Args:
            logits (torch.Tensor, optional): Raw logits of shape (batch_size, num_classes)
            labels (torch.Tensor): True labels of shape (batch_size,) - class indices
                or shape (batch_size, num_classes) - one-hot encoded vectors
            softmaxes (torch.Tensor, optional): Predicted probabilities of shape (batch_size, num_classes)
            
        Returns:
            torch.Tensor: Cross Entropy loss
        """
        # For CrossEntropyLoss, we need logits and class indices
        if logits is not None:
            # If labels are one-hot encoded, convert to class indices
            if len(labels.shape) > 1:
                _, class_indices = torch.max(labels, dim=1)
                return self.criterion(logits, class_indices)
            else:
                return self.criterion(logits, labels)
        elif softmaxes is not None:
            # If we have softmaxes, we need to compute cross entropy manually
            if len(labels.shape) > 1:
                # If labels are one-hot encoded, convert to class indices
                _, class_indices = torch.max(labels, dim=1)
                # Compute log probabilities
                log_probs = torch.log(softmaxes + 1e-7)
                # Gather log probabilities corresponding to the true labels
                log_probs_true = log_probs.gather(1, class_indices.unsqueeze(1)).squeeze(1)
                return -log_probs_true.mean()
            else:
                # Compute log probabilities
                log_probs = torch.log(softmaxes + 1e-7)
                # Gather log probabilities corresponding to the true labels
                log_probs_true = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                return -log_probs_true.mean()
        else:
            raise ValueError("Either logits or softmaxes must be provided") 