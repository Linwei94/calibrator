# focal.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """
    Compute Focal Loss
    
    Focal Loss is a modification of standard cross entropy loss that focuses training on hard examples
    by down-weighting easy examples. It helps address class imbalance by reducing the relative loss
    for well-classified examples and putting more focus on hard, misclassified examples.
    """
    def __init__(self, gamma=0, size_average=False):
        """
        Initialize Focal Loss
        
        Args:
            gamma (float): Focusing parameter that reduces the relative loss for well-classified examples
                and puts more focus on hard, misclassified examples. Default is 0.0.
            size_average (bool): If True, the loss is averaged over the batch. If False, the loss is summed.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum() 