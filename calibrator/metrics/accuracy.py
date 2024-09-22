# accuracy.py
import torch
import torch.nn as nn

class Accuracy(nn.Module):
    '''
    Compute accuracy as a loss (1 - accuracy) for optimization purposes
    '''
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, logits=None, labels=None, softmaxes=None, predictions=None, **kwargs):
        if predictions is None:
            if logits is not None:
                predictions = logits.argmax(dim=1)
            elif softmaxes is not None:
                predictions = softmaxes.argmax(dim=1)
            else:
                raise ValueError("Either logits or softmaxes must be provided")
        correct = predictions.eq(labels)
        accuracy = correct.float().mean()
        return accuracy.item()
