# nll.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLL(nn.Module):
    """
    Compute Negative Log Likelihood (NLL) Loss
    """
    def __init__(self):
        super(NLL, self).__init__()

    def forward(self, logits=None, labels=None, softmaxes=None, **kwargs):
        if logits is not None:
            log_probs = F.log_softmax(logits, dim=1)
        elif softmaxes is not None:
            log_probs = torch.log(softmaxes + 1e-12)  # Adding epsilon to prevent log(0)
        else:
            raise ValueError("Either logits or softmaxes must be provided")

        # Gather log probabilities corresponding to the true labels
        log_probs_true = log_probs.gather(1, labels.unsqueeze(1)).squeeze()
        nll_loss = -log_probs_true.mean()
        return nll_loss.item()
