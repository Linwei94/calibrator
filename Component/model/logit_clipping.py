'''
Code to perform logit clipping.
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from ..metrics import ECE, Accuracy
from .calibrator import Calibrator

class LogitClippingCalibrator(Calibrator):
    """
    A thin decorator, which wraps a model with logit clipping.
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(LogitClippingCalibrator, self).__init__()
        self.logit_clip = float("inf")


    def calibrate(self, logits, return_logits=False):
        if return_logits:
            return self.logit_clipping(logits)
        else:
            return F.softmax(self.logit_clipping(logits), dim=1)

    def logit_clipping(self, logits, clip_value=None):
        """
        Perform logit clipping on logits
        """
        if clip_value is None:
            clip_value = self.logit_clip
        return torch.clamp(logits, max=clip_value, min=-clip_value)

    def fit(self, val_logits, val_labels, cross_validate='ece'):
        """
        Tune the logit clipping threshold of the model (using the validation set) with cross-validation on ECE or NLL
        """
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECE().cuda()
        accuracy_criterion = Accuracy().cuda()
        before_clipping_acc = accuracy_criterion(val_logits, val_labels)

        nll_val_opt = float("inf")
        ece_val_opt = float("inf")
        C_opt_nll = float("inf")
        C_opt_ece = float("inf")
        C = max(torch.quantile(val_logits, 0.80).item(), 0.01)

        for _ in range(2000):
            clipped_logits = self.logit_clipping(val_logits, C)
            after_clipping_nll = nll_criterion(clipped_logits, val_labels)
            after_clipping_ece = ece_criterion(clipped_logits, val_labels)
            after_clipping_acc = accuracy_criterion(clipped_logits, val_labels)
            if (after_clipping_nll < nll_val_opt) and (after_clipping_acc > before_clipping_acc*0.95):
                C_opt_nll = C
                nll_val_opt = after_clipping_nll

            if (after_clipping_ece < ece_val_opt) and (after_clipping_acc > before_clipping_acc*0.95):
                C_opt_ece = C
                ece_val_opt = after_clipping_ece
            C += 0.01

        if cross_validate == 'ece':
            self.logit_clip = C_opt_ece
        else:
            self.logit_clip = C_opt_nll

        return self.logit_clip


    def get_logit_clip(self):
        return self.logit_clip