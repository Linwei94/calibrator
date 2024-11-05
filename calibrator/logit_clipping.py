'''
Code to perform logit clipping.
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from .metrics import ECE
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

    def logit_clipping(self, logits):
        """
        Perform logit clipping on logits
        """
        return torch.clamp(logits, max=self.logit_clip, min=-self.logit_clip)
    
    def fit(self, val_logits, val_labels, cross_validate='ece'):
        """
        Tune the logit clipping threshold of the model (using the validation set) with cross-validation on ECE or NLL
        """
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECE().cuda()

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        C_opt_nll = float("inf")
        C_opt_ece = float("inf")
        C = 0.01
        for _ in range(1000):
            self.logit_clip = C
            self.cuda()
            after_clipping_nll = nll_criterion(self.logit_clipping(val_logits), val_labels)
            after_clipping_ece = ece_criterion(self.logit_clipping(val_logits), val_labels)
            if nll_val > after_clipping_nll:
                C_opt_nll = C
                nll_val = after_clipping_nll

            if ece_val > after_clipping_ece:
                C_opt_ece = C
                ece_val = after_clipping_ece
            C += 0.01

        if cross_validate == 'ece':
            self.logit_clip = C_opt_ece
        else:
            self.logit_clip = C_opt_nll

        return self.logit_clip


    def get_logit_clip(self):
        return self.logit_clip