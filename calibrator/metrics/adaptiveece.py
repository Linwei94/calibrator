import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveECE(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECE, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels, softmaxes=None, confidences=None, predictions=None):
        if softmaxes is None:
            softmaxes = F.softmax(logits, dim=1)
        if confidences is None:
            confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].float().mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece