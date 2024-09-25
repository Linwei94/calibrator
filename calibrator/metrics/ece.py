import torch
import torch.nn as nn
import torch.nn.functional as F


class ECE(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        '''
        Args:
            n_bins: int
                The number of bins to use for the calibration
        '''
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits=None, labels=None, softmaxes=None):
        '''
        args:
            logits: torch.Tensor
                The logits to calibrate, the output of the model before softmax layer
            labels: torch.Tensor
                The labels of the test data
            softmaxes: torch.Tensor
                The softmaxes of the test data, if None, compute the softmaxes from logits

        Returns:
            ece: float
                The ECE value
        '''
        if softmaxes is None:
            softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=labels.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()