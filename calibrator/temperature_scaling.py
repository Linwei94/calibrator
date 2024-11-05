import torch
from torch import nn, optim
from torch.nn import functional as F

from .calibrator import Calibrator

class TemperatureScalingCalibrator(Calibrator):
    def __init__(self):
        super(TemperatureScalingCalibrator, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def calibrate(self, logits, return_logits=False):
        if return_logits:
            return self.temperature_scale(logits)
        else:
            return F.softmax(self.temperature_scale(logits), dim=1)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def fit(self, val_logits, val_labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(val_logits), val_labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        return self.temperature.item()

    def get_temperature(self):
        return self.temperature.item()
