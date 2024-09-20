'''
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
'''
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from .metrics import ECE

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = 1.0

    def fit(self, model, valid_loader, search_criteria='ece', verbose=False):
        '''
        Search the optimal temperature value for the calibration on validation set and set the optimal temperature value to self.temperature
        model: nn.Module, the model to be calibrated
        valid_loader: torch.utils.data.DataLoader, the validation data loader
        search_criteria: str, 'ece' or 'nll'
        verbose: bool, whether to print the search process
        '''
        self.cuda()
        model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECE().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            self.cuda()
            after_temperature_nll = nll_criterion(self.calibrate(logits), labels)
            after_temperature_ece = ece_criterion(self.calibrate(logits), labels)
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if search_criteria == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll
        self.cuda()

        return self.temperature

    def calibrate(self, logits):
        return logits / self.temperature
