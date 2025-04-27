# implementation of the parent calibrator class
import torch.nn as nn
from abc import ABC, abstractmethod
class Calibrator(nn.Module):

    def __init__(self):
        super(Calibrator, self).__init__()
        pass

    @abstractmethod
    def fit(self, val_logits, val_labels, **kwargs):
        '''
        Fit the calibrator on the validation set and return the optimal hyperparameter value
        '''
        pass

    @abstractmethod
    def calibrate(self, test_logits, return_logits=False, **kwargs):
        '''
        Calibrate the logits and return the calibrated probabilities or logits
        '''
        pass