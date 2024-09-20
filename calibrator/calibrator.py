# implementation of the parent calibrator class
import torch.nn as nn
from abc import ABC, abstractmethod
class Calibrator(nn.Module):

    def __init__(self):
        super(Calibrator, self).__init__()
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def calibrate(self, test_logits, **kwargs):
        pass
    
    