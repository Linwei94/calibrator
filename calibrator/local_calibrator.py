import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .calibrator import Calibrator
from .metrics import ECE

# Calibration error scores in the form of loss metrics
class LocalCalibrator(Calibrator):
    def __init__(self, aggregation='consistency', num_samples=1000, noise_type='gaussian'):
        super(LocalCalibrator, self).__init__()
        '''
        aggregation: str
            The aggregation method to use. Options are 'consistency' and 'mean'.
            'consistency' means the majority class is the final prediction, and the confidence is the ratio of the majority class.
            'mean' means the final prediction is the mean of the softmax output
        num_samples: int
            The number of samples to use for calibration
        noise_type: str
            The type of noise to use. Options are 'gaussian' and 'uniform'
        '''

        self.num_samples = num_samples
        self.aggregation = aggregation
        self.noise_type = noise_type
        self.eps = None # optimal epsilon value

    def fit(self, val_logits, val_labels, search_criteria='ece', verbose=False):
        '''
        Search the optimal epsilon value for the calibration on validation set and set the optimal epsilon value to self.eps
        '''
        if search_criteria == 'ece':
            criterion = ECE().cuda()
        elif search_criteria == 'nll':
            criterion = nn.NLLLoss()

        min_loss = float('inf')
        eps_search_space = np.linspace(0, 10, 100)
        for eps in eps_search_space:
            calibrated_probability = self.calibrate(val_logits, eps=eps)
            loss = criterion(labels=val_labels, softmaxes=calibrated_probability)
            if verbose:
                print('Epsilon: {}, {}: {}'.format(eps, search_criteria, loss))
            if self.eps is None or loss < min_loss:
                self.eps = eps
                min_loss = loss
        if verbose:
            print('--'*20)
            print('Optimal epsilon: {}, {}: {}'.format(self.eps, search_criteria, min_loss))
        return self.eps

    def calibrate(self, test_logits, eps=None):
        '''
        test_logits: torch.Tensor
            The logits to calibrate, the output of the model before softmax layer
        eps: float
            The epsilon value for noise, if None, use the self.eps value

        Returns:
        calibrated_probability: torch.Tensor
            The calibrated probability of the prediction
            Note that this method can only output probabilities (similar to softmax), not logits
        '''
        if eps is None:
            eps = self.eps

        device = test_logits.device
        num_samples = test_logits.size(0)
        num_classes = test_logits.size(1)
        softmaxes_mode_counts = torch.zeros(num_samples, num_classes, dtype=torch.int32).to(device)
        softmax_sum = torch.zeros(num_samples, num_classes).to(device)
        noise = torch.zeros_like(test_logits, device=device)

        for i in range(self.num_samples):
            # set noise
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(test_logits) * eps
            elif self.noise_type == 'uniform':
                noise.uniform_(-eps, eps)

            logits = (test_logits + noise).to(device)

            if self.aggregation == 'consistency':
                preds = logits.argmax(dim=1)
                softmaxes_mode_counts += F.one_hot(preds, num_classes=num_classes).int().to(device)

            elif self.aggregation == 'mean':
                softmaxes = F.softmax(logits, dim=1)
                softmax_sum += softmaxes

        if self.aggregation == 'consistency':
            return softmaxes_mode_counts / self.num_samples
        elif self.aggregation == 'mean':
            return softmax_sum / self.num_samples

    def get_eps(self):
        return self.eps

if __name__ == '__main__':
    # todo: add test code
    pass