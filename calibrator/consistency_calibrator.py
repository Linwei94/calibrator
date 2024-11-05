import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .calibrator import Calibrator
from .metrics import ECE

# Calibration error scores in the form of loss metrics
class ConsistencyCalibrator(Calibrator):
    def __init__(self, aggregation='consistency', num_samples=1000, noise_type=None):
        super(ConsistencyCalibrator, self).__init__()
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

        assert aggregation in ['consistency', 'mean'], "Invalid aggregation method"
        assert num_samples > 0, "Invalid number of samples"
        if noise_type:
            print("Warning: The 'noise_type' parameter is deprecated and will be removed in future versions. ConsistencyCalibrator will search both Gaussian and Uniform noise and set the noise type in fit function.")


        self.num_samples = num_samples
        self.aggregation = aggregation
        self.eps = None # optimal epsilon value

    def fit(self, val_logits, val_labels, search_criteria='ece', verbose=False, search_method='fine_grained', return_loss=False):
        '''
        Search the optimal epsilon value for the calibration on validation set and set the optimal epsilon value to self.eps, will search both Gaussian and Uniform noise and set the noise type
        '''
        g_eps, g_loss = self._fit(val_logits, val_labels, search_criteria, verbose, search_method, noise_type='gaussian')
        u_eps, u_loss = self._fit(val_logits, val_labels, search_criteria, verbose, search_method, noise_type='uniform')

        if g_loss < u_loss:
            self.noise_type = 'gaussian'
            self.eps = g_eps
            min_loss = g_loss
        else:
            self.noise_type = 'uniform'
            self.eps = u_eps
            min_loss = u_loss

        if return_loss:
            return self.eps, min_loss
        else:
            return self.eps


    def _fit(self, val_logits, val_labels, search_criteria='ece', verbose=False, search_method='fine_grained', noise_type='guassian'):
        '''
        Search the optimal epsilon value for the calibration on validation set and set the optimal epsilon value to self.eps
        '''

        if search_criteria == 'ece':
            criterion = ECE().cuda()
        elif search_criteria == 'nll':
            criterion = nn.NLLLoss()

        min_loss = float('inf')
        if search_method == 'grid_search':
            eps_search_space = np.linspace(0, 10, 100)
            for eps in eps_search_space:
                calibrated_probability = self.calibrate(val_logits, eps=eps, noise_type=noise_type)
                loss = criterion(labels=val_labels, softmaxes=calibrated_probability)
                if verbose:
                    print('Epsilon: {}, {}: {}'.format(eps, search_criteria, loss))
                if self.eps is None or loss < min_loss:
                    self.eps = eps
                    min_loss = loss

        elif search_method == 'fine_grained':
            fine_granularity = 20 # the search space is divided by 2^fine_granularity times
            fine_granularity_level = 3 # the search space is divided by 10^fine_granularity_level times
            eps_search_space = np.linspace(0, 10, fine_granularity)
            for eps in eps_search_space:
                calibrated_probability = self.calibrate(val_logits, eps=eps, noise_type=noise_type)
                loss = criterion(labels=val_labels, softmaxes=calibrated_probability)
                if verbose:
                    print('Epsilon: {}, {}: {}'.format(eps, search_criteria, loss))
                if self.eps is None or loss < min_loss:
                    self.eps = eps
                    min_loss = loss
            
            for i in range(fine_granularity_level):
                eps_search_space = np.linspace(self.eps - 1/10**i, self.eps + 1/10**i, fine_granularity)
                for eps in eps_search_space:
                    calibrated_probability = self.calibrate(val_logits, eps=eps, noise_type=noise_type)
                    loss = criterion(labels=val_labels, softmaxes=calibrated_probability)
                    if verbose:
                        print('Epsilon: {}, {}: {}'.format(eps, search_criteria, loss))
                    if self.eps is None or loss < min_loss:
                        self.eps = eps
                        min_loss = loss

        if verbose:
            print('--'*20)
            print('Optimal epsilon: {}, {}: {}'.format(self.eps, search_criteria, min_loss))
        return self.eps, min_loss

    def calibrate(self, test_logits, eps=None, noise_type=None, return_logits=False):
        '''
        test_logits: torch.Tensor
            The logits to calibrate, the output of the model before softmax layer
        eps: float
            The epsilon value for noise, if None, use the self.eps value
        noise_type: str
            The type of noise to use, if None, use the self._noise_type value
        return_logits: bool
            Whether to return the logits or the probabilities, cannot be True

        Returns:
        calibrated_probability: torch.Tensor
            The calibrated probability of the prediction
            Note that this method can only output probabilities (similar to softmax), not logits
        '''
        assert not return_logits, "ConsistencyCalibrator cannot return logits"

        if eps is None:
            eps = self.eps
        if noise_type is None:
            noise_type = self.noise_type

        device = test_logits.device
        num_samples = test_logits.size(0)
        num_classes = test_logits.size(1)
        softmaxes_mode_counts = torch.zeros(num_samples, num_classes, dtype=torch.int32).to(device)
        softmax_sum = torch.zeros(num_samples, num_classes).to(device)
        noise = torch.zeros_like(test_logits, device=device)
        # eps = (test_logits.max(dim=-1)[0] / 0.85).unsqueeze(1).expand_as(test_logits)

        for i in range(self.num_samples):
            # set noise
            if noise_type == 'gaussian':
                noise = torch.randn_like(test_logits) * eps
            elif noise_type == 'uniform':
                # noise.uniform_(-eps, eps)
                noise = torch.rand_like(test_logits) * eps

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

    def get_noise_type(self):
        return self.noise_type

if __name__ == '__main__':
    # todo: add test code
    pass
