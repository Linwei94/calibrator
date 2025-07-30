import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ECEDebiased(nn.Module):
    '''
    Compute ECE Debiased (Expected Calibration Error with bias correction)
    Based on "Verified uncertainty calibration" by Brocker, 2012; Ferro and Fricker, 2012
    '''
    def __init__(self, n_bins=15, p=2, resamples=1000):
        '''
        Args:
            n_bins: int
                The number of bins to use for the calibration
            p: int
                The norm to use (1 for ECE, 2 for L2 calibration error)
            resamples: int
                Number of resamples for normal debiased estimator (used when p != 2)
        '''
        super(ECEDebiased, self).__init__()
        self.n_bins = n_bins
        self.p = p
        self.resamples = resamples

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
            ece_debiased: float
                The debiased ECE value
        '''
        if softmaxes is None:
            softmaxes = F.softmax(logits, dim=1)
            
        # Convert to NumPy arrays if they're PyTorch tensors
        if torch.is_tensor(softmaxes):
            probs = softmaxes.detach().cpu().numpy()
        else:
            probs = softmaxes
            
        if torch.is_tensor(labels):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels
            
        # Get top-label confidences and predictions
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        correct = (predictions == labels_np).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Bin the data
        binned_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            if np.sum(in_bin) > 0:
                bin_confidences = confidences[in_bin]
                bin_correct = correct[in_bin]
                bin_data = list(zip(bin_confidences, bin_correct))
                binned_data.append(bin_data)
            else:
                binned_data.append([])
        
        # Apply debiasing
        if self.p == 2:
            return self._unbiased_l2_ce(binned_data)
        else:
            return self._normal_debiased_ce(binned_data, power=self.p)
    
    def _unbiased_l2_ce(self, binned_data):
        """
        Compute unbiased L2 calibration error
        """
        def bin_error(data):
            if len(data) < 2:
                return 0.0
            # Compute biased estimate
            pred_probs = [x[0] for x in data]
            labels = [x[1] for x in data]
            mean_pred = np.mean(pred_probs)
            mean_label = np.mean(labels)
            biased_estimate = abs(mean_pred - mean_label) ** 2
            
            # Compute variance correction
            variance = mean_label * (1.0 - mean_label) / (len(data) - 1.0)
            return biased_estimate - variance
        
        # Compute bin probabilities
        total_samples = sum(len(bin_data) for bin_data in binned_data)
        if total_samples == 0:
            return 0.0
            
        bin_probs = [len(bin_data) / total_samples for bin_data in binned_data]
        bin_errors = [bin_error(bin_data) for bin_data in binned_data]
        
        # Weighted sum of bin errors
        unbiased_square_ce = np.dot(bin_probs, bin_errors)
        return max(unbiased_square_ce, 0.0) ** 0.5
    
    def _normal_debiased_ce(self, binned_data, power=1):
        """
        Compute normal debiased calibration error using resampling
        """
        # Filter out empty bins and bins with only one sample
        non_empty_bins = [bin_data for bin_data in binned_data if len(bin_data) > 1]
        
        if len(non_empty_bins) == 0:
            return 0.0
        
        # Check if any bin has <= 1 sample
        bin_sizes = [len(bin_data) for bin_data in non_empty_bins]
        if min(bin_sizes) <= 1:
            # Fall back to plugin estimator
            return self._plugin_ce(binned_data, power)
        
        # Compute bin statistics
        label_means = []
        model_vals = []
        for bin_data in non_empty_bins:
            labels = [x[1] for x in bin_data]
            preds = [x[0] for x in bin_data]
            label_means.append(np.mean(labels))
            model_vals.append(np.mean(preds))
        
        label_means = np.array(label_means)
        model_vals = np.array(model_vals)
        bin_sizes = np.array(bin_sizes)
        
        # Compute standard deviations
        label_stddev = np.sqrt(label_means * (1 - label_means) / bin_sizes)
        
        # Compute plugin estimate
        ce = self._plugin_ce(binned_data, power)
        
        # Compute bin probabilities
        total_samples = sum(len(bin_data) for bin_data in binned_data)
        bin_probs = np.array([len(bin_data) / total_samples for bin_data in non_empty_bins])
        
        # Resample to estimate bias
        resampled_ces = []
        for _ in range(self.resamples):
            label_samples = np.random.normal(loc=label_means, scale=label_stddev)
            diffs = np.power(np.abs(label_samples - model_vals), power)
            cur_ce = np.power(np.dot(bin_probs, diffs), 1.0 / power)
            resampled_ces.append(cur_ce)
        
        mean_resampled = np.mean(resampled_ces)
        bias_corrected_ce = 2 * ce - mean_resampled
        return bias_corrected_ce
    
    def _plugin_ce(self, binned_data, power):
        """
        Compute standard plugin calibration error estimate
        """
        def bin_error(data):
            if len(data) == 0:
                return 0.0
            pred_probs = [x[0] for x in data]
            labels = [x[1] for x in data]
            mean_pred = np.mean(pred_probs)
            mean_label = np.mean(labels)
            return abs(mean_pred - mean_label) ** power
        
        # Compute bin probabilities
        total_samples = sum(len(bin_data) for bin_data in binned_data)
        if total_samples == 0:
            return 0.0
            
        bin_probs = [len(bin_data) / total_samples for bin_data in binned_data]
        bin_errors = [bin_error(bin_data) for bin_data in binned_data]
        
        return np.dot(bin_probs, bin_errors) ** (1.0 / power)