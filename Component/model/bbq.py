import torch
import numpy as np
from torch.nn import functional as F
from scipy import optimize
from scipy.special import gammaln

from .calibrator import Calibrator

class BBQCalibrator(Calibrator):
    def __init__(self, score_type='max_prob', n_bins_max=20):
        """
        Initialize the Bayesian Binning into Quantiles (BBQ) calibrator.
        
        Args:
            score_type (str): Type of score to use ('max_prob' or 'entropy')
            n_bins_max (int): Maximum number of bins to consider
        """
        super(BBQCalibrator, self).__init__()
        self.score_type = score_type
        self.n_bins_max = n_bins_max
        self.optimal_n_bins = None
        self.bin_boundaries = None
        self.bin_accuracies = None

    def _compute_score(self, probs):
        """
        Compute confidence score from probabilities.
        
        Args:
            probs (np.ndarray): Probability predictions
            
        Returns:
            np.ndarray: Confidence scores
        """
        if self.score_type == 'max_prob':
            return np.max(probs, axis=1)
        elif self.score_type == 'entropy':
            # Use negative entropy as confidence score
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            max_entropy = np.log(probs.shape[1])  # Maximum possible entropy
            return 1 - entropy / max_entropy  # Normalized to [0, 1]
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    def _log_marginal_likelihood(self, n_bins, scores, labels):
        """
        Compute log marginal likelihood for given number of bins.
        
        Args:
            n_bins (int): Number of bins
            scores (np.ndarray): Confidence scores
            labels (np.ndarray): True labels (binary: correct/incorrect predictions)
            
        Returns:
            float: Log marginal likelihood
        """
        n_samples = len(scores)
        
        # Create quantile-based bins
        if n_bins == 1:
            bin_boundaries = [0, 1]
        else:
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_boundaries = np.quantile(scores, quantiles)
            bin_boundaries[0] = 0  # Ensure first boundary is 0
            bin_boundaries[-1] = 1  # Ensure last boundary is 1
        
        log_marginal = 0
        
        for i in range(n_bins):
            # Find samples in this bin
            if i == 0:
                in_bin = (scores >= bin_boundaries[i]) & (scores <= bin_boundaries[i+1])
            else:
                in_bin = (scores > bin_boundaries[i]) & (scores <= bin_boundaries[i+1])
            
            n_bin = np.sum(in_bin)
            
            if n_bin > 0:
                n_correct = np.sum(labels[in_bin])
                n_incorrect = n_bin - n_correct
                
                # Beta-Binomial conjugate prior: Beta(1, 1) -> uniform prior
                # Marginal likelihood: Beta(n_correct + 1, n_incorrect + 1) / Beta(1, 1)
                # In log space: log B(n_correct + 1, n_incorrect + 1) - log B(1, 1)
                log_beta_numerator = gammaln(n_correct + 1) + gammaln(n_incorrect + 1) - gammaln(n_bin + 2)
                log_beta_denominator = gammaln(1) + gammaln(1) - gammaln(2)  # = log(1) = 0
                
                log_marginal += log_beta_numerator - log_beta_denominator
        
        return log_marginal

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Fit the BBQ calibrator on validation set using Bayesian model selection.
        
        Args:
            val_logits (torch.Tensor): Validation logits of shape [N, num_classes]
            val_labels (torch.Tensor): Validation labels of shape [N]
        
        Returns:
            dict: Fitted parameters
        """
        # Convert to probabilities
        if val_logits.dim() == 2:
            val_probs = F.softmax(val_logits, dim=1)
            predictions = torch.argmax(val_probs, dim=1)
        else:
            # Binary case
            val_probs = torch.sigmoid(val_logits.unsqueeze(1))
            val_probs = torch.cat([1 - val_probs, val_probs], dim=1)
            predictions = (val_probs[:, 1] > 0.5).long()
        
        # Convert to numpy
        val_probs_np = val_probs.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        val_labels_np = val_labels.detach().cpu().numpy()
        
        # Compute scores
        scores = self._compute_score(val_probs_np)
        
        # Binary labels: 1 if prediction is correct, 0 otherwise
        correct_predictions = (predictions_np == val_labels_np).astype(int)
        
        # Find optimal number of bins using Bayesian model selection
        best_log_marginal = -np.inf
        best_n_bins = 1
        
        for n_bins in range(1, min(self.n_bins_max + 1, len(scores) // 2 + 1)):
            log_marginal = self._log_marginal_likelihood(n_bins, scores, correct_predictions)
            
            if log_marginal > best_log_marginal:
                best_log_marginal = log_marginal
                best_n_bins = n_bins
        
        self.optimal_n_bins = best_n_bins
        
        # Fit with optimal number of bins
        if self.optimal_n_bins == 1:
            self.bin_boundaries = np.array([0, 1])
        else:
            quantiles = np.linspace(0, 1, self.optimal_n_bins + 1)
            self.bin_boundaries = np.quantile(scores, quantiles)
            self.bin_boundaries[0] = 0
            self.bin_boundaries[-1] = 1
        
        # Compute bin accuracies
        self.bin_accuracies = np.zeros(self.optimal_n_bins)
        
        for i in range(self.optimal_n_bins):
            if i == 0:
                in_bin = (scores >= self.bin_boundaries[i]) & (scores <= self.bin_boundaries[i+1])
            else:
                in_bin = (scores > self.bin_boundaries[i]) & (scores <= self.bin_boundaries[i+1])
            
            if np.sum(in_bin) > 0:
                self.bin_accuracies[i] = np.mean(correct_predictions[in_bin])
            else:
                # Use bin center if no samples
                self.bin_accuracies[i] = (self.bin_boundaries[i] + self.bin_boundaries[i+1]) / 2
        
        return {
            'optimal_n_bins': self.optimal_n_bins,
            'score_type': self.score_type,
            'bin_boundaries': self.bin_boundaries,
            'bin_accuracies': self.bin_accuracies,
            'log_marginal_likelihood': best_log_marginal
        }

    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate test logits using the fitted BBQ model.
        
        Args:
            test_logits (torch.Tensor): Test logits
            return_logits (bool): Whether to return logits or probabilities
            
        Returns:
            torch.Tensor: Calibrated predictions
        """
        if self.bin_boundaries is None:
            raise ValueError("Model must be fitted before calibration")
        
        device = test_logits.device
        
        # Convert to probabilities
        if test_logits.dim() == 2:
            test_probs = F.softmax(test_logits, dim=1)
            predictions = torch.argmax(test_probs, dim=1)
        else:
            # Binary case
            test_probs = torch.sigmoid(test_logits.unsqueeze(1))
            test_probs = torch.cat([1 - test_probs, test_probs], dim=1)
            predictions = (test_probs[:, 1] > 0.5).long()
        
        # Convert to numpy
        test_probs_np = test_probs.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        
        # Compute scores
        scores = self._compute_score(test_probs_np)
        
        # Apply BBQ calibration
        calibrated_confidences = np.zeros(len(scores))
        
        for i in range(self.optimal_n_bins):
            if i == 0:
                in_bin = (scores >= self.bin_boundaries[i]) & (scores <= self.bin_boundaries[i+1])
            else:
                in_bin = (scores > self.bin_boundaries[i]) & (scores <= self.bin_boundaries[i+1])
            
            if np.sum(in_bin) > 0:
                calibrated_confidences[in_bin] = self.bin_accuracies[i]
        
        # Convert back to torch
        calibrated_confidences = torch.tensor(calibrated_confidences, device=device, dtype=torch.float32)
        
        if test_logits.dim() == 2:
            # Multi-class case
            num_classes = test_logits.size(1)
            calibrated_probs = torch.zeros_like(test_probs)
            
            for i in range(len(calibrated_confidences)):
                pred_class = predictions[i]
                original_max_prob = torch.max(test_probs[i])
                new_confidence = calibrated_confidences[i]
                
                if original_max_prob > 0:
                    # Scale probabilities
                    scale_factor = new_confidence / original_max_prob
                    calibrated_probs[i] = test_probs[i] * scale_factor
                    calibrated_probs[i] = calibrated_probs[i] / torch.sum(calibrated_probs[i])
                else:
                    calibrated_probs[i] = torch.ones(num_classes, device=device) / num_classes
            
            if return_logits:
                calibrated_probs = torch.clamp(calibrated_probs, min=1e-8, max=1-1e-8)
                return torch.log(calibrated_probs)
            else:
                return calibrated_probs
        else:
            # Binary case
            if return_logits:
                calibrated_confidences = torch.clamp(calibrated_confidences, min=1e-8, max=1-1e-8)
                return torch.log(calibrated_confidences / (1 - calibrated_confidences))
            else:
                return calibrated_confidences