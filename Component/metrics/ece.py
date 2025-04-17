import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class KernelECE(nn.Module):
    '''
    Compute ECE (Expected Calibration Error) based on kernel density estimates
    This provides a continuous estimate of calibration error without binning
    '''
    def __init__(self, bandwidth=0.01, num_points=100):
        '''
        Args:
            bandwidth: float
                The bandwidth parameter for the kernel density estimation
            num_points: int
                The number of points to evaluate the kernel density
        '''
        super(KernelECE, self).__init__()
        self.bandwidth = bandwidth
        self.num_points = num_points
        self.eval_points = torch.linspace(0, 1, num_points)

    def gaussian_kernel(self, x, y):
        '''
        Compute the Gaussian kernel between x and y
        
        Args:
            x: torch.Tensor
                The first tensor
            y: torch.Tensor
                The second tensor
                
        Returns:
            kernel: torch.Tensor
                The kernel values
        '''
        return torch.exp(-0.5 * ((x.unsqueeze(1) - y.unsqueeze(0)) / self.bandwidth) ** 2)

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
            kernel_ece: float
                The kernel-based ECE value
        '''
        if softmaxes is None:
            softmaxes = F.softmax(logits, dim=1)
        
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels).float()
        
        # Move tensors to the same device as the model
        device = next(self.parameters()).device if list(self.parameters()) else confidences.device
        confidences = confidences.to(device)
        accuracies = accuracies.to(device)
        eval_points = self.eval_points.to(device)
        
        # Compute kernel density for confidences
        kernel_conf = self.gaussian_kernel(eval_points, confidences)
        density_conf = kernel_conf.mean(dim=1)
        
        # Compute kernel density for accuracies
        kernel_acc = self.gaussian_kernel(eval_points, accuracies)
        density_acc = kernel_acc.mean(dim=1)
        
        # Normalize densities
        density_conf = density_conf / (density_conf.sum() + 1e-10)
        density_acc = density_acc / (density_acc.sum() + 1e-10)
        
        # Compute ECE as the absolute difference between confidence and accuracy densities
        # weighted by the confidence density
        ece = torch.sum(torch.abs(eval_points - density_acc) * density_conf)
        
        return ece.item()