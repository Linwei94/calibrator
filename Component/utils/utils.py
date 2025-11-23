import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union
import sys
import os
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import metric implementations
from ..metrics import (
    ECE, AdaptiveECE, ClasswiseECE, NLL, Accuracy, ECEDebiased, ECESweep, RBS
)

def compute_ece(probs, labels, n_bins=15):
    """
    Compute ECE (Expected Calibration Error)
    
    Args:
        probs: numpy array or torch.Tensor of shape [n_samples, n_classes] with probabilities
        labels: numpy array or torch.Tensor of shape [n_samples] with ground truth labels
        n_bins: number of bins for confidence histogram
        
    Returns:
        Expected Calibration Error
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def compute_all_metrics(
    labels: torch.Tensor, 
    logits: Optional[torch.Tensor] = None, 
    probs: Optional[torch.Tensor] = None,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Compute all available metrics for the given logits/probs and labels.
    
    Args:
        labels (torch.Tensor): Target labels
        logits (torch.Tensor, optional): Input logits before softmax
        probs (torch.Tensor, optional): Probability distributions (softmax outputs)
        n_bins (int, optional): Number of bins for ECE calculation. Defaults to 15.
            
    Returns:
        Dict[str, float]: Dictionary containing all metric values
    """
    if logits is None and probs is None:
        raise ValueError("Either logits or probs must be provided")
    
    device = labels.device
    
    # If probs not provided, compute from logits
    if probs is None:
        probs = F.softmax(logits, dim=1)
    
    # Initialize metrics
    metrics = {
        'ece': ECE(n_bins=n_bins),
        'adaptive_ece': AdaptiveECE(n_bins=n_bins),
        'classwise_ece': ClasswiseECE(n_bins=n_bins),
        'ece_debiased': ECEDebiased(n_bins=n_bins),
        'ece_sweep': ECESweep(),
        'nll': NLL(),
        'accuracy': Accuracy(),
        'rbs': RBS()
    }
    
    # Set up basic logging
    logger = logging.getLogger(__name__)
    
    results = {}
    for name, metric in metrics.items():
        metric = metric.to(device)
        try:
            if name in ['nll', 'rbs']:
                if logits is not None:
                    value = metric(logits=logits, labels=labels)
                else:
                    value = metric(softmaxes=probs, labels=labels)
            elif name in ['ece', 'adaptive_ece', 'classwise_ece', 'ece_debiased', 'ece_sweep', 'accuracy']:
                value = metric(softmaxes=probs, labels=labels)
            else:
                logger.warning(f"Unknown metric type: {name}")
                continue
            
            # Convert to float if it's a tensor
            if torch.is_tensor(value):
                value = value.item()
            results[name] = value
        except Exception as e:
            logger.warning(f"Failed to compute {name}: {str(e)}")
            results[name] = None
            continue
            
    return results

def get_all_metrics(
    labels: torch.Tensor, 
    logits: Optional[torch.Tensor] = None, 
    probs: Optional[torch.Tensor] = None,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Get all metrics in a dictionary format compatible with the standard results structure.
    
    Args:
        labels (torch.Tensor): Target labels
        logits (torch.Tensor, optional): Input logits before softmax
        probs (torch.Tensor, optional): Probability distributions (softmax outputs)
        n_bins (int, optional): Number of bins for ECE calculation. Defaults to 15.
            
    Returns:
        Dict[str, float]: Dictionary containing the 8 standard metrics:
        {
            'ece': float,
            'accuracy': float,
            'adaece': float,
            'cece': float,
            'nll': float,
            'ece_debiased': float,
            'ece_sweep': float,
            'rbs': float
        }
    """
    metrics = compute_all_metrics(labels=labels, logits=logits, probs=probs, n_bins=n_bins)
    return {
        'ece': metrics.get('ece', None),
        'accuracy': metrics.get('accuracy', None),
        'adaece': metrics.get('adaptive_ece', None),
        'cece': metrics.get('classwise_ece', None),
        'nll': metrics.get('nll', None),
        'ece_debiased': metrics.get('ece_debiased', None),
        'ece_sweep': metrics.get('ece_sweep', None),
        'rbs': metrics.get('rbs', None)
    }

def get_all_metrics_multi_bins(
    labels: torch.Tensor, 
    logits: Optional[torch.Tensor] = None, 
    probs: Optional[torch.Tensor] = None,
    bins_list: list = [5, 10, 15, 20, 25, 30]
) -> Dict[str, float]:
    """
    Evaluate metrics across multiple bin sizes for ECE calculation.
    
    Args:
        labels (torch.Tensor): Target labels
        logits (torch.Tensor, optional): Input logits before softmax
        probs (torch.Tensor, optional): Probability distributions (softmax outputs)
        bins_list (list): List of bin sizes to evaluate. Defaults to [5, 10, 15, 20, 25, 30].
            
    Returns:
        Dict[str, float]: Dictionary with metrics for each bin size and base metrics:
        {
            'ece_5': float, 'ece_10': float, ..., 'ece_30': float,
            'adaece_5': float, 'adaece_10': float, ..., 'adaece_30': float,
            'cece_5': float, 'cece_10': float, ..., 'cece_30': float,
            'ece_debiased_5': float, 'ece_debiased_10': float, ..., 'ece_debiased_30': float,
            'accuracy': float,  # computed once with first bin size
            'nll': float,       # computed once with first bin size
            'rbs': float,       # computed once (Root Brier Score)
            'ece_sweep': float  # computed once (adaptive bin selection)
        }
    """
    results = {}
    
    # Evaluate for each bin size
    for i, n_bins in enumerate(bins_list):
        bin_metrics = get_all_metrics(
            labels=labels,
            logits=logits,
            probs=probs,
            n_bins=n_bins
        )
        
        # Store ECE and related metrics with bin suffix
        results[f'ece_{n_bins}'] = bin_metrics['ece']
        results[f'adaece_{n_bins}'] = bin_metrics['adaece']
        results[f'cece_{n_bins}'] = bin_metrics['cece']
        results[f'ece_debiased_{n_bins}'] = bin_metrics['ece_debiased']
        
        # For the first bin size, also store the base metrics (accuracy, nll, rbs, and ece_sweep don't depend on bins)
        if i == 0:
            results['accuracy'] = bin_metrics['accuracy']
            results['nll'] = bin_metrics['nll']
            results['ece_sweep'] = bin_metrics['ece_sweep']
            results['rbs'] = bin_metrics['rbs']
    
    return results