import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm, trange
import random
import logging
from typing import Dict, Any
from .calibrator import Calibrator
from ..metrics import (
    ECE, AdaptiveECE, ClasswiseECE, NLL, Accuracy,
    BrierLoss, FocalLoss, LabelSmoothingLoss, 
    CrossEntropyLoss, MSELoss, SoftECE
)
from ..metrics.WeightedSoftECE import WeightedSoftECE
from ..metrics.SmoothSoftECE import SmoothSoftECE
from ..metrics.GapIndexedSoftECE import GapIndexedSoftECE

# ------------------------------------------------------------------ #
# logging setup
# ------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)           # change to DEBUG for detailed logs
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)

class PTSCalibrator(Calibrator):
    """
    PyTorch implementation of Parameterized Temperature Scaling (PTS)
    """
    def __init__(self, steps=100000, lr=0.00005, weight_decay=0.0, batch_size=1000, nlayers=2, n_nodes=5, length_logits=None, top_k_logits=10, loss_fn=None, seed=42):
        """
        Args:
            steps (int): Number of optimization steps for PTS model tuning, default 100000 as per paper
            lr (float): Learning rate, default 0.00005 as per paper
            weight_decay (float): Weight decay coefficient (L2 regularization), default 0.0
            batch_size (int): Batch size for training, default 1000 as per paper
            nlayers (int): Number of fully connected layers in PTS model, default 2 as per paper
            n_nodes (int): Number of nodes in each hidden layer, default 5 as per paper
            length_logits (int): Length of input logits, will be set during fit if None
            top_k_logits (int): Number of top k elements to use from sorted logits, default 10 as per paper
            loss_fn (str or callable): Loss function to use, options:
                - 'mse' or 'mean_squared_error': Mean Squared Error
                - 'ce' or 'cross_entropy': Cross Entropy
                - 'soft_ece': Soft Expected Calibration Error
                - 'weighted_soft_ece': Weighted Soft Expected Calibration Error
                - 'smooth_soft_ece': Smooth Soft Expected Calibration Error
                - 'gap_indexed_soft_ece': Gap Indexed Soft Expected Calibration Error
                - 'brier': Brier Score
                - 'l1' or 'mean_absolute_error': L1 Loss
                - 'focal': Focal Loss (with gamma=2.0 by default)
                - 'label_smoothing': Label Smoothing (with alpha=0.1 by default)
                - Any callable function that takes (outputs, targets) as arguments
            seed (int): Random seed for reproducibility, default 42
        """
        super(PTSCalibrator, self).__init__()
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits
        self.seed = seed
        
        # Set random seeds for reproducibility
        self._set_seed(seed)
        
        # Set loss function
        self.loss_fn_type = loss_fn if isinstance(loss_fn, str) else None
        self.loss_fn = self._get_loss_function(loss_fn)
        
        # Build parameterized temperature branch: input is top k sorted logits
        layers = []
        input_dim = top_k_logits
        if self.nlayers > 0:
            # First layer
            layers.append(nn.Linear(input_dim, self.n_nodes))
            layers.append(nn.ReLU())
            # Subsequent hidden layers
            for _ in range(self.nlayers - 1):
                layers.append(nn.Linear(self.n_nodes, self.n_nodes))
                layers.append(nn.ReLU())
            # Final output layer: outputs scalar temperature
            layers.append(nn.Linear(self.n_nodes, 1))
        else:
            # If no hidden layers, directly map from top_k_logits to 1 number
            layers.append(nn.Linear(input_dim, 1))
        self.temp_branch = nn.Sequential(*layers)
        
        # Initialize weights with fixed seed
        self._init_weights()
        
        # Initialize all metrics
        self.metrics = {
            'ece': ECE(n_bins=15),
            'adaptive_ece': AdaptiveECE(n_bins=15),
            'classwise_ece': ClasswiseECE(n_bins=15),
            'nll': NLL(),
            'accuracy': Accuracy(),
            'brier': BrierLoss(),
            'focal': FocalLoss(),
            'label_smoothing': LabelSmoothingLoss(),
            'cross_entropy': CrossEntropyLoss(),
            'mse': MSELoss(),
            'soft_ece': SoftECE(),
            'weighted_soft_ece': WeightedSoftECE(),
            'smooth_soft_ece': SmoothSoftECE(),
            'gap_indexed_soft_ece': GapIndexedSoftECE()
        }
        
        logger.info("PTSCalibrator initialised: steps=%d  lr=%.6f  batch_size=%d  nlayers=%d  n_nodes=%d",
                    steps, lr, batch_size, nlayers, n_nodes)
        
        # Note: Since PyTorch's weight decay is set in the optimizer,
        # we don't need to specify regularization in each fully connected layer
    
    def _get_loss_function(self, loss_fn):
        """
        Get the appropriate loss function based on the input.
        
        Args:
            loss_fn (str or callable): Loss function specification
            
        Returns:
            callable: Loss function
        """
        if loss_fn is None:
            return MSELoss()
        elif isinstance(loss_fn, str):
            loss_fn_lower = loss_fn.lower()
            if loss_fn_lower in {"mse", "mean_squared_error"}:
                return MSELoss()
            elif loss_fn_lower in {"crossentropy", "cross_entropy", "ce"}:
                return CrossEntropyLoss()
            elif loss_fn_lower in {"l1", "l1loss", "mean_absolute_error"}:
                return nn.L1Loss()
            elif loss_fn_lower in {"soft_ece", "softece"}:
                return SoftECE()
            elif loss_fn_lower in {"weighted_soft_ece", "weightedsoftece"}:
                return WeightedSoftECE()
            elif loss_fn_lower in {"smooth_soft_ece", "smoothsoftece"}:
                return SmoothSoftECE()
            elif loss_fn_lower in {"gap_indexed_soft_ece", "gapindexedsoftece"}:
                return GapIndexedSoftECE()
            elif loss_fn_lower in {"brier", "brier_score"}:
                return BrierLoss()
            elif loss_fn_lower in {"focal", "focal_loss", "fl"}:
                return FocalLoss()
            elif loss_fn_lower in {"label_smoothing", "ls"}:
                return LabelSmoothingLoss(alpha=0.01)
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn}")
        else:
            # If loss_fn is a callable, then use it directly.
            return loss_fn
    
    def _set_seed(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _init_weights(self):
        """Initialize weights with fixed seed"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_logits):
        sorted_logits, _ = torch.sort(input_logits, 1, True)
        topk        = sorted_logits[:, :self.top_k_logits]

        t           = self.temp_branch(topk)     # (B,1)
        temperature = torch.abs(t)               # ← paper
        # temperature = torch.clamp(temperature, 1e-12, 1e12)
        temperature = torch.clamp(temperature, 0.1, 10)

        adjusted_logits = input_logits / temperature
        calibrated_probs= F.softmax(adjusted_logits, dim=1)
        return calibrated_probs, adjusted_logits


    def fit(self, val_logits, val_labels, **kwargs):
        """Tune (train) the PTS model."""
        clip    = kwargs.get('clip',      20)
        seed    = kwargs.get('seed',      self.seed)
        verbose = kwargs.get('verbose',   True)

        self._set_seed(seed)

        # -------- tensors & device --------
        if not torch.is_tensor(val_logits):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if not torch.is_tensor(val_labels):
            val_labels = torch.tensor(val_labels, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_logits, val_labels = val_logits.to(device), val_labels.to(device)
        self.to(device)

        # infer number of classes once
        if self.length_logits is None:
            self.length_logits = val_logits.shape[1]
        assert val_logits.size(1) == self.length_logits

        # keep a copy of original class indices for SoftECE
        original_labels = val_labels.clone()

        # labels: indices → one‑hot
        if val_labels.ndim == 1:
            oh = torch.zeros(
                val_labels.size(0), self.length_logits,
                dtype=torch.float32, device=device)
            oh.scatter_(1, val_labels.unsqueeze(1), 1.)
            val_labels = oh

        # clip logits
        val_logits = torch.clamp(val_logits, -clip, clip)

        # -------- dataloader --------
        dataset   = TensorDataset(val_logits, val_labels, original_labels)
        loader    = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(seed))

        optim = torch.optim.Adam(self.parameters(),
                         lr=self.lr,
                         weight_decay=self.weight_decay)

        # -------- training --------
        self.train()
        pbar, step = trange(self.steps, disable=not verbose, desc="Training PTS"), 0
        while step < self.steps:
            for batch_logits, batch_labels, batch_orig in loader:
                if step >= self.steps:
                    break

                optim.zero_grad()
                cal_probs, cal_logits = self.forward(batch_logits)

                # choose correct input for every loss
                if isinstance(self.loss_fn, (SoftECE, WeightedSoftECE, SmoothSoftECE, GapIndexedSoftECE)):
                    loss = self.loss_fn(logits=cal_logits, labels=batch_orig)
                elif isinstance(self.loss_fn, BrierLoss):
                    loss = self.loss_fn(logits=cal_logits, labels=batch_labels)
                elif isinstance(self.loss_fn, (FocalLoss, LabelSmoothingLoss)):
                    loss = self.loss_fn(softmaxes=cal_logits, labels=batch_labels)
                else:
                    # *** HERE: MSE gets probabilities, CE gets logits ***
                    loss_in = cal_probs if isinstance(self.loss_fn, nn.MSELoss) else cal_logits
                    loss    = self.loss_fn(loss_in, batch_labels)

                loss.backward()
                optim.step()

                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()



    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate logits using the trained PTS model
        
        Args:
            test_logits (np.array or torch.Tensor): shape (N, length_logits)
            return_logits (bool): Whether to return calibrated logits, defaults to False
            **kwargs: Optional additional parameters
                - clip (float): Clipping threshold, defaults to 1e2
        Return:
            If return_logits is False, returns calibrated probability distribution (torch.Tensor)
            If return_logits is True, returns calibrated logits (torch.Tensor)
        """
        clip = kwargs.get('clip', 20)
        
        if not torch.is_tensor(test_logits):
            test_logits = torch.tensor(test_logits, dtype=torch.float32)
        
        # Move tensor to the same device as the model
        device = next(self.parameters()).device
        test_logits = test_logits.to(device)
        
        assert test_logits.size(1) == self.length_logits, "Logits length must match length_logits!"
        test_logits = torch.clamp(test_logits, min=-clip, max=clip)
        
        self.eval()
        with torch.no_grad():
            calibrated_probs, calibrated_logits = self.forward(test_logits)
            
        if return_logits:
            return calibrated_logits
        return calibrated_probs
    
    def save(self, path="./"):
        """
        Save PTS model parameters
        """
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, "pts_model.pth")
        torch.save(self.state_dict(), save_path)
        print("Save PTS model to:", save_path)
    
    def load(self, path="./"):
        """
        Load PTS model parameters
        """
        load_path = os.path.join(path, "pts_model.pth")
        self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        print("Load PTS model from:", load_path)

    def compute_all_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute all available metrics for the given logits and labels.
        
        Args:
            logits (torch.Tensor): Input logits
            labels (torch.Tensor): Target labels
            
        Returns:
            Dict[str, float]: Dictionary containing all metric values
        """
        device = logits.device
        self.eval()
        with torch.no_grad():
            calibrated_probs, calibrated_logits = self.forward(logits)
        
        results = {}
        for name, metric in self.metrics.items():
            metric = metric.to(device)
            try:
                if name in ['nll', 'cross_entropy']:
                    value = metric(logits=calibrated_logits, labels=labels)
                elif name in ['brier', 'focal', 'label_smoothing', 'mse']:
                    value = metric(probs=calibrated_probs, labels=labels)
                elif name in ['ece', 'adaptive_ece', 'classwise_ece', 'soft_ece']:
                    value = metric(probs=calibrated_probs, labels=labels)
                elif name == 'accuracy':
                    value = metric(probs=calibrated_probs, labels=labels)
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

    def get_all_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Get all metrics in a dictionary format compatible with the results structure.
        
        Args:
            logits (torch.Tensor): Input logits
            labels (torch.Tensor): Target labels
            
        Returns:
            Dict[str, float]: Dictionary containing all metric values in the format:
            {
                'ece': float,
                'accuracy': float,
                'adaece': float,
                'cece': float,
                'nll': float
            }
        """
        metrics = self.compute_all_metrics(logits, labels)
        return {
            'ece': metrics.get('ece', None),
            'accuracy': metrics.get('accuracy', None),
            'adaece': metrics.get('adaptive_ece', None),
            'cece': metrics.get('classwise_ece', None),
            'nll': metrics.get('nll', None)
        }