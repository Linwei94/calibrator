import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm, trange
import random
from .calibrator import Calibrator
from ..metrics import (
    BrierLoss, FocalLoss, LabelSmoothingLoss, 
    CrossEntropyLoss, MSELoss, SoftECE
)

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
            elif loss_fn_lower in {"brier", "brier_score"}:
                return BrierLoss()
            elif loss_fn_lower in {"focal", "focal_loss", "fl"}:
                return FocalLoss()
            elif loss_fn_lower in {"label_smoothing", "ls"}:
                return LabelSmoothingLoss()
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
        """
        Forward pass:
          1. Sort input logits in descending order and take top k elements;
          2. Pass the selected elements through the fully connected network to get scalar temperature (with abs and clip);
          3. Divide original logits by the temperature and apply softmax to get calibrated probability distribution.
        """
        # Input shape: (batch_size, length_logits)
        # Sort logits in descending order
        sorted_logits, _ = torch.sort(input_logits, dim=1, descending=True)
        # Take top k elements
        topk = sorted_logits[:, :self.top_k_logits]  # shape: (batch_size, top_k_logits)
        
        # Get temperature through fully connected network (output shape: (batch_size, 1))
        t = self.temp_branch(topk)
        temperature = torch.abs(t)
        # Clip temperature to prevent division by zero or large values
        temperature = torch.clamp(temperature, min=1e-12, max=1e12)
        # Use broadcasting to apply temperature to all logits
        adjusted_logits = input_logits / temperature
        # Output softmax probability distribution
        calibrated_probs = F.softmax(adjusted_logits, dim=1)
        return calibrated_probs, adjusted_logits

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Tune (train) the PTS model
        
        Args:
            val_logits (np.array or torch.Tensor): shape (N, length_logits)
            val_labels (np.array or torch.Tensor): shape (N,) - class indices
                or shape (N, length_logits) - one-hot encoded vectors
            **kwargs: Optional additional parameters
                - clip (float): Clipping threshold for logits, defaults to 1e2
                - verbose (bool): Whether to display progress bars, defaults to True
                - seed (int): Random seed for reproducibility, defaults to the one set in __init__
                - focal_gamma (float): Gamma parameter for focal loss, defaults to 2.0
                - label_smoothing_alpha (float): Alpha parameter for label smoothing, defaults to 0.1
        """
        clip = kwargs.get('clip', 1e2)
        verbose = kwargs.get('verbose', True)
        seed = kwargs.get('seed', self.seed)
        focal_gamma = kwargs.get('focal_gamma', 2.0)
        label_smoothing_alpha = kwargs.get('label_smoothing_alpha', 0.1)
        
        # Set random seeds for reproducibility
        self._set_seed(seed)
        
        # Set length_logits if not already set
        if self.length_logits is None:
            self.length_logits = val_logits.shape[1]
        
        # Convert to tensor if input is not already a tensor (float type)
        if not torch.is_tensor(val_logits):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if not torch.is_tensor(val_labels):
            val_labels = torch.tensor(val_labels, dtype=torch.float32)
        
        # Move tensors to CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        self.to(device)  # Move model to the same device
        
        # Check input dimensions
        assert val_logits.size(1) == self.length_logits, "Logits length must match length_logits!"
        
        # Convert class indices to one-hot encoded vectors if needed
        if len(val_labels.shape) == 1:
            # Create one-hot encoded vectors from class indices
            one_hot_labels = torch.zeros(val_labels.size(0), self.length_logits, dtype=torch.float32, device=device)
            one_hot_labels.scatter_(1, val_labels.unsqueeze(1), 1)
            val_labels = one_hot_labels
        
        # Clip logits
        val_logits = torch.clamp(val_logits, min=-clip, max=clip)
        
        # Create DataLoader with fixed seed for shuffling
        dataset = TensorDataset(val_logits, val_labels)
        generator = torch.Generator().manual_seed(seed)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        
        # Define optimizer (weight_decay implements L2 regularization)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.train()
        step_count = 0
        epochs = 0
        
        # Create progress bar for steps
        pbar = trange(self.steps, desc="Training PTS", disable=not verbose)
        
        while step_count < self.steps:
            epoch_loss = 0.0
            # Use tqdm for the dataloader if verbose
            epoch_loader = tqdm(dataloader, desc=f"Epoch {epochs+1}", 
                                leave=False, disable=not verbose)
            
            for batch_logits, batch_labels in epoch_loader:
                if step_count >= self.steps:
                    break
                    
                optimizer.zero_grad()
                outputs, _ = self.forward(batch_logits)
                
                # Handle different loss function types
                if isinstance(self.loss_fn, FocalLoss):
                    # For Focal Loss, pass the gamma parameter
                    loss = self.loss_fn(softmaxes=outputs, labels=batch_labels)
                elif isinstance(self.loss_fn, LabelSmoothingLoss):
                    # For Label Smoothing, pass the alpha parameter
                    loss = self.loss_fn(softmaxes=outputs, labels=batch_labels)
                else:
                    # For other loss functions, use the outputs and targets directly
                    loss = self.loss_fn(softmaxes=outputs, labels=batch_labels)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_logits.size(0)
                step_count += 1
                
                # Update the main progress bar
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if step_count >= self.steps:
                    break
            
            epochs += 1
            avg_loss = epoch_loss / len(dataset)
            if verbose and epochs % 10 == 0:
                print(f"Completed epoch {epochs}, Average Loss: {avg_loss:.4f}, Steps: {step_count}/{self.steps}")
        
        # Close the progress bar
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
        clip = kwargs.get('clip', 1e2)
        
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