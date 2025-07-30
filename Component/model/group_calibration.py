import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from .calibrator import Calibrator

class GroupingNetwork(nn.Module):
    """Simple linear network to assign samples to groups based on logits."""
    
    def __init__(self, num_classes, num_groups):
        super(GroupingNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        
        # Simple linear layer without bias (matching original implementation)
        self.linear = nn.Linear(num_classes, num_groups, bias=False)
    
    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Input logits of shape [N, num_classes]
            
        Returns:
            torch.Tensor: Group assignment logits of shape [N, num_groups]
        """
        return self.linear(logits)

class GroupCalibrationCalibrator(Calibrator):
    def __init__(self, num_groups=2, num_partitions=20, weight_decay=0.1):
        """
        Initialize the Group Calibration calibrator.
        
        Args:
            num_groups (int): Number of groups to create (K=2 in paper)
            num_partitions (int): Number of partitions to train (U=20 in paper)
            weight_decay (float): Weight decay for grouping network (λ=0.1 in paper)
        """
        super(GroupCalibrationCalibrator, self).__init__()
        self.num_groups = num_groups
        self.num_partitions = num_partitions
        self.weight_decay = weight_decay
        
        self.w_net_list = None  # List of trained grouping networks
        self.num_classes = None

    def fit(self, val_logits, val_labels, max_iter=50, **kwargs):
        """
        Train multiple partitions (grouping networks) as in original implementation.
        
        Args:
            val_logits (torch.Tensor): Validation logits of shape [N, num_classes]
            val_labels (torch.Tensor): Validation labels of shape [N]
            max_iter (int): Maximum number of iterations for LBFGS
        
        Returns:
            dict: Fitted parameters
        """
        device = val_logits.device
        self.num_classes = val_logits.size(1)
        
        # Train multiple partitions (following original algorithm)
        self.w_net_list = []
        print(f"Training {self.num_partitions} partitions...")
        
        for partition_i in range(self.num_partitions):
            if partition_i % 5 == 0:
                print(f"Training partition {partition_i+1}/{self.num_partitions}")
            
            # Create new grouping network for this partition
            w_net = GroupingNetwork(self.num_classes, self.num_groups).to(device)
            
            # Initialize group temperatures (matching original: 1.5)
            tau = nn.Parameter(
                torch.tensor([1.5] * self.num_groups, device=device, requires_grad=True)
            )
            
            # Setup LBFGS optimizer (matching original)
            params = list(w_net.parameters()) + [tau]
            optimizer = optim.LBFGS(params, 
                                   line_search_fn="strong_wolfe",
                                   max_iter=max_iter)
            
            def closure():
                optimizer.zero_grad()
                
                # Calculate weight decay loss for grouping network
                reg_weight_decay = 0
                for name, param in w_net.named_parameters():
                    if "weight" in name:
                        reg_weight_decay += torch.mean(param ** 2)
                reg_weight_decay_loss = reg_weight_decay * self.weight_decay
                
                # Get group assignment logits
                group_logits = w_net(val_logits)
                
                # Compute calibrated logits using original algorithm
                calibrated_logits = self._calibrate_with_tau_and_w_logits(
                    val_logits, group_logits, tau
                )
                
                # Cross-entropy loss
                main_loss = F.cross_entropy(calibrated_logits, val_labels)
                
                # Total loss
                total_loss = main_loss + reg_weight_decay_loss
                
                total_loss.backward()
                return total_loss
            
            # Optimize this partition
            optimizer.step(closure)
            
            # Store the trained network and temperatures
            self.w_net_list.append({
                'w_net': w_net.cpu(),
                'tau': tau.detach().cpu()
            })
        
        print(f"Completed training {self.num_partitions} partitions")
        
        return {
            'num_groups': self.num_groups,
            'num_partitions': self.num_partitions,
            'weight_decay': self.weight_decay
        }

    def _calibrate_with_tau_and_w_logits(self, logits, w_logits, tau):
        """
        Core group calibration algorithm from original implementation.
        
        Args:
            logits (torch.Tensor): Input logits [N, num_classes]
            w_logits (torch.Tensor): Group assignment logits [N, num_groups]  
            tau (torch.Tensor): Temperature parameters [num_groups]
            
        Returns:
            torch.Tensor: Calibrated logits [N, num_classes]
        """
        N, num_classes = logits.shape
        num_groups = w_logits.shape[1]
        
        # Convert group assignment logits to log probabilities
        group_log_softmax = torch.log_softmax(w_logits, dim=1).view((N, num_groups, 1))
        
        # Expand to match dimensions
        group_log_softmax = group_log_softmax.expand((N, num_groups, num_classes))
        
        # Apply temperature scaling to logits for each group
        temp_logits = logits.view((N, 1, num_classes)) / tau.view((1, num_groups, 1))
        temp_log_softmax = torch.log_softmax(temp_logits, dim=2)
        
        # Combine using logsumexp (key insight from original code)
        calibrated_logits = torch.logsumexp(
            group_log_softmax + temp_log_softmax, dim=1
        )
        
        return calibrated_logits

    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate test logits using ensemble of trained partitions.
        
        Args:
            test_logits (torch.Tensor): Test logits
            return_logits (bool): Whether to return logits or probabilities
            
        Returns:
            torch.Tensor: Calibrated predictions
        """
        if self.w_net_list is None:
            raise ValueError("Model must be fitted before calibration")
        
        device = test_logits.device
        
        with torch.no_grad():
            # Collect predictions from all partitions
            all_calibrated_logits = []
            
            for partition in self.w_net_list:
                w_net = partition['w_net'].to(device)
                tau = partition['tau'].to(device)
                
                # Get group assignment logits for this partition
                group_logits = w_net(test_logits)
                
                # Apply group-based calibration
                calibrated_logits = self._calibrate_with_tau_and_w_logits(
                    test_logits, group_logits, tau
                )
                
                all_calibrated_logits.append(calibrated_logits)
            
            # Simple ensemble: average the log probabilities
            ensemble_logits = torch.stack(all_calibrated_logits, dim=0).mean(dim=0)
            
            if return_logits:
                return ensemble_logits
            else:
                # ensemble_logits is already in log-probability space
                return torch.exp(ensemble_logits)