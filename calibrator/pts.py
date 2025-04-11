import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from calibrator.calibrator import Calibrator

class PTSCalibrator(Calibrator):
    """
    PyTorch implementation of Parameterized Temperature Scaling (PTS)
    """
    def __init__(self, epochs, lr, weight_decay, batch_size, nlayers, n_nodes, length_logits, top_k_logits):
        """
        Args:
            epochs (int): Number of epochs for PTS model tuning
            lr (float): Learning rate
            weight_decay (float): Weight decay coefficient (L2 regularization)
            batch_size (int): Batch size for training
            nlayers (int): Number of fully connected layers in PTS model
            n_nodes (int): Number of nodes in each hidden layer
            length_logits (int): Length of input logits
            top_k_logits (int): Number of top k elements to use from sorted logits
        """
        super(PTSCalibrator, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

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
        
        # Note: Since PyTorch's weight decay is set in the optimizer,
        # we don't need to specify regularization in each fully connected layer

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
            val_labels (np.array or torch.Tensor): shape (N, length_logits)
                (Using MSE loss, so labels are typically one-hot encoded probability distributions)
            **kwargs: Optional additional parameters
                - clip (float): Clipping threshold for logits, defaults to 1e2
        """
        clip = kwargs.get('clip', 1e2)
        
        # Convert to tensor if input is not already a tensor (float type)
        if not torch.is_tensor(val_logits):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if not torch.is_tensor(val_labels):
            val_labels = torch.tensor(val_labels, dtype=torch.float32)
        
        # Check input dimensions
        assert val_logits.size(1) == self.length_logits, "Logits length must match length_logits!"
        assert val_labels.size(1) == self.length_logits, "Labels length must match length_logits!"
        
        # Clip logits
        val_logits = torch.clamp(val_logits, min=-clip, max=clip)
        
        # Create DataLoader
        dataset = TensorDataset(val_logits, val_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Define MSE loss function and Adam optimizer (weight_decay implements L2 regularization)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_logits, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs, _ = self.forward(batch_logits)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_logits.size(0)
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")
    
    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate logits using the trained PTS model
        
        Args:
            test_logits (np.array or torch.Tensor): shape (N, length_logits)
            return_logits (bool): Whether to return calibrated logits, defaults to False
            **kwargs: Optional additional parameters
                - clip (float): Clipping threshold, defaults to 1e2
        Return:
            If return_logits is False, returns calibrated probability distribution (np.array)
            If return_logits is True, returns calibrated logits (np.array)
        """
        clip = kwargs.get('clip', 1e2)
        
        if not torch.is_tensor(test_logits):
            test_logits = torch.tensor(test_logits, dtype=torch.float32)
        
        assert test_logits.size(1) == self.length_logits, "Logits length must match length_logits!"
        test_logits = torch.clamp(test_logits, min=-clip, max=clip)
        
        self.eval()
        with torch.no_grad():
            calibrated_probs, calibrated_logits = self.forward(test_logits)
            
        if return_logits:
            return calibrated_logits.cpu().numpy()
        return calibrated_probs.cpu().numpy()
    
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