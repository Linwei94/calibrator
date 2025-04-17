import torch
import torch.nn as nn
import numpy as np
from .calibrator import Calibrator
from ..metrics import ECE, Accuracy

class CTSCalibrator(Calibrator):
    """
    Implements Class-based Temperature Scaling (CTS).
    
    CTS assigns a separate temperature parameter for each class, allowing for more
    flexible calibration compared to standard temperature scaling.
    """
    def __init__(self, n_class, n_iter=5, n_bins=15):
        """
        Args:
            n_class (int): Number of classes.
            n_iter (int): Number of iterations for greedy search.
            n_bins (int): Number of bins used in the ECE computation.
        """
        super(CTSCalibrator, self).__init__()
        self.n_class = n_class
        self.n_iter = n_iter
        self.n_bins = n_bins
        # Initialize the learnable temperature parameters (one per class)
        self.T = nn.Parameter(torch.ones(n_class))
        # Initialize metrics
        self.ece_metric = ECE(n_bins=n_bins)
        self.accuracy_metric = Accuracy()
    def forward(self, x):
        """
        Forward pass: scales the input logits x by dividing each column (class score) 
        by its corresponding temperature.
        
        Args:
            x (torch.Tensor): Input logits of shape (batch_size, n_class)
            
        Returns:
            torch.Tensor: Scaled logits
        """
        # Broadcasting division; x's column i is divided by self.T[i]
        return x / self.T

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Fit the CTS calibrator on validation data using a greedy search approach.
        For each class separately, finds a temperature that minimizes ECE while
        maintaining accuracy.
        
        Args:
            val_logits (torch.Tensor): Validation logits of shape (N, n_class)
            val_labels (torch.Tensor): Validation labels of shape (N,)
            **kwargs: Additional arguments
                - grid (np.array): Temperature grid to search over, defaults to np.arange(0.1, 10.1, 0.1)
                
        Returns:
            np.array: Optimized temperature vector of shape (n_class,)
            dict: Dictionary containing:
                - 'final_ece': Final ECE value after optimization
                - 'final_accuracy': Final accuracy after optimization
        """
        # Convert inputs to tensors if they're not already
        if not torch.is_tensor(val_logits):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if not torch.is_tensor(val_labels):
            val_labels = torch.tensor(val_labels, dtype=torch.float32)
            
        # Get temperature grid from kwargs or use default
        grid = kwargs.get('grid', np.arange(0.1, 10.1, 0.1)) 
        '''
        optimal temperature can
        be found by a grid search over values between 0 and 10, with a
        step of 0.1, and finding the one that minimizes the validation set
        ECE. The grid search is done for each class separately, and the
        '''
        
        # Start with the current temperatures
        current_T = self.T.detach().cpu().numpy().copy()  # shape: (n_class,)
        
        # Compute initial probabilities, accuracy, and ECE using current_T
        scaled_logits = val_logits / torch.tensor(current_T, device=val_logits.device)
        probs = torch.nn.functional.softmax(scaled_logits, dim=1)
        base_acc = self.accuracy_metric(softmaxes=probs, labels=val_labels)
        base_ece = self.ece_metric(softmaxes=probs, labels=val_labels)
        print(f"Initial: Accuracy = {base_acc:.4f}, ECE = {base_ece:.4f}")
        
        # Greedy optimization loop
        for iteration in range(self.n_iter):
            updated = False
            print(f"\nIteration {iteration + 1}:")
            for i in range(self.n_class):
                best_candidate = current_T[i]
                best_ece = base_ece
                # Search candidate temperatures for class i over the grid
                for candidate in grid:
                    candidate_T = current_T.copy()
                    candidate_T[i] = candidate
                    candidate_scaled_logits = val_logits / torch.tensor(candidate_T, device=val_logits.device)
                    candidate_probs = torch.nn.functional.softmax(candidate_scaled_logits, dim=1)
                    candidate_ece = self.ece_metric(softmaxes=candidate_probs, labels=val_labels)
                    candidate_acc = self.accuracy_metric(softmaxes=candidate_probs, labels=val_labels)
                    # Update if candidate yields lower ECE and does not reduce accuracy
                    if candidate_acc >= base_acc and candidate_ece < best_ece:
                        best_candidate = candidate
                        best_ece = candidate_ece
                if best_candidate != current_T[i]:
                    print(f"  Class {i}: {current_T[i]:.2f} -> {best_candidate:.2f} (ECE: {best_ece:.4f})")
                    current_T[i] = best_candidate
                    # Refresh overall accuracy and ECE after this update
                    scaled_logits = val_logits / torch.tensor(current_T, device=val_logits.device)
                    probs = torch.nn.functional.softmax(scaled_logits, dim=1)
                    base_acc = self.accuracy_metric(softmaxes=probs, labels=val_labels)
                    base_ece = self.ece_metric(softmaxes=probs, labels=val_labels)
                    updated = True
            # If none of the classes got updated in this iteration, break out
            if not updated:
                print("No update in this iteration, converged.")
                break
            else:
                print(f"After iteration {iteration + 1}: Accuracy = {base_acc:.4f}, ECE = {base_ece:.4f}")
        
        # Finally, update the module's parameter with the optimized temperatures
        self.T.data = torch.from_numpy(current_T).to(self.T.device)
        print("\nOptimized class-based temperatures:", current_T)
        
        return current_T, {
            'final_ece': base_ece,
            'final_accuracy': base_acc
        }

    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Calibrate the test logits using the optimized temperatures.
        
        Args:
            test_logits (torch.Tensor): Test logits of shape (N, n_class)
            return_logits (bool): Whether to return calibrated logits instead of probabilities
            **kwargs: Additional arguments (not used)
            
        Returns:
            If return_logits is False: Calibrated probabilities of shape (N, n_class)
            If return_logits is True: Calibrated logits of shape (N, n_class)
        """
        # Convert input to tensor if it's not already
        if not torch.is_tensor(test_logits):
            test_logits = torch.tensor(test_logits, dtype=torch.float32)
            
        # Apply temperature scaling
        calibrated_logits = self.forward(test_logits)
        
        if return_logits:
            return calibrated_logits.detach().cpu().numpy()
        
        # Apply softmax to get probabilities
        calibrated_probs = torch.nn.functional.softmax(calibrated_logits, dim=1)
        return calibrated_probs.detach().cpu().numpy()
    
    def save(self, path="./"):
        """
        Save the CTS model parameters.
        
        Args:
            path (str): Directory to save the model
        """
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, "cts_model.pth")
        torch.save(self.state_dict(), save_path)
        print("Save CTS model to:", save_path)
    
    def load(self, path="./"):
        """
        Load the CTS model parameters.
        
        Args:
            path (str): Directory to load the model from
        """
        import os
        load_path = os.path.join(path, "cts_model.pth")
        self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
        print("Load CTS model from:", load_path)