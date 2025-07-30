import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from scipy import optimize

from .calibrator import Calibrator

# Import functions from the original Mix-n-Match implementation
def mse_t(t, *args):
    """Find optimal temperature with MSE loss function (from original code)"""
    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)  
    p = np.exp(logit)/n[:,None]
    mse = np.mean((p-label)**2)
    return mse

def ll_t(t, *args):
    """Find optimal temperature with Cross-Entropy loss function (from original code)"""
    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)  
    p = np.clip(np.exp(logit)/n[:,None],1e-20,1-1e-20)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce

def mse_w(w, *args):
    """Find optimal weight coefficients with MSE loss function (from original code)"""
    p0, p1, p2, label = args
    p = w[0]*p0+w[1]*p1+w[2]*p2
    p = p/np.sum(p,1)[:,None]
    mse = np.mean((p-label)**2)   
    return mse

def ll_w(w, *args):
    """Find optimal weight coefficients with Cross-Entropy loss function (from original code)"""
    p0, p1, p2, label = args
    p = (w[0]*p0+w[1]*p1+w[2]*p2)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce

def temperature_scaling(logit, label, loss):
    """Temperature scaling (from original code)"""
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
       t = optimize.minimize(ll_t, 1.0 , args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    if loss == 'mse':
        t = optimize.minimize(mse_t, 1.0 , args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    t = t.x
    return t

def ensemble_scaling(logit, label, loss, t, n_class):
    """Ensemble scaling (from original code)"""
    p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
    logit = logit/t
    p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
    p2 = np.ones_like(p0)/n_class
    
    bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
    def my_constraint_fun(x): return np.sum(x)-1
    constraints = { "type":"eq", "fun":my_constraint_fun,}
    if loss == 'ce':
        w = optimize.minimize(ll_w, (1.0, 0.0, 0.0) , args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
    if loss == 'mse':
        w = optimize.minimize(mse_w, (1.0, 0.0, 0.0) , args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
    w = w.x
    return w


class ETSCalibrator(Calibrator):
    """
    Ensemble Temperature Scaling (ETS) Calibrator.
    
    This implementation is based on "Mix-n-Match: Ensemble and Compositional Methods 
    for Uncertainty Calibration in Deep Learning". ETS combines three components:
    1. Temperature scaled probabilities
    2. Original softmax probabilities  
    3. Uniform distribution
    
    The optimal temperature and ensemble weights are learned via optimization.
    """
    
    def __init__(self, loss_type='mse', n_classes=None):
        """
        Initialize ETS calibrator.
        
        Args:
            loss_type (str): Loss function for optimization ('mse' or 'ce'). Default: 'mse'
            n_classes (int, optional): Number of classes. Will be inferred from data if not provided.
        """
        super(ETSCalibrator, self).__init__()
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.temperature = None
        self.weights = None
        self.is_fitted = False
        
    
    def fit(self, val_logits, val_labels, **kwargs):
        """
        Fit the ETS calibrator using validation data.
        
        Args:
            val_logits (torch.Tensor): Validation logits
            val_labels (torch.Tensor): Validation labels (can be one-hot or class indices)
            **kwargs: Additional arguments (unused)
            
        Returns:
            dict: Dictionary containing fitted temperature and weights
        """
        # Move to CPU for scipy optimization
        if torch.is_tensor(val_logits):
            logits_np = val_logits.detach().cpu().numpy()
        else:
            logits_np = val_logits
            
        if torch.is_tensor(val_labels):
            labels_np = val_labels.detach().cpu().numpy()
        else:
            labels_np = val_labels
            
        # Infer number of classes if not provided
        if self.n_classes is None:
            self.n_classes = logits_np.shape[1]
            
        # Convert labels to one-hot if needed
        if len(labels_np.shape) == 1:
            label_onehot = np.eye(self.n_classes)[labels_np]
        else:
            label_onehot = labels_np
            
        print(f"Fitting ETS calibrator with {self.n_classes} classes...")
        
        # Step 1: Find optimal temperature using original function
        # NOTE: According to original code, temperature scaling ALWAYS uses MSE loss
        print("Finding optimal temperature...")
        self.temperature = temperature_scaling(logits_np, label_onehot, loss='mse')[0]
        print(f"Optimal temperature: {self.temperature:.4f}")
        
        # Step 2: Find optimal ensemble weights using original function
        print("Finding optimal ensemble weights...")
        self.weights = ensemble_scaling(logits_np, label_onehot, self.loss_type, self.temperature, self.n_classes)
        print(f"Optimal weights: {self.weights}")
        
        self.is_fitted = True
        
        return {
            'temperature': self.temperature,
            'weights': self.weights.tolist()
        }
    
    def calibrate(self, logits, return_logits=False):
        """
        Apply ETS calibration to logits using the exact original implementation.
        
        Args:
            logits (torch.Tensor or numpy.ndarray): Input logits
            return_logits (bool): If True, return calibrated logits; otherwise return probabilities
            
        Returns:
            numpy.ndarray or torch.Tensor: Calibrated probabilities or logits
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calibration. Call fit() first.")
        
        # Convert to numpy if needed for original implementation
        input_was_tensor = torch.is_tensor(logits)
        if input_was_tensor:
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = logits.copy()
            
        # Use the exact original implementation logic
        # p1 = original softmax
        p1 = np.exp(logits_np) / np.sum(np.exp(logits_np), 1)[:, None]
        
        # p0 = temperature scaled
        logits_scaled = logits_np / self.temperature
        p0 = np.exp(logits_scaled) / np.sum(np.exp(logits_scaled), 1)[:, None]
        
        # p2 = uniform distribution
        p2 = np.ones_like(p0) / self.n_classes
        
        # Ensemble combination: p = w[0]*p0 + w[1]*p1 + w[2]*p2
        p_calibrated = self.weights[0] * p0 + self.weights[1] * p1 + self.weights[2] * p2
        
        if return_logits:
            # Convert back to logits (pseudo-logits)
            p_calibrated = np.clip(p_calibrated, 1e-8, 1-1e-8)
            result = np.log(p_calibrated)
        else:
            result = p_calibrated
            
        # Convert back to tensor if input was tensor
        if input_was_tensor:
            return torch.tensor(result, dtype=torch.float32, device=logits.device)
        else:
            return result
    
    def get_temperature(self):
        """Get the fitted temperature value."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted first.")
        return self.temperature
    
    def get_weights(self):
        """Get the fitted ensemble weights."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted first.")
        return self.weights.tolist()
    
    def get_params(self):
        """Get all fitted parameters."""
        return {
            'temperature': self.get_temperature(),
            'weights': self.get_weights(),
            'n_classes': self.n_classes,
            'loss_type': self.loss_type
        }