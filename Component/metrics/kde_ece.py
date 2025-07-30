import torch
import torch.nn as nn
import torch.nn.functional as F


class KDEECE(nn.Module):
    """
    Compute KDE-based Expected Calibration Error (ECE) using kernel density estimation.
    
    This implementation is based on "A Consistent and Differentiable Lp Canonical Calibration Error Estimator"
    and provides a differentiable calibration error metric that can be used both for evaluation and training.
    """
    
    def __init__(self, p=1, mc_type='canonical', bandwidth=None):
        """
        Initialize KDE ECE metric.
        
        Args:
            p (int): The p-norm to use (typically 1 or 2). Default: 1
            mc_type (str): Multiclass calibration type. Options: 'canonical', 'marginal', 'top_label'. Default: 'canonical'
            bandwidth (float, optional): Kernel bandwidth. If None, will be automatically selected. Default: None
        """
        super(KDEECE, self).__init__()
        self.p = p
        self.mc_type = mc_type
        self.bandwidth = bandwidth
        
    def forward(self, logits=None, labels=None, softmaxes=None):
        """
        Compute KDE-based ECE.
        
        Args:
            logits (torch.Tensor, optional): Logits from model output
            labels (torch.Tensor): Ground truth labels
            softmaxes (torch.Tensor, optional): Softmax probabilities. If None, computed from logits
            
        Returns:
            torch.Tensor: KDE ECE value
        """
        if softmaxes is None:
            if logits is None:
                raise ValueError("Either logits or softmaxes must be provided")
            softmaxes = F.softmax(logits, dim=1)
            
        # Ensure inputs are on the same device
        device = softmaxes.device
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, device=device)
        labels = labels.to(device)
        
        # Auto-select bandwidth if not provided
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = self._get_bandwidth(softmaxes, device)
            
        return self._get_ece_kde(softmaxes, labels, bandwidth, self.p, self.mc_type, device)
    
    def _get_bandwidth(self, f, device):
        """
        Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).
        
        Args:
            f (torch.Tensor): Probability scores [num_samples, num_classes]
            device (torch.device): Device type
            
        Returns:
            float: Optimal bandwidth
        """
        bandwidths = torch.cat((torch.logspace(start=-5, end=-1, steps=15, device=device), 
                               torch.linspace(0.2, 1, steps=5, device=device)))
        max_b = -1
        max_l = 0
        n = len(f)
        
        for b in bandwidths:
            log_kern = self._get_kernel(f, b, device)
            log_fhat = torch.logsumexp(log_kern, 1) - torch.log(torch.tensor(n-1, dtype=torch.float, device=device))
            l = torch.sum(log_fhat)
            if l > max_l:
                max_l = l
                max_b = b

        return max_b.item()
    
    def _get_ece_kde(self, f, y, bandwidth, p, mc_type, device):
        """
        Calculate an estimate of Lp calibration error.
        
        Args:
            f (torch.Tensor): Probability scores [num_samples, num_classes]
            y (torch.Tensor): Labels [num_samples]
            bandwidth (float): Kernel bandwidth
            p (int): The p-norm
            mc_type (str): Multiclass calibration type
            device (torch.device): Device type
            
        Returns:
            torch.Tensor: Lp calibration error estimate
        """
        self._check_input(f, bandwidth, mc_type)
        
        if f.shape[1] == 1:
            return 2 * self._get_ratio_binary(f, y, bandwidth, p, device)
        else:
            if mc_type == 'canonical':
                return self._get_ratio_canonical(f, y, bandwidth, p, device)
            elif mc_type == 'marginal':
                return self._get_ratio_marginal_vect(f, y, bandwidth, p, device)
            elif mc_type == 'top_label':
                return self._get_ratio_toplabel(f, y, bandwidth, p, device)
    
    def _get_ratio_binary(self, f, y, bandwidth, p, device):
        """Binary classification ECE calculation."""
        assert f.shape[1] == 1
        log_kern = self._get_kernel(f, bandwidth, device)
        return self._get_kde_for_ece(f, y, log_kern, p)
    
    def _get_ratio_canonical(self, f, y, bandwidth, p, device):
        """Canonical multiclass calibration error calculation."""
        if f.shape[1] > 60:
            # More numerically stable for large number of classes
            return self._get_ratio_canonical_log(f, y, bandwidth, p, device)
        
        log_kern = self._get_kernel(f, bandwidth, device)
        kern = torch.exp(log_kern)
        
        y_onehot = F.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
        kern_y = torch.matmul(kern, y_onehot)
        den = torch.sum(kern, dim=1)
        # Avoid division by 0
        den = torch.clamp(den, min=1e-10)
        
        ratio = kern_y / den.unsqueeze(-1)
        ratio = torch.sum(torch.abs(ratio - f)**p, dim=1)
        
        return torch.mean(ratio)
    
    def _get_ratio_canonical_log(self, f, y, bandwidth, p, device='cpu'):
        """Numerically stable canonical multiclass calibration error calculation."""
        log_kern = self._get_kernel(f, bandwidth, device)
        y_onehot = F.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
        log_y = torch.log(y_onehot + 1e-45)  # Add small epsilon to avoid log(0)
        log_den = torch.logsumexp(log_kern, dim=1)
        final_ratio = 0
        
        for k in range(f.shape[1]):
            log_kern_y = log_kern + (torch.ones([f.shape[0], 1], device=device) * log_y[:, k].unsqueeze(0))
            log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
            inner_ratio = torch.exp(log_inner_ratio)
            inner_diff = torch.abs(inner_ratio - f[:, k])**p
            final_ratio += inner_diff
            
        return torch.mean(final_ratio)
    
    def _get_ratio_marginal_vect(self, f, y, bandwidth, p, device):
        """Marginal multiclass calibration error calculation."""
        y_onehot = F.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
        log_kern_vect = self._beta_kernel(f, f, bandwidth).squeeze()
        log_kern_diag = torch.diag(torch.finfo(torch.float).min * torch.ones(len(f), device=device))
        
        # Multiclass case
        log_kern_diag_repeated = f.shape[1] * [log_kern_diag]
        log_kern_diag_repeated = torch.stack(log_kern_diag_repeated, dim=2)
        log_kern_vect = log_kern_vect + log_kern_diag_repeated
        
        return self._get_kde_for_ece_vect(f, y_onehot, log_kern_vect, p)
    
    def _get_ratio_toplabel(self, f, y, bandwidth, p, device):
        """Top-label calibration error calculation."""
        f_max, indices = torch.max(f, 1)
        f_max = f_max.unsqueeze(-1)
        y_max = (y == indices).to(torch.int)
        
        return self._get_ratio_binary(f_max, y_max, bandwidth, p, device)
    
    def _get_kde_for_ece_vect(self, f, y, log_kern, p):
        """KDE calculation for vectorized case."""
        log_kern_y = log_kern * y
        # Use -inf instead of 0 in log space
        log_kern_y[log_kern_y == 0] = torch.finfo(torch.float).min
        
        log_num = torch.logsumexp(log_kern_y, dim=1)
        log_den = torch.logsumexp(log_kern, dim=1)
        
        log_ratio = log_num - log_den
        ratio = torch.exp(log_ratio)
        ratio = torch.abs(ratio - f)**p
        
        return torch.sum(torch.mean(ratio, dim=0))
    
    def _get_kde_for_ece(self, f, y, log_kern, p):
        """KDE calculation for ECE."""
        f = f.squeeze()
        N = len(f)
        # Select entries where y == 1
        idx = torch.where(y == 1)[0]
        if not idx.numel():
            return torch.sum((torch.abs(-f))**p) / N
        
        if idx.numel() == 1:
            # Handle single positive example
            log_kern = torch.cat((log_kern[:idx], log_kern[idx+1:]))
            f_one = f[idx]
            f = torch.cat((f[:idx], f[idx+1:]))
            
        log_kern_y = torch.index_select(log_kern, 1, idx)
        
        log_num = torch.logsumexp(log_kern_y, dim=1)
        log_den = torch.logsumexp(log_kern, dim=1)
        
        log_ratio = log_num - log_den
        ratio = torch.exp(log_ratio)
        ratio = torch.abs(ratio - f)**p
        
        if idx.numel() == 1:
            return (ratio.sum() + f_one ** p) / N
            
        return torch.mean(ratio)
    
    def _get_kernel(self, f, bandwidth, device):
        """Get appropriate kernel based on input dimensions."""
        if f.shape[1] == 1:
            log_kern = self._beta_kernel(f, f, bandwidth).squeeze()
        else:
            log_kern = self._dirichlet_kernel(f, bandwidth).squeeze()
        # Use -inf on diagonal
        return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f), device=device))
    
    def _beta_kernel(self, z, zi, bandwidth=0.1):
        """Beta kernel for binary case."""
        p = zi / bandwidth + 1
        q = (1-zi) / bandwidth + 1
        z = z.unsqueeze(-2)
        
        log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
        log_num = (p-1) * torch.log(z + 1e-45) + (q-1) * torch.log(1-z + 1e-45)
        log_beta_pdf = log_num - log_beta
        
        return log_beta_pdf
    
    def _dirichlet_kernel(self, z, bandwidth=0.1):
        """Dirichlet kernel for multiclass case."""
        alphas = z / bandwidth + 1
        
        log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
        log_num = torch.matmul(torch.log(z + 1e-45), (alphas-1).T)
        log_dir_pdf = log_num - log_beta
        
        return log_dir_pdf
    
    def _check_input(self, f, bandwidth, mc_type):
        """Validate inputs."""
        assert not self._isnan(f), "Input contains NaN values"
        assert len(f.shape) == 2, "Input must be 2D tensor"
        assert bandwidth > 0, "Bandwidth must be positive"
        assert torch.min(f) >= 0, "Probabilities must be non-negative"
        assert torch.max(f) <= 1, "Probabilities must be <= 1"
        assert mc_type in ['canonical', 'marginal', 'top_label'], f"Invalid mc_type: {mc_type}"
    
    def _isnan(self, a):
        """Check for NaN values."""
        return torch.any(torch.isnan(a))