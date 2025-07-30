import torch
import torch.nn as nn
import torch.nn.functional as F


class KDEECELoss(nn.Module):
    """
    KDE-based Expected Calibration Error Loss for training.
    
    This is a training-optimized version of KDE ECE that can be used as a regularization
    term during model training. It includes numerical stability improvements and 
    gradient-friendly computations.
    """
    
    def __init__(self, p=1, mc_type='canonical', bandwidth=None, lambda_reg=1.0, 
                 auto_bandwidth=True, stable_mode=True):
        """
        Initialize KDE ECE Loss.
        
        Args:
            p (int): The p-norm to use (typically 1 or 2). Default: 1
            mc_type (str): Multiclass calibration type. Options: 'canonical', 'marginal', 'top_label'. Default: 'canonical'
            bandwidth (float, optional): Kernel bandwidth. If None, will be automatically selected. Default: None
            lambda_reg (float): Regularization weight for the ECE loss. Default: 1.0
            auto_bandwidth (bool): Whether to automatically select bandwidth during training. Default: True
            stable_mode (bool): Use numerically stable computation for large number of classes. Default: True
        """
        super(KDEECELoss, self).__init__()
        self.p = p
        self.mc_type = mc_type
        self.bandwidth = bandwidth
        self.lambda_reg = lambda_reg
        self.auto_bandwidth = auto_bandwidth
        self.stable_mode = stable_mode
        
        # Cache for bandwidth to avoid recomputation
        self._cached_bandwidth = None
        self._cache_size = 0
        
    def forward(self, logits, labels, primary_loss=None):
        """
        Compute KDE ECE loss for training.
        
        Args:
            logits (torch.Tensor): Model logits
            labels (torch.Tensor): Ground truth labels
            primary_loss (torch.Tensor, optional): Primary training loss to combine with
            
        Returns:
            torch.Tensor: Combined loss (primary_loss + lambda_reg * kde_ece_loss)
        """
        softmaxes = F.softmax(logits, dim=1)
        device = softmaxes.device
        
        # Handle bandwidth selection
        bandwidth = self._get_training_bandwidth(softmaxes, device)
        
        # Compute KDE ECE
        kde_ece = self._compute_kde_ece_stable(softmaxes, labels, bandwidth, device)
        
        # Apply regularization weight
        kde_ece_loss = self.lambda_reg * kde_ece
        
        if primary_loss is not None:
            return primary_loss + kde_ece_loss
        else:
            return kde_ece_loss
    
    def _get_training_bandwidth(self, softmaxes, device):
        """Get bandwidth for training, using caching for efficiency."""
        if self.bandwidth is not None:
            return self.bandwidth
            
        if not self.auto_bandwidth:
            return 0.1  # Default bandwidth
            
        # Use cached bandwidth if batch size hasn't changed significantly
        current_size = softmaxes.size(0)
        if (self._cached_bandwidth is not None and 
            abs(current_size - self._cache_size) < 10):
            return self._cached_bandwidth
            
        # Compute new bandwidth (simplified for training efficiency)
        self._cached_bandwidth = self._select_bandwidth_fast(softmaxes, device)
        self._cache_size = current_size
        return self._cached_bandwidth
    
    def _select_bandwidth_fast(self, f, device):
        """Fast bandwidth selection suitable for training."""
        # Simplified bandwidth selection with fewer candidates
        bandwidths = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5], device=device)
        max_b = 0.1
        max_l = -float('inf')
        n = f.size(0)
        
        # Subsample for efficiency in large batches
        if n > 100:
            indices = torch.randperm(n, device=device)[:100]
            f_sub = f[indices]
        else:
            f_sub = f
            
        for b in bandwidths:
            try:
                log_kern = self._get_kernel_fast(f_sub, b, device)
                log_fhat = torch.logsumexp(log_kern, 1) - torch.log(torch.tensor(f_sub.size(0)-1, dtype=torch.float, device=device))
                l = torch.sum(log_fhat)
                if l > max_l:
                    max_l = l
                    max_b = b.item()
            except:
                continue
                
        return max_b
    
    def _compute_kde_ece_stable(self, f, y, bandwidth, device):
        """Compute KDE ECE with numerical stability."""
        try:
            if f.shape[1] == 1:
                return 2 * self._get_ratio_binary_stable(f, y, bandwidth, device)
            else:
                if self.mc_type == 'canonical':
                    if self.stable_mode and f.shape[1] > 20:
                        return self._get_ratio_canonical_log_stable(f, y, bandwidth, device)
                    else:
                        return self._get_ratio_canonical_stable(f, y, bandwidth, device)
                elif self.mc_type == 'top_label':
                    return self._get_ratio_toplabel_stable(f, y, bandwidth, device)
                else:
                    # Fallback to canonical for unsupported types during training
                    return self._get_ratio_canonical_stable(f, y, bandwidth, device)
        except Exception as e:
            # Graceful fallback in case of numerical issues
            print(f"Warning: KDE ECE computation failed ({e}), using fallback")
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _get_ratio_canonical_stable(self, f, y, bandwidth, device):
        """Numerically stable canonical ratio computation."""
        log_kern = self._get_kernel_fast(f, bandwidth, device)
        
        # Clamp to avoid extreme values
        log_kern = torch.clamp(log_kern, min=-50, max=50)
        kern = torch.exp(log_kern)
        
        y_onehot = F.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
        kern_y = torch.matmul(kern, y_onehot)
        den = torch.sum(kern, dim=1)
        den = torch.clamp(den, min=1e-8)
        
        ratio = kern_y / den.unsqueeze(-1)
        diff = torch.abs(ratio - f)
        
        if self.p == 1:
            ratio_loss = torch.sum(diff, dim=1)
        else:
            ratio_loss = torch.sum(diff**self.p, dim=1)
            
        return torch.mean(ratio_loss)
    
    def _get_ratio_canonical_log_stable(self, f, y, bandwidth, device):
        """Log-space canonical ratio computation for numerical stability."""
        log_kern = self._get_kernel_fast(f, bandwidth, device)
        log_kern = torch.clamp(log_kern, min=-50, max=50)
        
        y_onehot = F.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
        log_y = torch.log(y_onehot + 1e-10)
        log_den = torch.logsumexp(log_kern, dim=1)
        
        final_ratio = 0
        for k in range(min(f.shape[1], 50)):  # Limit for training efficiency
            log_kern_y = log_kern + log_y[:, k].unsqueeze(0)
            log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
            inner_ratio = torch.exp(torch.clamp(log_inner_ratio, min=-50, max=50))
            inner_diff = torch.abs(inner_ratio - f[:, k])
            
            if self.p == 1:
                final_ratio += inner_diff
            else:
                final_ratio += inner_diff**self.p
                
        return torch.mean(final_ratio)
    
    def _get_ratio_binary_stable(self, f, y, bandwidth, device):
        """Stable binary ratio computation."""
        log_kern = self._get_kernel_fast(f, bandwidth, device)
        return self._get_kde_for_ece_stable(f, y, log_kern)
    
    def _get_ratio_toplabel_stable(self, f, y, bandwidth, device):
        """Stable top-label ratio computation."""
        f_max, indices = torch.max(f, 1)
        f_max = f_max.unsqueeze(-1)
        y_max = (y == indices).to(torch.int)
        return self._get_ratio_binary_stable(f_max, y_max, bandwidth, device)
    
    def _get_kde_for_ece_stable(self, f, y, log_kern):
        """Stable KDE computation for ECE."""
        f = f.squeeze()
        N = len(f)
        idx = torch.where(y == 1)[0]
        
        if not idx.numel():
            if self.p == 1:
                return torch.sum(torch.abs(f)) / N
            else:
                return torch.sum(torch.abs(f)**self.p) / N
        
        log_kern_y = torch.index_select(log_kern, 1, idx)
        log_kern = torch.clamp(log_kern, min=-50, max=50)
        log_kern_y = torch.clamp(log_kern_y, min=-50, max=50)
        
        log_num = torch.logsumexp(log_kern_y, dim=1)
        log_den = torch.logsumexp(log_kern, dim=1)
        
        log_ratio = log_num - log_den
        ratio = torch.exp(torch.clamp(log_ratio, min=-50, max=50))
        
        if self.p == 1:
            diff = torch.abs(ratio - f)
        else:
            diff = torch.abs(ratio - f)**self.p
            
        return torch.mean(diff)
    
    def _get_kernel_fast(self, f, bandwidth, device):
        """Fast kernel computation."""
        if f.shape[1] == 1:
            log_kern = self._beta_kernel_fast(f, f, bandwidth)
        else:
            log_kern = self._dirichlet_kernel_fast(f, bandwidth)
            
        # Add diagonal mask
        diag_mask = torch.diag(torch.full((f.size(0),), float('-inf'), device=device))
        return log_kern + diag_mask
    
    def _beta_kernel_fast(self, z, zi, bandwidth):
        """Fast beta kernel computation."""
        eps = 1e-8
        p = zi / bandwidth + 1
        q = (1 - zi) / bandwidth + 1
        z = z.unsqueeze(-2)
        
        log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
        log_num = (p - 1) * torch.log(z + eps) + (q - 1) * torch.log(1 - z + eps)
        log_beta_pdf = log_num - log_beta
        
        return log_beta_pdf.squeeze()
    
    def _dirichlet_kernel_fast(self, z, bandwidth):
        """Fast Dirichlet kernel computation."""
        eps = 1e-8
        alphas = z / bandwidth + 1
        
        log_beta = torch.sum(torch.lgamma(alphas), dim=1) - torch.lgamma(torch.sum(alphas, dim=1))
        log_num = torch.matmul(torch.log(z + eps), (alphas - 1).T)
        log_dir_pdf = log_num - log_beta
        
        return log_dir_pdf
    
    def set_lambda(self, lambda_reg):
        """Update the regularization weight."""
        self.lambda_reg = lambda_reg
    
    def get_lambda(self):
        """Get current regularization weight."""
        return self.lambda_reg