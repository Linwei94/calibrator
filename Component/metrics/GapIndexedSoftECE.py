# gap_indexed_soft_ece.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GapIndexedSoftECE(nn.Module):
    """
    SoftECE where the Gaussian kernel lives in a monotone reparameterization h(p).
    Choices: h(p)=p (identity), h(p)=logit(p) for sensitivity-aligned neighborhoods.
    The mismatch |avg_conf - avg_acc| is still computed in probability space.
    """
    def __init__(self, n_bins=15, sigma_h=0.2, mapping: str = "logit", eps=1e-6):
        """
        Args:
            sigma_h: bandwidth in h-space (note: not the same units as p-space).
            mapping: 'logit' or 'identity'
        """
        super().__init__()
        assert mapping in ("logit", "identity")
        self.n_bins = n_bins
        self.sigma_h = sigma_h
        self.mapping = mapping
        self.eps = eps

        # bins still defined in p-space; we transform them with h for the kernel
        self.register_buffer(
            'bin_centers_p',
            torch.linspace(1/(2*self.n_bins), 1 - 1/(2*self.n_bins), self.n_bins)
        )

    @staticmethod
    def _logit(u, eps):
        u = torch.clamp(u, eps, 1 - eps)
        return torch.log(u) - torch.log(1 - u)

    @staticmethod
    def _logit_prime(u, eps):
        u = torch.clamp(u, eps, 1 - eps)
        return 1.0 / (u * (1.0 - u))

    def _h(self, u):
        if self.mapping == "identity":
            return u
        return self._logit(u, self.eps)

    def _hprime(self, u):
        if self.mapping == "identity":
            return torch.ones_like(u)
        return self._logit_prime(u, self.eps)

    def forward(self, logits, labels):
        device = logits.device
        targets = labels.to(device)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        accuracies = (predictions == targets).float()

        # Transform to h-space for distances; include Jacobian h'(u) on bin centers
        h_conf = self._h(confidences)                                        # [B]
        h_bins = self._h(self.bin_centers_p)                                  # [n_bins]
        jac = self._hprime(self.bin_centers_p)                                # [n_bins]

        diff_h = h_conf.unsqueeze(1) - h_bins.unsqueeze(0)                    # [B, n_bins]
        weights = torch.exp(-0.5 * (diff_h**2) / (self.sigma_h**2)) * jac     # positive kernel
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)

        sum_weights_in_bin = weights.sum(dim=0)
        avg_conf_in_bin = (weights * confidences.unsqueeze(1)).sum(dim=0) / (sum_weights_in_bin + self.eps)
        avg_acc_in_bin  = (weights * accuracies .unsqueeze(1)).sum(dim=0) / (sum_weights_in_bin + self.eps)

        prop_in_bin = sum_weights_in_bin / (confidences.size(0) + self.eps)

        ece_per_bin = torch.abs(avg_conf_in_bin - avg_acc_in_bin)
        return torch.sum(ece_per_bin * prop_in_bin)
