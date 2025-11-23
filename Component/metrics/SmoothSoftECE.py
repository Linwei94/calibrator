# smooth_soft_ece.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothSoftECE(nn.Module):
    """
    SoftECE with a smooth convex discrepancy φδ.
    mode='softabs' uses φδ(r) = sqrt(r^2 + δ^2)  (>= |r|).
    mode='huber'  uses classic Huber; note Huber ≤ |r| for large r (different theory).
    """
    def __init__(self, n_bins=15, sigma=0.05, delta=1e-3, mode: str = "huber", eps=1e-6):
        super().__init__()
        assert mode in ("softabs", "huber")
        self.n_bins = n_bins
        self.sigma = sigma
        self.delta = delta
        self.mode = mode
        self.eps = eps
        self.register_buffer(
            'bin_centers',
            torch.linspace(1/(2*self.n_bins), 1 - 1/(2*self.n_bins), self.n_bins)
        )

    def _phi(self, r: torch.Tensor) -> torch.Tensor:
        # print(f"mode: {self.mode}")
        if self.mode == "softabs":
            return torch.sqrt(r*r + self.delta*self.delta)   # ≥ |r|
        # Huber
        absr = torch.abs(r)
        d = self.delta
        return torch.where(absr <= d, 0.5 * r*r / d, absr - 0.5 * d)

    def forward(self, logits, labels):
        device = logits.device
        targets = labels.to(device)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        accuracies = (predictions == targets).float()

        diff = confidences.unsqueeze(1) - self.bin_centers.unsqueeze(0)
        weights = torch.exp(-0.5 * (diff**2) / (self.sigma**2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)

        sum_weights_in_bin = weights.sum(dim=0)
        avg_conf_in_bin = (weights * confidences.unsqueeze(1)).sum(dim=0) / (sum_weights_in_bin + self.eps)
        avg_acc_in_bin  = (weights * accuracies.unsqueeze(1)).sum(dim=0)  / (sum_weights_in_bin + self.eps)

        prop_in_bin = sum_weights_in_bin / (confidences.size(0) + self.eps)

        resid = (avg_conf_in_bin - avg_acc_in_bin)
        # ece_per_bin = self._phi(resid)
        ece_per_bin = self._phi(resid) - self._phi(torch.zeros(1, device=resid.device))
        return torch.sum(ece_per_bin * prop_in_bin)
