# weighted_soft_ece.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSoftECE(nn.Module):
    """
    SoftECE with positive per-sample weights w(x).
    If sample_weights=None, it behaves like vanilla SoftECE.
    """
    def __init__(self, n_bins=15, sigma=0.05, eps=1e-6):
        super().__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.eps = eps
        self.register_buffer(
            'bin_centers',
            torch.linspace(1/(2*self.n_bins), 1 - 1/(2*self.n_bins), self.n_bins)
        )

    def forward(self, logits, labels, sample_weights: torch.Tensor | None = None):
        """
        Args:
            logits: [B, C]
            labels: [B] (int64)
            sample_weights: [B] non-negative (can depend on logits). If None -> ones.
        """
        device = logits.device
        B = logits.size(0)

        targets = labels.to(device)
        probs = F.softmax(logits, dim=1)                 # [B, C]
        confidences, predictions = probs.max(dim=1)      # [B]
        accuracies = (predictions == targets).float()    # [B]

        # Gaussian soft-binning in p-space
        diff = confidences.unsqueeze(1) - self.bin_centers.unsqueeze(0)  # [B, n_bins]
        weights = torch.exp(-0.5 * (diff**2) / (self.sigma**2))          # [B, n_bins]
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)

        # Sample weights (broadcast onto bins)
        if sample_weights is None:
            w = torch.ones(B, device=device)
        else:
            w = sample_weights.to(device)
        w = torch.clamp(w, min=0.0)                                      # ensure non-negative
        W = weights * w.unsqueeze(1)                                     # [B, n_bins]

        # Bin aggregates with sample weighting
        sum_weights_in_bin = W.sum(dim=0)                                # [n_bins]
        weighted_confidence = W * confidences.unsqueeze(1)
        avg_conf_in_bin = weighted_confidence.sum(dim=0) / (sum_weights_in_bin + self.eps)

        weighted_accuracy = W * accuracies.unsqueeze(1)
        avg_acc_in_bin = weighted_accuracy.sum(dim=0) / (sum_weights_in_bin + self.eps)

        # Proportion per bin relative to total sample weight
        prop_in_bin = sum_weights_in_bin / (w.sum() + self.eps)

        ece_per_bin = torch.abs(avg_conf_in_bin - avg_acc_in_bin)
        return torch.sum(ece_per_bin * prop_in_bin)

# ---------- Optional helpers to build sample weights ----------





# def soft_tail_proxy_g_tau(logits: torch.Tensor, tau: float = 5.0) -> torch.Tensor:
#     """
#     g_tau = z_(1) - (1/tau)*logsumexp(tau * z_(j), j>=2 )  (per sample).
#     Returns [B], larger means stronger strongest-competitor gap.
#     """
#     tau = float(tau)
#     z_sorted, _ = torch.sort(logits, dim=1, descending=True)  # [B, C]
#     z1 = z_sorted[:, 0:1]
#     tail = z_sorted[:, 1:]
#     lse = torch.logsumexp(tau * tail, dim=1, keepdim=True) / tau
#     g_tau = (z1 - lse).squeeze(1)
#     return g_tau

# def participation_ratio_k_eff(logits: torch.Tensor, tau: float = 5.0, eps: float = 1e-8) -> torch.Tensor:
#     """
#     k_eff = (sum exp(tau*z_tail))^2 / sum exp(2*tau*z_tail), tail over j>=2. Returns [B].
#     """
#     z_sorted, _ = torch.sort(logits, dim=1, descending=True)
#     tail = z_sorted[:, 1:]
#     w = torch.exp(tau * tail)
#     s1 = w.sum(dim=1)
#     s2 = (w*w).sum(dim=1) + eps
#     return (s1*s1) / s2
