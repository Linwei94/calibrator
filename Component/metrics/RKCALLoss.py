# rkcal_loss.py
from __future__ import annotations
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RKCALLoss(nn.Module):
    r"""
    RKHS Calibration Loss (RKCAL) with Random Fourier Features.

    Definition (population):
        For class c, let p_c ∈ [0,1] be the predicted probability and y_c ∈ {0,1}
        the one-hot label. For a universal kernel k on [0,1] with RKHS ℋ and
        feature map φ, define μ_c = E[(y_c - p_c) φ(p_c)] ∈ ℋ and

            L_RKCAL = Σ_{c=1}^K || μ_c ||_ℋ^2.

        L_RKCAL = 0  iff the model is calibrated classwise (E[y_c | p_c] = p_c a.s.).
        It upper-bounds a wide family of ECE-style errors via RKHS Cauchy–Schwarz.

    Mini-batch estimator (RFF approximation):
        Using features φ(u) ≈ sqrt(2/D) * cos( W * u + b ), with W ~ N(0, 1/ℓ^2),
        b ~ Uniform(0, 2π), define

            μ̂_c = (1/n) Σ_i (y_ic - p_ic) φ(p_ic)
            L̂_RKCAL ≈ Σ_c || μ̂_c ||_2^2.

    Args:
        num_classes: K.
        rff_dim: number of random features D (typ. 32–256).
        lengthscale: Gaussian kernel length-scale ℓ (>0). Larger = smoother.
        share_features: if True, all classes share the same (W, b); else per-class.
        clamp_eps: clamp probabilities into [eps, 1-eps] for numerical stability.
        reduction: 'sum' | 'mean' | 'none'  (over classes).
        from_logits: default behavior if not explicitly overridden in forward().
                     If True, inputs are logits; else probabilities in [0,1].
        device, dtype: optional initialization overrides for parameter buffers.

    Forward:
        loss = criterion(inputs, target, sample_weight=None, class_weight=None,
                         from_logits=None, return_per_class=False)

        inputs: (N, K) logits or probabilities.
        target: (N,) class indices   OR   (N, K) one-hot/soft labels.
        sample_weight (optional): (N,) or (N,1) per-sample nonnegative weights,
                                  multiplies residuals (y - p). Use to emphasize
                                  "sensitive" samples.
        class_weight (optional): (K,) nonnegative weights per class.
        from_logits (optional): overrides constructor default.
        return_per_class: if True, returns (loss, per_class_losses[K]).

    Notes:
        • This implements the (biased) V-statistic estimator described in the theory.
        • To re-sample RFF (e.g., per run/seed), call `resample_features()`.
        • You can tune `lengthscale` with a median heuristic over p_c on a held-out set.
    """

    def __init__(
        self,
        num_classes: int,
        rff_dim: int = 128,
        lengthscale: float = 0.2,
        share_features: bool = True,
        clamp_eps: float = 1e-6,
        reduction: str = "sum",
        from_logits: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert num_classes >= 2, "num_classes must be ≥ 2"
        assert rff_dim >= 1, "rff_dim must be ≥ 1"
        assert lengthscale > 0, "lengthscale must be > 0"
        assert reduction in {"sum", "mean", "none"}

        self.K = int(num_classes)
        self.D = int(rff_dim)
        self.lengthscale = float(lengthscale)
        self.share_features = bool(share_features)
        self.clamp_eps = float(clamp_eps)
        self.reduction = reduction
        self.default_from_logits = bool(from_logits)

        factory_kwargs = {"device": device, "dtype": dtype}

        if share_features:
            # Shared RFF parameters across classes
            self.register_buffer("W", torch.empty(self.D, **factory_kwargs))
            self.register_buffer("b", torch.empty(self.D, **factory_kwargs))
        else:
            # Per-class RFF parameters
            self.register_buffer("W", torch.empty(self.K, self.D, **factory_kwargs))
            self.register_buffer("b", torch.empty(self.K, self.D, **factory_kwargs))

        self.resample_features()  # initialize W, b

    # ----------------------- utilities -----------------------

    @torch.no_grad()
    def resample_features(self) -> None:
        """Resample the random Fourier features (W, b)."""
        sigma_inv = 1.0 / max(self.lengthscale, 1e-12)  # 1/ℓ
        if self.share_features:
            self.W.normal_(mean=0.0, std=sigma_inv)
            self.b.uniform_(0.0, 2.0 * math.pi)
        else:
            self.W.normal_(mean=0.0, std=sigma_inv)
            self.b.uniform_(0.0, 2.0 * math.pi)

    @torch.no_grad()
    def set_lengthscale(self, lengthscale: float, resample: bool = True) -> None:
        """Update lengthscale ℓ; optionally resample features with new ℓ."""
        assert lengthscale > 0
        self.lengthscale = float(lengthscale)
        if resample:
            self.resample_features()

    def _one_hot(self, y: torch.Tensor, K: int) -> torch.Tensor:
        if y.dim() == 1:
            return F.one_hot(y.to(torch.int64), num_classes=K).to(dtype=torch.get_default_dtype(), device=y.device)
        elif y.dim() == 2:
            assert y.size(1) == K, "target second dim must match num_classes"
            return y
        else:
            raise ValueError("target must be shape (N,) or (N,K)")

    def _rff_features(self, u: torch.Tensor, class_index: Optional[int] = None) -> torch.Tensor:
        """
        Compute φ(u) = sqrt(2/D) * cos(W*u + b) for scalar inputs u in [0,1].
        Shapes:
            u: (N,)
            returns: (N, D)
        """
        if self.share_features:
            W = self.W  # (D,)
            b = self.b  # (D,)
        else:
            assert class_index is not None, "class_index required when share_features=False"
            W = self.W[class_index]  # (D,)
            b = self.b[class_index]  # (D,)

        # Broadcasting: (N,1) * (D,) -> (N,D)
        proj = u.unsqueeze(-1) * W.unsqueeze(0) + b.unsqueeze(0)
        phi = torch.cos(proj) * math.sqrt(2.0 / self.D)
        return phi

    # ----------------------- forward -----------------------

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        class_weight: Optional[torch.Tensor] = None,
        *,
        from_logits: Optional[bool] = None,
        return_per_class: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute RKCAL loss (and optionally classwise components).

        Returns:
            loss  (scalar if reduction != 'none', else shape (K,))
            optionally: per_class (K,) always returned on request (pre-reduction)
        """
        if from_logits is None:
            from_logits = self.default_from_logits

        # inputs -> probabilities p ∈ (0,1), shape (N,K)
        if from_logits:
            p = F.softmax(inputs, dim=-1)
        else:
            p = inputs
        if self.clamp_eps is not None and self.clamp_eps > 0:
            p = p.clamp(min=self.clamp_eps, max=1.0 - self.clamp_eps)

        N, K = p.shape
        assert K == self.K, f"inputs second dim ({K}) != num_classes ({self.K})"

        # targets -> one-hot (N,K)
        y = self._one_hot(target, K=K).to(p.dtype)

        # residuals r = y - p, shape (N,K)
        r = y - p

        # optional per-sample weights (N,) or (N,1)
        if sample_weight is not None:
            if sample_weight.dim() == 1:
                sw = sample_weight.view(-1, 1)
            elif sample_weight.dim() == 2 and sample_weight.size(1) == 1:
                sw = sample_weight
            else:
                raise ValueError("sample_weight must be shape (N,) or (N,1)")
            r = r * sw

        # optional per-class weights (K,)
        if class_weight is not None:
            assert class_weight.dim() == 1 and class_weight.numel() == K
            cw = class_weight.view(1, K)
        else:
            cw = None

        # compute μ̂_c and its squared norm per class
        per_class = []
        for c in range(K):
            u_c = p[:, c]                    # (N,)
            phi_c = self._rff_features(u_c, class_index=None if self.share_features else c)  # (N,D)
            r_c = r[:, c].unsqueeze(1)      # (N,1)

            # μ̂_c = (1/N) Σ_i r_ic * φ(p_ic)
            mu_c = (r_c * phi_c).mean(dim=0)  # (D,)
            l_c = (mu_c * mu_c).sum()         # scalar
            if cw is not None:
                l_c = l_c * cw[0, c]
            per_class.append(l_c)

        per_class = torch.stack(per_class, dim=0)  # (K,)

        if self.reduction == "sum":
            loss = per_class.sum()
        elif self.reduction == "mean":
            loss = per_class.mean()
        elif self.reduction == "none":
            loss = per_class.clone()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        if return_per_class:
            return loss, per_class.detach()
        return loss


# ----------------------- quick usage example -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, K = 32, 5
    logits = torch.randn(N, K, requires_grad=True)
    labels = torch.randint(0, K, (N,))  # integer class labels

    # instantiate the loss
    crit = RKCALLoss(num_classes=K, rff_dim=128, lengthscale=0.3,
                     share_features=True, reduction="mean", from_logits=True)

    # optional: re-sample RFF or adjust lengthscale
    # crit.set_lengthscale(0.25, resample=True)

    # optional sample/class weights
    sample_w = None
    class_w = torch.ones(K)

    loss = crit(logits, labels, sample_weight=sample_w, class_weight=class_w)
    loss.backward()
    print("RKCAL loss:", float(loss))
