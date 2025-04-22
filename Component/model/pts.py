# pts_calibrator.py
import os, random, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange, tqdm

# ---- import your metric classes ------------------------------------------------
from .calibrator import Calibrator          # <- keep your own base‑class
from ..metrics import (                     # custom losses you already have
    BrierLoss, FocalLoss, LabelSmoothingLoss,
    CrossEntropyLoss, MSELoss, SoftECE)
from typing import Union, Optional
# -------------------------------------------------------------------------------

class PTSCalibrator(Calibrator):
    """
    Parameterised‑Temperature‑Scaling (PTS) — clean PyTorch implementation
    """

    # ----------------------------------------------------------------------- #
    #  constructor
    # ----------------------------------------------------------------------- #
    def __init__(
        self,
        steps: int           = 100_000,
        lr: float            = 5e-5,
        weight_decay: float  = 0.0,
        batch_size: int      = 1000,
        nlayers: int         = 2,
        n_nodes: int         = 5,
        length_logits: int   = None,
        top_k_logits: int    = 10,
        loss_fn: Optional[Union[str, nn.Module]] = None,
        seed: int            = 42,
    ):
        super().__init__()

        self.steps         = steps
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.batch_size    = batch_size
        self.nlayers       = nlayers
        self.n_nodes       = n_nodes
        self.length_logits = length_logits
        self.top_k_logits  = top_k_logits
        self.seed          = seed

        self._set_seed(seed)
        self.loss_fn = self._get_loss_function(loss_fn)

        # ---- temperature branch -------------------------------------------
        layers, inp = [], top_k_logits
        if nlayers > 0:
            layers += [nn.Linear(inp, n_nodes), nn.ReLU()]
            for _ in range(nlayers - 1):
                layers += [nn.Linear(n_nodes, n_nodes), nn.ReLU()]
            layers += [nn.Linear(n_nodes, 1)]
        else:
            layers += [nn.Linear(inp, 1)]
        self.temp_branch = nn.Sequential(*layers)
        self._init_weights()
    # ----------------------------------------------------------------------- #

    # helper: pick correct loss implementation ------------------------------ #
    def _get_loss_function(self, loss_fn):
        if loss_fn is None:
            return MSELoss()

        if isinstance(loss_fn, str):
            key = loss_fn.lower()
            if key in {"mse", "mean_squared_error"}:
                return MSELoss()
            if key in {"crossentropy", "cross_entropy", "ce"}:
                return CrossEntropyLoss()           # your custom CE
            if key in {"soft_ece"}:
                return SoftECE()
            if key in {"brier"}:
                return BrierLoss()
            if key in {"focal"}:
                return FocalLoss()
            if key in {"label_smoothing", "ls"}:
                return LabelSmoothingLoss(alpha=0.01)
            raise ValueError(f"Unsupported loss: {loss_fn}")

        return loss_fn
    # ----------------------------------------------------------------------- #

    def _set_seed(self, seed):
        random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ----------------------------------------------------------------------- #
    #  forward
    # ----------------------------------------------------------------------- #
    def forward(self, logits):
        # logits: (B, C)
        sorted_lgts, _ = torch.sort(logits, 1, True)
        topk           = sorted_lgts[:, : self.top_k_logits]

        t         = self.temp_branch(topk)      # (B,1)
        temp      = torch.abs(t).clamp(1e-12, 1e12)
        cal_lgts  = logits / temp
        cal_probs = F.softmax(cal_lgts, dim=1)
        return cal_probs, cal_lgts
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    #  fit
    # ----------------------------------------------------------------------- #
    def fit(self, val_logits, val_labels, **kwargs):
        clip    = kwargs.get("clip", 1e2)
        seed    = kwargs.get("seed", self.seed)
        verbose = kwargs.get("verbose", True)
        self._set_seed(seed)

        # tensors & device ---------------------------------------------------
        if not torch.is_tensor(val_logits):
            val_logits = torch.tensor(val_logits, dtype=torch.float32)
        if not torch.is_tensor(val_labels):
            val_labels = torch.tensor(val_labels, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_logits, val_labels = val_logits.to(device), val_labels.to(device)
        self.to(device)

        # classes ------------------------------------------------------------
        if self.length_logits is None:
            self.length_logits = val_logits.shape[1]
        assert val_logits.size(1) == self.length_logits

        #  original class indices (needed for CE / SoftECE) ------------------
        if val_labels.ndim == 2:                         # one‑hot provided
            orig_idx  = val_labels.argmax(dim=1).long()
        else:                                            # indices provided
            orig_idx  = val_labels.long()
        # one‑hot for those losses that need it
        if val_labels.ndim == 1:
            oh = torch.zeros(
                val_labels.size(0), self.length_logits, device=device
            )
            oh.scatter_(1, val_labels.unsqueeze(1), 1.)
            val_labels = oh

        # clip logits
        val_logits = torch.clamp(val_logits, -clip, clip)

        # dataloader
        ds     = TensorDataset(val_logits, val_labels, orig_idx)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # training loop ------------------------------------------------------
        self.train()
        pbar, step = trange(self.steps, disable=not verbose), 0
        while step < self.steps:
            for blgts, blabs, bidx in loader:
                if step >= self.steps:
                    break
                optim.zero_grad()
                probs, lgts = self.forward(blgts)

                if isinstance(self.loss_fn, SoftECE):
                    loss = self.loss_fn(logits=lgts, labels=bidx)

                elif isinstance(self.loss_fn, BrierLoss):
                    loss = self.loss_fn(probs, blabs)                     # p vs one‑hot

                elif isinstance(self.loss_fn, (FocalLoss, LabelSmoothingLoss)):
                    loss = self.loss_fn(softmaxes=probs, labels=blabs)    # expect probs

                elif isinstance(self.loss_fn, CrossEntropyLoss):
                    loss = self.loss_fn(lgts, bidx)                       # logits + idx

                elif isinstance(self.loss_fn, MSELoss):
                    diff = (probs - blabs).pow(2).sum(dim=1)   # shape (B,)
                    loss = diff.mean()

                else:   # any callable you plugged in
                    loss = self.loss_fn(lgts, blabs)

                loss.backward();  optim.step()
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    #  calibrate
    # ----------------------------------------------------------------------- #
    def calibrate(self, test_logits, *, return_logits=False, clip=1e2):
        if not torch.is_tensor(test_logits):
            test_logits = torch.tensor(test_logits, dtype=torch.float32)
        test_logits = test_logits.to(next(self.parameters()).device)
        test_logits = torch.clamp(test_logits, -clip, clip)

        self.eval()
        with torch.no_grad():
            probs, lgts = self.forward(test_logits)
        return lgts if return_logits else probs
    # ----------------------------------------------------------------------- #

    # small helpers ----------------------------------------------------------
    def save(self, path="./"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pts_model.pth"))

    def load(self, path="./"):
        self.load_state_dict(
            torch.load(os.path.join(path, "pts_model.pth"), map_location="cpu")
        )
