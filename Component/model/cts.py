# Network Calibration by Class-based Temperature Scaling
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict

from .calibrator import Calibrator
from ..metrics import (
    ECE, AdaptiveECE, ClasswiseECE, NLL, Accuracy,
    BrierLoss, FocalLoss, LabelSmoothingLoss,
    CrossEntropyLoss, MSELoss, SoftECE
)
from .temperature_scaling import TemperatureScalingCalibrator

# ------------------------------------------------------------------ #
# logging setup
# ------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)           # DEBUG for full details
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


class CTSCalibrator(Calibrator):
    """
    Class-based Temperature Scaling (CTS) with accuracy constraint:
    更新某类别温度时，只有当 (1) ECE 改进且 (2) 准确率不下降 (ΔAcc ≥ -ε)
    才接受该更新。
    """

    def __init__(self,
                 n_class: int,
                 n_iter: int = 5,
                 n_bins: int = 15,
                 acc_epsilon: float = 0.0,   # 允许的最大准确率跌幅
                 grid= None):
        super().__init__()
        self.n_class = n_class
        self.n_iter = n_iter
        self.n_bins = n_bins
        self.acc_epsilon = acc_epsilon
        # per-class temperature (buffer → no grad)
        self.register_buffer("T", torch.ones(n_class))
        # 基本度量

        self.metrics = {
            'ece': ECE(n_bins=n_bins),
            'adaptive_ece': AdaptiveECE(n_bins=n_bins),
            'classwise_ece': ClasswiseECE(n_bins=n_bins),
            'nll': NLL(),
            'accuracy': Accuracy(),
            'brier': BrierLoss(),
            'focal': FocalLoss(),
            'label_smoothing': LabelSmoothingLoss(),
            'cross_entropy': CrossEntropyLoss(),
            'mse': MSELoss(),
            'soft_ece': SoftECE()
        }

        self.ece_fn = ECE(n_bins=n_bins)
        self.acc_fn = Accuracy()
        # 网格
        self.default_grid = grid if grid is not None else np.arange(0.5, 5.1, 0.1)

        logger.info("CTSCalibrator initialised: classes=%d iters=%d bins=%d ε=%.3f",
                    n_class, n_iter, n_bins, acc_epsilon)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        return x / self.T

    def get_class_temperatures(self):
        return self.T.detach().cpu().tolist()

    # ------------------------------------------------------------------ #
    def _ece_and_acc(self, logits: torch.Tensor, labels: torch.Tensor):
        """计算 ECE 与 Accuracy（二者都返回 float）"""
        probs = F.softmax(logits, dim=1)
        ece = self.ece_fn(softmaxes=probs, labels=labels).item()
        acc = (probs.argmax(1) == labels).float().mean().item()
        return ece, acc

    # ------------------------------------------------------------------ #
    def fit(self,
            val_logits: torch.Tensor,
            val_labels: torch.Tensor,
            ts_loss: str = "nll",
            **kwargs) -> Dict[str, float]:
        
        # 选择设备，优先使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # 确保输入张量在正确的设备上
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        
        print(f"Using device: {device}")
        print("logit range  :", val_logits.min().item(), val_logits.max().item())
        print("first row    :", val_logits[0][:10])

        # 1-A.  How good is the *raw* model?
        raw_probs = F.softmax(val_logits, dim=1)
        raw_acc   = (raw_probs.argmax(1) == val_labels).float().mean().item()
        print("raw accuracy :", raw_acc)

        logger.info("CTS fit started")

        # ---------- 1-D TS 初始化 ----------
        print("TS loss type:", ts_loss)
        ts = TemperatureScalingCalibrator(loss_type=ts_loss)
        ts.to(device)  # 确保TS模型也在GPU上
        if "loss_fn" in kwargs:
            ts.loss_fn = kwargs["loss_fn"]
        ts.fit(val_logits, val_labels)

        with torch.no_grad():
            init_T = ts.temperature.clamp(0.5, 5.0)  # 裁剪一下防止极端欠置信
            self.T.fill_(init_T.item())
        
        logger.info("Initialised all class temps with TS value %.4f", init_T.item())

        # ---------- 计算初始指标 ----------
        cur_logits = val_logits / self.T            # (N, C)
        best_ece, best_acc = self._ece_and_acc(cur_logits, val_labels)
        logger.info("Initial ECE: %.6f | Acc: %.4f", best_ece, best_acc)

        # ---------- greedy optimisation ----------
        grid = kwargs.get("grid", self.default_grid)
        for it in range(self.n_iter):
            logger.info("Iter %d / %d", it + 1, self.n_iter)

            for cls in range(self.n_class):
                mask = (val_labels == cls)
                if mask.sum().item() == 0:
                    continue   # 无样本则跳过

                cls_best_temp = self.T[cls].item()   # 当前温度
                cls_best_ece  = best_ece
                cls_best_acc  = best_acc

                for temp in grid:
                    temp_vec = self.T.clone()
                    temp_vec[cls] = temp
                    cand_logits = val_logits / temp_vec
                    cand_ece, cand_acc = self._ece_and_acc(cand_logits, val_labels)

                    # 条件：ECE 改善 & Acc 不下降超过 epsilon
                    if (cand_ece < cls_best_ece) and (cand_acc >= best_acc - self.acc_epsilon):
                        cls_best_ece, cls_best_acc, cls_best_temp = cand_ece, cand_acc, temp

                # 若找到更优解则更新
                if cls_best_temp != self.T[cls].item():
                    with torch.no_grad():
                        self.T[cls] = cls_best_temp
                    best_ece, best_acc = cls_best_ece, cls_best_acc
                    logger.info("  Class %d: T=%.2f → %.2f | ECE=%.6f Acc=%.4f",
                                cls, self.T[cls].item(), cls_best_temp, best_ece, best_acc)

            logger.info("Iter %d done | ECE=%.6f Acc=%.4f", it + 1, best_ece, best_acc)

        # ---------- final report ----------
        logger.info("CTS fit complete | Final ECE=%.6f Acc=%.4f", best_ece, best_acc)
        temps = self.T.cpu().numpy()
        logger.info("Temperature stats -> min:%.3f max:%.3f mean:%.3f std:%.3f",
                    temps.min(), temps.max(), temps.mean(), temps.std())

        return {"final_ece": best_ece, "final_accuracy": best_acc}

    # ------------------------------------------------------------------ #
    def calibrate(self, test_logits, return_logits=False, **_):
        # 选择设备，优先使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        if not torch.is_tensor(test_logits):
            test_logits = torch.tensor(test_logits, dtype=torch.float32, device=device)
        else:
            test_logits = test_logits.to(device)
            
        logits = self.forward(test_logits)
        return logits if return_logits else F.softmax(logits, dim=1)

    # ------------------------------------------------------------------ #
    def save(self, path="./"):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/cts_model.pth")
        logger.info("CTS model saved to %s/cts_model.pth", path)

    def load(self, path="./"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(f"{path}/cts_model.pth", map_location=device))
        self.to(device)
        logger.info("CTS model loaded from %s/cts_model.pth and moved to %s", path, device)

    def compute_all_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute all available metrics for the given logits and labels.
        
        Args:
            logits (torch.Tensor): Input logits
            labels (torch.Tensor): Target labels
            
        Returns:
            Dict[str, float]: Dictionary containing all metric values
        """
        # 选择设备，优先使用GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # 确保输入张量在正确的设备上
        logits = logits.to(device)
        labels = labels.to(device)
        
        probs = F.softmax(logits / self.T, dim=1)
        
        results = {}
        for name, metric in self.metrics.items():
            metric = metric.to(device)
            try:
                if name in ['nll', 'cross_entropy']:
                    value = metric(logits=logits / self.T, labels=labels)
                elif name in ['brier', 'focal', 'label_smoothing', 'mse']:
                    value = metric(softmaxes=probs, labels=labels)
                elif name in ['ece', 'adaptive_ece', 'classwise_ece', 'soft_ece']:
                    value = metric(softmaxes=probs, labels=labels)
                elif name == 'accuracy':
                    value = metric(softmaxes=probs, labels=labels)
                else:
                    logger.warning(f"Unknown metric type: {name}")
                    continue
                
                # Convert to float if it's a tensor
                if torch.is_tensor(value):
                    value = value.item()
                results[name] = value
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {str(e)}")
                results[name] = None
                continue
                
        return results

    def get_all_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Get all metrics in a dictionary format compatible with the results structure.
        
        Args:
            logits (torch.Tensor): Input logits
            labels (torch.Tensor): Target labels
            
        Returns:
            Dict[str, float]: Dictionary containing all metric values in the format:
            {
                'ece': float,
                'accuracy': float,
                'adaece': float,
                'cece': float,
                'nll': float
            }
        """
        metrics = self.compute_all_metrics(logits, labels)
        return {
            'ece': metrics.get('ece', None),
            'accuracy': metrics.get('accuracy', None),
            'adaece': metrics.get('adaptive_ece', None),
            'cece': metrics.get('classwise_ece', None),
            'nll': metrics.get('nll', None)
        }
