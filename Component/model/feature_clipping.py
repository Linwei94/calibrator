"""
Feature Clipping Calibrator

Adapted from the original AAAI2025-FC implementation for use in the SMART framework.
Performs feature clipping on model features before classification to improve calibration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from calibrator.Component.metrics import ECE


class FeatureClippingCalibrator(nn.Module):
    def __init__(self, cross_validate='ece'):
        super(FeatureClippingCalibrator, self).__init__()
        self.cross_validate = cross_validate
        self.feature_clip = float("inf")
        self.ece_criterion = ECE()
        self.nll_criterion = nn.CrossEntropyLoss()

    def get_feature_clip(self):
        return self.feature_clip

    def set_feature_clip(self, features_val, logits_val, labels_val, classifier):
        device = features_val.device
        self.ece_criterion = self.ece_criterion.to(device)
        self.nll_criterion = self.nll_criterion.to(device)

        # Ensure all tensors are on the same device
        features_val = features_val.to(device)
        logits_val = logits_val.to(device)
        labels_val = labels_val.to(device)

        nll_val_opt = float("inf")
        ece_val_opt = float("inf")
        C_opt_nll = float("inf")
        C_opt_ece = float("inf")
        self.feature_clip = float("inf")

        before_clipping_acc = (F.softmax(logits_val, dim=1).argmax(dim=1) == labels_val).float().mean().item()

        C = 0.01
        for _ in range(2000):
            logits_after_clipping = classifier(self.feature_clipping(features_val, C))
            after_clipping_nll = self.nll_criterion(logits_after_clipping, labels_val).item()
            after_clipping_ece = self.ece_criterion(logits_after_clipping, labels_val).item()
            after_clipping_acc = (F.softmax(logits_after_clipping, dim=1).argmax(dim=1) == labels_val).float().mean().item()

            if (after_clipping_nll < nll_val_opt) and (after_clipping_acc > before_clipping_acc*0.99):
                C_opt_nll = C
                nll_val_opt = after_clipping_nll

            if (after_clipping_ece < ece_val_opt) and (after_clipping_acc > before_clipping_acc*0.99):
                C_opt_ece = C
                ece_val_opt = after_clipping_ece

            C += 0.01

        if self.cross_validate == 'ece':
            self.feature_clip = C_opt_ece
        elif self.cross_validate == 'nll':
            self.feature_clip = C_opt_nll

        return self.feature_clip

    def feature_clipping(self, features, c=None):
        """
        Perform feature clipping on features
        """
        if c is None:
            c = self.feature_clip
        return torch.clamp(features, min=-c, max=c)

    def forward(self, features, classifier, c=None):
        return classifier(self.feature_clipping(features, c))