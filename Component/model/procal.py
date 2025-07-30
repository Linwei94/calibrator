"""
ProCal (Proximity-Informed Calibration) implementation
Based on the paper: "Proximity-Informed Calibration for Deep Neural Networks" (Xiong et al., 2024)

This module implements two ProCal algorithms:
1. Density-Ratio Calibration (continuous)
2. Bin-Mean-Shift Calibration (discrete)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.special import softmax
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.isotonic import IsotonicRegression
import faiss

from .calibrator import Calibrator


class ProCalDensityRatioCalibrator(Calibrator):
    """
    ProCal Density-Ratio Calibration using Kernel Density Estimation
    
    This calibrator uses KDE to model p(confidence, proximity | correct) and 
    p(confidence, proximity | incorrect), then applies Bayes' rule to compute
    calibrated probabilities.
    """
    
    def __init__(self, k_neighbors=10, bandwidth='normal_reference', kernel='KDEMultivariate', 
                 distance_measure='L2', normalize_features=True):
        """
        Initialize ProCal Density-Ratio Calibrator
        
        Args:
            k_neighbors (int): Number of nearest neighbors for proximity calculation
            bandwidth (str or float): Bandwidth for KDE
            kernel (str): Kernel type ('KDEMultivariate', 'sklearn_kde', 'scipy_gaussian_kde')
            distance_measure (str): Distance measure for KNN ('L2', 'cosine')
            normalize_features (bool): Whether to normalize features before KNN
        """
        super(ProCalDensityRatioCalibrator, self).__init__()
        self.k_neighbors = k_neighbors
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.distance_measure = distance_measure
        self.normalize_features = normalize_features
        
        # Will be set during fit
        self.dens_true = None
        self.dens_false = None
        self.false_true_ratio = None
        self.index = None
        self.val_features = None

    def _extract_features(self, logits):
        """
        Extract features from logits for proximity calculation
        This is a fallback method when no features are provided.
        """
        return logits.detach().cpu().numpy()

    def _compute_proximity(self, features, reference_features=None):
        """
        Compute proximity using K-nearest neighbors
        
        Args:
            features (np.ndarray): Features to compute proximity for
            reference_features (np.ndarray): Reference features for KNN search
            
        Returns:
            np.ndarray: Proximity values (exponential of negative average distances)
        """
        if reference_features is None:
            reference_features = self.val_features
            
        # Normalize features if specified
        if self.normalize_features:
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            reference_features = reference_features / np.linalg.norm(reference_features, axis=1, keepdims=True)
        
        # Build FAISS index if not exists
        if self.index is None:
            dim = reference_features.shape[1]
            if self.distance_measure == "L2":
                self.index = faiss.IndexFlatL2(dim)
            elif self.distance_measure == "cosine": 
                self.index = faiss.IndexFlatIP(dim)
            else:
                raise NotImplementedError(f"Distance measure {self.distance_measure} not implemented")
            
            self.index.add(reference_features.astype(np.float32))
        
        # Search for K nearest neighbors
        k_search = self.k_neighbors + 1 if np.array_equal(features, reference_features) else self.k_neighbors
        D, I = self.index.search(features.astype(np.float32), k_search)
        
        # Remove self if searching in reference set
        if np.array_equal(features, reference_features):
            distances = D[:, 1:]  # Skip first column (self)
        else:
            distances = D
        
        # Compute proximity as exponential of negative average distance
        proximity = np.exp(-distances)
        proximity_mean = np.mean(proximity, axis=1)
        
        return proximity_mean

    def fit(self, val_logits, val_labels, val_features=None, **kwargs):
        """
        Fit the ProCal Density-Ratio calibrator
        
        Args:
            val_logits (torch.Tensor): Validation logits
            val_labels (torch.Tensor): Validation labels  
            val_features (torch.Tensor, optional): Validation features for proximity calculation
            **kwargs: Additional arguments
            
        Returns:
            dict: Fitted parameters
        """
        # Convert to numpy
        val_logits_np = val_logits.detach().cpu().numpy()
        val_labels_np = val_labels.detach().cpu().numpy()
        
        # Extract or use provided features
        if val_features is not None:
            self.val_features = val_features.detach().cpu().numpy()
        else:
            self.val_features = self._extract_features(val_logits)
        
        # Compute proximities
        proximities = self._compute_proximity(self.val_features)
        
        # Get predictions and confidences
        probs = softmax(val_logits_np, axis=-1)
        preds = np.argmax(val_logits_np, axis=-1)
        confs = np.max(probs, axis=-1)
        
        # Create dataframe for analysis
        val_df = pd.DataFrame({
            'ys': val_labels_np,
            'proximity': proximities, 
            'conf': confs,
            'pred': preds
        })
        
        val_df['correct'] = (val_df.pred == val_df.ys).astype('int')
        
        # Split into correct and incorrect predictions
        val_df_true = val_df[val_df['correct'] == 1]
        val_df_false = val_df[val_df['correct'] == 0]
        
        print(f"Correct predictions: {len(val_df_true)}, Incorrect: {len(val_df_false)}")
        
        # Fit KDE for correct predictions
        if len(val_df_true) > 0:
            true_data = np.array([val_df_true['conf'].values, val_df_true['proximity'].values]).T
            if self.kernel == 'KDEMultivariate':
                self.dens_true = sm.nonparametric.KDEMultivariate(
                    data=true_data, var_type='cc', bw=self.bandwidth
                )
            elif self.kernel == 'sklearn_kde':
                self.dens_true = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(true_data)
            else:
                raise NotImplementedError(f"Kernel {self.kernel} not implemented")
        
        # Fit KDE for incorrect predictions  
        if len(val_df_false) > 0:
            false_data = np.array([val_df_false['conf'].values, val_df_false['proximity'].values]).T
            if self.kernel == 'KDEMultivariate':
                self.dens_false = sm.nonparametric.KDEMultivariate(
                    data=false_data, var_type='cc', bw=self.bandwidth
                )
            elif self.kernel == 'sklearn_kde':
                self.dens_false = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(false_data)
            else:
                raise NotImplementedError(f"Kernel {self.kernel} not implemented")
        
        # Store the ratio of incorrect to correct predictions
        self.false_true_ratio = len(val_df_false) / max(len(val_df_true), 1)
        
        print(f"ProCal Density-Ratio calibration fitted. False/True ratio: {self.false_true_ratio:.4f}")
        
        return {
            'false_true_ratio': self.false_true_ratio,
            'n_correct': len(val_df_true),
            'n_incorrect': len(val_df_false)
        }

    def calibrate(self, test_logits, test_features=None, return_logits=False):
        """
        Calibrate test logits using fitted ProCal model
        
        Args:
            test_logits (torch.Tensor): Test logits to calibrate
            test_features (torch.Tensor, optional): Test features for proximity calculation  
            return_logits (bool): Whether to return logits or probabilities
            
        Returns:
            torch.Tensor: Calibrated probabilities or logits
        """
        if self.dens_true is None or self.dens_false is None:
            raise RuntimeError("Calibrator must be fitted before calibrating")
        
        # Convert to numpy
        test_logits_np = test_logits.detach().cpu().numpy()
        
        # Extract or use provided features
        if test_features is not None:
            test_features_np = test_features.detach().cpu().numpy()
        else:
            test_features_np = self._extract_features(test_logits)
        
        # Compute proximities
        proximities = self._compute_proximity(test_features_np)
        
        # Get predictions and confidences
        probs = softmax(test_logits_np, axis=-1)
        preds = np.argmax(test_logits_np, axis=-1)
        confs = np.max(probs, axis=-1)
        
        # Prepare data for KDE evaluation
        data = np.array([confs, proximities]).T
        
        # Get density estimates
        if self.kernel == 'KDEMultivariate':
            conf_reg_true = self.dens_true.pdf(data)
            conf_reg_false = self.dens_false.pdf(data)
        elif self.kernel == 'sklearn_kde':
            conf_reg_true = np.exp(self.dens_true.score_samples(data))
            conf_reg_false = np.exp(self.dens_false.score_samples(data))
        
        # Apply Bayes rule to get calibrated confidences
        eps = 1e-10
        conf_calibrated = conf_reg_true / np.maximum(
            conf_reg_true + conf_reg_false * self.false_true_ratio, eps
        )
        
        # Reconstruct calibrated probabilities
        calibrated_probs = probs.copy()
        
        # Zero out the predicted class probabilities
        mask = np.ones(probs.shape, dtype=bool)
        mask[range(probs.shape[0]), preds] = False
        calibrated_probs = calibrated_probs * mask
        
        # Rescale the non-predicted classes
        calibrated_probs = calibrated_probs * ((1 - conf_calibrated) / calibrated_probs.sum(axis=-1))[:, np.newaxis]
        
        # Set the calibrated confidence for predicted class
        calibrated_probs[range(probs.shape[0]), preds] = conf_calibrated
        
        # Convert back to torch tensor
        calibrated_probs = torch.from_numpy(calibrated_probs).float().to(test_logits.device)
        
        if return_logits:
            # Convert probabilities back to logits
            return torch.log(calibrated_probs + eps)
        else:
            return calibrated_probs


class ProCalBinMeanShiftCalibrator(Calibrator):
    """
    ProCal Bin-Mean-Shift Calibration
    
    This calibrator bins samples by proximity and applies calibration within each bin.
    """
    
    def __init__(self, base_calibrator_class=None, k_neighbors=10, proximity_bins=10, 
                 bin_strategy='quantile', distance_measure='L2', normalize_features=True, **base_kwargs):
        """
        Initialize ProCal Bin-Mean-Shift Calibrator
        
        Args:
            base_calibrator_class: Base calibration method class (e.g., IsotonicRegression)
            k_neighbors (int): Number of nearest neighbors for proximity calculation
            proximity_bins (int): Number of proximity bins
            bin_strategy (str): Binning strategy ('quantile', 'kmeans', 'uniform')
            distance_measure (str): Distance measure for KNN ('L2', 'cosine')  
            normalize_features (bool): Whether to normalize features before KNN
            **base_kwargs: Arguments for base calibrator
        """
        super(ProCalBinMeanShiftCalibrator, self).__init__()
        self.base_calibrator_class = base_calibrator_class or IsotonicRegression
        self.k_neighbors = k_neighbors
        self.proximity_bins = proximity_bins
        self.bin_strategy = bin_strategy
        self.distance_measure = distance_measure
        self.normalize_features = normalize_features
        self.base_kwargs = base_kwargs
        
        # Will be set during fit
        self.calibrators = None
        self.bin_edges = None
        self.index = None
        self.val_features = None

    def _extract_features(self, logits):
        """Extract features from logits for proximity calculation (fallback)"""
        return logits.detach().cpu().numpy()

    def _compute_proximity(self, features, reference_features=None):
        """Compute proximity using K-nearest neighbors"""
        if reference_features is None:
            reference_features = self.val_features
            
        # Normalize features if specified
        if self.normalize_features:
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            reference_features = reference_features / np.linalg.norm(reference_features, axis=1, keepdims=True)
        
        # Build FAISS index if not exists
        if self.index is None:
            dim = reference_features.shape[1]
            if self.distance_measure == "L2":
                self.index = faiss.IndexFlatL2(dim)
            elif self.distance_measure == "cosine": 
                self.index = faiss.IndexFlatIP(dim)
            else:
                raise NotImplementedError(f"Distance measure {self.distance_measure} not implemented")
            
            self.index.add(reference_features.astype(np.float32))
        
        # Search for K nearest neighbors
        k_search = self.k_neighbors + 1 if np.array_equal(features, reference_features) else self.k_neighbors
        D, I = self.index.search(features.astype(np.float32), k_search)
        
        # Remove self if searching in reference set
        if np.array_equal(features, reference_features):
            distances = D[:, 1:]  # Skip first column (self)
        else:
            distances = D
        
        # Compute proximity as exponential of negative average distance
        proximity = np.exp(-distances)
        proximity_mean = np.mean(proximity, axis=1)
        
        return proximity_mean

    def _get_bin_edges(self, proximity):
        """Get bin edges based on binning strategy"""
        if self.bin_strategy == 'quantile':
            quantiles = np.linspace(0, 100, self.proximity_bins + 1)
            bin_edges = np.asarray(np.percentile(proximity, quantiles))
        elif self.bin_strategy == 'kmeans':
            col_min, col_max = proximity.min(), proximity.max()
            uniform_edges = np.linspace(col_min, col_max, self.proximity_bins + 1)
            init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
            
            km = KMeans(n_clusters=self.proximity_bins, init=init, n_init=1, random_state=42)
            centers = km.fit(proximity[:, None]).cluster_centers_[:, 0]
            centers.sort()
            bin_edges = (centers[1:] + centers[:-1]) * 0.5
            bin_edges = np.r_[col_min, bin_edges, col_max]
        elif self.bin_strategy == 'uniform':
            col_min, col_max = proximity.min(), proximity.max()
            bin_edges = np.linspace(col_min, col_max, self.proximity_bins + 1)
        else:
            raise ValueError(f"Unknown bin strategy: {self.bin_strategy}")
        
        return bin_edges

    def fit(self, val_logits, val_labels, val_features=None, **kwargs):
        """
        Fit the ProCal Bin-Mean-Shift calibrator
        
        Args:
            val_logits (torch.Tensor): Validation logits
            val_labels (torch.Tensor): Validation labels
            val_features (torch.Tensor, optional): Validation features for proximity calculation
            **kwargs: Additional arguments
            
        Returns:
            dict: Fitted parameters
        """
        # Convert to numpy
        val_logits_np = val_logits.detach().cpu().numpy()
        val_labels_np = val_labels.detach().cpu().numpy()
        
        # Extract or use provided features
        if val_features is not None:
            self.val_features = val_features.detach().cpu().numpy()
        else:
            self.val_features = self._extract_features(val_logits)
        
        # Compute proximities
        proximities = self._compute_proximity(self.val_features)
        
        # Get bin edges
        self.bin_edges = self._get_bin_edges(proximities)
        
        # Assign samples to bins
        bin_assignments = np.searchsorted(self.bin_edges[1:-1], proximities, side="right")
        
        # Initialize calibrators for each bin
        self.calibrators = []
        bin_counts = []
        
        for bin_idx in range(self.proximity_bins):
            # Get samples for this bin
            bin_mask = (bin_assignments == bin_idx)
            bin_logits = val_logits_np[bin_mask]
            bin_labels = val_labels_np[bin_mask]
            
            bin_counts.append(bin_mask.sum())
            
            if bin_mask.sum() > 0:
                # Initialize and fit calibrator for this bin
                calibrator = self.base_calibrator_class(**self.base_kwargs)
                
                # Convert to probabilities for isotonic regression
                bin_probs = softmax(bin_logits, axis=-1)
                calibrator.fit(bin_probs, bin_labels)
                
                self.calibrators.append(calibrator)
            else:
                # No samples in this bin, use identity calibrator
                self.calibrators.append(None)
        
        print(f"ProCal Bin-Mean-Shift calibration fitted with {self.proximity_bins} bins")
        print(f"Bin counts: {bin_counts}")
        
        return {
            'proximity_bins': self.proximity_bins,
            'bin_counts': bin_counts,
            'bin_edges': self.bin_edges
        }

    def calibrate(self, test_logits, test_features=None, return_logits=False):
        """
        Calibrate test logits using fitted ProCal Bin-Mean-Shift model
        
        Args:
            test_logits (torch.Tensor): Test logits to calibrate
            test_features (torch.Tensor, optional): Test features for proximity calculation
            return_logits (bool): Whether to return logits or probabilities
            
        Returns:
            torch.Tensor: Calibrated probabilities or logits
        """
        if self.calibrators is None:
            raise RuntimeError("Calibrator must be fitted before calibrating")
        
        # Convert to numpy
        test_logits_np = test_logits.detach().cpu().numpy()
        
        # Extract or use provided features
        if test_features is not None:
            test_features_np = test_features.detach().cpu().numpy()
        else:
            test_features_np = self._extract_features(test_logits)
        
        # Compute proximities
        proximities = self._compute_proximity(test_features_np)
        
        # Assign samples to bins
        bin_assignments = np.searchsorted(self.bin_edges[1:-1], proximities, side="right")
        
        # Initialize calibrated probabilities
        test_probs = softmax(test_logits_np, axis=-1)
        calibrated_probs = np.zeros_like(test_probs)
        
        # Calibrate each bin separately
        for bin_idx in range(self.proximity_bins):
            bin_mask = (bin_assignments == bin_idx)
            
            if bin_mask.sum() > 0:
                bin_probs = test_probs[bin_mask]
                
                if self.calibrators[bin_idx] is not None:
                    # Apply calibration for this bin
                    bin_calibrated = self.calibrators[bin_idx].transform(bin_probs)
                    calibrated_probs[bin_mask] = bin_calibrated
                else:
                    # No calibrator for this bin, use original probabilities
                    calibrated_probs[bin_mask] = bin_probs
        
        # Convert back to torch tensor
        calibrated_probs = torch.from_numpy(calibrated_probs).float().to(test_logits.device)
        
        if return_logits:
            # Convert probabilities back to logits
            eps = 1e-10
            return torch.log(calibrated_probs + eps)
        else:
            return calibrated_probs