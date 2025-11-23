import torch
import torch.nn as nn
import numpy as np
from scipy import optimize
import time
import gc
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. DensityAwareCalibrator will not work without faiss.")

from sklearn.cluster import KMeans

from .calibrator import Calibrator


def torch_softmax(x):
    """PyTorch version of softmax"""
    return torch.softmax(x, dim=1)


class KNNScorer(object):
    def __init__(self, top_k=1, avg_top_k=False, return_dist_arr=False, gpu=True):
        """
        top_k:
            Pick top-k distance value as a measurement of density
        avg_top_k:
            if average top-k distances.
            Default: pick k-th distance value.
        return_dist_arr:
            if return distance matrix, instead of one value for each sample.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required for KNNScorer. Please install with: pip install faiss-cpu or pip install faiss-gpu")

        self.top_k = top_k
        self.avg_top_k = avg_top_k
        self.return_dist_arr = return_dist_arr
        self.gpu = gpu and torch.cuda.is_available()

        # knn
        self.ftrain_list = []

    def get_score(self, test_feats):
        return self.knn_score(test_feats)

    def set_train_feat(self, train_feats, train_labels=None, class_num=None, _type="single"):
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

        if _type == "single":
            train_feats = [train_feats]

        for train_feat in train_feats:
            # Convert to numpy if it's a tensor
            if isinstance(train_feat, torch.Tensor):
                train_feat = train_feat.detach().cpu().numpy()

            ftrain = prepos_feat(train_feat.astype(np.float32))
            self.ftrain_list += [ftrain]
            print(f"Set train features to KNNScorer: shape {train_feat.shape}")

        del train_feats
        gc.collect()

    def knn_score(self, test_feats):
        """
        test_feats: List of features (N, vector_length) or single tensor.
            Test features extracted from the classifier.
        """
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

        # Handle single tensor input
        if isinstance(test_feats, torch.Tensor):
            test_feats = [test_feats.detach().cpu().numpy()]
        elif isinstance(test_feats, list) and len(test_feats) > 0 and isinstance(test_feats[0], torch.Tensor):
            test_feats = [feat.detach().cpu().numpy() for feat in test_feats]

        assert len(test_feats) == len(self.ftrain_list)
        ftrain_list = self.ftrain_list
        ftest_list = [prepos_feat(feat.astype(np.float32)) for feat in test_feats]

        # Try to use GPU if requested and available
        use_gpu = False
        res = None
        if self.gpu:
            try:
                res = faiss.StandardGpuResources()
                use_gpu = True
            except AttributeError:
                print("Warning: faiss GPU not available, falling back to CPU")
                use_gpu = False

        D_list = []
        ood_scores = np.zeros((len(test_feats), test_feats[0].shape[0]))
        for i, (ftrain, ftest) in enumerate(zip(ftrain_list, ftest_list)):
            if use_gpu:
                try:
                    index_flat = faiss.IndexFlatL2(ftrain.shape[1])
                    index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                    index.add(ftrain)
                except Exception as e:
                    print(f"Warning: GPU indexing failed ({e}), falling back to CPU")
                    index = faiss.IndexFlatL2(ftrain.shape[1])
                    index.add(ftrain)
            else:
                index = faiss.IndexFlatL2(ftrain.shape[1])
                index.add(ftrain)
            D, _ = index.search(ftest, self.top_k)
            D_list.append(D[:, -1000:])
            if self.avg_top_k:
                ood_scores[i, :] = D[:, -self.top_k :].mean(1)
            else:
                ood_scores[i, :] = D[:, -self.top_k]

        if self.return_dist_arr:
            return D_list

        print("ood scores shape: ", ood_scores.shape)
        return ood_scores


class DAC(object):
    def __init__(
        self,
        ood_values_num=1,
        tol=1e-12,
        eps=1e-7,
        disp=False,  # to print optimization process
    ):
        """
        T = (w_i * knn_score_i) + w0
        p = softmax(logits / T)
        """
        self.method = "L-BFGS-B"

        self.ood_values_num = ood_values_num
        print("ood_values_num: ", self.ood_values_num)

        self.tol = tol
        self.eps = eps
        self.disp = disp

        self.bnds = [[0, 10000.0]] * self.ood_values_num + [[-100.0, 100.0]]
        self.init = [1.0] * self.ood_values_num + [1.0]

    def get_temperature(self, w, ood_score):
        if self.ood_values_num == 1:
            if type(ood_score).__module__ == np.__name__:
                if len(ood_score.shape) == 1:
                    ood_score = [ood_score]
                else:
                    ood_score = [ood_score[i, :] for i in range(ood_score.shape[0])]

        assert len(ood_score) == self.ood_values_num, (
            ood_score,
            len(ood_score),
            self.ood_values_num,
        )

        if len(ood_score) != 0:
            sample_size = len(ood_score[0])
            t = np.zeros(sample_size)

            for i in range(self.ood_values_num):
                t += w[i] * ood_score[i]
            t += w[-1]
        else:
            # temperature scaling
            t = np.zeros(1)
            t += w[-1]

        # temperature should be a positive value
        return np.clip(t, 1e-20, None)

    def mse_lf(self, w, *args):
        ## find optimal temperature with MSE loss function
        logit, label, ood_score = args
        t = self.get_temperature(w, ood_score)
        logit = logit / t[:, None]
        p = np_softmax(logit)
        mse = np.mean((p - label) ** 2)
        return mse

    def ll_lf(self, w, *args):
        ## find optimal temperature with Cross-Entropy loss function
        logit, label, ood_score = args
        t = self.get_temperature(w, ood_score)
        logit = logit / t[:, None]
        p = np_softmax(logit)
        N = p.shape[0]
        ce = -np.sum(label * np.log(p + 1e-12)) / N
        return ce

    def optimize(self, logit, label, ood_score, loss="ce"):
        """
        logit (N, C): classifier's outputs before softmax
        label (N, C): true labels, one-hot
        ood_score (N, number_of_layers):
            the value that represents how far the sample is in the feature space.
            we use KNN scoring strategy.
        """

        if not isinstance(self.eps, list):
            self.eps = [self.eps]

        if loss == "ce":
            func = self.ll_lf
        elif loss == "mse":
            func = self.mse_lf
        else:
            raise NotImplementedError

        # func:ll_t, 1.0:initial guess, args: args of the func, ..., tol: tolerence of minimization
        st = time.time()
        params = optimize.minimize(
            func,
            self.init,
            args=(logit, label, ood_score),
            method=self.method,
            bounds=self.bnds,
            tol=self.tol,
            options={"eps": self.eps, "disp": self.disp},
        )
        ed = time.time()

        w = params.x
        print("DAC Optimization done!: ({} sec)".format(ed - st))
        print(f"T = {w[:-1]} * ood_score_i + {w[-1]}")

        optim_value = params.fun
        self.w = w

        return self.get_optim_params()

    def calibrate(self, logits, ood_score):
        w = self.w
        t = self.get_temperature(w, ood_score)
        return np_softmax(logits / t[:, None])

    def calibrate_before_softmax(self, logits, ood_score):
        w = self.w
        t = self.get_temperature(w, ood_score)
        return logits / t[:, None]

    def get_optim_params(self):
        return self.w


def np_softmax(x):
    max_vals = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_vals)
    sum_vals = np.sum(e_x, axis=1, keepdims=True)
    return e_x / sum_vals


class DensityAwareCalibrator(Calibrator):
    def __init__(self, ood_values_num=1, loss_type='ce', knn_k=1, avg_top_k=False, gpu=True, base_calibrator=None):
        """
        Initialize the Density Aware Calibrator.

        Args:
            ood_values_num (int): Number of OOD values to use
            loss_type (str): Type of loss function ('ce' or 'mse')
            knn_k (int): K value for KNN scoring
            avg_top_k (bool): Whether to average top-k distances
            gpu (bool): Whether to use GPU for KNN computation
            base_calibrator: Optional base calibrator to combine with DAC (e.g., TemperatureScalingCalibrator)
                           If None, uses DAC alone. If provided, applies DAC first, then base calibrator.
        """
        super(DensityAwareCalibrator, self).__init__()

        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required for DensityAwareCalibrator. Please install with: pip install faiss-cpu or pip install faiss-gpu")

        self.ood_values_num = ood_values_num
        self.loss_type = loss_type
        self.knn_k = knn_k
        self.avg_top_k = avg_top_k
        self.gpu = gpu
        self.base_calibrator = base_calibrator

        # Initialize components
        self.knn_scorer = KNNScorer(
            top_k=knn_k,
            avg_top_k=avg_top_k,
            gpu=gpu
        )
        self.dac = DAC(ood_values_num=ood_values_num)

        # Store training features for later use
        self.train_features = None
        self.fitted = False

    def fit(self, val_logits, val_labels, val_features=None, train_features=None, **kwargs):
        """
        Fit the Density Aware Calibrator.

        Args:
            val_logits (torch.Tensor): Validation logits (N, C)
            val_labels (torch.Tensor): Validation labels (N,) or (N, C)
            val_features (torch.Tensor): Validation features for KNN scoring (N, D)
            train_features (torch.Tensor): Training features for KNN index (M, D)
            **kwargs: Additional arguments passed to base calibrator if provided
        """
        if val_features is None or train_features is None:
            raise ValueError("DensityAwareCalibrator requires both val_features and train_features")

        device = val_logits.device

        if self.base_calibrator is not None:
            print("=" * 50)
            print(f"Fitting DAC + {type(self.base_calibrator).__name__}")
            print("=" * 50)

        # Step 1: Fit DAC
        print("Step 1: Fitting DAC...")

        # Convert to numpy for DAC optimization
        val_logits_np = val_logits.detach().cpu().numpy()

        # Convert labels to one-hot if necessary
        if len(val_labels.shape) == 1:
            num_classes = val_logits.shape[1]
            val_labels_onehot = np.zeros((val_labels.shape[0], num_classes))
            val_labels_onehot[np.arange(val_labels.shape[0]), val_labels.cpu().numpy()] = 1
        else:
            val_labels_onehot = val_labels.detach().cpu().numpy()

        # Set training features for KNN
        self.knn_scorer.set_train_feat(train_features)

        # Get KNN scores for validation data
        ood_scores = self.knn_scorer.knn_score([val_features])

        # Optimize DAC parameters
        self.dac.optimize(val_logits_np, val_labels_onehot, ood_scores, loss=self.loss_type)
        print(f"DAC fitted with weights: {self.dac.w}")

        # Store for inference
        self.train_features = train_features

        # Step 2: Fit base calibrator if provided
        base_result = None
        if self.base_calibrator is not None:
            print(f"Step 2: Fitting base calibrator ({type(self.base_calibrator).__name__})...")

            # Get DAC-calibrated logits for validation set
            dac_calibrated_logits = self._apply_dac_only(val_logits, val_features, return_logits=True)

            # Fit base calibrator on DAC-calibrated logits
            base_result = self.base_calibrator.fit(dac_calibrated_logits, val_labels, **kwargs)
            print(f"Base calibrator fitted: {base_result}")

        self.fitted = True

        if self.base_calibrator is not None:
            print("=" * 50)
            print("DAC + Base Calibrator fitting completed!")
            print("=" * 50)
            return {
                'dac_weights': self.dac.get_optim_params(),
                'base_result': base_result
            }
        else:
            print("DAC fitting completed.")
            return self.dac.get_optim_params()

    def _apply_dac_only(self, test_logits, test_features, return_logits=False):
        """Apply only DAC (without base calibrator) to the logits"""
        device = test_logits.device

        # Get KNN scores for test data
        ood_scores = self.knn_scorer.knn_score([test_features])

        # Convert to numpy for DAC calibration
        test_logits_np = test_logits.detach().cpu().numpy()

        if return_logits:
            # Return calibrated logits
            calibrated_logits_np = self.dac.calibrate_before_softmax(test_logits_np, ood_scores)
            return torch.from_numpy(calibrated_logits_np).float().to(device)
        else:
            # Return calibrated probabilities
            calibrated_probs_np = self.dac.calibrate(test_logits_np, ood_scores)
            return torch.from_numpy(calibrated_probs_np).float().to(device)

    def calibrate(self, test_logits, test_features=None, return_logits=False, **kwargs):
        """
        Calibrate the logits using density aware calibration.

        Args:
            test_logits (torch.Tensor): Test logits (N, C)
            test_features (torch.Tensor): Test features for KNN scoring (N, D)
            return_logits (bool): Whether to return calibrated logits instead of probabilities
            **kwargs: Additional arguments passed to base calibrator if provided

        Returns:
            torch.Tensor: Calibrated probabilities or logits
        """
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before calibrating")

        if test_features is None:
            raise ValueError("DensityAwareCalibrator requires test_features for calibration")

        if self.base_calibrator is None:
            # Use DAC only
            return self._apply_dac_only(test_logits, test_features, return_logits)
        else:
            # Use DAC + base calibrator
            # Step 1: Apply DAC to get density-aware calibrated logits
            dac_calibrated_logits = self._apply_dac_only(test_logits, test_features, return_logits=True)

            # Step 2: Apply base calibrator to DAC-calibrated logits
            final_result = self.base_calibrator.calibrate(
                dac_calibrated_logits,
                return_logits=return_logits,
                **kwargs
            )

            return final_result

    def get_weights(self):
        """Get the optimized DAC weights"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted first")
        return self.dac.w

    def get_base_params(self):
        """Get parameters from the base calibrator (if available)"""
        if self.base_calibrator is None:
            return None
        elif hasattr(self.base_calibrator, 'get_temperature'):
            return self.base_calibrator.get_temperature()
        elif hasattr(self.base_calibrator, 'get_weights'):
            return self.base_calibrator.get_weights()
        else:
            return "Base calibrator parameters not accessible"