# Calibration
a package for calibrating the deep learning models

## Use case
```python
from calibrator import LocalCalibrator
import torch

val_logits = torch.randn(1000, 10)
val_labels = torch.randint(0, 10, (1000,))
test_logits = torch.randn(1000, 10)

calibrator = LocalCalibrator(aggregation='consistency', num_samples=1000, noise_type='gaussian')
eps_opt = calibrator.search_optimal_epsilon(val_logits, val_labels, verbose=True)
calibrated_probability = calibrator.calibrate(test_logits)
```