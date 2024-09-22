print("---Test LocalCalibrator---")


from calibrator import LocalCalibrator
import torch

val_logits, val_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
test_logits, test_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)

calibrator = LocalCalibrator()
eps_opt = calibrator.fit(val_logits, val_labels)
calibrated_probability = calibrator.calibrate(test_logits)

print("pass")