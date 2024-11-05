def test_ts():
    print("---Test Temperature Scaling---")

    from calibrator import TemperatureScalingCalibrator
    import torch
    from calibrator.metrics import ECE, AdaptiveECE, ClasswiseECE
    from .test_utils.cifar10 import get_train_valid_loader, get_test_loader
    from .test_utils.resnet import resnet50

    val_logits, val_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
    test_logits, test_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)

    calibrator = TemperatureScalingCalibrator()
    eps_opt = calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)

    print(f"Temperature: {calibrator.get_temperature()}")
    print("Uncalibrated ECE: ", ECE().cuda()(logits=test_logits, labels=test_labels))
    print("Calibrated ECE: ", ECE().cuda()(logits=calibrated_logits, labels=test_labels))
    print("!!! Pass Temperature Scaling Test !!!")

if __name__ == "__main__":
    test_ts()