from calibrator.metrics import ECE

def test_consistency_calibrator():
    print("---Test Consistency Calibrator---")


    from calibrator import ConsistencyCalibrator
    import torch

    val_logits, val_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
    test_logits, test_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)

    calibrator = ConsistencyCalibrator()
    eps_opt = calibrator.fit(val_logits, val_labels)
    calibrated_probability = calibrator.calibrate(test_logits)

    print(f"eps_opt: {eps_opt:.4f}, noise_type: {calibrator.noise_type}")
    print(f"Uncalibrated ECE: {ECE()(labels=test_labels, logits=test_logits):.4f}")
    print(f"Calibrated ECE: {ECE()(labels=test_labels, softmaxes=calibrated_probability):.4f}")
    print("!!! Pass Consistency Calibrator Test !!!")

if __name__ == "__main__":
    test_consistency_calibrator()