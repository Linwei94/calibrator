from calibrator.metrics import ECE

def test_pts_calibrator():
    print("---Test PTS Calibrator---")

    from calibrator import PTSCalibrator
    import torch
    import numpy as np

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load validation and test data
    val_logits, val_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
    test_logits, test_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)

    # Move data to the appropriate device
    val_logits = val_logits.to(device)
    val_labels = val_labels.to(device)
    test_logits = test_logits.to(device)
    test_labels = test_labels.to(device)

    # Get the number of classes from the logits shape
    num_classes = val_logits.shape[1]

    # Initialize the PTS calibrator with default parameters
    calibrator = PTSCalibrator(
        length_logits=num_classes,  # Only specify length_logits, use defaults for others,
        epochs=1000
    )

    # Fit the calibrator on validation data
    calibrator.fit(val_logits, val_labels)

    # Calibrate the test logits
    calibrated_probability = calibrator.calibrate(test_logits)

    # Calculate and print ECE metrics
    uncalibrated_ece = ECE()(labels=test_labels, logits=test_logits)
    calibrated_ece = ECE()(labels=test_labels, softmaxes=calibrated_probability)

    print(f"Uncalibrated ECE: {uncalibrated_ece:.4f}")
    print(f"Calibrated ECE: {calibrated_ece:.4f}")
    
    # Verify that calibration improved the ECE
    assert calibrated_ece < uncalibrated_ece, "Calibration should improve ECE"
    
    # Test with custom loss function
    custom_calibrator = PTSCalibrator(
        length_logits=num_classes,
        loss_fn=torch.nn.CrossEntropyLoss()
    )
    custom_calibrator.fit(val_logits, val_labels)
    custom_calibrated_probability = custom_calibrator.calibrate(test_logits)
    custom_calibrated_ece = ECE()(labels=test_labels, softmaxes=custom_calibrated_probability)
    print(f"Custom Loss Calibrated ECE: {custom_calibrated_ece:.4f}")
    
    # Test saving and loading the model
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the model
        calibrator.save(temp_dir)
        
        # Create a new calibrator instance with default parameters
        new_calibrator = PTSCalibrator(
            length_logits=num_classes
        )
        
        # Load the saved model
        new_calibrator.load(temp_dir)
        
        # Move the new calibrator to the same device
        new_calibrator.to(device)
        
        # Calibrate with the loaded model
        loaded_calibrated_probability = new_calibrator.calibrate(test_logits)
        
        # Verify that the loaded model produces the same results
        np.testing.assert_array_almost_equal(
            calibrated_probability.cpu().numpy(), 
            loaded_calibrated_probability.cpu().numpy(), 
            decimal=5, 
            err_msg="Loaded model should produce the same calibration results"
        )
    
    # Test calibrate with return_logits=True
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    
    # Verify that the returned logits can be converted to the same probabilities
    manual_calibrated_probs = torch.nn.functional.softmax(calibrated_logits, dim=1)
    np.testing.assert_array_almost_equal(
        calibrated_probability.cpu().numpy(), 
        manual_calibrated_probs.cpu().numpy(), 
        decimal=5, 
        err_msg="Manual softmax of returned logits should match calibrated probabilities"
    )
    
    print("!!! Pass PTS Calibrator Test !!!")

if __name__ == "__main__":
    test_pts_calibrator()
