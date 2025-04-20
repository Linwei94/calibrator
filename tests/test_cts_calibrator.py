import os
import sys

# Add the parent directory to the path so we can import the Component module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Component.metrics import ECE

def test_cts_calibrator():
    print("---Test CTS Calibrator---")

    from Component import CTSCalibrator
    import torch
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load validation and test data
    val_logits, val_labels = torch.load("calibrator/tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
    test_logits, test_labels = torch.load("calibrator/tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)
    
    # Ensure all tensors are on CPU
    val_logits = val_logits.cpu()
    val_labels = val_labels.cpu()
    test_logits = test_logits.cpu()
    test_labels = test_labels.cpu()

    # Get the number of classes from the logits shape
    num_classes = val_logits.shape[1]

    # Initialize the CTS calibrator with appropriate parameters
    calibrator = CTSCalibrator(
        n_class=num_classes,      # Number of classes
        n_iter=3,                 # Number of iterations for greedy search
        n_bins=15                 # Number of bins for ECE computation
    )
    
    # Fit the calibrator on validation data
    temperatures, metrics = calibrator.fit(val_logits, val_labels)
    print(f"Optimized temperatures: {temperatures}")
    print(f"Final ECE: {metrics['final_ece']:.4f}")
    print(f"Final Accuracy: {metrics['final_accuracy']:.4f}")

    # Verify temperatures are positive
    assert np.all(temperatures > 0), "All temperatures should be positive"

    # Calibrate the test logits
    calibrated_probability = calibrator.calibrate(test_logits)
    
    # Convert calibrated_probability back to PyTorch tensor for ECE calculation
    calibrated_probability_tensor = torch.tensor(calibrated_probability, dtype=torch.float32)

    # Calculate and print ECE metrics
    uncalibrated_ece = ECE()(labels=test_labels, logits=test_logits)
    calibrated_ece = ECE()(labels=test_labels, softmaxes=calibrated_probability_tensor)

    print(f"Uncalibrated ECE: {uncalibrated_ece:.4f}")
    print(f"Calibrated ECE: {calibrated_ece:.4f}")
    
    # Verify that calibration improved the ECE
    assert calibrated_ece < uncalibrated_ece, "Calibration should improve ECE"
    
    # Test saving and loading the model
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the model
        calibrator.save(temp_dir)
        
        # Create a new calibrator instance
        new_calibrator = CTSCalibrator(
            n_class=num_classes,
            n_iter=3,
            n_bins=15
        )
        
        # Load the saved model
        new_calibrator.load(temp_dir)
        
        # Verify that loaded temperatures match original temperatures
        np.testing.assert_array_almost_equal(
            calibrator.T.detach().cpu().numpy(),
            new_calibrator.T.detach().cpu().numpy(),
            decimal=5,
            err_msg="Loaded temperatures should match original temperatures"
        )
        
        # Calibrate with the loaded model
        loaded_calibrated_probability = new_calibrator.calibrate(test_logits)
        
        # Verify that the loaded model produces the same results
        np.testing.assert_array_almost_equal(
            calibrated_probability, 
            loaded_calibrated_probability, 
            decimal=5, 
            err_msg="Loaded model should produce the same calibration results"
        )
    
    # Test calibrate with return_logits=True
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)
    
    # Verify that the returned logits can be converted to the same probabilities
    manual_calibrated_probs = torch.nn.functional.softmax(torch.tensor(calibrated_logits), dim=1).numpy()
    np.testing.assert_array_almost_equal(
        calibrated_probability, 
        manual_calibrated_probs, 
        decimal=5, 
        err_msg="Manual softmax of returned logits should match calibrated probabilities"
    )
    
    # Test per-class temperature scaling
    # Take a single example and verify each class is scaled correctly
    test_example = test_logits[0:1]  # shape: (1, num_classes)
    calibrated_example = calibrator.calibrate(test_example, return_logits=True)
    
    # Verify each class is scaled by its corresponding temperature
    for class_idx in range(num_classes):
        expected_logit = test_example[0, class_idx] / temperatures[class_idx]
        np.testing.assert_almost_equal(
            calibrated_example[0, class_idx],
            expected_logit,
            decimal=5,
            err_msg=f"Class {class_idx} should be scaled by its corresponding temperature"
        )
    
    print("!!! Pass CTS Calibrator Test !!!")

if __name__ == "__main__":
    test_cts_calibrator() 