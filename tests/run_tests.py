#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
import time
from typing import List, Dict, Any, Callable

# Add the current directory to the path so we can import the calibrator module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
from calibrator.tests.test_metrics import test_metrics
from calibrator.tests.test_pts_calibrator import test_pts_calibrator
from calibrator.tests.test_cts_calibrator import test_cts_calibrator
from calibrator.tests.test_temperature_scaling_calibrator import test_ts
from calibrator.tests.test_logit_clipping_calibrator import test_lc
from calibrator.tests.test_consistency_calibrator import test_consistency_calibrator

# Dictionary mapping test names to their functions
TEST_FUNCTIONS = {
    "metrics": test_metrics,
    "pts": test_pts_calibrator,
    "cts": test_cts_calibrator,
    "temperature_scaling": test_ts,
    "logit_clipping": test_lc,
    "consistency": test_consistency_calibrator,
    "all": None  # Special case for running all tests
}

def run_test(test_name: str) -> bool:
    """
    Run a specific test by name.
    
    Args:
        test_name: Name of the test to run
        
    Returns:
        bool: True if test passed, False otherwise
    """
    if test_name not in TEST_FUNCTIONS:
        print(f"Error: Unknown test '{test_name}'")
        return False
    
    if test_name == "all":
        # Run all tests
        all_passed = True
        for name, func in TEST_FUNCTIONS.items():
            if name != "all":
                print(f"\n{'='*50}")
                print(f"Running test: {name}")
                print(f"{'='*50}\n")
                try:
                    func()
                    print(f"\nTest '{name}' passed successfully!")
                except Exception as e:
                    print(f"\nTest '{name}' failed with error: {e}")
                    all_passed = False
        return all_passed
    
    # Run a single test
    print(f"\n{'='*50}")
    print(f"Running test: {test_name}")
    print(f"{'='*50}\n")
    
    try:
        TEST_FUNCTIONS[test_name]()
        print(f"\nTest '{test_name}' passed successfully!")
        return True
    except Exception as e:
        print(f"\nTest '{test_name}' failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run calibrator tests")
    parser.add_argument(
        "test_name", 
        choices=list(TEST_FUNCTIONS.keys()),
        help="Name of the test to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    success = run_test(args.test_name)
    end_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"Test execution completed in {end_time - start_time:.2f} seconds")
    print(f"Overall result: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*50}\n")
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()