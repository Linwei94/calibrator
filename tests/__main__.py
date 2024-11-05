from .test_metrics import test_metrics
from .test_temperature_scaling_calibrator import test_ts
from .test_consistency_calibrator import test_consistency_calibrator    
from .test_logit_clipping_calibrator import test_lc
if __name__ == "__main__":
    # Metrics
    test_metrics()

    # Calibrators
    test_consistency_calibrator()
    test_ts()
    test_lc()
