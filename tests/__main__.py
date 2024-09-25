from .test_metrics import test_metrics
from .test_ts import test_ts
from .test_local_calibrator import test_local_calibrator    

if __name__ == "__main__":
    test_local_calibrator()
    test_metrics()
    test_ts()