print("---Test Temperature Scaling---")

from calibrator import TemperatureScaling
import torch
from calibrator.metrics import ECE, AdaptiveECE, ClasswiseECE
from .test_utils.cifar10 import get_train_valid_loader, get_test_loader
from .test_utils.resnet import resnet50

val_logits, val_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
test_logits, test_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = resnet50(num_classes=10).to(device)
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
pretrained_weight_path = "/home/linwei/local_calibration_measure/pretrained_weights/cifar10_resnet50_cross_entropy.model"
net.load_state_dict(torch.load(pretrained_weight_path, weights_only=True))

_, val_loader = get_train_valid_loader(
    batch_size=128,
    augment=True,
    random_seed=1,
)

test_loader = get_test_loader(
    batch_size=128,
    shuffle=False,
)

calibrator = TemperatureScaling()
eps_opt = calibrator.fit(net, val_loader)
calibrated_logits = calibrator.calibrate(test_logits)

print(ECE().cuda()(logits=test_logits, labels=test_labels))
print(ECE().cuda()(logits=calibrated_logits, labels=test_labels))
print("pass")

