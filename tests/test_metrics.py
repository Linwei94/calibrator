print("---Test Metrics---")

import torch
from calibrator.metrics import ECE, AdaptiveECE, ClasswiseECE, NLL, Accuracy

val_logits, val_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
test_logits, test_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)
softmaxes = torch.nn.functional.softmax(val_logits, dim=1)

print(ECE().cuda()(logits=val_logits, labels=val_labels))
print(AdaptiveECE().cuda()(logits=val_logits, labels=val_labels))
print(ClasswiseECE().cuda()(logits=val_logits, labels=val_labels))
print("---"*20)
print(NLL().cuda()(logits=val_logits, labels=val_labels))
print(NLL().cuda()(softmaxes=softmaxes, labels=val_labels))
print(torch.nn.functional.cross_entropy(val_logits, val_labels))
print(Accuracy().cuda()(logits=val_logits, labels=val_labels))
print(Accuracy().cuda()(softmaxes=softmaxes, labels=val_labels))
print("pass")