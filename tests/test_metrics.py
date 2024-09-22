def test_metrics():
    print("---Test Metrics---")

    import torch
    from calibrator.metrics import ECE, AdaptiveECE, ClasswiseECE, NLL, Accuracy

    val_logits, val_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_val_0.1_vanilla.pt", weights_only=False)
    test_logits, test_labels = torch.load("tests/test_logits/resnet50_cifar10_cross_entropy_test_0.9_vanilla.pt", weights_only=False)
    softmaxes = torch.nn.functional.softmax(val_logits, dim=1)

    print("ECE: ", ECE().cuda()(logits=val_logits, labels=val_labels))
    print("AdaptiveECE: ", AdaptiveECE().cuda()(logits=val_logits, labels=val_labels))
    print("ClasswiseECE: ", ClasswiseECE().cuda()(logits=val_logits, labels=val_labels))
    print("NLL: ", NLL().cuda()(logits=val_logits, labels=val_labels))
    print("Accuracy: ", Accuracy().cuda()(logits=val_logits, labels=val_labels))
    print("!!! Pass Metrics Test !!!")

if __name__ == "__main__":
    test_metrics()
