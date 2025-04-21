# soft_ece.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftECE(nn.Module):
    """
    Soft-binned Expected Calibration Error loss (using a Gaussian kernel).
    通过"软分桶"来计算 ECE，使梯度对置信度更加平滑可传播。
    """
    def __init__(self, n_bins=15, sigma=0.05, eps=1e-6):
        """
        Args:
            n_bins: 将 [0,1] 区间大致分为多少个 bin
            sigma: 高斯核的带宽 (标准差)
            eps: 避免除 0 的小量
        """
        super(SoftECE, self).__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.eps = eps
        
        # 这里使用 bin 的中心点而不是边界
        # 若希望严格对齐 [0,1] 两端，也可以选择别的排布方式
        self.register_buffer('bin_centers', torch.linspace(1/(2*self.n_bins), 
                                          1 - 1/(2*self.n_bins), 
                                          self.n_bins))
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes] 的网络输出
            targets: [batch_size] 的整型真实标签
        
        Returns:
            soft ECE loss (可参与反向传播)
        """
        # Ensure targets is on the same device as logits
        targets = targets.to(logits.device)
        
        # 1) logits -> 概率分布
        probs = F.softmax(logits, dim=1)  # [B, C]
        
        # 2) 获取每个样本的最高置信度
        confidences, predictions = torch.max(probs, dim=1)  # [B]
        
        # 3) 计算该预测是否正确
        accuracies = (predictions == targets).float()       # [B]
        
        # 4) 计算对每个 bin 的软分配权重 (Gaussian kernel)
        #    shape: [batch_size, n_bins]
        #    weights[j, i] = exp(- (confidences[j] - bin_centers[i])^2 / (2*sigma^2))
        #    后续会再进行归一化
        diff = confidences.unsqueeze(1) - self.bin_centers.unsqueeze(0)  # [B, n_bins]
        weights = torch.exp(-0.5 * (diff**2) / (self.sigma**2))     # [B, n_bins]
        
        # 归一化：对每个样本，使所有 bin 的权重之和=1
        # 这样每个样本会对所有 bin 有一个分布(soft assignment)
        weights_sum = weights.sum(dim=1, keepdim=True) + self.eps   # [B, 1]
        weights_norm = weights / weights_sum                        # [B, n_bins]
        
        # 5) 分别计算每个 bin 的"平均置信度"和"平均准确率"
        #    avg_confidence_in_bin[i] = \sum_j (weights_norm[j,i] * confidences[j]) / \sum_j (weights_norm[j,i])
        #    下面用按列求和的方式实现
        weighted_confidence = weights_norm * confidences.unsqueeze(1)   # [B, n_bins]
        sum_conf_in_bin = weighted_confidence.sum(dim=0)                # [n_bins]
        sum_weights_in_bin = weights_norm.sum(dim=0)                    # [n_bins]
        avg_confidence_in_bin = sum_conf_in_bin / (sum_weights_in_bin + self.eps)
        
        # 同理计算平均准确率
        # 确保 accuracies 的形状与 weights_norm 兼容
        weighted_accuracy = weights_norm * accuracies.unsqueeze(1)      # [B, n_bins]
        sum_acc_in_bin = weighted_accuracy.sum(dim=0)                   # [n_bins]
        avg_accuracy_in_bin = sum_acc_in_bin / (sum_weights_in_bin + self.eps)
        
        # 6) 计算 ECE：对每个 bin 的误差 * bin 的权重占比 再累加
        #    prop_in_bin = \sum_j weights_norm[j, i] / batch_size
        prop_in_bin = sum_weights_in_bin / confidences.size(0)          # [n_bins]
        
        # soft ece = sum_i ( | avg_conf_in_bin[i] - avg_acc_in_bin[i] | * prop_in_bin[i] )
        ece_per_bin = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
        soft_ece = torch.sum(ece_per_bin * prop_in_bin)
        
        return soft_ece
