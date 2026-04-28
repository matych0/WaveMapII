# losses/focal_loss_tv.py

import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )