from torch import nn
import torch
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """ Computes Focal Tversky Loss (GDL) as described in https://arxiv.org/pdf/1706.05721.pdf.
    """

    def __init__(
            self,
            alpha: float,
            beta: float,
            gamma: float,
            smooth: float = 1,
            dim: int = -1,
            clamp: float = 0.001
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.clamp = clamp
        self.dim = dim

    def forward(self, y, target, mask=None):
        assert y.shape == target.shape

        if mask is not None:
            y = mask * y

        tp = torch.sum(y * target, self.dim)
        fn = torch.sum((1. - y) * target, self.dim)
        fp = torch.sum(y * (1. - target), self.dim)

        score = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        score = score.clamp(min=self.clamp, max=1-self.clamp)

        loss = 1 - score

        if self.gamma != 1:
            return loss.pow(self.gamma).mean()
        else:
            return loss.mean()