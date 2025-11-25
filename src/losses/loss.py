from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchtuples import TupleTree


class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, g_case, g_control, shrink):
        """CoxCC and CoxTime loss, but with only a single control.
        https://github.com/havakv/pycox/blob/master/pycox/models/loss.py
        """
        loss = F.softplus(g_control - g_case).mean()
        if shrink != 0:
            loss += shrink * (g_case.abs().mean() + g_control.abs().mean())
        return loss
    

def cox_cc_loss_single_ctrl(g_case: Tensor, g_control: Tensor, shrink: float = 0.) -> Tensor:
    """CoxCC and CoxTime loss, but with only a single control.
    """
    loss = F.softplus(g_control - g_case).mean()
    if shrink != 0:
        loss += shrink * (g_case.abs().mean() + g_control.abs().mean())
    return loss
    

def cox_cc_loss(g_case: Tensor, g_control: Tensor, shrink : float = 0.,
                clamp: Tuple[float, float] = (-3e+38, 80.)) -> Tensor:
    """Torch loss function for the Cox case-control models.
    For only one control, see `cox_cc_loss_single_ctrl` instead.
    
    Arguments:
        g_case {torch.Tensor} -- Result of net(input_case)
        g_control {torch.Tensor} -- Results of [net(input_ctrl1), net(input_ctrl2), ...]
    
    Keyword Arguments:
        shrink {float} -- Shrinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})
    
    Returns:
        [type] -- [description]
    """
    control_sum = 0.
    shrink_control = 0.
    if g_case.shape != g_control[0].shape:
        raise ValueError(f"Need `g_case` and `g_control[0]` to have same shape. Got {g_case.shape}"+
                         f" and {g_control[0].shape}")
    for ctr in g_control:
        shrink_control += ctr.abs().mean()
        ctr = ctr - g_case
        ctr = torch.clamp(ctr, *clamp)  # Kills grads for very bad cases (should instead cap grads!!!).
        control_sum += torch.exp(ctr)
    loss = torch.log(1. + control_sum)
    shrink_zero = shrink * (g_case.abs().mean() + shrink_control) / len(g_control)
    return torch.mean(loss) + shrink_zero.abs()


class CoxCCLoss(torch.nn.Module):
    """Torch loss function for the Cox case-control models.

    loss_func = LossCoxCC()
    loss = loss_func(g_case, g_control)
    
    Keyword Arguments:
        shrink {float} -- Shrinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})
    https://github.com/havakv/pycox/blob/master/pycox/models/loss.py
    """
    def __init__(self, shrink: float = 0., clamp: Tuple[float, float] = (-3e+38, 80.)) -> Tensor:
        super().__init__()
        self.shrink = shrink
        self.clamp = clamp

    @property
    def shrink(self) -> float:
        return self._shrink
    
    @shrink.setter
    def shrink(self, shrink: float) -> None:
        if shrink < 0:
            raise ValueError(f"Need shrink to be non-negative, got {shrink}.")
        self._shrink = shrink

    def forward(self, g_case: Tensor, g_control: TupleTree) -> Tensor:
        single = False
        if hasattr(g_control, 'shape'):
            if g_case.shape == g_control.shape:
                return cox_cc_loss_single_ctrl(g_case, g_control, self.shrink)
        elif (len(g_control) == 1) and (g_control[0].shape == g_case.shape):
                return cox_cc_loss_single_ctrl(g_case, g_control[0], self.shrink)
        return cox_cc_loss(g_case, g_control, self.shrink, self.clamp)


if __name__ == "__main__":
    g_case = torch.randn([4])
    g_control = torch.randn([8, 4])
    
    loss = CoxCCLoss(shrink=0.1)
    
    print(loss(g_case=g_case, g_control=g_control))