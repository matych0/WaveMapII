import torch
import torch.nn as nn
from typing import Tuple, Dict

from model.building_blocks import AttentionPooling
from model.pyramid_resnet import LocalActivationResNet



class CoxAttentionResnet(nn.Module):
    """
    A model combining a LocalActivationResNet and an AttentionPooling layer,
    designed for Cox Proportional Hazards modeling.
    """

    def __init__(self, resnet_params: Dict, amil_params: Dict):
        """
        Initializes the CoxAttentionResnet.

        Args:
            resnet_params: Parameters for the LocalActivationResNet.
            amil_params: Parameters for the AttentionPooling layer.
        """
        super().__init__()

        # Initialize the LocalActivationResNet
        self.resnet: nn.Module = LocalActivationResNet(**resnet_params)

        # Initialize the AMIL layer
        self.amil: nn.Module = AttentionPooling(**amil_params)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CoxAttentionResnet.

        Args:
            x: Input tensor.
            mask: Mask tensor. This is used to ignore padding in the input.

        Returns:
            A tuple containing the risk and attention weights.
        """
        # Pass input through ResNet
        x = self.resnet(x)

        # Transpose dimensions for compatibility
        x = x.transpose(-2, -1)

        # Pass through AMIL
        risk, attention_weights = self.amil(x, mask)

        return risk, attention_weights