from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.model.building_blocks import (AttentionPooling, AveragePoolingBlock,
                                   MaxPoolingBlock)
from src.model.pyramid_resnet import LocalActivationResNet


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
    
    
class CoxMaxResnet(nn.Module):
    """
    A model combining a LocalActivationResNet and a MaxPooling layer,
    designed for Cox Proportional Hazards modeling.
    """

    def __init__(self, resnet_params: Dict, maxmil_params: Dict):
        """
        Initializes the CoxMaxResnet.

        Args:
            resnet_params: Parameters for the LocalActivationResNet.
            maxmil_params: Parameters for the MaxPooling layer.
        """
        super().__init__()

        # Initialize the LocalActivationResNet
        self.resnet: nn.Module = LocalActivationResNet(**resnet_params)

        # Initialize the MaxPooling layer
        self.maxmil: nn.Module = MaxPoolingBlock(**maxmil_params)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CoxMaxResnet.

        Args:
            x: Input tensor.
            mask: Mask tensor. This is used to ignore padding in the input.

        Returns:
            Risk scores from the MaxPooling layer.
        """
        # Pass input through ResNet
        x = self.resnet(x)

        # Transpose dimensions for compatibility
        x = x.transpose(-2, -1)

        # Pass through MaxPooling layer
        risk = self.maxmil(x, mask)

        return risk, None
    
    
class CoxAvgResnet(nn.Module):
    """
    A model combining a LocalActivationResNet and an AveragePooling layer,
    designed for Cox Proportional Hazards modeling.
    """

    def __init__(self, resnet_params: Dict, avgmil_params: Dict):
        """
        Initializes the CoxAvgResnet.

        Args:
            resnet_params: Parameters for the LocalActivationResNet.
            avgmil_params: Parameters for the AveragePooling layer.
        """
        super().__init__()

        # Initialize the LocalActivationResNet
        self.resnet: nn.Module = LocalActivationResNet(**resnet_params)

        # Initialize the AveragePooling layer
        self.avgmil: nn.Module = AveragePoolingBlock(**avgmil_params)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CoxAvgResnet.

        Args:
            x: Input tensor.
            mask: Mask tensor. This is used to ignore padding in the input.

        Returns:
            Risk scores from the AveragePooling layer.
        """
        # Pass input through ResNet
        x = self.resnet(x)

        # Transpose dimensions for compatibility
        x = x.transpose(-2, -1)

        # Pass through AveragePooling layer
        risk = self.avgmil(x, mask)

        return risk, None
    

if __name__ == "__main__":
    x = torch.rand([2500, 24, 203])  # [batch, channels, instances, samples]

    resnet_params = {
        "in_features": 24,
        "dim": 1,
        "kernel_size": 5,
        "stem_kernel_size": 17,
        "blocks": [3, 4, 6, 3],
        "features": [24, 32, 64, 128],
        "activation": "LReLU",
        "normalization": "BatchN",
        "downsampling_factor": 2,
        "preactivation": False
    }

    amil_params = {
        "input_size": 128,
        "hidden_size": 128,
        "attention_hidden_size": 64,
        "output_size": 1,
        "dropout": False,
    }

    model = CoxAttentionResnet(resnet_params, amil_params)
    risk, attention_weights = model(x, None)
    print("CoxAttentionResnet output________")
    print(risk.shape)