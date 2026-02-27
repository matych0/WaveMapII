from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.model.building_blocks import (AttentionPooling, AveragePoolingBlock,
                                   MaxPoolingBlock)
from src.model.pyramid_resnet import LocalActivationResNet


class FeedforwardMIL(nn.Module):
    def __init__(self, amil_params: Dict):
        super().__init__()
    
        # Initialize the AMIL layer
        self.amil: nn.Module = AttentionPooling(**amil_params)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, batch_size=None) -> Tuple[torch.Tensor, torch.Tensor]:

        x = x.unsqueeze(-1)        # [B, I, 1]

        risk, attention_weights = self.amil(x, mask)

        return risk, attention_weights