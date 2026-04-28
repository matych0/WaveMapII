from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

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


class AFibAttentionNet(nn.Module):
    def __init__(self, embedding_dim=64, attention_dim=32, **kwargs):
        super(AFibAttentionNet, self).__init__()
        
        # 1. Instance Encoder
        # Maps the handcrafted features to a higher dimensional space
        self.feature_extractor = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )

        # 2. Gated Attention Mechanism
        self.attention_V = nn.Sequential(nn.Linear(embedding_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embedding_dim, attention_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attention_dim, 1)

        # 3. Risk Predictor (Cox Head)
        self.risk_layer = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Log-hazard ratio
        )

    def forward(self, x, mask=None, **kwargs):
        # x: [B, I]
        B, I, CH = x.shape
        
        # Add the feature dimension: [B, I] -> [B, I, 1]
        # x = x.unsqueeze(-1)
        
        # Flatten for the MLP: [B*I, 1]
        x_flat = x.view(-1, CH)
        H = self.feature_extractor(x_flat) # [B*I, embedding_dim]
        
        # Compute Attention
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U) # [B*I, 1]
        
        # Reshape attention scores back to [B, I]
        A = A.view(B, I)
        
        if mask is not None:
            # mask should be [B, I] boolean tensor
            # Set padded indices to -infinity so softmax makes them 0
            if mask.dtype != torch.bool:
                mask = mask.bool()
            
            A = A.masked_fill(~mask, float('-inf'))
            
        # Normalize weights across instances for each patient
        A_weights = F.softmax(A, dim=1) # [B, I]
        
        # Reshape H to [B, I, embedding_dim]
        H = H.view(B, I, -1)
        
        # Weighted average of instance embeddings -> Patient embedding
        # [B, 1, I] @ [B, I, embedding_dim] -> [B, 1, embedding_dim]
        M = torch.bmm(A_weights.unsqueeze(1), H).squeeze(1) 
        
        # Final Risk
        risk = self.risk_layer(M) # [B, 1]
        
        return risk, A_weights