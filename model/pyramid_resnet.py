from model import building_blocks as bb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from model.building_blocks import AttentionPooling


class LocalActivationResNet(nn.Module):

    def __init__(
            self,
            in_features: int,
            kernel_size: int,
            blocks: Union[Tuple, List],
            features: Union[Tuple, List],
            stem_kernel_size: int,
            activation: str,
            normalization: str,
            downsampling_factor: int,
            preactivation: bool,
            trace_stages: bool,
            **kwargs,
    ) -> None:

        super().__init__()
        self._name = kwargs.pop('name', 'defMdl')
        self._version = kwargs.pop('version', '0.0.1')

        self.trace_stages = trace_stages

        # input gate
        stem_in_planes, stem_out_planes = in_features, features[0]

        self.stem = bb.Basic1dStem(
            in_channels=stem_in_planes,
            out_channels=stem_out_planes,
            kernel_size=stem_kernel_size,
            normalization=None,
            activation=None,
        )
        
        self.downsample = bb.MaxAntialiasDownsampling(
            in_channels=stem_out_planes,
            out_channels=stem_out_planes,
            stride=downsampling_factor,
            normalization=normalization,
        )

        # encoder
        self.backbone = bb.ResNet(
            kernel_size=kernel_size,
            blocks=blocks,
            features=features,
            stem_kernel_size=stem_kernel_size,
            activation=activation,
            normalization=normalization,
            preactivation=preactivation,
            trace_stages=trace_stages,
        )


        # Custom layer weights init
        self.init_weights(orthogonal=False)

    def init_weights(self, orthogonal: bool = False):
        """_summary_
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                if orthogonal:
                    nn.init.orthogonal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)

            if isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def cosine_regularization(self):
        cos_losses = list()
        for name, module in self.named_modules():
            if name.split('.')[0] not in {'stem'}:
                continue

            if isinstance(module, nn.Conv1d):
                eye_mask = torch.eye(module.weight.shape[0], device=module.weight.device,
                                     dtype=torch.bool).logical_not()

                flatten_w = torch.flatten(module.weight, 1, 2)
                norm = torch.linalg.vector_norm(flatten_w, dim=1).unsqueeze(1)
                normalized_w = flatten_w / torch.clamp(norm, min=1e-8)
                w_cos = torch.matmul(normalized_w, torch.transpose(normalized_w, 0, 1).contiguous())
                w_cos = eye_mask * w_cos
                cos_losses.append(torch.mean(torch.pow(w_cos, 2)))

        return torch.tensor(cos_losses, device=module.weight.device).mean()

    def forward(self, x):
        # Stem
        x = self.stem(x)
        # Downsample
        x = self.downsample(x)
        # Backbone
        x = self.backbone(x)
        
        # Global average pooling    
        return torch.mean(x,dim=3,keepdim=False)
    
    
    
if __name__ == "__main__":
    
    x = torch.rand([10,1,40,2035])
    
    resnet = LocalActivationResNet(
        in_features=1,
        kernel_size=(1,5),
        stem_kernel_size=(1,17),
        blocks=[9,12,18,9],
        features=[16,32,64,128],
        activation="LReLU",
        normalization="BatchN2D",
        downsampling_factor=4,
        preactivation=False,
        trace_stages=True
    )

    
    x = resnet(x)
    print("ResNet output________")
    print(x.shape)
    
    x = x.transpose(-2,-1)
    print("Transpose__________")
    print(x.shape)
    
    amil = AttentionPooling(
        input_size=128,
        hidden_size=128,
        attention_hidden_size=64,
        output_size=1,
        dropout=False,
        dropout_prob=0.25
    )
    
    risk, a = amil(x)
    print("AMIL output________________")
    print(risk)
    print(a.shape)
    