from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvOps:
    def __init__(self, dim: int):
        assert dim in (1, 2)
        self.dim = dim

        self.Conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.BatchNorm = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
        self.MaxPool = nn.MaxPool1d if dim == 1 else nn.MaxPool2d
        self.AvgPool = nn.AvgPool1d if dim == 1 else nn.AvgPool2d

        self.conv_fn = F.conv1d if dim == 1 else F.conv2d


def get_activation(activation: str):
    if not isinstance(activation, str):
        return None
    # !!! inplace set to False, it might cause excessive model complexity !!!
    return {
        "ReLU": nn.ReLU(inplace=False),
        "LReLU": nn.LeakyReLU(negative_slope=0.1, inplace=False),
        "PReLU": nn.PReLU(num_parameters=1, init=0.1),
    }[activation]


def get_normalization(normalization: str, dim: int, num_channels: int = 0, num_groups: int = 1):
    if normalization is None:
        return None

    if normalization == "BatchN":
        return nn.BatchNorm1d(num_channels) if dim == 1 else nn.BatchNorm2d(num_channels)

    if normalization == "GroupN":
        return nn.GroupNorm(num_groups, num_channels)

    if normalization == "InstaN":
        return nn.GroupNorm(num_channels, num_channels)

    if normalization == "LayerN":
        return nn.GroupNorm(1, num_channels)

    raise ValueError(f"Unknown normalization {normalization}")


def temporal_stride(stride, dim):
    return stride if dim == 1 else (1, stride)

def temporal_kernel(k, dim):
    return k if dim == 1 else (1, k)

def temporal_padding(p, dim):
    return p if dim == 1 else (0, p)


#Needs modification
class ConvNdWrapper(nn.Module): # Replace with 1d
    """ Conv1d wrapper
    """

    def __init__(
            self,
            dim: int,
            in_channels: int,
            out_channels: int,
            kernel_size,
            dilation: int = 1,
            **kwargs,
    ):
        
        super().__init__()

        self.ops = ConvOps(dim)

        normalization = kwargs.pop("normalization", None)
        activation = kwargs.pop("activation", None)
        preactivation = kwargs.pop("preactivation", False)
        padding = kwargs.pop("padding", None)
        normalization_groups = kwargs.pop("normalization_groups", 1)

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2 * dilation
            else:
                padding = tuple(kernel_dimension // 2 * dilation for kernel_dimension in kernel_size)

        self.conv = self.ops.Conv(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            **kwargs,
        )

        self.preactivation = preactivation
        self.activation = get_activation(activation)

        if self.preactivation:
            norm_channels = in_channels
        else:
            norm_channels = out_channels

        self.normalization = get_normalization(normalization, dim, norm_channels, normalization_groups)

    def forward(self, x):

        if not self.preactivation:
            x = self.conv(x)

        if self.normalization is not None:
            x = self.normalization(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.preactivation:
            x = self.conv(x)

        return x


class ReshapeTensor(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


""" class MaxAntialiasDownsampling(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 2,
        normalization: Optional[str] = None,
        ) -> None:
        kernel_size = stride + 1
        padding = kernel_size // 2
        
        super().__init__(
        nn.MaxPool2d(kernel_size=(1, kernel_size), stride=1, padding=(0, padding)),
        nn.AvgPool2d(kernel_size=(1, kernel_size), stride=1, padding=(0, padding)),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, stride), bias=False),
        get_normalization(normalization,out_channels,1),
        ) # Replace with 1d """


class MaxAntialiasDownsampling(nn.Sequential):
    """ MaxBlur pooling inspired by "Making Convolutional Networks Shift-Invariant Again" at https://arxiv.org/pdf/1904.11486.

    Box filter is used as weak anti-alias filter instead of Finite Impulse Response filters. 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        normalization: Optional[str] = None,
        dim: int = 1,
    ):
        ops = ConvOps(dim)

        kernel_size = stride + 1
        padding = kernel_size // 2

        super().__init__(
            ops.MaxPool(
                kernel_size=temporal_kernel(kernel_size, dim),
                stride=1,
                padding=temporal_padding(padding, dim),
            ),
            ops.AvgPool(
                kernel_size=temporal_kernel(kernel_size, dim),
                stride=1,
                padding=temporal_padding(padding, dim),
            ),
            ops.Conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=temporal_stride(stride, dim),
                bias=False,
            ),
            get_normalization(normalization, dim, out_channels),
        )

class Basic1dStem(nn.Module):
    """
    ResNet input gate.
    """

    def __init__(self, dim, in_channels, out_channels, kernel_size, normalization: str = 'BatchN', activation: str = 'LReLU'):
        """
        """
        super().__init__()

        self.conv = ConvNdWrapper(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=temporal_kernel(kernel_size, dim),
            stride=temporal_stride(2, dim),
            bias=False,
            normalization=normalization,
            activation=activation,
            preactivation=False
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ReshapeTensor(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class ResidualBlock1D(nn.Module):
    """
    Super class for building general ResNet blocks.
    Original residual block is replaced by pre-activation variant (No layers after addition).
    """
    expansion = 2

    def __init__(
            self,
            dim: int,
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            **kwargs,
    ):
        raw_stride = kwargs.get("stride", 1)

        if dim == 1:
            self.has_stride = raw_stride > 1
        else:
            self.has_stride = isinstance(raw_stride, tuple) and raw_stride[1] > 1

        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        stride = kwargs.pop('stride', 1)
        activation = kwargs.pop('activation', None)
        normalization = kwargs.pop('normalization', None)
        preactivation = kwargs.pop('preactivation', None)

        planes = in_planes // self.expansion

        # residual block
        self.bottleneck = ConvNdWrapper(
            dim,
            in_planes,
            planes,
            kernel_size=temporal_kernel(1, dim),
            activation=activation,
            normalization=normalization,
            preactivation=preactivation,
            bias=False,
            **kwargs,
        )

        self.receptive_block = ConvNdWrapper(
            dim,
            planes,
            planes,
            kernel_size,
            activation=activation,
            normalization=normalization,
            preactivation=preactivation,
            bias=True,
            **kwargs,
        )

        self.pool = nn.AvgPool2d(kernel_size=stride) if self.has_stride else nn.Identity()

        self.expansion_block = ConvNdWrapper(
            dim,
            planes,
            out_planes,
            kernel_size=temporal_kernel(1, dim),
            activation=activation,
            normalization=normalization,
            preactivation=preactivation,
            bias=False,
            **kwargs,
        )

        # filters for reshaping identity data
        if self.has_stride or self.in_planes != self.out_planes:
            self.resample = nn.Sequential(
                get_normalization(
                    normalization,
                    dim,
                    in_planes
                ),

                nn.AvgPool2d(kernel_size=stride) if self.has_stride else nn.Identity(),

                ConvNdWrapper(
                    dim,
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    activation=None,
                    normalization=None,
                    preactivation=False,
                    **kwargs,
                )
            )

    def forward(self, x):
        # hold and/or resample input data
        if self.has_stride or self.in_planes != self.out_planes:
            identity = self.resample(x)
        else:
            identity = x

        # forward pass through residual block
        x = self.bottleneck(x)
        x = self.receptive_block(x)
        x = self.pool(x)
        x = self.expansion_block(x)

        # addition
        x += identity

        return x


class ResidualStage1D(nn.Module):
    """ Wrapper for stacking multiple residual block together
    """

    def __init__(
            self,
            dim: int,
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            layer_depth: int,
            **kwargs,
    ):
        super().__init__()

        self.stage = nn.Sequential()
        for block_idx in range(layer_depth):
            if block_idx == 0:
                stride = temporal_stride(2, dim)
            else:
                in_planes = out_planes
                stride = 1

            self.stage.append(
                ResidualBlock1D(
                    dim=dim,
                    in_planes=in_planes,
                    out_planes=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    **kwargs
                )
            )

    def forward(self, x):
        return self.stage(x)


class ResNet(nn.Module):
    """
    https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/resnet.py
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        blocks: Union[Tuple, List] = [3, 6, 32, 6],
        features: Union[Tuple, List] = [16, 32, 64, 128],
        activation: str = 'LReLU',
        normalization: str = 'BatchN',
        preactivation: bool = True,
        normalization_groups: int = 1,
        ** kwargs,
    ) -> None:
        """_summary_

        Args:
            in_features (int): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
            blocks (list, optional): _description_. Defaults to [3, 6, 32, 6].
            features (list, optional): _description_. Defaults to [16, 32, 64, 128].
            stem_kernel_size (int, optional): _description_. Defaults to 7.
            activation (str, optional): _description_. Defaults to 'LReLU'.
            normalization (str, optional): _description_. Defaults to 'BatchN'.
            preactivation (bool, optional): _description_. Defaults to True.
            normalization_groups (list, optional): _description_. Defaults to None.
        """

        assert len(blocks) == len(features)


        super().__init__()        
        
        # create encoding layers
        self.backbone = nn.ModuleList()

        for layer_depth, in_planes, out_planes in zip(
                blocks,
                [features[0]] + features[:-1],
                features,
        ):
            # append layer to encoder
            self.backbone.append(
                ResidualStage1D(
                    dim=dim,
                    in_planes=in_planes,
                    out_planes=in_planes,
                    kernel_size=temporal_kernel(kernel_size, dim),
                    layer_depth=layer_depth,
                    activation=activation,
                    normalization=normalization,
                    preactivation=preactivation,
                    padding=None,
                    normalization_groups=normalization_groups,
                )
            )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_
            sample_lengths (_type_): _description_

        Returns:
            _type_: _description_
        """
        # ResNet backbone
        for stage in self.backbone:
            x = stage(x)

        return x
    
    
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation="ReLU"):
        super(ProjectionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = get_activation(activation)

    def forward(self, x):
        return self.activation(self.fc(x))


class AttentionNetGated(nn.Module):
    """
    Gated Attention Network with optional Dropout.

    Args:
        input_size(int): Dimension of the input.
        middle_size(int): Dimension of the hidden layer.
        output_size(int): Dimension of the output.
        dropout (bool): Whether to use dropout.
        dropout_prob (float): Dropout probability.
    """

    def __init__(self, input_size=256, middle_size=128, output_size=1, dropout=False, dropout_prob=0.25):
        super(AttentionNetGated, self).__init__()

        self.attention_u = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob) if dropout else nn.Identity()
        )
        self.attention_v = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.Sigmoid(),
            nn.Dropout(dropout_prob) if dropout else nn.Identity()
        )
        self.attention_z = nn.Linear(middle_size, output_size)

    def forward(self, x):
        u = self.attention_u(x)
        v = self.attention_v(x)
        uv = u * v
        z = self.attention_z(uv)
        return z


class AttentionPooling(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 attention_hidden_size,
                 output_size,
                 dropout=False,
                 dropout_prob=0.25):
        """
        Attention pooling layer that applies a projection, attention mechanism, and a final prediction layer.
        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer after projection.
            attention_hidden_size (int): Size of the hidden layer in the attention mechanism.
            output_size (int): Size of the output (usually 1 for cox regression risk score).
            dropout (bool): Whether to apply dropout.   
            dropout_prob (float): Probability of dropout if applied.
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout else nn.Identity()
        )

        # Pass dropout params into AttentionNetGated
        self.attention_pool = AttentionNetGated(hidden_size, attention_hidden_size, 1, dropout=dropout,
                                                dropout_prob=dropout_prob)

        self.predictor = nn.Sequential(
            nn.Dropout(dropout_prob) if dropout else nn.Identity(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, h, mask=None):
        h = self.projection(h)
        attention_weights = self.attention_pool(h)
        attention_weights = torch.transpose(attention_weights, -2, -1)

        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_weights.masked_fill_(mask == 0, float('-inf'))

        attention_weights_softmax = F.softmax(attention_weights, dim=-1)
        avg_instances = torch.matmul(attention_weights_softmax, h)
        risk = self.predictor(avg_instances)

        return risk, (h, attention_weights_softmax)


class MaxPoolingBlock(nn.Module):
    """
    MaxPooling aggregation followed by a fully connected layer to produce a single risk score.

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output (usually 1 for cox regression risk score).
    """
    def __init__(self, input_size, hidden_size, output_size=1, dropout=False, dropout_prob=0.25):
        super(MaxPoolingBlock, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout else nn.Identity()
        )
        
        self.predictor = nn.Sequential(
            nn.Dropout(dropout_prob) if dropout else nn.Identity(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, h, mask=None):
        """
        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, num_instances, feature_dim).
            mask (torch.Tensor, optional): Binary mask of shape (batch_size, num_instances) where 1=valid, 0=padding.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        
        h = self.projection(h)  # shape: (batch_size, num_instances, hidden_size)

        if mask is not None:
            # Mask invalid positions by setting them to -inf before max
            mask = mask.unsqueeze(-1)  # (batch_size, num_instances, 1)
            h = h.masked_fill(mask == 0, float('-inf'))

        # Max pooling across instances (dim=1)
        max_pooled, _ = torch.max(h, dim=1)  # shape: (batch_size, feature_dim)

        # Fully connected layer to get final score
        risk = self.predictor(max_pooled)  # shape: (batch_size, output_size)

        return risk


class AveragePoolingBlock(nn.Module):
    """
    AveragePooling aggregation followed by a fully connected layer to produce a single risk score.

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output (usually 1 for cox regression risk score).
    """
    def __init__(self, input_size, hidden_size, output_size=1, dropout=False, dropout_prob=0.25):
        super(AveragePoolingBlock, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout else nn.Identity()
        )
        
        self.predictor = nn.Sequential(
            nn.Dropout(dropout_prob) if dropout else nn.Identity(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, h, mask=None):
        """
        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, num_instances, feature_dim).
            mask (torch.Tensor, optional): Binary mask of shape (batch_size, num_instances), where 1=valid, 0=padding.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size, 1)
        """
        
        h = self.projection(h)

        if mask is not None:
            # Expand mask for broadcasting
            mask = mask.unsqueeze(-1).float()  # (batch_size, num_instances, 1)
            h = h * mask  # Zero out invalid positions

            # Compute mean only over valid instances
            summed = torch.sum(h, dim=1)  # (batch_size, feature_dim)
            counts = torch.clamp(mask.sum(dim=1), min=1e-6)  # (batch_size, 1) to avoid division by zero
            avg_pooled = summed / counts
        else:
            # Simple mean across instances
            avg_pooled = torch.mean(h, dim=1)

        # Fully connected layer to get final score
        risk = self.predictor(avg_pooled)  # shape: (batch_size, output_size)

        return risk


if __name__ == "__main__":
    x = torch.randn(10,400,128)
    attn_block = AttentionPooling(128,128,64,1)
    output = attn_block(x)
    print(output[0].shape)

    