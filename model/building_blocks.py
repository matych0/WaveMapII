import torch
import torch.nn as nn
import torch.nn.functional as F
#from nnAudio import features
from typing import Optional, List, Tuple, Union


def get_activation(activation: str):
    if not isinstance(activation, str):
        return None

    return {
        "ReLU": nn.ReLU(inplace=True),
        "LReLU": nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "PReLU": nn.PReLU(num_parameters=1, init=0.1),
    }[activation]


def get_normalization(normalization: str, num_channels: int = 0, num_groups: int = 1):
    if not isinstance(normalization, str):
        return None

    choices = nn.ModuleDict({
        'conv': nn.Conv2d(10, 10, 3),
        'pool': nn.MaxPool2d(3)
    })

    return {
        "BatchN": nn.BatchNorm1d(num_channels),
        "BatchN2D": nn.BatchNorm2d(num_channels),
        "GroupN": nn.GroupNorm(num_groups, num_channels),
        "InstaN": nn.GroupNorm(num_channels, num_channels),
        "LayerN": nn.GroupNorm(1, num_channels),
    }[normalization]

#Needs modification
class Conv1dWrapper(nn.Conv2d):
    """ Conv1d wrapper
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, ...],
            dilation: int = 1,
            **kwargs,
    ):
        """
        """
        normalization = kwargs.pop("normalization", None)
        activation = kwargs.pop("activation", None)
        preactivation = kwargs.pop("preactivation", False)
        padding = kwargs.pop("padding", None)
        normalization_groups = kwargs.pop("normalization_groups", 1)

        if padding is None:
            padding = tuple(kernel_dimension // 2 * dilation for kernel_dimension in kernel_size)

        super().__init__(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, **kwargs)

        self.preactivation = preactivation
        self.activation = get_activation(activation)

        if self.preactivation:
            norm_channels = in_channels
        else:
            norm_channels = out_channels

        self.normalization = get_normalization(normalization, norm_channels, normalization_groups)

    def forward(self, x):

        if not self.preactivation:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.normalization is not None:
            x = self.normalization(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.preactivation:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x


class ReshapeTensor(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Basic1dStem(nn.Module):
    """
    ResNet input gate.
    """

    def __init__(self, in_channels, out_channels, kernel_size, normalization: str = 'BatchN2D', activation: str = 'LReLU'):
        """
        """
        super().__init__()

        self.conv = Conv1dWrapper(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1,2),
            bias=False,
            normalization=normalization,
            activation=activation,
            preactivation=False
        )

    def forward(self, x):
        x = self.conv(x)
        return F.avg_pool2d(x, kernel_size=(1,3), stride=(1,2), padding=(0,1))

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
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            **kwargs,
    ):
        self.has_stride = True if kwargs.get('stride', (1,1))[1] > 1 else False
        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        stride = kwargs.pop('stride', (1,1))
        activation = kwargs.pop('activation', None)
        normalization = kwargs.pop('normalization', None)
        preactivation = kwargs.pop('preactivation', None)

        planes = in_planes // self.expansion

        # residual block
        self.bottleneck = Conv1dWrapper(
                in_planes,
                planes,
                kernel_size=(1,1),
                activation=activation,
                normalization=normalization,
                preactivation=preactivation,
                bias=False,
                **kwargs,
            )

        self.receptive_block = Conv1dWrapper(
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

        self.expansion_block = Conv1dWrapper(
            planes,
            out_planes,
            kernel_size=(1,1),
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
                    in_planes,
                ),

                nn.AvgPool2d(kernel_size=stride) if self.has_stride else nn.Identity(),

                Conv1dWrapper(
                    in_planes,
                    out_planes,
                    kernel_size=(1,1),
                    stride=(1,1),
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
                stride = (1,2)
            else:
                in_planes = out_planes
                stride = (1,1)

            self.stage.append(
                ResidualBlock1D(
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
        kernel_size: Union[Tuple[int, ...], int] = (1,3),
        blocks: Union[Tuple, List] = [3, 6, 32, 6],
        features: Union[Tuple, List] = [16, 32, 64, 128],
        activation: str = 'LReLU',
        normalization: str = 'BatchN2D',
        preactivation: bool = True,
        trace_stages: bool = True,                        
        normalization_groups: int = 1,
        kernel_dimension: int = 2,
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
            trace_stages (bool, optional): _description_. Defaults to True.
            normalization_groups (list, optional): _description_. Defaults to None.
        """

        assert len(blocks) == len(features)

        if kernel_dimension == 2:
            residual_class = ResidualStage1D
        elif kernel_dimension == 1:
            raise NotImplementedError
        else:
            raise ValueError(f'Incorrect kernel size. Expected `int` or `tuple`. Got {type(kernel_size)}')

        super().__init__()        
        self.trace_stages = trace_stages
        
        # create encoding layers
        self.backbone = nn.ModuleList()

        for layer_depth, in_planes, out_planes in zip(
                blocks,
                [features[0]] + features[:-1],
                features,
        ):
            # append layer to encoder
            self.backbone.append(
                residual_class(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
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


class PyramidFeatures(nn.Module):
    """
    A ResNet layer with N stacked blocks.
    """

    def __init__(
            self,
            features: Union[Tuple, List],
            compression: int,
            kernel_size: int,
            activation: str = 'LReLU',
            normalization: str = 'BatchN',
            preactivation: bool = True,
            **kwargs,
    ):
        """_summary_

        Args:
            features (Union[Tuple, List]): _description_
            compression (int): _description_
            activation (str, optional): _description_. Defaults to 'LReLU'.
            normalization (str, optional): _description_. Defaults to 'BatchN'.
            preactivation (bool, optional): _description_. Defaults to True.
        """
        super().__init__()

        in_planes = features

        # convolution heads with fixed kernels
        self.compression_head = nn.ModuleList()
        self.output_head = nn.ModuleList()
        self.concat_head = nn.ModuleList()

        # append in reversed order because tensors from each stage need to be processed backwards
        for ip in reversed(in_planes):
            # input 1x1 convolution
            self.compression_head.append(
                Conv1dWrapper(
                    in_channels=ip,
                    out_channels=compression,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    activation=activation,
                    normalization=normalization,
                    preactivation=preactivation,
                )
            )

            self.output_head.append(
                ResidualBlock1D(
                    in_planes=compression,
                    out_planes=compression,
                    kernel_size=kernel_size,
                    stride=1,
                    activation=activation,
                    normalization=normalization,
                    preactivation=preactivation,
                    **kwargs
                )
            )

    def forward(self, x):
        assert len(x) == len(self.compression_head)

        x.reverse()

        # get sizes for upsampling
        w = [i.shape[-1] for i in x]

        trace = []
        for idx, (t, compression_conv, out_conv) in enumerate(zip(
                x,
                self.compression_head,
                self.output_head,
        )):

            # input convolution
            t = compression_conv(t)

            if idx != 0:
                t = t + yt

            if idx < len(x) - 1:
                yt = F.interpolate(t, w[idx + 1], mode='linear')

            # output convolution
            t = out_conv(t)

        return trace, t


class DilatedReceptiveLayer(nn.Module):
    """
    """
    expansion = 4

    def __init__(
            self,
            in_channels: int,
            dilations: Union[Tuple, List],
            kernel_size: int,                        
            activation: str = 'LReLU',
            normalization: str = 'BatchN',
            preactivation: bool = True,                        
            **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): _description_
            dilations (Union[Tuple, List]): _description_
            kernel_size (int): _description_
            activation (str, optional): _description_. Defaults to 'LReLU'.
            normalization (str, optional): _description_. Defaults to 'BatchN'.
            preactivation (bool, optional): _description_. Defaults to True.
        """
        super().__init__()

        compressed_channels = in_channels // self.expansion

        # compression convolutions with constant kernel
        self.compression_gate = Conv1dWrapper(
            in_channels=in_channels,
            out_channels=compressed_channels,
            kernel_size=1,            
            activation=activation,
            normalization=normalization,
            preactivation=preactivation,
            bias=False,
            )

        # dilated convolutions for increased receptive field
        self.dilated_gates = nn.ModuleList()
        for d in dilations:            
            self.dilated_gates.append(
                Conv1dWrapper(
                    in_channels=compressed_channels,
                    out_channels=compressed_channels,
                    kernel_size=kernel_size,                    
                    dilation=d,
                    activation=activation,
                    normalization=normalization,
                    preactivation=preactivation,
                    bias=False,
                )
            )        
        
        # output convolutions with constant kernel
        out_channels = len(dilations) * compressed_channels

        self.output_gate = Conv1dWrapper(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=1,
            activation=activation,
            normalization=normalization,
            preactivation=preactivation,
            )

    def forward(self, x):        
        x = self.compression_gate(x)

        trace = []
        for conv in self.dilated_gates:
            trace.append(conv(x))

        # Concat output
        y = torch.cat(trace, dim=1)        

        # Output gate
        y = self.output_gate(y)

        return y


class ClassificationLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        """_summary_

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
        """
        super().__init__()
        
        self.pooling = nn.Sequential(            
            nn.AdaptiveAvgPool1d(1)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(in_channels, 64, bias=True),            
            nn.Dropout(0.15),
            nn.Linear(64, out_channels, bias=False),
        )

    def forward(self, x, mask: torch.Tensor = None):

        if mask is not None:
            x = x * mask

        # forward net
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)

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
    Gated Attention Network.

    Args:
        input_size: Dimension of the input.
        middle_size: Dimension of the hidden layer.
        output_size: Dimension of the output.
    """
    def __init__(self, input_size=256, middle_size=128, output_size=1):
        super(AttentionNetGated, self).__init__()
        self.attention_u = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.Tanh()
        )
        self.attention_v = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.Sigmoid()
        )
        self.attention_z = nn.Linear(middle_size, output_size)

    def forward(self, x):
        u = self.attention_u(x)
        v = self.attention_v(x)
        uv = u * v 
        z = self.attention_z(uv)
        return z


class AttentionPooling(nn.Module):
    """
    Attention-based Multi-Instance Learning network.

    Args:
        input_size (int): Dimension of the input.
        hidden_size (int): Dimension of the hidden layer in the feature extraction network.
        attention_hidden_size (int): Dimension of the hidden layer in the attention network.
        output_size (int): Dimension of the output.
        dropout (bool, optional): Flag to use dropout layer. Defaults to False.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.25.
    """

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 attention_hidden_size, 
                 output_size, 
                 dropout=False, 
                 dropout_prob=0.25):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout else nn.Identity())
            
        self.attention_pool = AttentionNetGated(hidden_size, attention_hidden_size, 1) 
        
        self.predictor = nn.Linear(hidden_size, output_size)

    def forward(self, h):
        """
        Forward pass of the AMIL network.

        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, num_instances, input_size).

        Returns:
            tuple: (risk, attention_weights_softmax)
        """

        h = self.projection(h)
        
        #batch_size, num_instances, _ = h.shape  # Get batch size and number of instances

        attention_weights = self.attention_pool(h)  # (batch_size, num_instances, hidden_size)
        attention_weights = torch.transpose(attention_weights, -2, -1)  # (batch_size, hidden_size, num_instances)
        attention_weights_softmax = F.softmax(attention_weights, dim=-1) 

        # Batched matrix multiplication
        avg_instances = torch.matmul(attention_weights_softmax, h)  # (batch_size, hidden_size, input_size)
        # (attention_weights_softmax * h[:, 0]).sum()
        risk = self.predictor(avg_instances)  # (batch_size, output_size)

        return risk, attention_weights_softmax



if __name__ == "__main__":
    """ conv_layer = Conv1dWrapper(
        in_channels=16,
        out_channels=32,
        kernel_size=(1,3),
        dilation=1,
        normalization="BatchN2D",
        activation="ReLU",
        preactivation=True,
    )
    x = torch.randn(2,1,4000,2035)  # Batch size 8, 16 channels, sequence length 50
    output = conv_layer(x) """
    
    
    x = torch.randn(10,400,256)
    attn_block = AttentionPooling(256,64,32,1)
    output = attn_block(x)
    print(output[0].shape)
    