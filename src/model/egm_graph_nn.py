from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.model.graph_nn import DynamicEdgeConvGNN
from src.model.pyramid_resnet import LocalActivationResNet


class ResNetEdgeConvGNN(nn.Module):
    """
    Encodes each node's EGM with a LocalActivationResNet and feeds the embeddings
    (optionally concatenated with handcrafted features and coordinates) into DynamicEdgeConvGNN.
    """

    # column layout of data.x built by GraphFeatureDataset
    HANDCRAFTED_COLUMNS = slice(0, 5)  # Vpp, dvdt_max, delta_LAT, egm_duration, deflections
    COORD_COLUMNS = slice(5, 8)        # standardized x, y, z

    def __init__(
        self,
        resnet: Dict,
        gnn: Dict,
        proj_dim: int = None,
        use_handcrafted: bool = True,
        use_coords: bool = True,
    ):
        """
        Args:
            resnet: Parameters for the LocalActivationResNet.
            gnn: Parameters for the DynamicEdgeConvGNN (in_channels is derived here).
            proj_dim: Size of the linear projection of the EGM embedding, None keeps the ResNet output size.
            use_handcrafted: Concatenate handcrafted EGM features to the node embedding.
            use_coords: Concatenate standardized coordinates to the node embedding.
        """
        super().__init__()

        self.dim = resnet["dim"]
        self.use_handcrafted = use_handcrafted
        self.use_coords = use_coords

        self.resnet = LocalActivationResNet(**resnet)

        embedding_dim = resnet["features"][-1]
        if proj_dim:
            self.projection = nn.Linear(embedding_dim, proj_dim)
            embedding_dim = proj_dim
        else:
            self.projection = nn.Identity()

        in_channels = embedding_dim
        if use_handcrafted:
            in_channels += self.HANDCRAFTED_COLUMNS.stop - self.HANDCRAFTED_COLUMNS.start
        if use_coords:
            in_channels += self.COORD_COLUMNS.stop - self.COORD_COLUMNS.start

        self.gnn = DynamicEdgeConvGNN(in_channels=in_channels, **gnn)

    def encode_traces(self, traces: torch.Tensor) -> torch.Tensor:
        """ [N_total, S] -> [N_total, C], every EGM is encoded independently."""
        if self.dim == 1:
            return self.resnet(traces.unsqueeze(1))  # [N_total, 1, S] -> [N_total, C]

        x = self.resnet(traces.view(1, 1, *traces.shape))  # [1, 1, N_total, S] -> [1, C, N_total]
        return x.squeeze(0).transpose(0, 1)

    def forward(self, data):
        x = self.projection(self.encode_traces(data.traces))

        node_features = [x]
        if self.use_handcrafted:
            node_features.append(data.x[:, self.HANDCRAFTED_COLUMNS])
        if self.use_coords:
            node_features.append(data.x[:, self.COORD_COLUMNS])

        # new Data object so the input batch is not modified
        graph = Data(
            x=torch.cat(node_features, dim=1),
            edge_index=data.edge_index,
            batch=data.batch,
        )

        return self.gnn(graph)
