import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import EdgeConv, GlobalAttention, knn_graph
from torch_geometric.nn import global_max_pool, global_mean_pool

class EdgeConvGNN(nn.Module):

    def __init__(
        self, 
        in_channels,
        **kwargs
    ):

        super().__init__()

        # EdgeConv layers
        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2*in_channels, 16),
                nn.ReLU(),
                nn.Linear(16, 16)
            ), 
            aggr='max'
        )

        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2*16, 32),
                nn.ReLU(),
                nn.Linear(32, 32)
            ),
            aggr='max'
        )

        """ self.conv3 = EdgeConv(
            nn.Sequential(
                nn.Linear(2*128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
        ) """

        # attention pooling
        gate_nn = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.pool = GlobalAttention(gate_nn)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )


    def forward(self, data, **kwargs):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))

        # graph embedding
        x = self.pool(x, batch)

        out = self.classifier(x)

        return out.squeeze(-1)


# -------------------------
# Utility: MLP builder
# -------------------------

def build_mlp(in_channels, layer_dims, activation="leakyrelu", use_bn=True):
    layers = []

    for out_channels in layer_dims:
        layers.append(nn.Linear(in_channels, out_channels))

        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))

        if activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Unknown activation: {activation}")

        in_channels = out_channels

    return nn.Sequential(*layers)


# -------------------------
# More flexible EdgeConvGNN with configurable architecture
# -------------------------


class DynamicEdgeConvGNN(nn.Module):

    def __init__(
        self,
        in_channels,
        conv_channels=[64, 64, 128, 256],
        embedding_dims=[1024],   # now configurable
        k=20,
        aggr="max",
        activation="leakyrelu",
        use_bn=True,
        graph_recompute=True,
        pooling="attention",
        classifier_dims=[512, 256],
        dropout=0.5
    ):
        super().__init__()

        self.k = k
        self.pooling_type = pooling
        self.graph_recompute = graph_recompute

        # -------------------------
        # EdgeConv layers
        # -------------------------
        self.convs = nn.ModuleList()

        prev_channels = in_channels
        for out_channels in conv_channels:

            mlp = build_mlp(
                2 * prev_channels,
                [out_channels, out_channels],
                activation,
                use_bn
            )

            self.convs.append(EdgeConv(mlp, aggr=aggr))
            prev_channels = out_channels

        # -------------------------
        # Feature projection (NO HARDCODE)
        # -------------------------
        total_conv_dim = sum(conv_channels)

        self.feature_proj = build_mlp(
            total_conv_dim,
            embedding_dims,
            activation,
            use_bn
        )

        final_embedding_dim = embedding_dims[-1]

        # -------------------------
        # Pooling
        # -------------------------
        if pooling == "attention":
            gate_nn = build_mlp(
                final_embedding_dim,
                [final_embedding_dim // 2, 1],
                activation,
                use_bn
            )
            self.pool = GlobalAttention(gate_nn)

        elif pooling == "max":
            self.pool = global_max_pool

        elif pooling == "mean":
            self.pool = global_mean_pool

        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # -------------------------
        # Classifier
        # -------------------------
        cls_layers = []
        in_dim = final_embedding_dim

        for dim in classifier_dims:
            cls_layers.append(nn.Linear(in_dim, dim))

            if use_bn:
                cls_layers.append(nn.BatchNorm1d(dim))

            if activation == "leakyrelu":
                cls_layers.append(nn.LeakyReLU())
            else:
                cls_layers.append(nn.ReLU())

            cls_layers.append(nn.Dropout(dropout))
            in_dim = dim

        cls_layers.append(nn.Linear(in_dim, 1))

        self.classifier = nn.Sequential(*cls_layers)

    def forward(self, data):

        x, batch, edge_index = data.x, data.batch, data.edge_index

        features = []


        for conv in self.convs:
            if self.graph_recompute:
                edge_index = knn_graph(x, k=self.k, batch=batch)
            x = conv(x, edge_index)
            features.append(x)

        x = torch.cat(features, dim=1)

        x = self.feature_proj(x)

        if self.pooling_type == "attention":
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)

        out = self.classifier(x)

        return out.squeeze(-1)