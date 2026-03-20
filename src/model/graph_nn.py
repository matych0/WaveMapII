import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import EdgeConv
from torch_geometric.nn import GlobalAttention


class EdgeConvGNN(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        # EdgeConv layers
        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2*in_channels, 16),
                nn.ReLU(),
                nn.Linear(16, 16)
            ), 
            aggr='add'
        )

        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2*16, 32),
                nn.ReLU(),
                nn.Linear(32, 32)
            ),
            aggr='add'
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