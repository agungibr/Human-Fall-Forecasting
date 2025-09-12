# models/stgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.tgcn import ConvTemporalGraphical
from models.utils.graph import Graph


class STGCNModel(nn.Module):
    """ ST-GCN model (configurable, heavy version for max performance). """

    def __init__(self, 
                 in_channels=2,
                 num_class=2,
                 graph_args={'layout': 'coco', 'strategy': 'uniform'},
                 edge_importance_weighting=True,
                 hidden_dims=[1024, 2048, 2048, 4096],   # 🔥 heavy setup
                 dropout=0.25):
        super().__init__()

        # ----- Graph -----
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9   # bigger temporal receptive field
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # input normalization
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # ----- ST-GCN Layers -----
        self.st_gcn_networks = nn.ModuleList()
        input_dim = in_channels
        for i, hdim in enumerate(hidden_dims):
            self.st_gcn_networks.append(
                st_gcn(input_dim, hdim, kernel_size, stride=2 if i % 2 == 1 else 1,
                       dropout=dropout, residual=(i != 0))
            )
            input_dim = hdim

        # ----- Edge importance weighting -----
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # ----- Prediction heads -----
        final_dim = hidden_dims[-1]

        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Conv1d(final_dim, final_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(final_dim // 2, 34, kernel_size=1)   # 17 joints * 2 coords
        )

        # Classification head
        self.classifier = nn.Conv2d(final_dim, num_class, kernel_size=1)

    def forward(self, x):
        # x: (N, T, 34)
        N, T, D = x.shape
        V = 17
        C = 2
        M = 1

        # reshape to (N, C, T, V, M)
        x = x.view(N, T, V, C)          # (N, T, V, 2)
        x = x.permute(0, 3, 1, 2)       # (N, C, T, V)
        x = x.unsqueeze(-1)             # (N, C, T, V, M)

        # ---- normalization ----
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # ---- GCN layers ----
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # ---- Global pooling ----
        x = F.avg_pool2d(x, x.size()[2:])   # (N*M, final_dim, 1, 1)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)   # (N, final_dim, 1, 1)

        # ---- Classification ----
        class_out = self.classifier(x)
        class_out = class_out.view(class_out.size(0), -1)

        # ---- Forecasting ----
        feat = x.view(x.size(0), -1)        # (N, final_dim)
        forecast_out = feat.unsqueeze(-1).repeat(1, 1, T)  # repeat over T
        forecast_out = self.forecast_head(forecast_out)    # (N, 34, T)
        forecast_out = forecast_out.permute(0, 2, 1)       # (N, T, 34)

        return forecast_out, class_out


class st_gcn(nn.Module):
    """ Single ST-GCN block. """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.25,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A