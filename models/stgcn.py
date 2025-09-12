# models/stgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.tgcn import ConvTemporalGraphical
from models.utils.graph import Graph

# source: https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
class STGCNModel(nn.Module):
    """Lightweight ST-GCN aligned with (N, T, 34) input."""

    def __init__(self, forecast_window, output_class_size, layout='openpose'):
        super().__init__()

        # Graph
        self.graph = Graph(layout=layout)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Kernel size
        kernel_size = (1, A.size(0))  # temporal kernel = 1, spatial kernel = K
        in_channels = 2   # x,y per joint
        num_joints = self.graph.num_node

        # GCN layers
        self.gcn1 = st_gcn(in_channels, 32, kernel_size, residual=False)
        self.gcn2 = st_gcn(32, 64, kernel_size)
        self.gcn3 = st_gcn(64, 64, kernel_size)
        self.gcn4 = st_gcn(64, 128, kernel_size)

        # Heads
        self.forecast_head = nn.Linear(128, num_joints * in_channels)  # (34)
        self.classifier = nn.Linear(128, output_class_size)

        self.forecast_window = forecast_window
        self.num_joints = num_joints
        self.in_channels = in_channels

    def forward(self, x):
        # Input x: (N, T, 34)
        N, T, D = x.shape
        V, C = self.num_joints, self.in_channels
        assert D == V * C, f"Expected {V*C}, got {D}"

        # Reshape to (N, C, T, V)
        x = x.view(N, T, V, C).permute(0, 3, 1, 2).contiguous()

        # ST-GCN forward
        for gcn in [self.gcn1, self.gcn2, self.gcn3, self.gcn4]:
            x, _ = gcn(x, self.A)

        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])   # (N, 128, 1, 1)
        x = x.view(N, -1)                   # (N, 128)

        # Forecast output (N, T, 34)
        forecast_out = self.forecast_head(x)          # (N, 34)
        forecast_out = forecast_out.unsqueeze(1).repeat(1, T, 1)

        # Classification output (N, num_classes)
        class_out = self.classifier(x)

        return forecast_out, class_out


class st_gcn(nn.Module):
    """Single ST-GCN block (lightweight)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
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
