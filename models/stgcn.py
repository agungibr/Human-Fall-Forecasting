import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.graph import Graph 

class STGCNModel(nn.Module):
    r"""Spatial temporal graph convolutional networks adapted for dual-task learning.

    Args:
        in_channels (int): Number of channels in the input data (e.g., 2 for (x, y))
        graph_args (dict): The arguments for building the graph
        forecast_window (int): The number of future frames to predict
        output_class_size (int): Number of classes for the classification task
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    """

    def __init__(self, in_channels, forecast_window, output_class_size, 
                 graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph 
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks 
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node)
        
        # The main ST-GCN layers, now scaled up to produce 1024 features
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 128, kernel_size, 1, residual=False, **kwargs), 
            st_gcn(128, 128, kernel_size, 1, **kwargs),                   
            st_gcn(128, 128, kernel_size, 1, **kwargs),                   
            st_gcn(128, 256, kernel_size, 1, **kwargs),                   
            st_gcn(256, 256, kernel_size, 2, **kwargs),                   
            st_gcn(256, 512, kernel_size, 1, **kwargs),                   
            st_gcn(512, 512, kernel_size, 1, **kwargs),                   
            st_gcn(512, 1024, kernel_size, 2, **kwargs),                  
            st_gcn(1024, 1024, kernel_size, 1, **kwargs),                 
            st_gcn(1024, 1024, kernel_size, 1, **kwargs),                 
        ))

        # initialize parameters for edge importance weighting 
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # This now reflects the output of the last ST-GCN layer
        final_feature_dim = 1024 # Changed 256 -> 1024
        
        # 1. Forecasting Head (automatically adapts to new final_feature_dim)
        forecast_output_dim = forecast_window * self.graph.num_node * 2
        self.fc_forecast = nn.Linear(final_feature_dim, forecast_output_dim)
        
        # 2. Classification Head (automatically adapts to new final_feature_dim)
        self.fc_class = nn.Linear(final_feature_dim, output_class_size)
        # --------------------------------------------

    def forward(self, x):
        # data normalization
        # Input shape: (N, C, T, V, M) 
        # N: batch, C: channels, T: frames, V: joints, M: people
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous() # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous() # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        # forward GCN
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # --- MODIFICATION FOR DUAL-TASK OUTPUT ---
        # Global pooling to get a summary feature vector
        x = F.avg_pool2d(x, x.size()[2:]) # (N*M, 256, 1, 1)
        x = x.view(N, M, -1).mean(dim=1)  # (N, 256)

        # 1. Forecasting Output
        forecast = self.fc_forecast(x)
        # Reshape to (N, forecast_window, num_joints * coords) e.g. (N, 15, 34)
        forecast = forecast.view(N, -1, self.graph.num_node * 2)

        # 2. Classification Output
        classify = self.fc_class(x) # (N, num_classes)
        # -------------------------------------

        return forecast, classify

# --- Helper classes from the paper's code, slightly refactored ---
class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1,
                 t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A