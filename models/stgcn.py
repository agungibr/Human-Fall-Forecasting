import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Graph:
    """
    The Graph to model the skeletons of human body/pose.
    """
    def __init__(self, layout='coco', strategy='spatial'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [
            (10, 8), (8, 6), (9, 7), (7, 5), # Arms
            (4, 2), (2, 0), (3, 1), (1, 0), # Legs
            (12, 11), (11, 5), (12, 6), (11, 13), (13, 15), (12, 14), (14, 16),
            (6, 5) # Torso
        ]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.distance_matrix = self._get_distance_matrix()
        self.A = self.get_adjacency_matrix(strategy)

    def _get_distance_matrix(self):
        dist = np.full((self.num_node, self.num_node), np.inf)
        for i in range(self.num_node):
            dist[i, i] = 0
        for i, j in self.neighbor:
            dist[i, j] = 1
        for k in range(self.num_node):
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    def get_adjacency_matrix(self, strategy):
        valid_hop = range(0, self.num_node - 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.get_hop_distance(hop)] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            return A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_far = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.get_distance(i, j) == hop:
                            if self.get_distance(j, 0) > self.get_distance(i, 0):
                                a_close[j, i] = normalize_adjacency[j, i]
                            elif self.get_distance(j, 0) < self.get_distance(i, 0):
                                a_far[j, i] = normalize_adjacency[j, i]
                            else:
                                a_root[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_far)
            return np.stack(A)
        else:
            raise ValueError("Strategy does not exist")

    def get_hop_distance(self, hop):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.neighbor:
            A[i, j] = 1
            A[j, i] = 1
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        return (arrive_mat[hop] ^ arrive_mat[hop-1]) if hop > 0 else arrive_mat[hop]

    def get_distance(self, i, j):
        return self.distance_matrix[i, j]

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        return np.dot(A, Dn)

class ST_GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_spatial_partitions, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = nn.Conv2d(in_channels * num_spatial_partitions, out_channels, kernel_size=1)
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
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
        N, C, T, V = x.size()
        x = torch.einsum('nctv,kvw->nkctw', (x, A))
        x = x.permute(0, 1, 2, 3, 4).contiguous().view(N, -1, T, V)
        x = self.gcn(x)
        x = self.tcn(x) + res
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, forecast_window, edge_importance_weighting=True):
        super().__init__()
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            ST_GCN_block(in_channels, 64, kernel_size, spatial_kernel_size, stride=1, residual=False),
            ST_GCN_block(64, 64, kernel_size, spatial_kernel_size, stride=1),
            ST_GCN_block(64, 128, kernel_size, spatial_kernel_size, stride=2),
            ST_GCN_block(128, 128, kernel_size, spatial_kernel_size, stride=1),
            ST_GCN_block(128, 256, kernel_size, spatial_kernel_size, stride=2),
            ST_GCN_block(256, 256, kernel_size, spatial_kernel_size, stride=1),
        ))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        self.fc_shared = nn.Linear(256, 512)
        self.fc_forecast = nn.Linear(512, forecast_window * self.graph.num_node * in_channels)
        self.forecast_window = forecast_window
        self.fc_class = nn.Linear(512, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)
        x = F.relu(self.fc_shared(x))
        forecast_out = self.fc_forecast(x).view(N, self.forecast_window, -1)
        class_out = self.fc_class(x)
        return forecast_out, class_out