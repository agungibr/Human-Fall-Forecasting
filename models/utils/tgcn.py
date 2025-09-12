import torch
import torch.nn as nn

# Source: https://github.com/yysijie/st-gcn/blob/master/net/utils/tgcn.py
class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            bias=bias
        )
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x, A):
        # x: (N, C, T, V), A: (K, V, V)
        x = self.conv(x)   # (N, out_channels*K, T, V)
        N, KC, T, V = x.size()
        x = x.view(N, self.kernel_size, self.out_channels, T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # graph conv
        return x.contiguous(), A
