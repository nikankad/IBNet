import torch.nn as nn
import torch

class IBConv(nn.Module):
    """Inverted Bottleneck 1D Time-Channel Separable Convolution Module"""
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, expand=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        mid_channel = in_channel * expand

        self.net = nn.Sequential(
            # 1) Pointwise expand: C → C*t
            nn.Conv1d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(),

            # 2) Depthwise conv in expanded space
            nn.Conv1d(mid_channel, mid_channel, kernel_size, stride, padding,
                      groups=mid_channel, bias=False),
            nn.BatchNorm1d(mid_channel),
            nn.ReLU(),

            # 3) Pointwise compress: C*t → C_out (no ReLU — linear bottleneck)
            nn.Conv1d(mid_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

        self.use_residual = (in_channel == out_channel and stride == 1)

    def forward(self, x):
        out = self.net(x)
        if self.use_residual:
            out = out + x
        return out

class IBBlock(nn.Module):
    """Block of R inverted bottleneck modules with block-level residual"""
    def __init__(self, in_channel, out_channel, kernel_size, R=3, expand=2):
        super().__init__()

        # First module handles channel change (in_channel → out_channel)
        self.layer1 = IBConv(in_channel, out_channel, kernel_size, expand=expand)
        # Remaining R-1 modules: same channels, each has internal per-module residual
        self.layers = nn.ModuleList([
            IBConv(out_channel, out_channel, kernel_size, expand=expand) for _ in range(R-1)
        ])

        # Block-level residual (same as QuartzNet)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        out = self.layer1(x)
        for layer in self.layers:
            out = layer(out)
        out = out + self.residual(x)
        return torch.relu(out)

class IBNet(nn.Module):
    def __init__(self, n_mels=64, n_classes=29, R=3, expand=2, C=192):
        super().__init__()
        C2 = C * 2
        C3 = C2 * 2
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, C, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(),
            IBBlock(C, C, kernel_size=33, R=R, expand=expand),
            IBBlock(C, C, kernel_size=39, R=R, expand=expand),
            IBBlock(C, C2, kernel_size=51, R=R, expand=expand),
            IBBlock(C2, C2, kernel_size=63, R=R, expand=expand),
            IBBlock(C2, C2, kernel_size=75, R=R, expand=expand),
            IBConv(C2, C2, kernel_size=87, expand=expand),
            nn.Conv1d(C2, C3, kernel_size=1, bias=False),
            nn.BatchNorm1d(C3),
            nn.ReLU(),
            nn.Conv1d(C3, n_classes, dilation=2, kernel_size=1),
        )

    def forward(self, x):
        x = x.squeeze(1)
        return self.net(x)
