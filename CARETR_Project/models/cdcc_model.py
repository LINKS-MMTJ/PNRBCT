"""CDCC (Crack-Grid Distribution Classifier) and the position-agnostic baseline.

CDCC: Conv2d(1 -> 8, 2x2) + ReLU followed by a three-layer MLP.
PositionAgnosticMLP: the same three-layer MLP with randomly permuted inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_ROWS, GRID_COLS = 6, 8


class CDCC(nn.Module):
    """Crack-grid distribution classifier: Conv2d(1 -> 8, 2x2) + three-layer MLP."""

    def __init__(self, in_ch=1, conv_kernels=8, hidden_dim1=256, hidden_dim2=256,
                 num_classes=3, drop=0.25):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, conv_kernels, kernel_size=2)
        self.act = nn.ReLU()
        feat_dim = conv_kernels * GRID_ROWS * GRID_COLS        # 8*6*8 = 384
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_dim2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # (B, 48) -> (B, 1, 6, 8)
        if x.dim() == 2:
            x = x.view(-1, 1, GRID_ROWS, GRID_COLS)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        v = self.act(self.conv(F.pad(x, (0, 1, 0, 1))))    # (B, 8, 6, 8)
        return self.net(v.flatten(1))


class PositionAgnosticMLP(nn.Module):
    """Three-layer MLP with randomly permuted grid inputs.

    Architecture: 48 -> hidden -> hidden -> 3, identical in capacity to a
    standard three-layer MLP. The 48 input values are randomly permuted for
    each sample, so the model cannot exploit the spatial arrangement of the
    crack grid; it can only rely on permutation-invariant statistics.

    At inference a fixed permutation seed is used so that evaluation is
    reproducible.
    """

    def __init__(self, input_dim=48, hidden_dim1=256, hidden_dim2=256,
                 num_classes=3, drop=0.25, permute_seed=1234):
        super().__init__()
        self.input_dim = input_dim
        self.permute_seed = permute_seed
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_dim2, num_classes),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _permute(self, x):
        if self.training:
            r = torch.rand(x.size(0), self.input_dim, device=x.device)
        else:
            g = torch.Generator(device='cpu').manual_seed(self.permute_seed)
            r = torch.rand(x.size(0), self.input_dim, generator=g).to(x.device)
        idx = torch.argsort(r, dim=1)
        return torch.gather(x, 1, idx)

    def forward(self, x):
        return self.net(self._permute(x))
