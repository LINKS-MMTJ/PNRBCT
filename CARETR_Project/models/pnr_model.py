
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionAwareNoiseRemover(nn.Module):
    def __init__(self, grid_rows=6, grid_cols=8, in_ch=2, conv_kernels=8, num_classes=2):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_classes = num_classes
       
        self.conv = nn.Conv2d(in_ch, conv_kernels, kernel_size=2)
        self.dropout = nn.Dropout(0.3)   
        self.fc = nn.Linear(conv_kernels * grid_rows * grid_cols,
                            grid_rows * grid_cols * num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, G):
        """G: (B, 2, 6, 8)  ->  logits k: (B, 48, 2)"""
        f = F.relu(self.conv(F.pad(G, (0, 1, 0, 1))))        
        k = self.fc(self.dropout(f.flatten(1)))              
        return k.view(-1, self.grid_rows * self.grid_cols, self.num_classes)

    @torch.no_grad()
    def clean(self, G, return_mask=False, prob_limit=0.85, keep_channels=False):
        
        k = self.forward(G.unsqueeze(0)).squeeze(0)         
        is_noise = k.argmax(dim=-1).view(self.grid_rows, self.grid_cols) == 1  

        candidate = (G[0] == 1)                              
        if prob_limit is not None:                          
            candidate = candidate & (G[1] < prob_limit)
        remove = candidate & is_noise

        if keep_channels:
            Gc = G.clone()
            Gc[:, remove] = 0.0
            return (Gc, remove) if return_mask else Gc

       
        grid = G[0].clone()
        grid[remove] = 0.0
        return (grid, remove) if return_mask else grid

    @torch.no_grad()
    def noise_prob(self, G):
       
        k = self.forward(G.unsqueeze(0)).squeeze(0)
        return F.softmax(k, dim=-1)[:, 1]
