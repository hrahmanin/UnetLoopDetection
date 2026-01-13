
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.insert(3, nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth

        # encoder
        enc: List[nn.Module] = []
        ch = in_channels
        feat = base_channels
        for _ in range(depth):
            enc.append(DoubleConv(ch, feat, dropout=dropout))
            ch = feat
            feat *= 2
        self.enc = nn.ModuleList(enc)
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DoubleConv(ch, feat, dropout=dropout)

        # decoder
        dec_up: List[nn.Module] = []
        dec_conv: List[nn.Module] = []
        for _ in range(depth):
            dec_up.append(nn.ConvTranspose2d(feat, feat // 2, kernel_size=2, stride=2))
            dec_conv.append(DoubleConv(feat, feat // 2, dropout=dropout))
            feat //= 2
        self.dec_up = nn.ModuleList(dec_up)
        self.dec_conv = nn.ModuleList(dec_conv)

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = list(reversed(skips))
        for up, conv, skip in zip(self.dec_up, self.dec_conv, skips):
            x = up(x)
            # pad if needed (odd sizes)
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.shape[-2] - x.shape[-2]
                diffX = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.head(x)
