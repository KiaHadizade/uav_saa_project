'''
Lightweight U-Net; outputs logits for n_classes
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, n_classes=5, base=32):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)

        self.b = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.u3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.u2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.u1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self,x):
        d1 = self.d1(x) # [B,base,H,W]
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        b  = self.b(self.p3(d3))
        x  = self.up3(b); x = torch.cat([x, d3], dim=1); x = self.u3(x)
        x  = self.up2(x); x = torch.cat([x, d2], dim=1); x = self.u2(x)
        x  = self.up1(x); x = torch.cat([x, d1], dim=1); x = self.u1(x)
        return self.out(x)
