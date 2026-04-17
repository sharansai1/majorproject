"""
Hybrid GAT U-Net Model Architecture
Pancreatic Tumor Segmentation Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.W(x)
        a1 = self.a.weight[0, :h.size(-1)].view(-1, 1)
        a2 = self.a.weight[0, h.size(-1):].view(-1, 1)
        score1 = torch.matmul(h, a1)
        score2 = torch.matmul(h, a2)
        e = self.leakyrelu(score1 + score2.transpose(1, 2))
        attention = F.softmax(e, dim=-1)
        h_prime = torch.bmm(attention, h)
        return F.elu(h_prime)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(skip_ch)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = self.psi(F.relu(g + s, inplace=True))
        return self.bn(skip * attn)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class HybridGATUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.b_conv = ConvBlock(256, 512, dropout=0.5)
        self.gat = GraphAttentionLayer(512, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(256, 256, 128)
        self.d1 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(128, 128, 64)
        self.d2 = ConvBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(64, 64, 32)
        self.d3 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

        self.ds1 = nn.Conv2d(256, 1, kernel_size=1)
        self.ds2 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool1(x1))
        x3 = self.e3(self.pool2(x2))

        b = self.b_conv(self.pool3(x3))
        B, C, H, W = b.size()
        nodes = b.view(B, C, -1).permute(0, 2, 1)
        gat_out = self.gat(nodes)
        gat_out = gat_out.permute(0, 2, 1).view(B, C, H, W)
        b = b + gat_out

        d1 = self.up1(b)
        d1 = torch.cat([d1, self.ag1(d1, x3)], dim=1)
        d1 = self.d1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, self.ag2(d2, x2)], dim=1)
        d2 = self.d2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, self.ag3(d3, x1)], dim=1)
        d3 = self.d3(d3)

        main_out = self.out(d3)

        if self.training:
            target_size = main_out.shape[2:]
            ds1_out = F.interpolate(self.ds1(d1), size=target_size,
                                    mode="bilinear", align_corners=False)
            ds2_out = F.interpolate(self.ds2(d2), size=target_size,
                                    mode="bilinear", align_corners=False)
            return main_out, ds1_out, ds2_out

        return main_out
