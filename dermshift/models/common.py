"""Building blocks shared by several models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """(Conv3x3 - BN - ReLU) x 2."""

    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class AttentionGate(nn.Module):
    """Additive attention gate: alpha = sigmoid(BN(psi(ReLU(Wx x + Wg g)))), output = alpha * x.
    ``g`` is bilinearly resized to the spatial size of ``x`` when they differ."""

    def __init__(self, c_skip: int, c_gate: int, c_inter: int):
        super().__init__()
        self.Wx = nn.Conv2d(c_skip, c_inter, 1, bias=False)
        self.Wg = nn.Conv2d(c_gate, c_inter, 1, bias=True)
        self.psi = nn.Conv2d(c_inter, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        gx = self.Wg(g)
        if gx.shape[-2:] != x.shape[-2:]:
            gx = F.interpolate(gx, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.bn(self.psi(F.relu(self.Wx(x) + gx))))
        return alpha * x, alpha


def expand_first_conv(conv: nn.Conv2d, in_channels: int, extra_init: str = "kaiming") -> nn.Conv2d:
    """Return a copy of ``conv`` accepting ``in_channels`` (>= original); the first
    channels keep the (pretrained) weights, the extra ones are initialised with ``extra_init``
    ('kaiming', or 'mean' = mean of the original filters)."""
    c_old = conv.in_channels
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :c_old] = conv.weight
        if extra_init == "mean":
            new.weight[:, c_old:] = conv.weight.mean(1, keepdim=True)
        else:
            nn.init.kaiming_normal_(new.weight[:, c_old:], mode="fan_out", nonlinearity="relu")
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new
