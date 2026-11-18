"""DeepLabV3+, Attention U-Net and the two-stage cascade that combines them.

The cascade feeds the thresholded coarse mask (no gradient) to the refinement network as a fourth input
channel and gates the refinement encoder features (ResNet-50 layer3) with per-channel weights computed from
the coarse decoder features: ``f <- (1 + w) * f``, ``w = sigmoid(W2 relu(W1 gap(f_coarse)))``.
Every model returns a dict with ``logits``; the cascade also returns ``coarse_logits`` and the gate ``w``.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common import AttentionGate, ConvBlock, expand_first_conv


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with a ResNet-50 encoder, output stride 16."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None,
                                     in_channels=3, classes=1, encoder_output_stride=16)
        self.out_channels = 256

    def forward_features(self, x: torch.Tensor):
        fc = self.net.decoder(self.net.encoder(x))           # (B, 256, H/4, W/4)
        return self.net.segmentation_head(fc), fc

    def forward(self, x: torch.Tensor) -> dict:
        return {"logits": self.forward_features(x)[0]}

    def encoder_parameters(self):
        return self.net.encoder.parameters()


class AttentionUNet(nn.Module):
    """Attention U-Net with a torchvision ResNet-50 encoder and three attention gates."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        r = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_channels != 3:
            r.conv1 = expand_first_conv(r.conv1, in_channels, extra_init="kaiming")
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.ag1 = AttentionGate(1024, 2048, 512)
        self.dec1 = ConvBlock(2048 + 1024, 512)
        self.ag2 = AttentionGate(512, 512, 256)
        self.dec2 = ConvBlock(512 + 512, 256)
        self.ag3 = AttentionGate(256, 256, 128)
        self.dec3 = ConvBlock(256 + 256, 128)
        self.head = nn.Conv2d(128, 1, 1)
        self.fe_channels = 1024

    def encoder_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def forward(self, x: torch.Tensor, gate=None) -> dict:
        H, W = x.shape[-2:]
        s1 = self.layer1(self.pool(self.stem(x)))   # 256  @ H/4
        s2 = self.layer2(s1)                        # 512  @ H/8
        s3 = self.layer3(s2)                        # 1024 @ H/16
        s4 = self.layer4(s3)                        # 2048 @ H/32
        if gate is not None:
            s3 = gate(s3)
        up = F.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat([up, self.ag1(s3, s4)[0]], 1))
        up = F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([up, self.ag2(s2, up)[0]], 1))
        up = F.interpolate(d, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat([up, self.ag3(s1, up)[0]], 1))
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return {"logits": self.head(d)}


class ChannelGate(nn.Module):
    """Per-channel weights for the refinement features, computed from the coarse decoder features."""

    def __init__(self, c_coarse: int = 256, c_refine: int = 1024, reduction: int = 16):
        super().__init__()
        self.W1 = nn.Linear(c_coarse, c_coarse // reduction, bias=False)
        self.W2 = nn.Linear(c_coarse // reduction, c_refine, bias=True)
        nn.init.kaiming_normal_(self.W1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_normal_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)

    def forward(self, fe: torch.Tensor, fc: torch.Tensor):
        w = torch.sigmoid(self.W2(F.relu(self.W1(F.adaptive_avg_pool2d(fc, 1).flatten(1)))))
        return fe * (1.0 + w[:, :, None, None]), w


class Cascade(nn.Module):
    """DeepLabV3+ (coarse) -> Attention U-Net (refinement), trained jointly."""

    def __init__(self, pretrained: bool = True, coarse_threshold: float = 0.5, gate_reduction: int = 16):
        super().__init__()
        self.coarse = DeepLabV3Plus(pretrained)
        self.refine = AttentionUNet(4, pretrained)
        self.fusion = ChannelGate(self.coarse.out_channels, self.refine.fe_channels, gate_reduction)
        self.tau = coarse_threshold

    def forward(self, x: torch.Tensor) -> dict:
        coarse_logits, fc = self.coarse.forward_features(x)
        mask = (torch.sigmoid(coarse_logits.float()) >= self.tau).to(x.dtype).detach()
        store = {}

        def gate(fe):
            out, store["w"] = self.fusion(fe, fc)
            return out

        out = self.refine(torch.cat([x, mask], 1), gate=gate)
        return {"logits": out["logits"], "coarse_logits": coarse_logits, "gate": store.get("w")}

    def encoder_parameters(self):
        yield from self.coarse.encoder_parameters()
        yield from self.refine.encoder_parameters()

    def fusion_parameters(self):
        return self.fusion.parameters()
