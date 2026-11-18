"""Single-stage baselines. Each returns {"logits": (B, 1, H, W)} and exposes ``encoder_parameters()``.

* UNet          Ronneberger et al. (2015) with batch normalisation, trained from scratch.
* ResNet50-UNet U-Net decoder on an ImageNet ResNet-50 encoder (segmentation_models_pytorch).
* TransUNet     Chen et al. (2021), R50-ViT-B/16 encoder (ImageNet-21k), cascaded upsampler, three skips.
* SwinUnet      Cao et al. (2022), Swin-T encoder (ImageNet-1k), symmetric decoder with patch expanding;
                decoder stages initialised from the encoder stages of the same width.

DeepLabV3+ and Attention U-Net are in ``cascade.py``.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock


# ----------------------------------------------------------------------------- U-Net
class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([ConvBlock(in_channels, c[0])] + [ConvBlock(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([ConvBlock(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return {"logits": self.head(x)}

    def encoder_parameters(self):
        return iter(())  # no pretrained part


# ----------------------------------------------------------------------------- smp wrappers
class SmpModel(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return {"logits": self.net(x)}

    def encoder_parameters(self):
        return self.net.encoder.parameters()


def resnet50_unet(pretrained: bool = True) -> SmpModel:
    return SmpModel(smp.Unet("resnet50", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1))


# ----------------------------------------------------------------------------- TransUNet
class _UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        name = "vit_base_r50_s16_224.orig_in21k"
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        bb = self.vit.patch_embed.backbone
        self._feats = {}
        for key, mod in (("x2", bb.stem.norm), ("x4", bb.stages[0]), ("x8", bb.stages[1])):
            mod.register_forward_hook(lambda m, i, o, key=key: self._feats.__setitem__(key, o))
        self.conv_more = nn.Sequential(nn.Conv2d(768, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.dec = nn.ModuleList([_UpBlock(512, 512, 256), _UpBlock(256, 256, 128),
                                  _UpBlock(128, 64, 64), _UpBlock(64, 0, 16)])
        self.head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        self._feats.clear()
        tok = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens:]   # (B, N, 768)
        h, w = H // 16, W // 16
        y = self.conv_more(tok.transpose(1, 2).reshape(x.shape[0], 768, h, w))
        skips = [self._feats["x8"], self._feats["x4"], self._feats["x2"], None]
        for blk, s in zip(self.dec, skips):
            y = blk(y, s)
        return {"logits": self.head(y)}

    def encoder_parameters(self):
        return self.vit.parameters()


# ----------------------------------------------------------------------------- Swin-Unet
class PatchExpand(nn.Module):
    """(B,H,W,C) -> (B,2H,2W,C/scale_div) via linear expansion + pixel shuffle (Swin-Unet)."""

    def __init__(self, dim: int, factor: int = 2, out_dim: int | None = None):
        super().__init__()
        self.factor = factor
        self.out_dim = out_dim or dim // factor
        self.expand = nn.Linear(dim, self.out_dim * factor * factor, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        B, H, W, _ = x.shape
        f = self.factor
        x = self.expand(x).view(B, H, W, f, f, self.out_dim).permute(0, 1, 3, 2, 4, 5)
        return self.norm(x.reshape(B, H * f, W * f, self.out_dim))


class SwinUnet(nn.Module):
    def __init__(self, img_size=(192, 256), pretrained: bool = True, window_size: int = 7,
                 dec_depths=(2, 2, 2)):
        super().__init__()
        from timm.models.swin_transformer import SwinTransformerStage

        self.encoder = timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=pretrained,
                                         features_only=True, img_size=img_size)
        dims = [96, 192, 384, 768]
        heads = [3, 6, 12, 24]
        H4, W4 = img_size[0] // 4, img_size[1] // 4
        res = [(H4 // 2 ** i, W4 // 2 ** i) for i in range(4)]
        self.ups, self.concat_back, self.stages = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for j, i in enumerate((2, 1, 0)):              # decoder at stage-3, stage-2, stage-1 resolution
            self.ups.append(PatchExpand(dims[i + 1], 2, dims[i]))
            self.concat_back.append(nn.Linear(2 * dims[i], dims[i]))
            self.stages.append(SwinTransformerStage(dim=dims[i], out_dim=dims[i], input_resolution=res[i],
                                                    depth=dec_depths[j], downsample=False, num_heads=heads[i],
                                                    window_size=window_size))
        self.norm_up = nn.LayerNorm(dims[0])
        self.final_up = PatchExpand(dims[0], 4, dims[0])
        self.head = nn.Conv2d(dims[0], 1, 1, bias=False)
        if pretrained:
            self._mirror_init()

    def _mirror_init(self):
        enc_stages = self.encoder.layers_0, self.encoder.layers_1, self.encoder.layers_2
        for dec, i in zip(self.stages, (2, 1, 0)):
            src = enc_stages[i].blocks
            for k, blk in enumerate(dec.blocks):
                if k < len(src):
                    blk.load_state_dict(src[k].state_dict(), strict=False)

    def forward(self, x):
        feats = self.encoder(x)                         # NHWC: /4, /8, /16, /32
        y = feats[3]
        for up, cat, stage, skip in zip(self.ups, self.concat_back, self.stages, (feats[2], feats[1], feats[0])):
            y = up(y)
            y = cat(torch.cat([y, skip], -1))
            y = stage(y)
        y = self.final_up(self.norm_up(y)).permute(0, 3, 1, 2)
        return {"logits": self.head(y)}

    def encoder_parameters(self):
        return self.encoder.parameters()
