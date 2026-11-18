"""Model factory and optimiser parameter groups."""
from __future__ import annotations

import torch.nn as nn

from . import baselines
from .cascade import AttentionUNet, Cascade, DeepLabV3Plus

MODELS = ("unet", "resnet50_unet", "deeplabv3plus", "attention_unet", "transunet", "swin_unet", "cascade")


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    name, pre = m["name"], bool(m["pretrained"])
    hw = (cfg["data"]["height"], cfg["data"]["width"])
    builders = {
        "unet": lambda: baselines.UNet(),
        "resnet50_unet": lambda: baselines.resnet50_unet(pre),
        "deeplabv3plus": lambda: DeepLabV3Plus(pre),
        "attention_unet": lambda: AttentionUNet(3, pre),
        "transunet": lambda: baselines.TransUNet(pre),
        "swin_unet": lambda: baselines.SwinUnet(hw, pre),
        "cascade": lambda: Cascade(pre, m["coarse_threshold"], m["gate_reduction"]),
    }
    if name not in builders:
        raise ValueError(f"unknown model '{name}'")
    return builders[name]()


def param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """Pretrained encoder -> lr_encoder; cascade gate -> lr_gate; everything else -> lr_decoder.
    Encoders trained from scratch use lr_decoder."""
    o = cfg["optim"]
    pretrained = bool(cfg["model"]["pretrained"])
    trainable = [p for p in (model.trainable_parameters() if hasattr(model, "trainable_parameters")
                             else model.parameters()) if p.requires_grad]
    tid = {id(p) for p in trainable}
    enc = {id(p) for p in model.encoder_parameters()} & tid if pretrained else set()
    fus = {id(p) for p in model.fusion_parameters()} & tid if hasattr(model, "fusion_parameters") else set()
    groups = [
        {"name": "encoder", "params": [p for p in trainable if id(p) in enc], "lr": o["lr_encoder"]},
        {"name": "fusion", "params": [p for p in trainable if id(p) in fus], "lr": o["lr_gate"]},
        {"name": "decoder", "params": [p for p in trainable if id(p) not in enc | fus], "lr": o["lr_decoder"]},
    ]
    return [g for g in groups if g["params"]]
