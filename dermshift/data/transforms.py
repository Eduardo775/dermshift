"""Augmentation and normalisation. Inputs are already 192x256, so spatial transforms act at the final
resolution. Normalisation uses ImageNet mean and standard deviation.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_BORDER = {"constant": cv2.BORDER_CONSTANT, "reflect101": cv2.BORDER_REFLECT_101}


def _normalize(cfg: dict) -> A.Normalize:
    norm = cfg["data"].get("normalization", "imagenet")
    if norm == "imagenet":
        return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    raise ValueError(norm)


def train_transform(cfg: dict) -> A.Compose:
    a = cfg["augmentation"]
    t = a["affine_translate"]
    return A.Compose([
        A.HorizontalFlip(p=a["hflip_p"]),
        A.VerticalFlip(p=a["vflip_p"]),
        A.Affine(scale=tuple(a["affine_scale"]), translate_percent=(-t, t), rotate=a["affine_rotate"],
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                 border_mode=_BORDER[a["affine_border"]], fill=0, fill_mask=0, p=a["affine_p"]),
        A.CLAHE(clip_limit=a["clahe_clip"], tile_grid_size=(a["clahe_grid"], a["clahe_grid"]), p=a["clahe_p"]),
        A.RandomGamma(gamma_limit=tuple(a["gamma_limit"]), p=a["gamma_p"]),
        A.Sharpen(alpha=tuple(a["sharpen_alpha"]), lightness=tuple(a["sharpen_lightness"]), p=a["sharpen_p"]),
        A.OneOf([A.Blur(blur_limit=(a["blur_limit"], a["blur_limit"])),
                 A.MotionBlur(blur_limit=(a["blur_limit"], a["blur_limit"]))], p=a["blur_p"]),
        A.HueSaturationValue(hue_shift_limit=a["hsv_hue"], sat_shift_limit=a["hsv_sat"],
                             val_shift_limit=a["hsv_val"], p=a["hsv_p"]),
        _normalize(cfg),
        ToTensorV2(),
    ])


def eval_transform(cfg: dict) -> A.Compose:
    return A.Compose([_normalize(cfg), ToTensorV2()])


def seeded_apply(tf: A.Compose, seed: int, image: np.ndarray, mask: np.ndarray) -> dict:
    """Apply ``tf`` with a per-sample seed so augmentation is a pure function of
    (run seed, epoch, sample index): identical across DataLoader workers and after resuming."""
    tf.set_random_seed(seed)
    return tf(image=image, mask=mask)
