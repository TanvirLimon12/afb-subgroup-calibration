"""Bounded cross-style augmentation and corruption transforms.

Transformation strength is bounded on purpose: strong hue shifts destroy the
stain signal the detector depends on (P46 rationale). The same primitives drive
the Stage-3 consistency regulariser (bounded) and the Experiment-5 corruption
sweep (larger, test-time only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torchvision.transforms.functional as TF


def _uneven_illumination(image: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
    c, h, w = image.shape
    g = torch.Generator(device=image.device).manual_seed(seed)
    ax = torch.linspace(-1, 1, w, device=image.device)
    ay = torch.linspace(-1, 1, h, device=image.device)
    gx = (torch.rand(1, generator=g).item() * 2 - 1)
    gy = (torch.rand(1, generator=g).item() * 2 - 1)
    field = 1.0 + strength * (gx * ax[None, :] + gy * ay[:, None])
    return (image * field.clamp(min=0.2)).clamp(0, 1)


def style_view(image: torch.Tensor, params: Dict[str, float], seed: int = 0) -> torch.Tensor:
    out = image
    if params.get("brightness", 0):
        out = TF.adjust_brightness(out, 1.0 + params["brightness"])
    if params.get("contrast", 0):
        out = TF.adjust_contrast(out, 1.0 + params["contrast"])
    if params.get("saturation", 0):
        out = TF.adjust_saturation(out, 1.0 + params["saturation"])
    if params.get("hue", 0):
        out = TF.adjust_hue(out, params["hue"])
    if params.get("blur", 0) and params["blur"] > 0:
        k = int(params["blur"]) * 2 + 1
        out = TF.gaussian_blur(out, kernel_size=k, sigma=max(1e-3, params["blur"]))
    if params.get("noise", 0):
        g = torch.Generator(device=out.device).manual_seed(seed)
        out = (out + torch.randn(out.shape, generator=g, device=out.device) * params["noise"]).clamp(0, 1)
    if params.get("illumination", 0):
        out = _uneven_illumination(out, params["illumination"], seed)
    if params.get("jpeg_quality", 0):
        out = _jpeg(out, int(params["jpeg_quality"]))
    return out.clamp(0, 1)


def _jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    try:
        from torchvision.io import decode_jpeg, encode_jpeg

        u8 = (image * 255).round().to(torch.uint8)
        return decode_jpeg(encode_jpeg(u8, quality=quality)).float() / 255.0
    except Exception:
        return image


@dataclass
class CrossStyleAugment:
    max_hue: float = 0.03
    max_sat: float = 0.2
    max_blur: float = 1.5
    max_brightness: float = 0.15
    max_noise: float = 0.02

    def _sample(self, g: torch.Generator) -> Dict[str, float]:
        def u(scale):
            return (torch.rand(1, generator=g, device=g.device).item() * 2 - 1) * scale

        return {
            "brightness": u(self.max_brightness),
            "contrast": u(self.max_brightness),
            "saturation": u(self.max_sat),
            "hue": u(self.max_hue),
            "blur": abs(u(self.max_blur)),
            "noise": abs(u(self.max_noise)),
        }

    def two_views(self, image: torch.Tensor, seed: int = 0):
        g = torch.Generator(device=image.device).manual_seed(seed)
        v1 = style_view(image, self._sample(g), seed=seed)
        v2 = style_view(image, self._sample(g), seed=seed + 1)
        return v1, v2


CORRUPTIONS = {
    "brightness": {"brightness": 0.4},
    "contrast": {"contrast": -0.4},
    "hue_shift": {"hue": 0.1},
    "white_balance": {"saturation": 0.5},
    "defocus_blur": {"blur": 3.0},
    "motion_blur": {"blur": 2.0},
    "sensor_noise": {"noise": 0.06},
    "compression": {"jpeg_quality": 25},
    "uneven_illumination": {"illumination": 0.6},
    "reduced_resolution": {"blur": 1.5},
}


def apply_corruption(image: torch.Tensor, name: str, seed: int = 0) -> torch.Tensor:
    return style_view(image, CORRUPTIONS[name], seed=seed)
