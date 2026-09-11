"""DINOv2 backbone loading and intermediate-layer feature fusion.

The register-token variant is the default (Darcet et al., ICLR 2024 - P08), a
separate work from the DINOv2 backbone origin (Oquab et al., TMLR 2024 - P44).
The non-register checkpoint is a single ablation row, not asserted as mandatory.

Features are pulled from several transformer blocks and combined with learned
weights: final blocks emphasise semantic identity, earlier blocks retain texture
and local shape, both of which matter for rod-shaped, reddish, textured bacilli.

An offline stub backbone (small ViT with the same intermediate-layer interface)
keeps the pipeline runnable without network access / for CI.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..utils.logging import get_logger

logger = get_logger(__name__)


class _StubBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x


class _StubViT(nn.Module):
    """Small ViT exposing get_intermediate_layers, for offline/CI use only.

    Blocks use plain nn.Linear MLPs (like real DINOv2) so LoRA injection applies;
    the nn.MultiheadAttention module is left intact by the LoRA injector.
    """

    def __init__(self, embed_dim: int = 128, depth: int = 4, patch: int = 14, heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([_StubBlock(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.n_blocks = depth

    def get_intermediate_layers(self, x, n, reshape=False, return_class_token=False, norm=True):
        wanted = set(n if isinstance(n, (list, tuple)) else range(self.n_blocks - n, self.n_blocks))
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        outputs = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i in wanted:
                t = self.norm(tokens) if norm else tokens
                patch = t[:, 1:]
                outputs.append((patch, t[:, 0]) if return_class_token else patch)
        return outputs


class OfflineStubRequested(RuntimeError):
    """Raised when the caller explicitly asked for a real backbone but the stub
    was used anyway. Used as the control-flow signal inside ``build_dino``."""


def build_dino(name: str = "dinov2_vitb14_reg", offline_stub: bool = True) -> nn.Module:
    """Build the DINOv2 verifier backbone.

    ``offline_stub=True`` (default, CI/offline): a small random-init ViT with the
    real DINOv2 intermediate-layer interface — keeps the pipeline runnable without
    network access. It carries NO pretrained signal, so it must NEVER be used for
    any result that claims a foundation-model contribution.

    ``offline_stub=False``: load the real checkpoint via ``torch.hub``. If the load
    fails for ANY reason (network, hub, missing weights, Python-version-incompatible
    upstream code), this RAISES — it does not silently fall back. A silent fallback
    here would produce a full run that looks successful but contains no real DINOv2,
    which is the worst possible failure mode for this project (every prior suite run
    was contaminated this way). Catch ``OfflineStubRequested`` only if you have a
    deliberate, logged reason to accept the stub on a real run.
    """
    if offline_stub:
        logger.warning(
            "using offline ViT stub for verifier backbone — this carries NO "
            "pretrained signal; set verifier.offline_stub=False for any real run.")
        return _StubViT()
    try:
        model = torch.hub.load("facebookresearch/dinov2", name)
    except Exception as exc:  # network unavailable / hub error / upstream py-incompat
        raise RuntimeError(
            f"DINOv2 hub load failed (offline_stub=False): {exc}. The verifier "
            f"requires the real '{name}' checkpoint for any foundation-model claim. "
            f"Either (a) fix network/hub access and retry, or (b) explicitly accept "
            f"the stub by setting verifier.offline_stub=True — but do NOT report any "
            f"result from a stub run as an FM-RobustAFB result."
        ) from exc
    logger.info("loaded DINOv2 checkpoint: %s", name)
    return model


class DinoFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, fuse_layers: List[int]):
        super().__init__()
        self.backbone = backbone
        self.fuse_layers = [max(0, i - 1) for i in fuse_layers]
        self.embed_dim = getattr(backbone, "embed_dim", 768)
        self.layer_weights = nn.Parameter(torch.zeros(len(self.fuse_layers)))

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        layers = self.backbone.get_intermediate_layers(
            crops, self.fuse_layers, reshape=False, return_class_token=True, norm=True)
        pooled = []
        for patch, cls in layers:
            pooled.append(0.5 * patch.mean(dim=1) + 0.5 * cls)
        stacked = torch.stack(pooled, dim=0)
        weights = torch.softmax(self.layer_weights, dim=0)[:, None, None]
        return (stacked * weights).sum(dim=0)
