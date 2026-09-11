"""Typed configuration loaded from YAML with dotted-key overrides."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, get_type_hints

import yaml


@dataclass
class DataConfig:
    root: str = "data/afb"
    ann_train: str = "data/afb/train.json"
    ann_val: str = "data/afb/val.json"
    ann_test: str = "data/afb/test.json"
    cameras: List[str] = field(default_factory=lambda: ["hayear", "optilab"])
    backgrounds: List[str] = field(default_factory=lambda: ["b0", "b1", "b2", "b3"])
    num_classes: int = 1
    tile_size: int = 640
    tile_overlap: float = 0.18
    normalize_stain: bool = False
    max_side: int = 0  # 0 = native resolution; >0 caps the long side (feasibility only)


@dataclass
class AuditConfig:
    phash_hamming_max: int = 6
    embed_cosine_min: float = 0.98


@dataclass
class AssignerConfig:
    center_radius: float = 2.5
    candidate_topk: int = 10


@dataclass
class DetLossConfig:
    cls_weight: float = 1.0
    box_weight: float = 2.0
    ctr_weight: float = 1.0


@dataclass
class DetectorConfig:
    backbone: str = "resnet18"
    pretrained: bool = False
    fpn_min_level: int = 2
    fpn_max_level: int = 5
    fpn_channels: int = 128
    head_convs: int = 2
    num_classes: int = 1
    score_thresh: float = 0.05
    nms_iou: float = 0.5
    max_det_per_tile: int = 300
    cross_tile_nms_iou: float = 0.5
    assigner: AssignerConfig = field(default_factory=AssignerConfig)
    loss: DetLossConfig = field(default_factory=DetLossConfig)


@dataclass
class VerifierConfig:
    backbone: str = "dinov2_vitb14_reg"
    offline_stub: bool = True
    crop_context: float = 2.5
    crop_size: int = 224
    fuse_layers: List[int] = field(default_factory=lambda: [6, 9, 12])
    adaptation: str = "lora"
    lora_rank: int = 8
    prototypes_per_class: int = 8
    head: str = "linear"
    hard_negative_ratio: float = 1.0


@dataclass
class SegmentationConfig:
    enabled: bool = False
    models: List[str] = field(default_factory=lambda: ["micro_sam", "cellsam"])
    pilot_dice_min: float = 0.6
    consensus_dice_min: float = 0.7
    containment_min: float = 0.7
    elongation_min: float = 2.0


@dataclass
class GroupDROConfig:
    step_size: float = 0.01
    l2: float = 0.0005


@dataclass
class ConsistencyConfig:
    enabled: bool = True
    weight: float = 0.5
    max_hue: float = 0.03
    max_sat: float = 0.2
    max_blur: float = 1.5


@dataclass
class RobustConfig:
    method: str = "group_balanced"
    group_dro: GroupDROConfig = field(default_factory=GroupDROConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)


@dataclass
class FusionConfig:
    calibration: str = "temperature"
    model: str = "logistic"
    use_group_feature: bool = True


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_iters: int = 200
    num_workers: int = 4
    amp: bool = True


@dataclass
class EvalConfig:
    ap_iou: List[float] = field(default_factory=lambda: [0.5, 0.75])
    dece_bins: int = 15
    dece_binning: str = "confidence"
    coverage_points: List[float] = field(default_factory=lambda: [1.0, 0.95, 0.9, 0.8])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs/default"
    data: DataConfig = field(default_factory=DataConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    robust: RobustConfig = field(default_factory=RobustConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _from_dict(cls, data: dict):
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: List[str] | None = None) -> Config:
    """Load a Config from YAML, then apply ``key.subkey=value`` overrides."""
    data: dict = {}
    if path is not None:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    cfg = _from_dict(Config, data)
    for ov in overrides or []:
        _apply_override(cfg, ov)
    return cfg


def _apply_override(cfg: Any, override: str) -> None:
    key, _, raw = override.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got: {override}")
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = getattr(node, p)
    leaf = parts[-1]
    current = getattr(node, leaf)
    setattr(node, leaf, _coerce(raw, current))


def _coerce(raw: str, template: Any) -> Any:
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        parsed = raw
    if isinstance(template, bool):
        return bool(parsed)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(parsed)
    if isinstance(template, float):
        return float(parsed)
    return parsed


def to_dict(cfg: Any) -> dict:
    if is_dataclass(cfg):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, list):
        return [to_dict(v) for v in cfg]
    return copy.deepcopy(cfg)
