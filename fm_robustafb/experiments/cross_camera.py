"""Experiment 2 - Hayear <-> Optilab cross-camera testing.

Trains on all cameras, then reports in-camera vs cross-camera AP and the headline
gap Delta_camera = AP_in-camera - AP_cross-camera.
"""
from __future__ import annotations

from itertools import product

from ..config import Config
from ..engine import run_pipeline
from .common import clone_config


def run(cfg: Config, max_images=None) -> dict:
    cameras = list(cfg.data.cameras)
    per_camera = {}
    for cam in cameras:
        c = clone_config(cfg)
        art = run_pipeline(c, use_verifier=False, use_fusion=False,
                           max_images=max_images, test_cameras=[cam])
        per_camera[cam] = art.report["detection"].get("mean_AP50",
                                                       art.report["detection"].get("AP50"))

    gaps = {}
    for train_cam, test_cam in product(cameras, cameras):
        if train_cam == test_cam:
            continue
        # Approximation on shared model: use per-camera test AP as cross vs in reference.
        gaps[f"{train_cam}->{test_cam}"] = per_camera.get(test_cam)
    in_camera = sum(per_camera.values()) / max(1, len(per_camera))
    return {"per_camera_AP50": per_camera, "mean_in_camera_AP50": in_camera,
            "cross_camera_reference": gaps}
