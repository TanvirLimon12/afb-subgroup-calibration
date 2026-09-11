"""Stage-2 verifier training over detector-proposed candidates.

Candidates are labelled by IoU against ground truth; detector false positives
provide natural hard negatives (stain deposits, fibers, debris). Ground-truth
crops are added as positives so the verifier sees confirmed bacilli even where the
detector misses them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..utils.boxes import box_iou_np
from ..verifier import DINOVerifier
from ..verifier.verifier import extract_crops
from .infer import tiled_inference


@dataclass
class Candidate:
    image_id: int
    box: np.ndarray
    label: int
    det_logit: float
    group: int
    source: str
    iou: float = 0.0  # best IoU against GT (0 for GT positives / unmatched); for LaECE/LRP


class CandidateBank:
    def __init__(self, records: List[Candidate], dataset):
        self.records = records
        self.dataset = dataset
        self._id_to_index = {im["id"]: i for i, im in enumerate(dataset.images)}

    def __len__(self):
        return len(self.records)

    def image_for(self, image_id: int) -> torch.Tensor:
        return self.dataset[self._id_to_index[image_id]]["image"]

    def arrays(self):
        det = [r for r in self.records if r.source == "det"]
        return {
            "det_logits": np.array([r.det_logit for r in det], dtype=np.float64),
            "labels": np.array([r.label for r in det], dtype=np.int64),
            "groups": np.array([r.group for r in det], dtype=np.int64),
            "ious": np.array([r.iou for r in det], dtype=np.float64),
            "records": det,
        }


def build_candidate_bank(
    detector, dataset, device: torch.device, cfg: Config, iou_pos: float = 0.5,
    max_images: Optional[int] = None, add_gt_positives: bool = True, corruption=None,
) -> CandidateBank:
    results = tiled_inference(detector, dataset, device, cfg.data.tile_size,
                              cfg.data.tile_overlap, cfg.detector.cross_tile_nms_iou,
                              corruption=corruption, max_images=max_images)
    records: List[Candidate] = []
    for r in results:
        if len(r.boxes):
            ious = box_iou_np(r.boxes, r.gt_boxes) if len(r.gt_boxes) else np.zeros((len(r.boxes), 0))
            max_iou = ious.max(axis=1) if ious.shape[1] > 0 else np.zeros(len(r.boxes))
            for k in range(len(r.boxes)):
                records.append(Candidate(r.image_id, r.boxes[k], int(max_iou[k] >= iou_pos),
                                         float(r.det_logits[k]), r.group, "det", iou=float(max_iou[k])))
        if add_gt_positives:
            for gb in r.gt_boxes:
                records.append(Candidate(r.image_id, gb, 1, float("nan"), r.group, "gt", iou=1.0))
    return CandidateBank(records, dataset)


class _CropDataset(Dataset):
    def __init__(self, bank: CandidateBank, context: float, size: int):
        self.bank = bank
        self.context = context
        self.size = size

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        rec = self.bank.records[i]
        image = self.bank.image_for(rec.image_id)
        crop = extract_crops(image, torch.as_tensor(rec.box[None], dtype=torch.float32),
                             self.context, self.size)[0]
        return crop, rec.label, rec.group


def train_verifier(cfg: Config, bank: CandidateBank, device: torch.device, epochs: int = 5):
    verifier = DINOVerifier(cfg.verifier).to(device)
    ds = _CropDataset(bank, cfg.verifier.crop_context, cfg.verifier.crop_size)

    # Training-free NN head: populate the memory bank from labelled candidate
    # features and skip gradient descent entirely (AnomalyDINO-style, arXiv:2405.14529).
    if cfg.verifier.head == "nn":
        _fit_nn_head(verifier, ds, device)
        return verifier

    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.train.num_workers)

    if cfg.verifier.head == "prototype":
        _init_prototypes(verifier, ds, device)

    opt = torch.optim.AdamW(verifier.trainable_parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    verifier.train()
    for _ in range(epochs):
        for crops, labels, _ in loader:
            crops = crops.to(device)
            labels = labels.float().to(device)
            logits = verifier(crops)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return verifier


@torch.no_grad()
def _init_prototypes(verifier: DINOVerifier, ds: _CropDataset, device, max_samples: int = 256):
    feats, labels = [], []
    for i in range(min(len(ds), max_samples)):
        crop, label, _ = ds[i]
        feats.append(verifier.features(crop[None].to(device))[0].cpu())
        labels.append(label)
    if feats:
        verifier.head.init_from_features(torch.stack(feats), torch.tensor(labels))


@torch.no_grad()
def _fit_nn_head(verifier: DINOVerifier, ds: _CropDataset, device, max_samples: int = 1024):
    """Populate the NN memory head's positive/negative banks from labelled crops.

    Uses more samples than prototype init because the NN head's quality scales with
    bank coverage; the head itself caps each class at max_per_class for inference.
    """
    feats, labels = [], []
    for i in range(min(len(ds), max_samples)):
        crop, label, _ = ds[i]
        feats.append(verifier.features(crop[None].to(device))[0].cpu())
        labels.append(label)
    if feats:
        verifier.head.fit_from_features(torch.stack(feats), torch.tensor(labels))


@torch.no_grad()
def verifier_logits(verifier: DINOVerifier, bank: CandidateBank, device, records) -> np.ndarray:
    verifier.eval()
    logits = []
    for rec in records:
        image = bank.image_for(rec.image_id)
        logit = verifier.verify_boxes(image.to(device), torch.as_tensor(rec.box[None], dtype=torch.float32).to(device))
        logits.append(float(logit[0]) if logit.numel() else 0.0)
    return np.array(logits, dtype=np.float64)
