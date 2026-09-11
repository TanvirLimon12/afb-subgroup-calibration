import numpy as np
import torch

from afb_calibration.config import DetectorConfig, RobustConfig, VerifierConfig
from afb_calibration.detector import AFBDetector
from afb_calibration.robust import build_reducer
from afb_calibration.segmentation import dice, mask_morphology
from afb_calibration.verifier import DINOVerifier


def test_detector_forward_and_backward():
    det = AFBDetector(DetectorConfig(fpn_channels=32))
    out = det(torch.rand(2, 3, 128, 128))
    targets = [{"boxes": torch.tensor([[10, 10, 20, 25.0]]), "labels": torch.tensor([0])},
               {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    loss = det.loss(out, targets)
    loss["loss"].backward()
    assert loss["per_image"].shape[0] == 2


def test_detector_has_p2_level():
    det = AFBDetector(DetectorConfig(fpn_min_level=2))
    assert 2 in det.levels and det.level_strides[0] == 4


def test_extract_crops_handles_degenerate_boxes():
    from afb_calibration.verifier.verifier import extract_crops
    img = torch.rand(3, 60, 80)
    # zero-height box, edge box, and out-of-range box
    boxes = torch.tensor([[10, 20, 30, 20.0], [79, 59, 90, 70], [0, 0, 1, 1.0]])
    crops = extract_crops(img, boxes, context=2.0, size=32)
    assert crops.shape == (3, 3, 32, 32)
    assert torch.isfinite(crops).all()


def test_verifier_produces_logits():
    v = DINOVerifier(VerifierConfig(offline_stub=True, crop_size=42, fuse_layers=[2, 4]))
    logits = v.verify_boxes(torch.rand(3, 120, 120), torch.tensor([[10, 10, 18, 26.0]]))
    assert logits.shape[0] == 1


def test_group_dro_updates_weights():
    reducer = build_reducer(RobustConfig(method="group_dro"), num_groups=4)
    before = reducer.group_weights.clone()
    per_ex = torch.tensor([1.0, 0.1, 0.5, 0.2])
    groups = torch.tensor([0, 1, 2, 3])
    reducer.reduce(per_ex, groups)
    assert not torch.allclose(before, reducer.group_weights)


def test_morphology_elongation_of_rod():
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:22, 5:35] = True
    score = mask_morphology(mask, np.array([5, 18, 35, 22.0]))
    assert score.elongation > 2.0
    assert dice(mask, mask) == 1.0
