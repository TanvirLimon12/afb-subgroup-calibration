import numpy as np
import torch

from fm_robustafb.utils.boxes import box_iou, box_iou_np, expand_boxes, nms_boxes


def test_box_iou_identity():
    a = torch.tensor([[0, 0, 10, 10.0]])
    assert torch.isclose(box_iou(a, a)[0, 0], torch.tensor(1.0))


def test_box_iou_disjoint():
    a = torch.tensor([[0, 0, 10, 10.0]])
    b = torch.tensor([[20, 20, 30, 30.0]])
    assert box_iou(a, b)[0, 0].item() == 0.0


def test_box_iou_np_matches_torch():
    a = np.array([[0, 0, 10, 10.0], [5, 5, 15, 15]])
    b = np.array([[0, 0, 10, 10.0]])
    t = box_iou(torch.tensor(a), torch.tensor(b)).numpy()
    assert np.allclose(t, box_iou_np(a, b), atol=1e-5)


def test_expand_boxes_clips():
    boxes = torch.tensor([[5, 5, 7, 9.0]])
    out = expand_boxes(boxes, 3.0, height=20, width=20)
    assert out[0, 0] >= 0 and out[0, 2] <= 20


def test_nms_removes_overlap():
    boxes = torch.tensor([[0, 0, 10, 10.0], [1, 1, 11, 11.0]])
    scores = torch.tensor([0.9, 0.8])
    keep = nms_boxes(boxes, scores, 0.5)
    assert keep.numel() == 1
