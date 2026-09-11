import torch

from fm_robustafb.data.tiling import build_grid, fuse_tile_detections, tile_image


def test_grid_covers_image():
    grid = build_grid(500, 500, 200, 0.2)
    xs = [t[2] for t in grid.tiles]
    ys = [t[3] for t in grid.tiles]
    assert max(xs) == 500 and max(ys) == 500


def test_tile_image_shapes():
    image = torch.rand(3, 400, 400)
    grid = build_grid(400, 400, 256, 0.1)
    crops = tile_image(image, grid)
    assert len(crops) == len(grid)


def test_fuse_dedupes_seam_boxes():
    grid = build_grid(400, 200, 200, 0.0)
    dets = [
        {"boxes": torch.tensor([[10, 10, 20, 20.0]]), "scores": torch.tensor([0.9]),
         "logits": torch.tensor([2.0]), "feats": torch.zeros(1, 4)},
        {"boxes": torch.tensor([[10, 10, 20, 20.0]]), "scores": torch.tensor([0.8]),
         "logits": torch.tensor([1.5]), "feats": torch.zeros(1, 4)},
    ]
    fused = fuse_tile_detections(dets, grid, 0.5)
    assert fused["boxes"].shape[0] == 2  # different tiles -> different image coords
    assert fused["boxes"][1, 1] >= 200  # second tile shifted down
