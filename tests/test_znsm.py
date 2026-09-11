import cv2
import numpy as np

from afb_calibration.data.znsm import _microscope_of, convert_znsm, extract_boxes


def _draw_annotated(path, size=(600, 800)):
    img = np.full((size[0], size[1], 3), 200, np.uint8)
    img[..., 0] = 180  # bluish background, not black
    # oval (single bacillus) -> positive
    cv2.ellipse(img, (150, 150), (25, 15), 0, 0, 360, (0, 0, 0), 2)
    # rectangle (occluded) -> positive
    cv2.rectangle(img, (400, 120), (450, 180), (0, 0, 0), 2)
    # diamond (artifact) -> negative
    pts = np.array([[600, 300], [640, 340], [600, 380], [560, 340]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)


def test_microscope_parsing():
    assert _microscope_of("a/Mannual_Microscope-2_set1/3.jpg") == "micro2"
    assert _microscope_of("foo/bar.jpg") == "unknown"


def test_shape_recovery(tmp_path):
    p = tmp_path / "Microscope-1" / "img.jpg"
    p.parent.mkdir(parents=True)
    _draw_annotated(p)
    pos, art = extract_boxes(p)
    assert len(pos) >= 2   # oval + rectangle recovered as bacilli
    assert len(art) >= 1   # diamond recovered as artifact


def test_convert_znsm_camera_groups(tmp_path):
    for cam in ("Microscope-1", "Microscope-2", "Microscope-3"):
        for i in range(4):
            d = tmp_path / "imgs" / f"Mannual_{cam}_set1"
            d.mkdir(parents=True, exist_ok=True)
            _draw_annotated(d / f"{i}.jpg")
    paths = convert_znsm(tmp_path / "imgs", tmp_path / "out", seed=0)
    import json
    cams = set()
    for p in paths.values():
        for im in json.load(open(p))["images"]:
            cams.add(im["camera"])
    assert cams == {"micro1", "micro2", "micro3"}
