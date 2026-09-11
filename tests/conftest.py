import pytest

from fm_robustafb.config import load_config
from fm_robustafb.data.synthetic import generate_dataset


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("afb")
    paths = generate_dataset(root, n_per_group=2, size=192, seed=3)
    return {"root": str(root), "paths": paths}


@pytest.fixture
def smoke_config(synthetic_dataset):
    cfg = load_config("configs/default.yaml")
    root = synthetic_dataset["root"]
    cfg.data.root = root
    cfg.data.ann_train = synthetic_dataset["paths"]["train"]
    cfg.data.ann_val = synthetic_dataset["paths"]["val"]
    cfg.data.ann_test = synthetic_dataset["paths"]["test"]
    cfg.device = "cpu"
    cfg.train.epochs = 1
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0
    cfg.detector.fpn_channels = 32
    cfg.data.tile_size = 192
    cfg.detector.score_thresh = 0.03
    cfg.verifier.offline_stub = True
    cfg.verifier.crop_size = 42
    cfg.verifier.fuse_layers = [2, 4]
    cfg.verifier.head = "linear"
    cfg.robust.method = "group_balanced"
    cfg.robust.consistency.enabled = False
    return cfg
