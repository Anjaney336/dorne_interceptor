from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.perception.infer import resolve_model_path
from drone_interceptor.perception.train import create_dataset_yaml, resolve_project_path


def test_create_dataset_yaml_writes_expected_config(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "visdrone_yolo"
    for directory in (
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ):
        directory.mkdir(parents=True)

    yaml_path = create_dataset_yaml(dataset_root=dataset_root)
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert config["path"] == str(dataset_root.resolve())
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert len(config["names"]) == 11
    assert config["names"][0] == "pedestrian"
    assert config["names"][-1] == "others"


def test_resolve_model_path_prefers_explicit_or_default_checkpoint(tmp_path: Path) -> None:
    explicit_checkpoint = tmp_path / "explicit.pt"
    explicit_checkpoint.write_text("checkpoint", encoding="utf-8")
    assert resolve_model_path(explicit_checkpoint) == explicit_checkpoint.resolve()


def test_resolve_project_path_uses_project_root_for_relative_paths(tmp_path: Path) -> None:
    resolved = resolve_project_path("models/test.pt", project_root=tmp_path)
    assert resolved == (tmp_path / "models" / "test.pt").resolve()
