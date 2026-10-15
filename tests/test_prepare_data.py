import json
from pathlib import Path

import pytest
from PIL import Image


def source_images(root):
    for split in ("train", "test"):
        for label in ("a", "b"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                Image.new("RGB", (24, 24), (i * 60, 120, 30)).save(folder / f"{i}.png")
    return root


def test_import_is_verified_and_consumable_by_training(tmp_path):
    from matu.prepare_data import prepare, validate
    from matu.training import load_tasks, DEFAULTS

    result = prepare(
        "fixture", tmp_path / "data", source=source_images(tmp_path / "raw")
    )
    assert validate(result)["counts"] == {"train": 4, "test": 4}
    tasks = load_tasks(
        dict(
            DEFAULTS,
            backend="imagefolder",
            model="vit_base_patch32_224.augreg_in21k",
            manifest=str(result / "tasks.json"),
        )
    )
    train, test, classes, targets = tasks["fixture"]
    assert (len(train), len(test), classes) == (4, 4, 2)
    assert train[0][0].shape == (3, 224, 224)
    (result / "train/a/00000000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed|decode"):
        validate(result)


def test_import_refuses_overlap_overwrite_and_invalid_split_paths(tmp_path):
    from matu.prepare_data import prepare

    raw = source_images(tmp_path / "raw")
    prepare("fixture", tmp_path / "data", source=raw)
    with pytest.raises(FileExistsError):
        prepare("fixture", tmp_path / "data", source=raw)
    (raw / "test/a/alias.png").symlink_to(raw / "train/a/0.png")
    with pytest.raises(ValueError, match="overlap"):
        prepare("overlap", tmp_path / "data", source=raw)
    split = tmp_path / "splits.csv"
    split.write_text("path,split,label\n../secret.png,train,a\n")
    with pytest.raises(ValueError, match="outside|relative"):
        prepare("escape", tmp_path / "data", source=raw, split_file=split)
    assert not (tmp_path / "data/overlap").exists()


def test_training_rejects_symlink_overlap_without_preparation(tmp_path):
    from matu.training import load_tasks, DEFAULTS

    raw = source_images(tmp_path / "raw")
    (raw / "test/a/alias.png").symlink_to(raw / "train/a/0.png")
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {"fixture": {"train": str(raw / "train"), "test": str(raw / "test")}}
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        load_tasks(
            dict(
                DEFAULTS,
                backend="imagefolder",
                model="vit_base_patch32_224.augreg_in21k",
                manifest=str(manifest),
            )
        )


def test_app_resolves_config_and_output_for_worker_processes(tmp_path):
    from matu.configuration import load_app_config

    cfg = tmp_path / "vision.json"
    cfg.write_text(
        json.dumps(
            {"backend": "imagefolder", "manifest": "tasks.json", "output": "old"}
        )
    )
    result = load_app_config({"config": str(cfg), "output": str(tmp_path / "out")})
    assert result["manifest"] == str(tmp_path / "tasks.json")
    assert result["output"] == str(tmp_path / "out")


def test_csv_splits_and_combined_tasks_are_portable(tmp_path):
    from matu.prepare_data import prepare, combine

    raw = source_images(tmp_path / "raw")
    split = tmp_path / "splits.csv"
    rows = ["path,split,label"]
    rows.extend(
        f"{p.relative_to(raw).as_posix()},{p.relative_to(raw).parts[0]},{p.parent.name}"
        for p in sorted(raw.rglob("*.png"))
    )
    split.write_text("\n".join(rows))
    one = prepare("first", tmp_path / "data", source=raw, split_file=split)
    two = prepare("second", tmp_path / "data", source=raw)
    manifest = tmp_path / "configs/tasks.json"
    combine([one / "tasks.json", two / "tasks.json"], manifest)
    tasks = json.loads(manifest.read_text())
    assert set(tasks) == {"first", "second"}
    assert (manifest.parent / tasks["first"]["train"]).resolve() == one / "train"
    with pytest.raises(FileExistsError):
        combine([one / "tasks.json"], manifest)


def test_imagefolder_rejects_external_source_symlink(tmp_path):
    from matu.prepare_data import prepare
    raw = source_images(tmp_path / 'raw')
    external = tmp_path / 'outside.png'
    Image.new('RGB', (24,24), 'blue').save(external)
    (raw / 'train/a/external.png').symlink_to(external)
    with pytest.raises(ValueError, match='outside source'):
        prepare('escape', tmp_path / 'data', source=raw)
    assert not (tmp_path / 'data/escape').exists()


@pytest.mark.parametrize('replacement', [
    {'fixture': {'train': '../../unverified/train', 'test': 'test'}},
    {'renamed': {'train': 'train', 'test': 'test'}},
    {'fixture': {'train': 'test', 'test': 'train'}},
])
def test_combine_rejects_task_manifest_not_bound_to_validated_dataset(tmp_path, replacement):
    from matu.prepare_data import prepare, combine
    prepared = prepare('fixture', tmp_path / 'data', source=source_images(tmp_path / 'raw'))
    manifest = prepared / 'tasks.json'
    manifest.write_text(json.dumps(replacement))
    with pytest.raises(ValueError, match='manifest|dataset|split'):
        combine([manifest], tmp_path / 'combined.json')
    assert not (tmp_path / 'combined.json').exists()


def test_svhn_dependency_is_declared():
    import tomllib
    metadata = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    assert any(requirement.startswith('scipy') for requirement in metadata['project']['dependencies'])


def test_downloaded_tar_rejects_escape_and_restores_filter(tmp_path, monkeypatch):
    import io
    import tarfile
    from torchvision import datasets
    from torchvision.datasets.utils import extract_archive
    from matu.prepare_data import downloaded_records

    archive = tmp_path / 'malicious.tar'
    (tmp_path / 'extract').mkdir()
    with tarfile.open(archive, 'w') as stream:
        member = tarfile.TarInfo('../outside.png')
        member.size = 1
        stream.addfile(member, io.BytesIO(b'x'))

    def fake_dtd(*args, **kwargs):
        extract_archive(archive, tmp_path / 'extract')
        return []

    monkeypatch.setattr(datasets, 'DTD', fake_dtd)
    previous = vars(tarfile.TarFile).get('extraction_filter')
    with pytest.raises(tarfile.FilterError):
        list(downloaded_records('dtd', tmp_path / 'cache'))
    assert not (tmp_path / 'outside.png').exists()
    assert vars(tarfile.TarFile).get('extraction_filter') is previous


def test_svhn_mat_fixture_can_be_loaded(tmp_path):
    import numpy as np
    from scipy.io import savemat
    from torchvision.datasets import SVHN

    images = np.zeros((32, 32, 3, 2), dtype=np.uint8)
    labels = np.array([[1], [10]], dtype=np.uint8)
    path = tmp_path / "train_32x32.mat"
    savemat(path, {"X": images, "y": labels})
    # Test the real scipy-backed constructor without downloading the corpus.
    from hashlib import md5
    class FixtureSVHN(SVHN):
        split_list = {"train": ("unused", path.name, md5(path.read_bytes()).hexdigest())}

    data = FixtureSVHN(tmp_path, split="train", download=False)
    assert len(data) == 2
    assert data[0][0].size == (32, 32)
    assert [data[index][1] for index in range(2)] == [1, 0]
