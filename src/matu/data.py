"""Verified file/native dataset manifests and task loading without image copies."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .catalog import canonical_name


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_digest(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def content_overlap_report(rows, policy="reject"):
    if policy not in {"reject", "preserve-source"}:
        raise ValueError("content overlap policy must be reject or preserve-source")
    hashes = {}
    for row in rows:
        digest = row["sha256"]
        if not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid content hash")
        entry = hashes.setdefault(digest, {"train": 0, "test": 0, "labels": set()})
        entry[row["split"]] += 1
        entry["labels"].add(row["label"])
    overlap = [x for x in hashes.values() if x["train"] and x["test"]]
    if overlap and policy == "reject":
        raise ValueError(
            f"Identical content overlaps train/test ({len(overlap)} hashes); use an explicit preserve-source policy only to retain a documented source split"
        )
    return {
        "policy": policy,
        "cross_split_hashes": len(overlap),
        "cross_split_pairs": sum(x["train"] * x["test"] for x in overlap),
        "conflicting_label_hashes": sum(len(x["labels"]) > 1 for x in overlap),
    }


def source_path(source, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid relative source path: {relative}")
    target = (source / relative).resolve()
    if not target.is_relative_to(source.resolve()):
        raise ValueError(f"Path outside source: {relative}")
    return target


def manifest_path(name, root):
    return Path(root).expanduser().resolve() / canonical_name(name) / "dataset.json"


def verify_task(name, root, *, full=True):
    """Check metadata, split disjointness, class mapping and source identity.

    Full verification hashes and decodes every image. Loaders use structural and
    file-size checks; call this before an experiment for complete integrity QA.
    """
    name = canonical_name(name)
    path = manifest_path(name, root)
    metadata = json.loads(path.read_text())
    if metadata.get("schema") != 1 or metadata.get("dataset") != name:
        raise ValueError("Unsupported or mismatched dataset manifest")
    classes = metadata["classes"]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Invalid class mapping")
    source = (path.parent / metadata["source_root"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset source is unavailable: {source}")
    if full:
        for filename, expected in metadata.get("source_manifest_hashes", {}).items():
            if sha256_file(source_path(source, filename)) != expected:
                raise ValueError(f"Source assignment hash changed: {filename}")
    if metadata["backend"] == "huggingface":
        from .hub_data import hub_file, validate_hub_files

        validate_hub_files(source, name, metadata["hub"]["files"])
        for record in metadata["hub"]["files"]:
            item = hub_file(source, name, record["path"])
            if item.stat().st_size != record["bytes"] or (
                full and sha256_file(item) != record["sha256"]
            ):
                raise ValueError(f"Collection shard changed: {item}")
    counts = {"train": 0, "test": 0}
    labels = {"train": set(), "test": set()}
    seen = set()
    identities = set()
    native = {}
    for row in metadata["samples"]:
        split, label = row["split"], row["label"]
        if (
            split not in counts
            or type(label) is not int
            or not 0 <= label < len(classes)
        ):
            raise ValueError("Invalid split or class label")
        if metadata["backend"] == "files":
            item = source_path(source, row["path"])
            stat = item.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in identities:
                raise ValueError(f"Duplicate source image: {item}")
            identities.add(identity)
            key = row["path"]
            if stat.st_size != row["size"]:
                raise ValueError(f"Source changed: {item}")
            if full:
                if sha256_file(item) != row["sha256"]:
                    raise ValueError(f"Source hash changed: {item}")
                with Image.open(item) as image:
                    image.load()
        elif metadata["backend"] in {"torchvision", "huggingface"}:
            if (
                type(row["index"]) is not int
                or row["index"] < 0
                or row["source_split"] not in {"train", "test", "all"}
            ):
                raise ValueError("Invalid native source index or split")
            key = (row["source_split"], row["index"])
            if full:
                if row["source_split"] not in native:
                    if metadata["backend"] == "torchvision":
                        from .preparation import native_dataset

                        native[row["source_split"]] = native_dataset(
                            name, source, row["source_split"], False
                        )
                    else:
                        from .hub_data import hub_dataset

                        native[row["source_split"]] = hub_dataset(
                            metadata, source, row["source_split"]
                        )
                value = native[row["source_split"]][row["index"]]
                image, target = (
                    (value["image"], value["label"])
                    if isinstance(value, dict)
                    else value
                )
                if int(target) != label or image_digest(image) != row["sha256"]:
                    raise ValueError("Native dataset image/label hash changed")
        else:
            raise ValueError("Unknown dataset backend")
        if key in seen:
            raise ValueError("Duplicate source record or overlapping splits")
        seen.add(key)
        counts[split] += 1
        labels[split].add(label)
    if counts != metadata["counts"] or not all(counts.values()):
        raise ValueError("Nonempty train/test splits and matching counts are required")
    if labels["train"] != set(range(len(classes))) or labels["test"] != labels["train"]:
        raise ValueError("Train/test must contain every declared class")
    recorded = metadata.get("content_overlap")
    overlap = content_overlap_report(
        metadata["samples"], recorded["policy"] if recorded else "reject"
    )
    if recorded is not None and recorded != overlap:
        raise ValueError("Content overlap counts differ from the manifest")
    return metadata


class ManifestDataset(Dataset):
    def __init__(self, metadata, manifest, split, transform=None):
        self.metadata = metadata
        self.source = (manifest.parent / metadata["source_root"]).resolve()
        self.rows = [r for r in metadata["samples"] if r["split"] == split]
        self.targets = np.asarray([r["label"] for r in self.rows], dtype=np.int64)
        self.classes = list(metadata["classes"])
        self.class_names = self.classes
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform
        self._native = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.metadata["backend"] == "files":
            with Image.open(source_path(self.source, row["path"])) as item:
                image = item.convert("RGB")
        else:
            split = row["source_split"]
            if split not in self._native:
                if self.metadata["backend"] == "torchvision":
                    from .preparation import native_dataset

                    self._native[split] = native_dataset(
                        self.metadata["dataset"], self.source, split, False
                    )
                else:
                    from .hub_data import hub_dataset

                    self._native[split] = hub_dataset(self.metadata, self.source, split)
            value = self._native[split][row["index"]]
            image, label = (
                (value["image"], value["label"]) if isinstance(value, dict) else value
            )
            if int(label) != row["label"]:
                raise ValueError("Native labels differ from prepared manifest")
            image = image.convert("RGB")
        return (self.transform(image) if self.transform else image), row["label"]


@dataclass
class Task:
    train: Dataset
    test: Dataset
    classes: list[str]
    targets: np.ndarray
    metadata: dict


def load_task(name, root, transform, split):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    metadata = verify_task(name, root, full=False)
    return ManifestDataset(metadata, manifest_path(name, root), split, transform)


def load_tasks(names, root, train_transform=None, test_transform=None):
    result = {}
    for value in names:
        name = canonical_name(value)
        if name in result:
            raise ValueError(f"Duplicate requested task: {name}")
        metadata = verify_task(name, root, full=False)
        manifest = manifest_path(name, root)
        train = ManifestDataset(metadata, manifest, "train", train_transform)
        test = ManifestDataset(metadata, manifest, "test", test_transform)
        result[name] = Task(train, test, train.classes, train.targets, metadata)
    return result
