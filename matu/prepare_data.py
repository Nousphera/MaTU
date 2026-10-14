"""Prepare verified, portable ImageFolder tasks without modifying raw data."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root, value):
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Expected a relative path inside the source: {value}")
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside source: {value}")
    return path


def source_records(source, split_file=None):
    source = Path(source).resolve()
    if split_file:
        with Path(split_file).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "split", "label"} <= set(reader.fieldnames or []):
                raise ValueError("Split CSV requires path,split,label columns")
            for row in reader:
                yield row["split"], row["label"], safe_path(source, row["path"])
    else:
        for split in ("train", "test"):
            for path in sorted((source / split).rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    rel = path.relative_to(source / split)
                    if len(rel.parts) < 2:
                        raise ValueError("Images must be inside class directories")
                    yield split, rel.parts[0], path


def downloaded_records(dataset, cache):
    from torchvision import datasets

    factories = {
        "mnist": lambda split: datasets.MNIST(
            cache, train=split == "train", download=True
        ),
        "svhn": lambda split: datasets.SVHN(cache, split=split, download=True),
        "gtsrb": lambda split: datasets.GTSRB(cache, split=split, download=True),
        "dtd": lambda split: datasets.DTD(
            cache, split=split, partition=1, download=True
        ),
    }
    if dataset not in factories:
        raise ValueError(
            "Automatic download supports mnist, svhn, gtsrb and dtd; use --source/--split-file for other tasks"
        )
    for split in ("train", "test"):
        data = factories[dataset](split)
        # DTD validation is kept out of the maintained training split.
        for index in range(len(data)):
            image, label = data[index]
            yield split, str(int(label)), image, f"{dataset}/{split}/{index}"


def prepare(dataset, root, *, source=None, split_file=None, download=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset):
        raise ValueError(
            "Task name must contain only letters, numbers, hyphens or underscores"
        )
    if bool(source) == bool(download):
        raise ValueError("Choose exactly one of --source or --download")
    if split_file and not source:
        raise ValueError("--split-file requires --source")
    root = Path(root).resolve()
    target = root / dataset
    if target.exists():
        raise FileExistsError(
            f"{target} exists; use --validate-only or choose a new root"
        )
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset}-", dir=root))
    records, counts, labels, seen = (
        [],
        {"train": 0, "test": 0},
        {"train": set(), "test": set()},
        {},
    )
    try:
        if download:
            items = downloaded_records(dataset, root / "_downloads")
        else:
            items = (
                (split, label, path, str(path.resolve()))
                for split, label, path in source_records(source, split_file)
            )
        for split, label, item, source_id in items:
            if split not in counts or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", label
            ):
                raise ValueError(
                    "Split must be train/test; labels must be safe class directory names"
                )
            if isinstance(item, Path):
                stat = item.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    raise ValueError(f"Duplicate or overlapping source image: {item}")
                seen[identity] = split
                with Image.open(item) as image:
                    image.load()
                    image = image.convert("RGB")
            else:
                image = item.convert("RGB")
            destination = staging / split / label / f"{counts[split]:08d}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)
            records.append(
                {
                    "path": destination.relative_to(staging).as_posix(),
                    "split": split,
                    "label": label,
                    "source": source_id,
                    "sha256": digest(destination),
                }
            )
            counts[split] += 1
            labels[split].add(label)
        if not all(counts.values()) or labels["train"] != labels["test"]:
            raise ValueError(
                "Nonempty train/test splits with identical classes are required"
            )
        metadata = {
            "schema": 1,
            "dataset": dataset,
            "source": str(Path(source).resolve())
            if source
            else f"torchvision:{dataset}",
            "split_policy": "torchvision official train/test (DTD partition 1; validation excluded)"
            if download
            else "user supplied train/test; no repartitioning",
            "counts": counts,
            "classes": sorted(labels["train"]),
            "files": records,
        }
        (staging / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (staging / "tasks.json").write_text(
            json.dumps({dataset: {"train": "train", "test": "test"}}, indent=2) + "\n"
        )
        validate(staging)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return target


def validate(root):
    root = Path(root).resolve()
    data = json.loads((root / "dataset.json").read_text())
    counts = {"train": 0, "test": 0}
    labels = {"train": set(), "test": set()}
    seen = set()
    for record in data["files"]:
        path = safe_path(root, record["path"])
        if path in seen:
            raise ValueError("Duplicate file in manifest")
        seen.add(path)
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Missing or changed file: {path}")
        with Image.open(path) as image:
            image.load()
        split, label = record["split"], record["label"]
        if split not in counts or Path(record["path"]).parts[:2] != (split, label):
            raise ValueError("Invalid split/class in manifest")
        counts[split] += 1
        labels[split].add(label)
    actual = {
        p.resolve()
        for split in counts
        for p in (root / split).rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }
    if (
        actual != seen
        or counts != data["counts"]
        or not all(counts.values())
        or labels["train"] != labels["test"]
        or sorted(labels["train"]) != data["classes"]
    ):
        raise ValueError("Dataset membership or class mapping changed")
    return data


def combine(manifests, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    combined = {}
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        validate(manifest.parent)
        for name, splits in json.loads(manifest.read_text()).items():
            if name in combined:
                raise ValueError(f"Duplicate task: {name}")
            combined[name] = {
                split: os.path.relpath(manifest.parent / path, destination.parent)
                for split, path in splits.items()
            }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(combined, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Task name, e.g. mnist, sun397 or resisc45")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--source",
        type=Path,
        help="Raw train/test ImageFolders or root for the split CSV",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="CSV: path,split,label; paths relative to source",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Official torchvision MNIST/SVHN/GTSRB/DTD splits",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--combine", nargs="+", type=Path, metavar="TASKS_JSON")
    parser.add_argument("--output", type=Path, help="Destination for --combine")
    args = parser.parse_args()
    if args.combine:
        if not args.output:
            parser.error("--combine requires --output")
        combine(args.combine, args.output)
        print(args.output)
    elif not args.dataset:
        parser.error("--dataset is required")
    elif args.validate_only:
        print(json.dumps(validate(args.root / args.dataset)["counts"]))
    else:
        print(
            prepare(
                args.dataset,
                args.root,
                source=args.source,
                split_file=args.split_file,
                download=args.download,
            )
            / "tasks.json"
        )


if __name__ == "__main__":
    main()
