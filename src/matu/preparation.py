"""Index acquired data or download official datasets, preserving source pixels."""

import argparse
from contextlib import contextmanager
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import tarfile

from PIL import Image

from .catalog import BENCHMARKS, DATASETS, canonical_name
from .data import (
    content_overlap_report,
    image_digest,
    manifest_path,
    sha256_file,
    source_path,
    verify_task,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".webp", ".tif", ".tiff"}


@contextmanager
def safe_archives():
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError(
            "Downloads require an updated Python with tarfile.data_filter"
        )
    previous = vars(tarfile.TarFile).get("extraction_filter")
    tarfile.TarFile.extraction_filter = staticmethod(tarfile.data_filter)
    try:
        yield
    finally:
        tarfile.TarFile.extraction_filter = previous


def native_dataset(name, root, split, download):
    from torchvision import datasets

    name = canonical_name(name)
    factory = DATASETS[name].torchvision
    if not factory:
        raise ValueError(
            f"{name} requires acquired data from {DATASETS[name].url}; use --source-dir or --source-map"
        )
    kwargs = {"root": str(root), "download": download}
    if name in {"mnist", "cifar10", "cifar100", "fashionmnist", "emnist"}:
        kwargs["train"] = split == "train"
        if name == "emnist":
            kwargs["split"] = "balanced"
    elif name in {"eurosat", "sun397"}:
        if split != "all":
            raise ValueError(f"{name} requires source_split=all")
    else:
        kwargs["split"] = "trainval" if name == "pets" and split == "train" else split
        if name == "dtd":
            kwargs["partition"] = 1
    if download:
        with safe_archives():
            return getattr(datasets, factory)(**kwargs)
    return getattr(datasets, factory)(**kwargs)


def _file_records(source, split_file=None):
    """Return original order and explicit labels; never trust folder sort over IDs."""
    if split_file:
        with Path(split_file).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "split", "label"} <= set(reader.fieldnames or []):
                raise ValueError("Split CSV requires path,split,label columns")
            raw = list(reader)
        classes = sorted({r["label"] for r in raw})
        mapping = {name: i for i, name in enumerate(classes)}
        return (
            classes,
            [
                {"path": r["path"], "split": r["split"], "label": mapping[r["label"]]}
                for r in raw
            ],
            "split-csv",
        )
    if (source / "samples.txt").is_file():
        mapping = {}
        for line in (source / "class_names.txt").read_text().splitlines():
            name, index = line.split("\t")
            if name in mapping:
                raise ValueError("Duplicate class name")
            mapping[name] = int(index)
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError(
                "Class IDs must be unique contiguous integers starting at zero"
            )
        classes = [
            name for name, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]
        rows = []
        for line in (source / "samples.txt").read_text().splitlines():
            path, name, index, split = line.split("\t")
            if name not in mapping or mapping[name] != int(index):
                raise ValueError("Manifest label disagrees with class_names.txt")
            rows.append({"path": path, "label": int(index), "split": split})
        return classes, rows, "emr-tsv"
    split_dirs = [
        s for s in ("train", "test", "val", "validation") if (source / s).is_dir()
    ]
    roots = (
        [(s if s != "validation" else "val", source / s) for s in split_dirs]
        if split_dirs
        else [("none", source)]
    )
    classes = sorted(
        {p.name for _, root in roots for p in root.iterdir() if p.is_dir()}
    )
    rows = []
    for split, root in roots:
        for label, name in enumerate(classes):
            for path in sorted((root / name).rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    rows.append(
                        {
                            "path": path.relative_to(source).as_posix(),
                            "label": label,
                            "split": split,
                        }
                    )
    return classes, rows, "imagefolder"


def _assign_splits(rows, policy, seed, fraction):
    if policy not in {"official", "stratified", "legacy-emr"}:
        raise ValueError("split_policy must be official, stratified or legacy-emr")
    if not 0 < fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if any(r["split"] not in {"train", "test", "val", "none"} for r in rows):
        raise ValueError("Unknown split in source manifest")
    excluded = sum(r["split"] == "val" for r in rows)
    unsplit = [r for r in rows if r["split"] == "none"]
    selected = [r.copy() for r in rows if r["split"] in {"train", "test"}]
    if unsplit:
        if policy == "official":
            raise ValueError(
                "Source has no official split; explicitly choose --split-policy stratified or legacy-emr"
            )
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            unsplit,
            test_size=fraction,
            random_state=seed,
            stratify=[r["label"] for r in unsplit],
        )
        selected.extend(
            {**row, "split": split}
            for split, group in [("train", train), ("test", test)]
            for row in group
        )
    if policy == "legacy-emr":
        from sklearn.model_selection import train_test_split

        tests = [r for r in selected if r["split"] == "test"]
        if len(tests) > 5000:
            tests, _ = train_test_split(
                tests,
                train_size=5000,
                random_state=seed,
                stratify=[r["label"] for r in tests],
            )
            selected = [r for r in selected if r["split"] == "train"] + tests
    return selected, {
        "name": policy,
        "seed": seed,
        "test_fraction": fraction if unsplit else None,
        "max_test_samples": 5000 if policy == "legacy-emr" else None,
        "excluded_validation": excluded,
        "algorithm": (
            "sklearn.model_selection.train_test_split"
            if unsplit or policy == "legacy-emr"
            else "source assignments"
        ),
        "sklearn_version": (
            importlib.metadata.version("scikit-learn")
            if unsplit or policy == "legacy-emr"
            else None
        ),
    }


def prepare_dataset(
    name,
    root,
    *,
    source=None,
    source_format="auto",
    split_file=None,
    split_policy="official",
    seed=42,
    test_fraction=0.2,
    download=False,
    content_overlap_policy=None,
):
    """Prepare one atomic manifest; repeated identical requests verify and reuse it.

    Imported images remain in their source directory. Native torchvision arrays
    remain in the download cache. No resized or recompressed image copies appear.
    """
    name = canonical_name(name)
    if bool(source) == bool(download):
        raise ValueError("Choose an acquired source or download=True")
    if source_format not in {"auto", "torchvision"}:
        raise ValueError("source_format must be auto or torchvision")
    if split_file and (not source or source_format == "torchvision"):
        raise ValueError("split_file requires an acquired source")
    overlap_policy = content_overlap_policy or (
        "preserve-source" if download or source_format == "torchvision" else "reject"
    )
    target = manifest_path(name, root)
    source = (
        Path(source).expanduser().resolve()
        if source
        else Path(root).expanduser().resolve() / "_downloads" / name
    )
    request = {
        "source": str(source),
        "download": download,
        "split_policy": split_policy,
        "source_format": source_format,
        "seed": seed,
        "test_fraction": test_fraction,
        "split_file": str(Path(split_file).resolve()) if split_file else None,
        "content_overlap_policy": overlap_policy,
    }
    if split_file:
        request["split_file_sha256"] = sha256_file(split_file)
    if target.exists():
        existing = verify_task(name, root)
        if existing.get("request") != request:
            raise ValueError(
                f"Prepared settings differ at {target}; choose another --root"
            )
        return target
    native = {}
    if download or source_format == "torchvision":
        source.mkdir(parents=True, exist_ok=True)
        source_splits = ["all"] if name in {"eurosat", "sun397"} else ["train", "test"]
        rows = []
        classes = None
        for split in source_splits:
            dataset = native_dataset(name, source, split, download)
            native[split] = dataset
            candidate = getattr(dataset, "classes", None)
            if candidate is not None:
                candidate = [str(c) for c in candidate]
                if classes is not None and candidate != classes:
                    raise ValueError("Native train/test class mappings differ")
                classes = candidate
            for index in range(len(dataset)):
                image, label = dataset[index]
                rows.append(
                    {
                        "source_split": split,
                        "index": index,
                        "label": int(label),
                        "split": "none" if split == "all" else split,
                        "sha256": image_digest(image),
                    }
                )
        if classes is None:
            classes = [str(i) for i in range(max(r["label"] for r in rows) + 1)]
        backend, source_format = "torchvision", "torchvision"
    else:
        if not source.is_dir():
            raise ValueError(f"Source must be a dataset directory: {source}")
        classes, rows, source_format = _file_records(source, split_file)
        backend = "files"
        seen = set()
        for row in rows:
            path = source_path(source, row["path"])
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                raise ValueError(f"Duplicate source image: {path}")
            seen.add(identity)
            with Image.open(path) as image:
                image.load()
            row.update(size=stat.st_size, sha256=sha256_file(path))
    rows, policy = _assign_splits(rows, split_policy, seed, test_fraction)
    overlap = content_overlap_report(rows, overlap_policy)
    counts = {
        split: sum(r["split"] == split for r in rows) for split in ("train", "test")
    }
    labels = {
        split: {r["label"] for r in rows if r["split"] == split} for split in counts
    }
    if (
        not all(counts.values())
        or labels["train"] != set(range(len(classes)))
        or labels["test"] != labels["train"]
    ):
        raise ValueError(
            "Nonempty train/test splits containing every class are required; provide explicit split assignments"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": 1,
        "dataset": name,
        "backend": backend,
        "source_root": os.path.relpath(source, target.parent),
        "source_format": source_format,
        "source_url": DATASETS[name].url,
        "classes": classes,
        "counts": counts,
        "split_policy": policy,
        "request": request,
        "samples": rows,
        "content_overlap": overlap,
        "source_provenance": {
            "kind": (
                "official-downloader"
                if download
                else "native-cache" if backend == "torchvision" else "custom-import"
            ),
            "catalog_identity_verified": False,
        },
    }
    if backend == "torchvision":
        metadata["torchvision_version"] = importlib.metadata.version("torchvision")
    else:
        metadata["source_manifest_hashes"] = {
            filename: sha256_file(source / filename)
            for filename in ("samples.txt", "class_names.txt")
            if (source / filename).is_file()
        }
    descriptor, temporary = tempfile.mkstemp(
        prefix=".dataset-", suffix=".json", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(metadata, stream, separators=(",", ":"))
            stream.write("\n")
        # Exclusive creation preserves another preparer's manifest if it won.
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", nargs="+", action="extend", help="Dataset names; aliases accepted"
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--hf-repo",
        nargs="?",
        const="Nousphera/MaTU-30",
        help="Read a collection config; defaults to Nousphera/MaTU-30 when supplied without a value",
    )
    parser.add_argument("--revision", help="Pin the Hugging Face collection revision")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="One dataset directory, or parent containing task directories",
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        help="JSON mapping names to paths or {path,format,split_file,split_policy} objects",
    )
    parser.add_argument(
        "--source-format",
        choices=["auto", "torchvision"],
        default="auto",
        help="Use torchvision to reuse an existing native cache",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="CSV path,split,label for a single acquired task",
    )
    parser.add_argument(
        "--split-policy",
        choices=["official", "stratified", "legacy-emr"],
        default="official",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--content-overlap",
        choices=["reject", "preserve-source"],
        help="File imports reject identical train/test content by default; native official splits preserve and report it",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="List all acquisition routes"
    )
    args = parser.parse_args(argv)
    if args.list:
        for spec in DATASETS.values():
            print(
                f"{spec.name}: {'automatic torchvision' if spec.torchvision else 'acquire then import'} — {spec.url}"
            )
        return
    if args.dataset and args.benchmark:
        parser.error("Choose --dataset or --benchmark")
    if args.source_dir and args.source_map:
        parser.error("Choose --source-dir or --source-map")
    if args.revision and not args.hf_repo:
        parser.error("--revision requires --hf-repo")
    names = [
        canonical_name(n)
        for n in (args.dataset or BENCHMARKS[args.benchmark or "small"])
    ]
    if len(set(names)) != len(names):
        parser.error("Duplicate task names")
    if args.split_file and (len(names) != 1 or not args.source_dir):
        parser.error("--split-file requires one task and --source-dir")
    sources = {}
    if args.source_map:
        for key, value in json.loads(args.source_map.read_text()).items():
            name = canonical_name(key)
            if name in sources:
                parser.error(f"Duplicate canonical source-map name: {name}")
            entry = {"path": value} if isinstance(value, str) else dict(value)
            if "path" not in entry or set(entry) - {
                "path",
                "format",
                "split_file",
                "split_policy",
            }:
                parser.error(
                    "Source-map objects require path; optional keys are format, split_file, split_policy"
                )
            for field in ("path", "split_file"):
                if field in entry:
                    path = Path(entry[field]).expanduser()
                    entry[field] = (
                        path
                        if path.is_absolute()
                        else args.source_map.resolve().parent / path
                    )
            sources[name] = entry
    failures = []
    for name in names:
        entry = sources.get(name, {})
        source = entry.get("path")
        if args.source_map and not entry and not args.hf_repo:
            parser.error(f"Source map has no entry for {name}")
        if args.source_dir:
            candidate = args.source_dir if len(names) == 1 else args.source_dir / name
            source = (
                candidate
                if len(names) == 1 or candidate.exists() or not args.hf_repo
                else None
            )
        try:
            if args.verify_only:
                metadata = verify_task(name, args.root)
                print(f"{name}: verified {metadata['counts']}")
                continue
            if args.hf_repo and source is None:
                from .hub_data import prepare_huggingface

                path = prepare_huggingface(name, args.root, args.hf_repo, args.revision)
            else:
                path = prepare_dataset(
                    name,
                    args.root,
                    source=source,
                    download=source is None,
                    source_format=entry.get("format", args.source_format),
                    split_file=entry.get("split_file", args.split_file),
                    split_policy=entry.get("split_policy", args.split_policy),
                    seed=args.seed,
                    test_fraction=args.test_fraction,
                    content_overlap_policy=args.content_overlap,
                )
        except (ValueError, OSError, RuntimeError) as error:
            failures.append(f"{name}: {error}")
            continue
        print(f"{name}: {path}")
    if failures:
        parser.exit(1, "Tasks needing attention:\n" + "\n".join(failures) + "\n")


if __name__ == "__main__":
    main()
