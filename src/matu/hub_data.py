"""Load the versioned MaTU-30 collection without assuming every image is mirrored."""

import json
import os
from pathlib import Path
import re
import tempfile

from .catalog import DATASETS, canonical_name
from .data import (
    content_overlap_report,
    image_digest,
    manifest_path,
    sha256_file,
    verify_task,
)


def hub_file(source, name, relative):
    path = Path(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:2] != ("data", name)
        or path.suffix != ".parquet"
    ):
        raise ValueError(f"Invalid collection data path: {relative}")
    # The Hub cache uses symlinks into its adjacent content-addressed blob store.
    return source / path


def hub_dataset(metadata, source, split):
    from datasets import load_dataset

    validate_hub_files(source, metadata["dataset"], metadata["hub"]["files"])
    files = [
        str(hub_file(source, metadata["dataset"], f["path"]))
        for f in metadata["hub"]["files"]
        if f["split"] == split
    ]
    return load_dataset("parquet", data_files={split: files}, split=split)


def validate_hub_files(source, name, records):
    seen, splits = set(), set()
    for record in records:
        path = hub_file(source, name, record["path"])
        if path in seen:
            raise ValueError(f"Duplicate collection shard: {record['path']}")
        seen.add(path)
        if record["split"] not in {"train", "test"}:
            raise ValueError("Collection shard split must be train or test")
        splits.add(record["split"])
        if (
            type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or not re.fullmatch("[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError("Invalid collection shard size or hash")
    if splits != {"train", "test"}:
        raise ValueError("Collection requires train and test shards")


def prepare_huggingface(name, root, repo_id="Nousphera/MaTU-30", revision=None):
    """Download one pixel-bearing config, pin its revision and verify all records.

    A local collection directory is also accepted for offline mirrors. Metadata
    configs use an available official downloader, recording that native source
    membership can differ from the collection. Other configs report acquisition.
    """
    name = canonical_name(name)
    root = Path(root).expanduser().resolve()
    target = manifest_path(name, root)
    if target.exists():
        metadata = verify_task(name, root)
        reference = metadata.get("hub", metadata.get("hub_reference", {}))
        if reference.get("repo_id") != str(repo_id) or (
            revision
            and revision
            not in {reference.get("revision"), reference.get("requested_revision")}
        ):
            raise ValueError("Prepared Hub source differs; choose another --root")
        return target
    local = Path(repo_id).expanduser()
    if local.is_dir():
        source = local.resolve()
        catalog_path = source / "catalog.json"
        resolved_revision = "local-" + sha256_file(catalog_path)
    else:
        from huggingface_hub import hf_hub_download

        catalog_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename="catalog.json",
                revision=revision,
                cache_dir=root / "_hub_cache",
            )
        )
        resolved_revision = catalog_path.parent.name
        source = catalog_path.parent
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema") != 1 or name not in catalog.get("configs", {}):
        raise ValueError("Unsupported collection catalog or missing config")
    config = catalog["configs"][name]
    if not config.get("pixels_available"):
        if DATASETS[name].torchvision:
            from .preparation import prepare_dataset

            declared = config.get("split_policy", {})
            # SUN397 has no native train/test assignment: honor the catalog's
            # explicit policy. Other native datasets retain published splits.
            policy = (
                declared.get("name", "official")
                if name in {"eurosat", "sun397"}
                else "official"
            )
            path = prepare_dataset(
                name,
                root,
                download=True,
                split_policy=policy,
                seed=declared.get("seed", 42),
                test_fraction=declared.get("test_fraction") or 0.2,
            )
            metadata = json.loads(path.read_text())
            metadata["hub_reference"] = {
                "repo_id": str(repo_id),
                "revision": resolved_revision,
                "requested_revision": revision,
                "catalog_sha256": sha256_file(catalog_path),
                "counts": config.get("counts"),
                "split_policy": declared,
                "collection_membership_reused": False,
                "note": "Acquired canonical native source; membership and counts may differ from collection metadata.",
            }
            descriptor, temporary = tempfile.mkstemp(
                prefix=".dataset-", suffix=".json", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(metadata, stream, separators=(",", ":"))
                    stream.write("\n")
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return path
        raise ValueError(
            f"{name} contains acquisition metadata only. Acquire {config.get('source_url', '')} and use --source-dir/--source-map; the collection does not redistribute these images."
        )
    validate_hub_files(source, name, config["files"])
    if not local.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=resolved_revision,
                cache_dir=root / "_hub_cache",
                allow_patterns=["catalog.json"] + [f["path"] for f in config["files"]],
            )
        )
    metadata = {
        "schema": 1,
        "dataset": name,
        "backend": "huggingface",
        "source_root": os.path.relpath(source, target.parent),
        "source_url": config["source_url"],
        "classes": config["classes"],
        "counts": config["counts"],
        "split_policy": config.get(
            "split_policy", {"name": "published collection membership"}
        ),
        "source_provenance": {
            "kind": "pinned-hf-collection",
            "catalog_identity_verified": True,
        },
        "hub": {
            "repo_id": str(repo_id),
            "revision": resolved_revision,
            "requested_revision": revision,
            "catalog_sha256": sha256_file(catalog_path),
            "files": config["files"],
        },
        "samples": [],
    }
    for record in config["files"]:
        path = hub_file(source, name, record["path"])
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Collection shard hash changed: {path}")
    counts = {}
    for split in ("train", "test"):
        dataset = hub_dataset(metadata, source, split)
        counts[split] = len(dataset)
        if (
            "image" not in dataset.column_names
            or dataset.features["label"].names != config["classes"]
        ):
            raise ValueError("Collection image feature or class mapping differs")
        for index, item in enumerate(dataset):
            metadata["samples"].append(
                {
                    "split": split,
                    "source_split": split,
                    "index": index,
                    "label": int(item["label"]),
                    "sha256": image_digest(item["image"]),
                }
            )
    if counts != config["counts"]:
        raise ValueError("Collection counts differ from its catalog")
    labels = {
        split: {r["label"] for r in metadata["samples"] if r["split"] == split}
        for split in counts
    }
    if (
        not all(counts.values())
        or labels["train"] != set(range(len(config["classes"])))
        or labels["test"] != labels["train"]
    ):
        raise ValueError("Collection train/test splits must contain every class")
    metadata["content_overlap"] = content_overlap_report(
        metadata["samples"], "preserve-source"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".dataset-", suffix=".json", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(metadata, stream, separators=(",", ":"))
            stream.write("\n")
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
    return target
