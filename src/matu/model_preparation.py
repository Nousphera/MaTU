"""Download and verify the immutable ImageNet-21k ViT-B/32 checkpoint."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

MODEL_ID = "google/vit-base-patch32-224-in21k"
MODEL_REVISION = "98cb6f7099d372694c8547bf7002289b36857dc7"
DEFAULT_WEIGHTS_DIR = "checkpoints/vit-b32-in21k"
MODEL_FILES = {
    "config.json": {
        "size": 502,
        "sha256": "62101f81166bbbd06fddcf9fecef1ddb14e63cb408cd18757b33880777676077",
    },
    "preprocessor_config.json": {
        "size": 160,
        "sha256": "90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2",
    },
    "model.safetensors": {
        "size": 352205741,
        "sha256": "82d6433fcb417924480236806bd22e3bc9b21f578df839cae78d79e24db61c1d",
    },
}


def _verify_files(directory):
    for name, expected in MODEL_FILES.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint file {path}; run prepare_model.py"
            )
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 checksum mismatch: {path}")


def validate_model(directory):
    """Reject incomplete, changed or differently sourced checkpoint directories."""
    directory = Path(directory).expanduser().resolve()
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint manifest at {path}; run prepare_model.py"
        )
    manifest = json.loads(path.read_text())
    expected = {
        "schema": 1,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": MODEL_FILES,
    }
    if manifest != expected:
        raise ValueError(
            "Checkpoint manifest does not match the pinned model and hashes"
        )
    _verify_files(directory)
    return manifest


def _download(name, destination):
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
    request = Request(url, headers={"User-Agent": "MaTU-model-preparation"})
    with urlopen(request, timeout=60) as response, destination.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def prepare_model(directory=DEFAULT_WEIGHTS_DIR):
    """Publish verified files atomically; existing checkpoints are only validated."""
    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        return validate_model(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent)
    )
    try:
        for name in MODEL_FILES:
            _download(name, temporary / name)
        _verify_files(temporary)
        manifest = {
            "schema": 1,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "files": MODEL_FILES,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if directory.exists():
            raise FileExistsError(
                f"Checkpoint destination appeared during preparation: {directory}"
            )
        os.rename(temporary, directory)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
