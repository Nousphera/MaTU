"""Resolve configuration paths before handing work to worker processes."""

import json
from .training import DEFAULTS


def load_app_config(settings):
    from pathlib import Path

    path = Path(settings["config"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path = path.resolve()
    cfg = {**DEFAULTS, **json.loads(path.read_text())}
    if cfg.get("manifest"):
        cfg["manifest"] = str((path.parent / cfg["manifest"]).resolve())
    cfg["output"] = str(Path(settings.get("output") or cfg["output"]).resolve())
    return cfg
