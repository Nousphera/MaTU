"""Run real-image MaTU experiments with the native Flower local simulation."""

import argparse
from contextlib import contextmanager
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from .catalog import BENCHMARKS, canonical_name
from .config import METHODS, MODEL, defaults, validate
from .partitioning import assign_tasks, build_partitions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    cfg = defaults()
    p.add_argument("--method", choices=METHODS, default=cfg["method"])
    selection = p.add_mutually_exclusive_group()
    selection.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        help="Dataset names; defaults to the small benchmark",
    )
    selection.add_argument("--benchmark", choices=BENCHMARKS)
    p.add_argument("--model", choices=[MODEL], default=MODEL)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=cfg["device"])
    p.add_argument(
        "--num-clients",
        "--nr-clients",
        "--clients",
        dest="num_clients",
        type=int,
        default=cfg["num_clients"],
    )
    p.add_argument(
        "--rounds",
        "--num-rounds",
        type=int,
        help="Default: 300 for thirty tasks or MaT-FL, otherwise 100",
    )
    for key in (
        "seed",
        "local_epochs",
        "batch_size",
        "eval_batch_size",
        "head_rounds",
        "rank",
        "tasks_per_client",
        "top_k",
        "matfl_top_k",
        "eval_every",
        "torch_threads",
        "num_workers",
    ):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=cfg[key])
    for key in (
        "participation",
        "learning_rate",
        "weight_decay",
        "head_learning_rate",
        "lora_alpha",
        "lora_dropout",
        "task_alpha",
        "dirichlet_alpha",
        "rho",
        "similarity_threshold",
        "prox_mu",
        "server_learning_rate",
        "reply_timeout",
        "registration_timeout",
    ):
        p.add_argument(
            "--" + key.replace("_", "-"),
            type=float,
            default=None if key in {"participation", "task_alpha"} else cfg[key],
        )
    p.add_argument("--data-root", type=Path, default=Path(cfg["data_root"]))
    p.add_argument("--weights-dir", type=Path, default=Path(cfg["weights_dir"]))
    p.add_argument(
        "--output",
        type=Path,
        help="New result directory; defaults to results/METHOD-BENCHMARK-seedSEED",
    )
    p.add_argument(
        "--assignments",
        type=Path,
        help="JSON list of dataset-name lists, one per client",
    )
    p.add_argument("--parallel-clients", type=int, default=1)
    p.add_argument("--client-cpus", type=float, default=1.0)
    p.add_argument(
        "--client-gpus",
        type=float,
        help="Ray GPU allocation per client; defaults to 1 for CUDA, otherwise 0",
    )
    p.add_argument("--superlink", default="local")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration, ownership and command without reading data or training",
    )
    return p


def _resources(args, cfg):
    if (
        args.parallel_clients < 1
        or not math.isfinite(args.client_cpus)
        or args.client_cpus <= 0
    ):
        raise ValueError(
            "Client CPU allocation and parallel client count must be positive"
        )
    gpus = (
        args.client_gpus
        if args.client_gpus is not None
        else (1.0 if cfg["device"] == "cuda" else 0.0)
    )
    if not math.isfinite(gpus) or gpus < 0 or (cfg["device"] == "cuda" and gpus <= 0):
        raise ValueError(
            "GPU allocation must be finite and nonnegative; CUDA requires positive --client-gpus"
        )
    return gpus


def ownership(cfg):
    return assign_tasks(
        cfg["datasets"],
        cfg["num_clients"],
        cfg["task_alpha"],
        cfg["seed"],
        tasks_per_client=cfg["tasks_per_client"],
        assignments=cfg["assignments"] or None,
    )


def build_config(args):
    cfg = defaults()
    if args.benchmark == "thirty":
        cfg.update(rounds=300, participation=1.0, task_alpha=0.2)
    if args.method == "matfl":
        cfg["rounds"] = 300
    for key in cfg:
        if key not in ("datasets", "assignments", "output"):
            value = getattr(args, key, None)
            if value is not None:
                cfg[key] = value
    cfg["datasets"] = [
        canonical_name(name)
        for name in (args.datasets or BENCHMARKS[args.benchmark or "small"])
    ]
    if args.assignments is not None:
        rows = json.loads(args.assignments.expanduser().read_text())
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError("Assignments must be a JSON list of lists")
        cfg["assignments"] = [[canonical_name(name) for name in row] for row in rows]
    if cfg["method"] == "centralized":
        cfg.update(
            num_clients=1,
            participation=1.0,
            tasks_per_client=len(cfg["datasets"]),
            assignments=[list(cfg["datasets"])],
        )
    label = args.benchmark or ("-".join(cfg["datasets"]) if args.datasets else "small")
    cfg["output"] = (
        args.output or Path("results") / f"{cfg['method']}-{label}-seed{cfg['seed']}"
    )
    for key in ("data_root", "weights_dir", "output"):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    cfg = validate(cfg)
    _resources(args, cfg)
    ownership(cfg)
    return cfg


def verify_inputs(cfg):
    """Verify prepared assets and partition feasibility without loading a backbone."""
    import numpy as np
    import torch
    from .data import verify_task
    from .model_preparation import validate_model

    labels = {}
    for task in cfg["datasets"]:
        metadata = verify_task(task, cfg["data_root"], full=True)
        labels[task] = np.asarray(
            [row["label"] for row in metadata["samples"] if row["split"] == "train"],
            dtype=np.int64,
        )
    validate_model(cfg["weights_dir"])
    owners = ownership(cfg)
    partitions = build_partitions(labels, owners, cfg["dirichlet_alpha"], cfg["seed"])
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu or --device mps")
    if cfg["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; choose --device cpu")
    return owners, partitions


def prepare_output(cfg):
    path = Path(cfg["output"])
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output is not a directory: {path}")
    for name in ("metrics.json", "model.pt", "clients"):
        if (path / name).exists():
            raise FileExistsError(
                f"Result already exists: {path / name}; choose a new --output"
            )
    config = path / "config.json"
    if config.exists():
        if json.loads(config.read_text()) != cfg:
            raise FileExistsError(f"Existing configuration differs: {config}")
        return
    path.mkdir(parents=True, exist_ok=True)
    with config.open("x") as stream:
        json.dump(cfg, stream, indent=2, sort_keys=True)
        stream.write("\n")


@contextmanager
def flower_app(cfg):
    """Build Flower's descriptor in a temporary directory, from installed sources."""
    import tomli_w

    dist = distribution("matu")
    package = Path(__file__).resolve().parent
    assets = package.parent.parent
    if not (assets / "setup.py").is_file():
        assets = Path(sys.prefix) / "share" / "matu"
    descriptor = {
        "build-system": {
            "requires": ["setuptools>=77", "wheel"],
            "build-backend": "setuptools.build_meta",
        },
        "project": {
            "name": "matu",
            "version": dist.version,
            "description": dist.metadata["Summary"],
            "requires-python": dist.metadata["Requires-Python"],
            "dependencies": dist.requires or [],
            "license": "MIT",
            "readme": {"text": dist.metadata["Summary"], "content-type": "text/plain"},
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "nousphera",
                    "components": {
                        "serverapp": "matu.server_app:app",
                        "clientapp": "matu.client_app:app",
                    },
                    "config": {"config": json.dumps(cfg, sort_keys=True)},
                }
            }
        },
    }
    with TemporaryDirectory(prefix="matu-flower-") as directory:
        root = Path(directory)
        for source in package.rglob("*.py"):
            target = root / "matu" / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for name in ("setup.py", "requirements.txt", "LICENSE"):
            shutil.copy2(assets / name, root / name)
        (root / "pyproject.toml").write_text(tomli_w.dumps(descriptor))
        yield root


def flower_command(args, cfg, application):
    executable = Path(sys.executable).with_name("flwr")
    if not executable.is_file():
        raise RuntimeError("Install requirements.txt in the active environment first")
    gpus = _resources(args, cfg)
    federation = (
        f"num-supernodes={cfg['num_clients']} client-resources-num-cpus={args.client_cpus:g} "
        f"client-resources-num-gpus={gpus:g} init-args-num-cpus={args.client_cpus * args.parallel_clients:g}"
    )
    return [
        str(executable),
        "run",
        str(application),
        args.superlink,
        "--federation-config",
        federation,
        "--stream",
    ]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "config": cfg,
                        "ownership": ownership(cfg),
                        "command": flower_command(
                            args, cfg, "<temporary-native-flower-app>"
                        ),
                    },
                    indent=2,
                )
            )
            return
        verify_inputs(cfg)
        prepare_output(cfg)
        env = os.environ.copy()
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        env.setdefault("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", "1")
        with flower_app(cfg) as application:
            command = flower_command(args, cfg, application)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True, env=env)
        output = Path(cfg["output"])
        if not all((output / name).is_file() for name in ("metrics.json", "model.pt")):
            raise RuntimeError(
                "Flower ended without saved results; inspect the run logs"
            )
        print(f"Saved {output}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()
