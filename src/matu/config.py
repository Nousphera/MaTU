"""Validated experiment settings shared by the launcher and Flower apps."""

import math
import re

MODEL = "google/vit-base-patch32-224-in21k"
METHODS = ("matu", "fedavg", "fedprox", "fedper", "matfl", "ntk-fedavg", "centralized")


def defaults():
    return dict(
        method="matu",
        datasets=["mnist", "svhn", "gtsrb", "dtd", "eurosat"],
        model=MODEL,
        weights_dir="checkpoints/vit-b32-in21k",
        data_root="data",
        output="results/matu",
        device="cpu",
        seed=42,
        num_clients=30,
        rounds=100,
        participation=0.2,
        local_epochs=1,
        batch_size=32,
        eval_batch_size=64,
        learning_rate=0.0001,
        weight_decay=0.0,
        head_rounds=1,
        head_learning_rate=0.001,
        rank=16,
        lora_alpha=16.0,
        lora_dropout=0.1,
        task_alpha=0.5,
        dirichlet_alpha=0.1,
        tasks_per_client=0,
        assignments=[],
        top_k=2,
        rho=0.4,
        similarity_threshold=0.5,
        matfl_top_k=3,
        prox_mu=0.01,
        server_learning_rate=1.0,
        eval_every=1,
        torch_threads=1,
        num_workers=0,
        reply_timeout=3600.0,
        registration_timeout=120.0,
    )


def validate(config):
    missing = set(defaults()) - set(config)
    extra = set(config) - set(defaults())
    if missing or extra:
        raise ValueError(
            f"Invalid configuration keys: missing={sorted(missing)}, unknown={sorted(extra)}"
        )
    cfg = dict(config)
    integers = {
        "seed": 0,
        "num_clients": 1,
        "rounds": 1,
        "local_epochs": 1,
        "batch_size": 1,
        "eval_batch_size": 1,
        "head_rounds": 1,
        "rank": 1,
        "tasks_per_client": 0,
        "top_k": 0,
        "matfl_top_k": 1,
        "eval_every": 1,
        "torch_threads": 1,
        "num_workers": 0,
    }
    for key, minimum in integers.items():
        if (
            isinstance(cfg[key], bool)
            or not isinstance(cfg[key], int)
            or cfg[key] < minimum
        ):
            raise ValueError(f"{key} must be an integer >= {minimum}")
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
        value = cfg[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")
        cfg[key] = float(value)
    if any(
        cfg[k] <= 0
        for k in (
            "learning_rate",
            "head_learning_rate",
            "lora_alpha",
            "dirichlet_alpha",
            "server_learning_rate",
            "reply_timeout",
            "registration_timeout",
        )
    ):
        raise ValueError(
            "Learning rates, adapter scale, class concentration and timeouts must be positive"
        )
    if any(cfg[k] < 0 for k in ("weight_decay", "task_alpha", "prox_mu")):
        raise ValueError("Regularization and task concentration must be nonnegative")
    if not 0 < cfg["participation"] <= 1 or not 0 <= cfg["lora_dropout"] < 1:
        raise ValueError("Invalid participation or LoRA dropout")
    if any(not 0 <= cfg[k] <= 1 for k in ("rho", "similarity_threshold")):
        raise ValueError("Agreement and similarity thresholds must be in [0,1]")
    if (
        cfg["method"] not in METHODS
        or cfg["model"] != MODEL
        or cfg["device"] not in ("cpu", "cuda", "mps")
    ):
        raise ValueError("Unsupported method, model or device")
    names = cfg["datasets"]
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(n, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", n)
            for n in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError("datasets must contain unique canonical task names")
    cfg["datasets"] = list(names)
    if cfg["tasks_per_client"] > len(names):
        raise ValueError("tasks_per_client exceeds the number of tasks")
    if not isinstance(cfg["assignments"], list):
        raise ValueError("assignments must be a list of client task lists")
    for key in ("weights_dir", "data_root", "output"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
    if cfg["rank"] > 768 or cfg["seed"] >= 2**32:
        raise ValueError("rank must be <= 768 and seed < 2**32")
    return cfg
