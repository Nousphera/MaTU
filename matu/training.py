"""Seeded many-task training, with synthetic smoke and ImageFolder benchmarks."""

import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .core import ClientUpdate, UnifiedVector, aggregate, unify

DEFAULTS = dict(
    backend="synthetic",
    seed=42,
    rounds=3,
    clients=3,
    tasks=3,
    samples=96,
    local_epochs=1,
    batch_size=16,
    learning_rate=0.01,
    rank=4,
    participation=1.0,
    dirichlet_alpha=0.5,
    top_k=2,
    rho=0.4,
    agreement="absolute",
    combine="sum",
    head_rounds=2,
    device="cpu",
    output="runs/smoke",
    model="vit_base_patch32_224.augreg_in21k",
    manifest=None,
)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_backbone(config):
    seed_all(config["seed"])
    if config["backend"] == "synthetic":
        model = nn.Sequential(LoRALinear(nn.Linear(12, 24), config["rank"]), nn.Tanh())
        return model, 24
    if config["backend"] != "imagefolder":
        raise ValueError("backend must be synthetic or imagefolder")
    import timm

    model = timm.create_model(
        config["model"], pretrained=config.get("pretrained", True), num_classes=0
    )
    model.requires_grad_(False)
    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name == "qkv" and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, config["rank"]))
                replaced += 1
    if not replaced:
        raise ValueError("Model has no supported qkv projection; use a timm ViT")
    return model, model.num_features


def adapter_vector(model):
    return (
        torch.cat(
            [p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad]
        )
        .numpy()
        .copy()
    )


def set_adapter(model, values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite adapter weights")
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                n = parameter.numel()
                if offset + n > len(values):
                    raise ValueError("Adapter size mismatch")
                parameter.copy_(
                    torch.from_numpy(
                        values[offset : offset + n].reshape(parameter.shape)
                    ).to(parameter)
                )
                offset += n
    if offset != len(values):
        raise ValueError("Adapter size mismatch")


def dirichlet_partition(labels, clients, alpha, seed):
    """Disjoint class-Dirichlet allocation, with explicit nonempty repair."""
    if clients < 1 or alpha <= 0 or len(labels) < clients:
        raise ValueError("Need positive alpha and at least one sample per client")
    rng = np.random.default_rng(seed)
    result = [[] for _ in range(clients)]
    labels = np.asarray(labels)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        sizes = rng.multinomial(len(indices), rng.dirichlet(np.repeat(alpha, clients)))
        for part, split in zip(result, np.split(indices, np.cumsum(sizes)[:-1])):
            part.extend(split.tolist())
    for part in result:
        if not part:
            donor = max(result, key=len)
            part.append(donor.pop())
    return [np.asarray(part, dtype=np.int64) for part in result]


def load_tasks(config):
    """Separate test sets are never used for fitting or model selection."""
    if config["backend"] == "synthetic":
        generator = torch.Generator().manual_seed(config["seed"] + 19)
        tasks = {}
        for index in range(config["tasks"]):
            teacher = torch.randn(12, 3, generator=generator)
            x = torch.randn(config["samples"] * 2, 12, generator=generator)
            y = (x @ teacher).argmax(1)
            n = config["samples"]
            tasks[f"task-{index}"] = (
                TensorDataset(x[:n], y[:n]),
                TensorDataset(x[n:], y[n:]),
                3,
                y[:n].numpy(),
            )
        return tasks
    from torchvision.datasets import ImageFolder
    from timm.data import create_transform, resolve_model_data_config
    import timm

    # No second pretrained download is needed to resolve preprocessing.
    specification = timm.create_model(config["model"], pretrained=False, num_classes=0)
    transform = create_transform(
        **resolve_model_data_config(specification), is_training=False
    )
    del specification
    path = Path(config["manifest"]).resolve()
    manifest = json.loads(path.read_text())
    tasks = {}
    for name, settings in manifest.items():
        train_path = (path.parent / settings["train"]).resolve()
        test_path = (path.parent / settings["test"]).resolve()
        if train_path == test_path:
            raise ValueError(f"{name}: train and test must be different directories")
        train, test = (
            ImageFolder(train_path, transform),
            ImageFolder(test_path, transform),
        )
        if train.class_to_idx != test.class_to_idx:
            raise ValueError(f"{name}: train/test class mappings differ")
        tasks[name] = (train, test, len(train.classes), np.asarray(train.targets))
    if not tasks:
        raise ValueError("No tasks in manifest")
    return tasks


def build_clients(tasks, config):
    names = sorted(tasks)
    assignments = config.get("assignments") or [
        names[:] for _ in range(config["clients"])
    ]
    if len(assignments) != config["clients"] or any(
        not a or len(set(a)) != len(a) for a in assignments
    ):
        raise ValueError(
            "assignments must contain nonempty unique task lists for every client"
        )
    if set().union(*(set(a) for a in assignments)) != set(names):
        raise ValueError("Assignments must cover exactly the configured tasks")
    clients = [{} for _ in assignments]
    partition_record = {}
    for t, name in enumerate(names):
        holders = [i for i, a in enumerate(assignments) if name in a]
        parts = dirichlet_partition(
            tasks[name][3], len(holders), config["dirichlet_alpha"], config["seed"] + t
        )
        partition_record[name] = {}
        for cid, indices in zip(holders, parts):
            clients[cid][name] = Subset(tasks[name][0], indices.tolist())
            partition_record[name][str(cid)] = indices.tolist()
    return clients, partition_record


def train_local(model, head, dataset, config, *, seed, probe=False):
    model.eval()  # keep backbone dropout/statistics fixed; adapters still receive gradients
    head.train()
    head.requires_grad_(probe)
    parameters = (
        list(head.parameters())
        if probe
        else [p for p in model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=0.0
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    device = config["device"]
    for _ in range(config["local_epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if probe:
                with torch.no_grad():
                    features = model(x)
            else:
                features = model(x)
            loss = nn.functional.cross_entropy(head(features), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local training loss")
            loss.backward()
            optimizer.step()


def evaluate(model, head, dataset, config):
    model.eval()
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=config["batch_size"]):
            prediction = head(model(x.to(config["device"]))).argmax(1).cpu()
            correct += int((prediction == y).sum())
            total += len(y)
    return correct / total


class Experiment:
    """Same task-level state is used by sequential and Flower execution."""

    def __init__(self, config):
        self.config = {**DEFAULTS, **config}
        cfg = self.config
        if (
            not 0 < cfg["participation"] <= 1
            or min(
                cfg["rounds"],
                cfg["clients"],
                cfg["rank"],
                cfg["batch_size"],
                cfg["local_epochs"],
            )
            < 1
        ):
            raise ValueError("Invalid training configuration")
        self.model, dimension = make_backbone(cfg)
        self.origin = adapter_vector(self.model)
        self.model.to(cfg["device"])
        self.tasks = load_tasks(cfg)
        self.clients, self.partitions = build_clients(self.tasks, cfg)
        seed_all(cfg["seed"] + 23)
        self.heads = {
            t: nn.Linear(dimension, item[2]).to(cfg["device"])
            for t, item in sorted(self.tasks.items())
        }
        self.global_vectors = {t: np.zeros_like(self.origin) for t in self.tasks}
        self.probe_bytes = 0
        self.upload_bytes = self.download_bytes = 0
        self.history = []

    def probe_heads(self):
        """Task-level federated linear probing; explicitly extra to MaTU equations."""
        cfg = self.config
        for rnd in range(cfg["head_rounds"]):
            for task in sorted(self.tasks):
                states, counts = [], []
                for cid, data in enumerate(self.clients):
                    if task not in data:
                        continue
                    head = copy.deepcopy(self.heads[task])
                    train_local(
                        self.model,
                        head,
                        data[task],
                        cfg,
                        seed=cfg["seed"] + rnd * cfg["clients"] + cid,
                        probe=True,
                    )
                    states.append(
                        {k: v.detach().clone() for k, v in head.state_dict().items()}
                    )
                    counts.append(len(data[task]))
                    self.probe_bytes += 2 * sum(
                        v.numel() * v.element_size() for v in states[-1].values()
                    )
                average = {
                    k: sum(s[k] * n / sum(counts) for s, n in zip(states, counts))
                    for k in states[0]
                }
                self.heads[task].load_state_dict(average)

    def client_fit(self, cid, packet, rnd):
        cfg = self.config
        local = {}
        for offset, task in enumerate(sorted(self.clients[cid])):
            set_adapter(self.model, self.origin + packet.reconstruct(task))
            train_local(
                self.model,
                self.heads[task],
                self.clients[cid][task],
                cfg,
                seed=cfg["seed"] + rnd * 10000 + cid * 100 + offset,
            )
            local[task] = adapter_vector(self.model) - self.origin
        return ClientUpdate(
            unify(local), {t: len(d) for t, d in self.clients[cid].items()}
        )

    def apply(self, updates, rnd):
        cfg = self.config
        self.global_vectors.update(
            aggregate(
                updates,
                top_k=cfg["top_k"],
                rho=cfg["rho"],
                agreement=cfg["agreement"],
                combine=cfg["combine"],
            )
        )
        accuracy = {}
        for task in sorted(self.tasks):
            set_adapter(self.model, self.origin + self.global_vectors[task])
            accuracy[task] = evaluate(
                self.model, self.heads[task], self.tasks[task][1], cfg
            )
        self.history.append(
            {
                "round": rnd,
                "accuracy": accuracy,
                "macro_accuracy": float(np.mean(list(accuracy.values()))),
            }
        )

    def save(self, frozen_digest):
        cfg = self.config
        output = Path(cfg["output"])
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "rounds": len(self.history),
            "history": self.history,
            "upload_bytes": self.upload_bytes,
            "download_bytes": self.download_bytes,
            "probe_bytes": self.probe_bytes,
            "adapter_change": float(
                sum(np.linalg.norm(v) for v in self.global_vectors.values())
            ),
            "backbone_unchanged": frozen_hash(self.model) == frozen_digest,
            "backbone_sha256": frozen_digest,
            "scope": "execution smoke"
            if cfg["backend"] == "synthetic"
            else "user dataset run; not paper-result reproduction",
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (output / "partitions.json").write_text(json.dumps(self.partitions) + "\n")
        torch.save(
            {
                "config": cfg,
                "origin": torch.from_numpy(self.origin),
                "task_vectors": {
                    k: torch.from_numpy(v) for k, v in self.global_vectors.items()
                },
                "heads": {
                    k: {n: v.cpu() for n, v in h.state_dict().items()}
                    for k, h in self.heads.items()
                },
            },
            output / "final.pt",
        )
        return result


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def check_output_directory(config):
    """Protect previous experiments before training starts."""
    output = Path(config.get("output", DEFAULTS["output"]))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"Output already contains files: {output}. Choose a new --output directory."
        )


def run(config):
    check_output_directory(config)
    experiment = Experiment(config)
    cfg = experiment.config
    frozen = frozen_hash(experiment.model)
    experiment.probe_heads()
    rng = np.random.default_rng(cfg["seed"])
    for rnd in range(1, cfg["rounds"] + 1):
        chosen = sorted(
            rng.choice(
                cfg["clients"],
                max(1, math.ceil(cfg["clients"] * cfg["participation"])),
                replace=False,
            )
        )
        updates = []
        for cid in chosen:
            packet = unify(
                {t: experiment.global_vectors[t] for t in experiment.clients[cid]}
            )
            payload = packet.to_bytes()
            experiment.download_bytes += len(payload)
            update = experiment.client_fit(cid, UnifiedVector.from_bytes(payload), rnd)
            wire = update.packet.to_bytes()
            experiment.upload_bytes += len(wire) + len(
                json.dumps(update.counts).encode()
            )
            updates.append(ClientUpdate(UnifiedVector.from_bytes(wire), update.counts))
        experiment.apply(updates, rnd)
    return experiment.save(frozen)
