"""Shared PyTorch optimization, task partitions and held-out evaluation."""

from functools import lru_cache
from io import BytesIO
import json
import random
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import validate
from .data import load_tasks
from .models import (
    adapter_parameters,
    adapter_vector,
    build_backbone,
    linearized_features,
    make_head,
    model_transforms,
    personalized_mask,
    set_adapter,
)
from .partitioning import assign_tasks, build_partitions


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def optimizer_bytes(state):
    stream = BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


def optimizer_from_bytes(value):
    return torch.load(BytesIO(value), weights_only=True, map_location="cpu")


def train_local(
    model, head, data, cfg, *, seed, probe=False, reference=None, optimizer_state=None
):
    """Optimize a task head or its LoRA factors without modifying the backbone."""
    seed_everything(seed, cfg["torch_threads"])
    device = torch.device(cfg["device"])
    model.to(device)
    head.to(device)
    adapters = adapter_parameters(model)
    for parameter in adapters.values():
        parameter.requires_grad_(not probe)
    head.requires_grad_(probe or cfg["method"] == "fedper")
    parameters = [
        p for p in [*adapters.values(), *head.parameters()] if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg["head_learning_rate"] if probe else cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    loader = DataLoader(
        data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=cfg["num_workers"],
    )
    model.train(not probe)
    head.train()
    use_ntk = not probe and cfg["method"] == "ntk-fedavg"
    if not probe and cfg["method"] in ("fedprox", "ntk-fedavg") and reference is None:
        reference = adapter_vector(model)
    reference_tensor = (
        None if reference is None else torch.as_tensor(reference, device=device)
    )
    total = total_prox = 0.0
    seen = 0
    for _ in range(1 if probe else cfg["local_epochs"]):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if probe:
                with torch.no_grad():
                    features = model(images)
            else:
                features = (
                    linearized_features(model, images, reference)
                    if use_ntk
                    else model(images)
                )
            loss = nn.functional.cross_entropy(head(features), labels)
            proximal = torch.zeros((), device=device)
            if not probe and cfg["method"] == "fedprox":
                values = torch.cat([p.reshape(-1) for p in adapters.values()])
                proximal = (
                    0.5 * cfg["prox_mu"] * (values - reference_tensor).square().sum()
                )
            objective = loss + proximal
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training loss")
            objective.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in parameters
            ):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            n = len(labels)
            total += float(loss.detach()) * n
            total_prox += float(proximal.detach()) * n
            seen += n
    if not seen:
        raise ValueError("Cannot train an empty client task")
    return {
        "count": len(data),
        "loss": total / seen,
        "proximal_loss": total_prox / seen,
        "optimizer": optimizer.state_dict(),
    }


def evaluate(model, head, data, cfg, *, reference=None):
    device = torch.device(cfg["device"])
    model.to(device).eval()
    head.to(device).eval()
    correct = count = 0
    loss = 0.0
    with torch.no_grad():
        for images, labels in DataLoader(
            data, batch_size=cfg["eval_batch_size"], num_workers=cfg["num_workers"]
        ):
            images, labels = images.to(device), labels.to(device)
            features = (
                model(images)
                if reference is None
                else linearized_features(model, images, reference)
            )
            scores = head(features)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite evaluation scores")
            loss += float(nn.functional.cross_entropy(scores, labels, reduction="sum"))
            correct += int((scores.argmax(1) == labels).sum())
            count += len(labels)
    if not count:
        raise ValueError("Empty test split")
    return {
        "count": count,
        "correct": correct,
        "accuracy": correct / count,
        "loss": loss / count,
    }


class Runtime:
    def __init__(self, cfg):
        self.cfg = validate(cfg)
        seed_everything(cfg["seed"], cfg["torch_threads"])
        self.model, _ = build_backbone(cfg)
        self.origin = adapter_vector(self.model)
        self.private = personalized_mask(self.model)
        train_transform, test_transform = model_transforms()
        self.tasks = load_tasks(
            cfg["datasets"], cfg["data_root"], train_transform, test_transform
        )
        self.ownership = assign_tasks(
            cfg["datasets"],
            cfg["num_clients"],
            cfg["task_alpha"],
            cfg["seed"],
            tasks_per_client=cfg["tasks_per_client"],
            assignments=cfg["assignments"] or None,
        )
        self.partitions = build_partitions(
            {t: d.targets for t, d in self.tasks.items()},
            self.ownership,
            cfg["dirichlet_alpha"],
            cfg["seed"],
        )
        self.heads = {
            t: make_head(len(d.classes), (cfg["seed"] + i) % 2**32)
            for i, (t, d) in enumerate(self.tasks.items())
        }
        groups = {}
        offset = 0
        for name, p in adapter_parameters(self.model).items():
            layer = int(re.search(r"\.layer\.(\d+)\.", name).group(1))
            groups.setdefault(layer, [offset, offset])
            offset += p.numel()
            groups[layer][1] = offset
        self.layers = [slice(*bounds) for bounds in groups.values()]

    def train_data(self, cid, task):
        return Subset(self.tasks[task].train, self.partitions[cid][task])

    def counts(self, cid):
        return {t: len(indices) for t, indices in self.partitions[cid].items()}

    def head_state(self, tasks=None):
        return {
            f"{t}/{name}": p.detach().cpu().clone()
            for t in (tasks or self.tasks)
            for name, p in self.heads[t].state_dict().items()
        }

    def load_heads(self, state, tasks):
        expected = {
            f"{t}/{n}": p for t in tasks for n, p in self.heads[t].state_dict().items()
        }
        if set(state) != set(expected):
            raise ValueError("Task head keys mismatch")
        for key, value in state.items():
            if value.shape != expected[key].shape or not torch.isfinite(value).all():
                raise ValueError("Invalid task classifier")
        for t in tasks:
            self.heads[t].load_state_dict(
                {n: state[f"{t}/{n}"] for n in self.heads[t].state_dict()}
            )


@lru_cache(maxsize=1)
def runtime(serialized_config):
    return Runtime(json.loads(serialized_config))
