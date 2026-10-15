"""Typed model records for the native Flower apps."""

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord
from .core import UnifiedVector, unify


def encode(vectors, compressed=False):
    if compressed:
        return ConfigRecord({"packet": unify(vectors).to_bytes()})
    return ArrayRecord(
        {task: torch.from_numpy(value.copy()) for task, value in vectors.items()}
    )


def decode(record, tasks, dimension, compressed=False):
    if compressed:
        if set(record) != {"packet"} or not isinstance(record["packet"], bytes):
            raise ValueError("Invalid compressed model record")
        packet = UnifiedVector.from_bytes(record["packet"])
        if set(packet.masks) != set(tasks) or packet.vector.size != dimension:
            raise ValueError("Compressed task set or dimension mismatch")
        return {t: packet.reconstruct(t) for t in tasks}
    values = record.to_torch_state_dict()
    if set(values) != set(tasks):
        raise ValueError("Task vector keys mismatch")
    result = {}
    for t, value in values.items():
        if (
            value.dtype != torch.float32
            or value.shape != (dimension,)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid task vector")
        result[t] = value.numpy().copy()
    return result


def payload_bytes(record):
    if isinstance(record, ConfigRecord):
        return len(record["packet"])
    return sum(
        value.numel() * value.element_size()
        for value in record.to_torch_state_dict().values()
    )
