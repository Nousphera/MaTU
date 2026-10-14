"""MaTU task unification and many-task aggregation.

Equations refer to arXiv:2502.06376v3. See docs/reproduction.md for
the explicit interpretation of ambiguous Eq. 3 and Eq. 7 conventions.
No model, network, filesystem, or random-state side effects occur here.
"""

from dataclasses import dataclass
import io
import json
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class UnifiedVector:
    vector: np.ndarray
    masks: dict[str, np.ndarray]
    scales: dict[str, float]

    def reconstruct(self, task: str) -> np.ndarray:
        return self.vector * self.masks[task] * self.scales[task]

    def to_bytes(self) -> bytes:
        """Portable float32 vector, bit-packed masks and JSON scalar metadata.

        NPZ container overhead is included in reported transport bytes.
        Pickle/object arrays are deliberately unsupported.
        """
        tasks = sorted(self.masks)
        metadata = json.dumps(
            {"tasks": tasks, "scales": [self.scales[t] for t in tasks]}
        )
        arrays = {
            "vector": self.vector.astype("<f4"),
            "metadata": np.frombuffer(metadata.encode(), dtype=np.uint8),
        }
        arrays.update(
            {
                f"mask_{i}": np.packbits(self.masks[t], bitorder="little")
                for i, t in enumerate(tasks)
            }
        )
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        return stream.getvalue()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "UnifiedVector":
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            vector = np.asarray(archive["vector"], dtype=np.float32)
            metadata = json.loads(archive["metadata"].tobytes())
            tasks, scales = metadata["tasks"], metadata["scales"]
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError("Invalid unified vector")
            if len(tasks) != len(scales) or len(set(tasks)) != len(tasks) or not tasks:
                raise ValueError("Invalid task metadata")
            if any(not isinstance(t, str) or not t for t in tasks):
                raise ValueError("Tasks must be nonempty strings")
            if not np.isfinite(scales).all() or any(s < 0 for s in scales):
                raise ValueError("Scales must be finite and nonnegative")
            masks = {}
            for i, task in enumerate(tasks):
                packed = archive[f"mask_{i}"]
                if packed.dtype != np.uint8 or packed.size != (vector.size + 7) // 8:
                    raise ValueError("Invalid mask payload length")
                masks[task] = np.unpackbits(packed, bitorder="little")[
                    : vector.size
                ].astype(bool)
        return cls(vector, masks, dict(zip(tasks, scales)))


@dataclass(frozen=True)
class ClientUpdate:
    packet: UnifiedVector
    counts: dict[str, int]


def unify(vectors: Mapping[str, np.ndarray]) -> UnifiedVector:
    """Elect an aligned maximum-magnitude vector and task modulators (Eq. 2).

    A sign tie is zero. An entirely erased task has scale zero rather than
    dividing by zero; it cannot be reconstructed by this representation.
    """
    if not vectors or any(not isinstance(t, str) or not t for t in vectors):
        raise ValueError("At least one named task is required")
    tasks = sorted(vectors)
    arrays = [np.asarray(vectors[t], dtype=np.float32) for t in tasks]
    if any(a.ndim != 1 or not a.size for a in arrays):
        raise ValueError("Task vectors must be nonempty one-dimensional arrays")
    if len({a.shape for a in arrays}) != 1 or not all(
        np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("Task vectors must have equal shapes and finite entries")
    stacked = np.stack(arrays)
    sign = np.sign(stacked.sum(axis=0, dtype=np.float64))
    aligned = np.sign(stacked) == sign
    vector = (sign * np.where(aligned, np.abs(stacked), 0).max(axis=0)).astype(
        np.float32
    )
    masks, scales = {}, {}
    for task, original in zip(tasks, arrays):
        mask = original * vector > 0
        denominator = np.abs(vector[mask]).sum(dtype=np.float64)
        masks[task] = mask
        scales[task] = (
            float(np.abs(original).sum(dtype=np.float64) / denominator)
            if denominator
            else 0.0
        )
    return UnifiedVector(vector, masks, scales)


def aggregate(
    updates: list[ClientUpdate],
    *,
    top_k: int = 2,
    rho: float = 0.4,
    similarity_threshold: float = 0.5,
    agreement: str = "absolute",
    combine: str = "sum",
) -> dict[str, np.ndarray]:
    """Count-weighted same-task aggregation followed by related-task transfer.

    Default agreement uses the magnitude of mean signs: both consistently
    positive and consistently negative directions have agreement one.
    `signed` reproduces the literal printed Eq. 3 (which can reverse signs).
    `sum` follows Eq. 7; `mean` follows the overview's averaging prose.
    Similarity weights in Eq. 6 are not renormalized. A task excludes itself.
    """
    if (
        not updates
        or top_k < 0
        or not 0 <= rho <= 1
        or not 0 <= similarity_threshold <= 1
    ):
        raise ValueError("Invalid aggregation settings")
    if agreement not in {"absolute", "signed"} or combine not in {"sum", "mean"}:
        raise ValueError("Unknown aggregation convention")
    dimension = updates[0].packet.vector.size
    for update in updates:
        packet = update.packet
        if set(update.counts) != set(packet.masks) or set(packet.scales) != set(
            packet.masks
        ):
            raise ValueError("Every task requires one mask, scale, and sample count")
        if any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0
            for n in update.counts.values()
        ):
            raise ValueError("Task counts must be positive integers")
        if packet.vector.shape != (dimension,) or not np.isfinite(packet.vector).all():
            raise ValueError("Invalid client vector")
        if any(
            m.shape != (dimension,) or not np.isin(m, [0, 1]).all()
            for m in packet.masks.values()
        ):
            raise ValueError("Invalid task mask")
        if not np.isfinite(list(packet.scales.values())).all() or any(
            s < 0 for s in packet.scales.values()
        ):
            raise ValueError("Invalid task scale")
    tasks = sorted(set().union(*(u.counts for u in updates)))
    same, task_masks = {}, {}
    for task in tasks:
        clients = [u for u in updates if task in u.counts]
        signs = np.stack(
            [np.sign(u.packet.vector * u.packet.masks[task]) for u in clients]
        )
        alpha = signs.mean(axis=0)
        if agreement == "absolute":
            alpha = np.abs(alpha)
        task_masks[task] = np.where(alpha >= rho, 1.0, alpha)
        total = sum(u.counts[task] for u in clients)
        same[task] = (
            sum((u.counts[task] / total) * u.packet.reconstruct(task) for u in clients)
            * task_masks[task]
        )
    result = {}
    for task in tasks:
        related = []
        for other in tasks:
            if other == task:
                continue
            similarity = float(
                (np.sign(same[task]) @ np.sign(same[other]) / dimension + 1) / 2
            )
            if similarity > similarity_threshold:
                related.append((similarity, other))
        related.sort(key=lambda item: (-item[0], item[1]))
        neighbors = related[:top_k]
        cross = sum(
            (s * task_masks[task] * same[t] for s, t in neighbors), np.zeros(dimension)
        )
        value = same[task] + cross
        if combine == "mean" and neighbors:
            value *= 0.5
        result[task] = value.astype(np.float32)
    return result
