"""Pure aggregation for FedAvg and the published MaT-FL layer-voting method."""

from collections.abc import Mapping

import numpy as np

from .core import _counts, _integer, _name, _vector, _vectors


def aggregate_fedavg(
    vectors: dict[str, np.ndarray], counts: dict[str, int]
) -> np.ndarray:
    """One sample-weighted adapter across all client-task outputs, without masks."""
    arrays = _vectors(vectors, identifiers=True)
    counts = _counts(counts, arrays)
    total = sum(counts.values())
    return _vector(
        sum(
            (counts[name] / total) * value.astype(np.float64)
            for name, value in arrays.items()
        )
    )


def aggregate_matfl(
    vectors: dict[str, np.ndarray],
    counts: dict[str, int],
    tasks: dict[str, str],
    layers: list[slice],
    top_k: int,
) -> dict[str, np.ndarray]:
    """Layer-ranked friend voting, adaptive aggregation, then task aggregation.

    Reference: Cai et al., CVPRW 2023, Algorithm 1 and Eq. 5–6. `vectors`
    are round-local adapter deltas (trained minus received), NOT full model
    weights. Add each returned task delta to that task's pre-round state.
    A physical client's separate tasks are virtual client-task participants;
    use only this round's selected participants, not cached inactive deltas.

    Explicit ViT/many-task conventions: caller-specified nonoverlapping layer
    slices; self-inclusive top-k; deterministic self-first/name tie breaks;
    nonnegative cosine weights with a self fallback for zero rows; and sample-
    weighted within-task averaging. The paper does not specify zero/negative
    normalization or these multi-task/unequal-data details. This implements
    layer voting, not the official repo's different flattened clustering.
    """
    arrays = _vectors(vectors, identifiers=True)
    counts = _counts(counts, arrays)
    if not isinstance(tasks, Mapping) or set(tasks) != set(arrays):
        raise ValueError("Every update needs exactly one task identity")
    for task in tasks.values():
        _name(task)
    names = list(arrays)
    values = np.stack(list(arrays.values())).astype(np.float64)
    n, dimension = values.shape
    k = min(_integer(top_k), n)
    if not isinstance(layers, list) or not layers:
        raise ValueError("At least one layer slice is required")
    occupied = np.zeros(dimension, dtype=bool)
    for layer in layers:
        if not isinstance(layer, slice) or layer.step not in (None, 1):
            raise ValueError("Layers must be contiguous slices")
        start = _integer(layer.start, minimum=0)
        stop = _integer(layer.stop)
        if start >= stop or stop > dimension or occupied[start:stop].any():
            raise ValueError(
                "Layer slices must be nonempty, in bounds, and nonoverlapping"
            )
        occupied[start:stop] = True
    votes = np.zeros((n, n), dtype=np.int64)
    averaged_weights = np.zeros((n, n), dtype=np.float64)
    for layer in layers:
        block = values[:, layer]
        norms = np.linalg.norm(block, axis=1)
        denominator = norms[:, None] * norms[None, :]
        similarity = np.divide(
            block @ block.T, denominator, out=np.zeros((n, n)), where=denominator > 0
        )
        similarity = np.clip(similarity, -1.0, 1.0)
        np.fill_diagonal(similarity, 1.0)
        for i in range(n):
            ranked = sorted(
                range(n), key=lambda j: (-similarity[i, j], j != i, names[j])
            )
            for rank, j in enumerate(ranked[:k]):
                votes[i, j] += k - rank
        positive = np.maximum(similarity, 0.0)
        averaged_weights += positive / positive.sum(axis=1, keepdims=True) / len(layers)
    personalized = {}
    for i, name in enumerate(names):
        friends = sorted(range(n), key=lambda j: (-votes[i, j], j != i, names[j]))[:k]
        weights = averaged_weights[i, friends]
        total = weights.sum()
        if total <= 0:
            personalized[name] = values[i].copy()
        else:
            personalized[name] = (weights / total) @ values[friends]
    result = {}
    for task in sorted(set(tasks.values())):
        holders = [name for name in names if tasks[name] == task]
        total = sum(counts[name] for name in holders)
        result[task] = _vector(
            sum((counts[name] / total) * personalized[name] for name in holders)
        )
    return result
