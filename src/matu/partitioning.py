"""Seeded task ownership and disjoint class-Dirichlet data partitions."""

import math
import numpy as np


def assign_tasks(
    names, num_clients, alpha, seed, *, tasks_per_client=0, assignments=None
):
    names = sorted(names)
    if (
        not names
        or len(set(names)) != len(names)
        or num_clients < 1
        or not math.isfinite(alpha)
        or alpha < 0
        or tasks_per_client < 0
    ):
        raise ValueError("Invalid task assignment settings")
    if assignments:
        if (
            len(assignments) != num_clients
            or any(
                not isinstance(row, list) or not row or len(row) != len(set(row))
                for row in assignments
            )
            or set().union(*map(set, assignments)) != set(names)
        ):
            raise ValueError(
                "Explicit assignments must cover all tasks with nonempty unique lists"
            )
        return [list(row) for row in assignments]
    rng = np.random.default_rng(seed)
    if tasks_per_client or alpha == 0:
        k = tasks_per_client or 1
        if k > len(names) or num_clients * k < len(names):
            raise ValueError(
                "Not enough client task slots to cover the requested datasets"
            )
        order = rng.permutation(names).tolist()
        rows = [
            sorted(order[(cid * k + j) % len(order)] for j in range(k))
            for cid in range(num_clients)
        ]
        return [rows[i] for i in rng.permutation(num_clients)]
    rows = [[] for _ in range(num_clients)]
    for name in names:
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        selected = np.flatnonzero(proportions > 0.97 / num_clients)
        if not len(selected):
            selected = np.array([int(proportions.argmax())])
        for cid in selected:
            rows[int(cid)].append(name)
    # Repair empty clients by adding an owner, without removing a task's only owner.
    for row in rows:
        if not row:
            counts = {name: sum(name in items for items in rows) for name in names}
            choices = [name for name in names if counts[name] == min(counts.values())]
            row.append(str(rng.choice(choices)))
    return [sorted(row) for row in rows]


def build_partitions(labels, ownership, alpha, seed):
    if not math.isfinite(alpha) or alpha <= 0 or not ownership:
        raise ValueError("Class concentration and client count must be positive")
    if any(not row or len(row) != len(set(row)) for row in ownership) or set().union(
        *map(set, ownership)
    ) != set(labels):
        raise ValueError(
            "Every client and task must be represented exactly by its ownership list"
        )
    output = [{} for _ in ownership]
    for task_index, name in enumerate(sorted(labels)):
        values = np.asarray(labels[name])
        owners = [cid for cid, row in enumerate(ownership) if name in row]
        if (
            values.ndim != 1
            or len(values) < len(owners)
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError(
                f"{name}: insufficient or invalid labels for client allocation"
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, task_index]))
        parts = [[] for _ in owners]
        for label in np.unique(values):
            indices = np.flatnonzero(values == label)
            rng.shuffle(indices)
            counts = rng.multinomial(
                len(indices), rng.dirichlet(np.full(len(owners), alpha))
            )
            for part, piece in zip(parts, np.split(indices, np.cumsum(counts)[:-1])):
                part.extend(piece.tolist())
        for part in parts:
            if not part:
                donor = max(parts, key=len)
                part.append(donor.pop())
        for cid, part in zip(owners, parts):
            output[cid][name] = sorted(part)
    return output
