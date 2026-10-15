"""MaTU equations 2–7 and its portable unified-vector packet.

The rendered Eq. 3 uses absolute mean signs. Eq. 7 adds the same-task and
cross-task terms without normalizing similarity weights. Tensor coordinates
must refer to one common, immutable model/adapter origin.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import numbers
import unicodedata
import zipfile

import numpy as np


def _name(value, *, identifier=False):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Names must be nonempty strings without outer whitespace")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("Names must be valid UTF-8") from error
    if (
        len(encoded) > 256
        or any(unicodedata.category(c).startswith("C") for c in value)
        or (not identifier and (value in {".", ".."} or "/" in value or "\\" in value))
    ):
        raise ValueError("Invalid task or update name")
    return value


def _integer(value, *, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise ValueError(f"Expected an integer >= {minimum}")
    return int(value)


def _real(value, *, minimum=0.0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError("Expected a finite real scalar")
    value = float(value)
    if (
        not np.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError("Scalar is outside the permitted finite range")
    return value


def _vector(value):
    array = np.asarray(value)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "fiu":
        raise ValueError("Vectors must be nonempty one-dimensional real arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("Vector values must be finite and representable as float32")
    return array


def _vectors(values, *, identifiers=False):
    if not isinstance(values, Mapping) or not values:
        raise ValueError("At least one named vector is required")
    for name in values:
        _name(name, identifier=identifiers)
    arrays = {name: _vector(values[name]) for name in sorted(values)}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("All vectors must have the same dimension")
    return arrays


def _counts(values, keys):
    if not isinstance(values, Mapping) or set(values) != set(keys):
        raise ValueError("Every vector/task requires exactly one sample count")
    return {name: _integer(values[name]) for name in sorted(keys)}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON metadata key")
        result[key] = value
    return result


@dataclass(frozen=True)
class UnifiedVector:
    vector: np.ndarray
    masks: dict[str, np.ndarray]
    scales: dict[str, float]

    def __post_init__(self):
        vector, masks, scales = self._validated()
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "scales", scales)

    def _validated(self):
        vector = _vector(self.vector)
        if (
            not isinstance(self.masks, Mapping)
            or not self.masks
            or not isinstance(self.scales, Mapping)
        ):
            raise ValueError("A packet needs task masks and scales")
        if set(self.masks) != set(self.scales):
            raise ValueError("Task mask and scale keys must match")
        for task in self.masks:
            _name(task)
        masks, scales = {}, {}
        for task in sorted(self.masks):
            mask = np.asarray(self.masks[task])
            if (
                mask.shape != vector.shape
                or mask.dtype.kind not in "biu"
                or not np.isin(mask, [0, 1]).all()
            ):
                raise ValueError(
                    "Task masks must be binary vectors matching the unified vector"
                )
            masks[task] = mask.astype(bool, copy=True)
            scales[task] = _real(self.scales[task])
        return vector, masks, scales

    def reconstruct(self, task: str) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            values = (
                self.vector.astype(np.float64) * self.masks[task] * self.scales[task]
            )
        return _vector(values)

    def to_bytes(self) -> bytes:
        """Encode float32 coordinates, bit-packed masks, and strict JSON metadata."""
        vector, masks, scales = self._validated()
        tasks = list(masks)
        metadata = json.dumps(
            {
                "version": 1,
                "dimension": vector.size,
                "tasks": tasks,
                "scales": [scales[t] for t in tasks],
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        arrays = {
            "vector": vector.astype("<f4"),
            "metadata": np.frombuffer(metadata.encode("utf-8"), dtype=np.uint8),
        }
        arrays.update(
            {
                f"mask_{i}": np.packbits(masks[t], bitorder="little")
                for i, t in enumerate(tasks)
            }
        )
        stream = io.BytesIO()
        np.savez(stream, **arrays)
        return stream.getvalue()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "UnifiedVector":
        """Reject malformed, duplicate, extra, nonfinite, or noncanonical fields."""
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("Packet must be nonempty bytes")
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as container:
                names = container.namelist()
                if len(names) != len(set(names)) or any(
                    "/" in n or not n.endswith(".npy") for n in names
                ):
                    raise ValueError("Invalid or duplicate NPZ members")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if "vector" not in archive.files or "metadata" not in archive.files:
                    raise ValueError("Packet is missing required arrays")
                vector, raw = archive["vector"], archive["metadata"]
                if (
                    vector.dtype != np.dtype("<f4")
                    or vector.ndim != 1
                    or not vector.size
                ):
                    raise ValueError("Packet vector must be a nonempty float32 vector")
                if raw.dtype != np.uint8 or raw.ndim != 1 or not raw.size:
                    raise ValueError("Metadata must be a UTF-8 byte vector")
                metadata = json.loads(
                    raw.tobytes().decode("utf-8"), object_pairs_hook=_unique_object
                )
                if not isinstance(metadata, dict) or set(metadata) != {
                    "version",
                    "dimension",
                    "tasks",
                    "scales",
                }:
                    raise ValueError("Unsupported packet metadata schema")
                if (
                    _integer(metadata["version"]) != 1
                    or _integer(metadata["dimension"]) != vector.size
                ):
                    raise ValueError("Unsupported packet version or dimension")
                tasks, scales = metadata["tasks"], metadata["scales"]
                if (
                    not isinstance(tasks, list)
                    or not tasks
                    or not isinstance(scales, list)
                    or len(tasks) != len(scales)
                ):
                    raise ValueError("Invalid task metadata")
                for task in tasks:
                    _name(task)
                if len(tasks) != len(set(tasks)):
                    raise ValueError("Duplicate packet task")
                expected = {
                    "vector",
                    "metadata",
                    *(f"mask_{i}" for i in range(len(tasks))),
                }
                if set(archive.files) != expected or len(archive.files) != len(
                    expected
                ):
                    raise ValueError("Unexpected or missing packet array")
                masks = {}
                for i, task in enumerate(tasks):
                    packed = archive[f"mask_{i}"]
                    if packed.dtype != np.uint8 or packed.shape != (
                        (vector.size + 7) // 8,
                    ):
                        raise ValueError("Invalid packed mask shape or type")
                    remainder = vector.size % 8
                    if remainder and int(packed[-1]) >> remainder:
                        raise ValueError("Unused mask padding bits must be zero")
                    masks[task] = np.unpackbits(
                        packed, bitorder="little", count=vector.size
                    ).astype(bool)
                return cls(vector, masks, dict(zip(tasks, scales)))
        except (
            KeyError,
            TypeError,
            UnicodeError,
            OSError,
            EOFError,
            zipfile.BadZipFile,
        ) as error:
            raise ValueError("Invalid unified-vector packet") from error


@dataclass(frozen=True)
class ClientUpdate:
    packet: UnifiedVector
    counts: dict[str, int]

    def __post_init__(self):
        if not isinstance(self.packet, UnifiedVector):
            raise ValueError("Expected a unified-vector packet")
        object.__setattr__(self, "counts", _counts(self.counts, self.packet.masks))


def unify(vectors: Mapping[str, np.ndarray]) -> UnifiedVector:
    """Elect maximum aligned magnitudes, then derive masks and global L1 scales.

    Exact sign ties and entirely erased tasks reconstruct as zero. Sign equality
    implements the positive-product mask without underflow on tiny coordinates.
    """
    arrays = _vectors(vectors)
    stacked = np.stack(list(arrays.values()))
    sign = np.sign(stacked.sum(axis=0, dtype=np.float64))
    aligned = np.sign(stacked) == sign
    vector = (sign * np.where(aligned, np.abs(stacked), 0).max(axis=0)).astype(
        np.float32
    )
    masks, scales = {}, {}
    for task, original in arrays.items():
        mask = (np.sign(original) == sign) & (sign != 0)
        denominator = np.abs(vector[mask]).sum(dtype=np.float64)
        masks[task] = mask
        scales[task] = (
            float(np.abs(original).sum(dtype=np.float64) / denominator)
            if denominator
            else 0.0
        )
    return UnifiedVector(vector, masks, scales)


def _update_key(update):
    """Stable reduction order without requiring client identity in this pure API."""
    digest = hashlib.sha256(update.packet.vector.tobytes())
    for task in sorted(update.counts):
        digest.update(
            json.dumps(
                [task, update.counts[task], update.packet.scales[task]],
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(
            np.packbits(update.packet.masks[task], bitorder="little").tobytes()
        )
    return digest.digest()


def aggregate(
    updates: list[ClientUpdate],
    *,
    top_k: int = 2,
    rho: float = 0.4,
    similarity_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Aggregate current-round packets with absolute agreement and Eq. 7's sum.

    No self-neighbor or stale-task donor is included. Same-task sample weights
    and unweighted sign agreement are distinct. Similarity ties use task names.
    The caller retains previous state for tasks absent from this round.
    """
    top_k = _integer(top_k, minimum=0)
    rho = _real(rho, maximum=1)
    similarity_threshold = _real(similarity_threshold, maximum=1)
    if not updates:
        raise ValueError("At least one client update is required")
    dimensions = set()
    for update in updates:
        if not isinstance(update, ClientUpdate):
            raise ValueError("Expected ClientUpdate values")
        vector, _, _ = update.packet._validated()
        _counts(update.counts, update.packet.masks)
        dimensions.add(vector.size)
    if len(dimensions) != 1:
        raise ValueError("Client vector dimensions differ")
    dimension = dimensions.pop()
    updates = sorted(updates, key=_update_key)
    tasks = sorted(set().union(*(update.counts for update in updates)))
    same, masks = {}, {}
    for task in tasks:
        holders = [update for update in updates if task in update.counts]
        signs = np.stack(
            [np.sign(u.packet.vector) * u.packet.masks[task] for u in holders]
        )
        agreement = np.abs(signs.mean(axis=0, dtype=np.float64))
        masks[task] = np.where(agreement >= rho, 1.0, agreement)
        total = sum(u.counts[task] for u in holders)
        average = sum(
            (u.counts[task] / total) * u.packet.reconstruct(task).astype(np.float64)
            for u in holders
        )
        same[task] = average * masks[task]
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
        cross = sum(
            (score * masks[task] * same[other] for score, other in related[:top_k]),
            np.zeros(dimension, dtype=np.float64),
        )
        result[task] = _vector(same[task] + cross)
    return result
