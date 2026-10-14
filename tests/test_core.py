import numpy as np
import pytest
from matu import core


def test_election_uses_maximum_aligned_magnitude_not_average():
    result = core.unify(
        {"a": np.array([2.0, -3.0, 1.0, 0.0]), "b": np.array([1.0, -1.0, -2.0, 0.0])}
    )
    np.testing.assert_array_equal(result.vector, [2.0, -3.0, -2.0, 0.0])
    np.testing.assert_array_equal(result.masks["a"], [1, 1, 0, 0])
    assert result.scales["a"] == pytest.approx(6 / 5)
    assert result.scales["b"] == pytest.approx(4 / 7)


def test_zero_and_tied_coordinates_remain_finite():
    result = core.unify({"a": np.array([1.0, 0.0]), "b": np.array([-1.0, 0.0])})
    np.testing.assert_array_equal(result.vector, [0.0, 0.0])
    assert result.scales == {"a": 0.0, "b": 0.0}
    assert np.isfinite(result.reconstruct("a")).all()


def test_one_task_round_preserves_consistent_negative_updates():
    a = core.ClientUpdate(core.unify({"a": np.array([-2.0, 4.0])}), {"a": 1})
    b = core.ClientUpdate(core.unify({"a": np.array([-4.0, 2.0])}), {"a": 3})
    result = core.aggregate([a, b], top_k=0)
    np.testing.assert_allclose(result["a"], [-3.5, 2.5])


def test_cross_task_threshold_excludes_conflicting_tasks_and_self():
    update = core.ClientUpdate(core.unify({"a": np.array([2.0, 3.0])}), {"a": 1})
    conflict = core.ClientUpdate(core.unify({"b": np.array([-2.0, -3.0])}), {"b": 1})
    result = core.aggregate([update, conflict], top_k=5)
    np.testing.assert_allclose(result["a"], [2.0, 3.0])
    np.testing.assert_allclose(result["b"], [-2.0, -3.0])


def test_payload_roundtrip_packs_binary_modulators():
    packet = core.unify({"long task name": np.arange(1.0, 18.0), "b": -np.arange(17.0)})
    payload = packet.to_bytes()
    restored = core.UnifiedVector.from_bytes(payload)
    np.testing.assert_array_equal(packet.vector, restored.vector)
    for task in packet.masks:
        np.testing.assert_array_equal(
            packet.reconstruct(task), restored.reconstruct(task)
        )
    assert len(payload) > 0


def test_rejects_nonfinite_vector_and_invalid_counts():
    with pytest.raises(ValueError):
        core.unify({"a": np.array([np.nan])})
    update = core.ClientUpdate(core.unify({"a": np.ones(2)}), {"a": 0})
    with pytest.raises(ValueError):
        core.aggregate([update])
