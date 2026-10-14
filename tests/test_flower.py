import pytest

pytest.importorskip(
    "flwr", reason="Install the flower extra for transport integration tests"
)

import json
import numpy as np
from flwr.app import Context, Message, RecordDict, ConfigRecord
from flwr.supercore.task_identity import TaskIdentity
from matu import flower
from matu.core import unify, UnifiedVector
from matu.training import DEFAULTS, Experiment


def test_flower_client_trains_real_adapter_from_wire_message():
    TaskIdentity.run_id, TaskIdentity.node_id, TaskIdentity.task_id = 0, 0, 0
    cfg = dict(DEFAULTS, clients=2, tasks=2, samples=16, head_rounds=0)
    experiment = Experiment(cfg)
    packet = unify(experiment.global_vectors)
    request = Message(
        content=RecordDict(
            {
                "config": ConfigRecord(
                    {
                        "config": json.dumps(cfg),
                        "cid": 0,
                        "round": 1,
                        "phase": "train",
                        "packet": packet.to_bytes(),
                    }
                ),
                "heads": flower.head_record(experiment.heads),
            }
        ),
        dst_node_id=10,
        message_type="train",
    )
    ctx = Context(
        run_id=0, node_id=10, node_config={}, state=RecordDict(), run_config={}
    )
    reply = flower.train(request, ctx)
    update = UnifiedVector.from_bytes(reply.content["result"]["packet"])
    assert np.linalg.norm(update.vector) > 0
    assert json.loads(reply.content["result"]["counts"]) == {
        t: len(d) for t, d in experiment.clients[0].items()
    }


def test_transport_size_counts_arrays_and_metadata():
    from flwr.app import ArrayRecord

    content = RecordDict(
        {
            "array": ArrayRecord([np.ones(4, dtype=np.float32)]),
            "config": ConfigRecord({"round": 1}),
        }
    )
    assert flower.content_bytes(content) > 16


def test_duplicate_reply_cannot_replace_missing_participant():
    from types import SimpleNamespace
    import pytest

    reply = lambda node: SimpleNamespace(
        metadata=SimpleNamespace(src_node_id=node), has_error=lambda: False
    )
    with pytest.raises(RuntimeError):
        flower.validate_replies([reply(1), reply(1)], {1, 2})
    flower.validate_replies([reply(2), reply(1)], {1, 2})
