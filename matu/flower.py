"""Flower 1.40 Message API: task-head probing followed by MaTU rounds."""

import copy
import json
import math

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp

from .core import ClientUpdate, UnifiedVector, unify
from .training import (
    DEFAULTS,
    Experiment,
    frozen_hash,
    train_local,
    check_output_directory,
)

client_app = ClientApp()
server_app = ServerApp()


def content_bytes(content):
    """Count serialized record payloads, not Python object allocation."""
    return sum(record.count_bytes() for record in content.values())


def validate_replies(replies, expected_nodes):
    actual = [reply.metadata.src_node_id for reply in replies]
    if (
        len(actual) != len(expected_nodes)
        or set(actual) != set(expected_nodes)
        or any(reply.has_error() for reply in replies)
    ):
        raise RuntimeError(
            "Each selected client must supply exactly one successful reply"
        )


def head_record(heads):
    return ArrayRecord(
        {
            f"{task}/{name}": value.detach().cpu()
            for task, head in heads.items()
            for name, value in head.state_dict().items()
        }
    )


def load_heads(heads, record):
    state = record.to_torch_state_dict()
    for task, head in heads.items():
        head.load_state_dict(
            {name: state[f"{task}/{name}"] for name in head.state_dict()}
        )


@client_app.train()
def train(message, context):
    settings = message.content["config"]
    cfg = json.loads(settings["config"])
    experiment = Experiment(cfg)
    cid, rnd = int(settings["cid"]), int(settings["round"])
    if not 0 <= cid < len(experiment.clients):
        raise ValueError("Client assignment out of range")
    load_heads(experiment.heads, message.content["heads"])
    counts = {t: len(d) for t, d in experiment.clients[cid].items()}
    if settings["phase"] == "probe":
        heads = {}
        for task, data in experiment.clients[cid].items():
            head = copy.deepcopy(experiment.heads[task])
            train_local(
                experiment.model,
                head,
                data,
                cfg,
                seed=cfg["seed"] + rnd * cfg["clients"] + cid,
                probe=True,
            )
            heads[task] = head
        content = RecordDict(
            {
                "heads": head_record(heads),
                "result": ConfigRecord({"counts": json.dumps(counts)}),
            }
        )
    elif settings["phase"] == "train":
        packet = UnifiedVector.from_bytes(settings["packet"])
        update = experiment.client_fit(cid, packet, rnd)
        content = RecordDict(
            {
                "result": ConfigRecord(
                    {
                        "packet": update.packet.to_bytes(),
                        "counts": json.dumps(update.counts),
                    }
                )
            }
        )
    else:
        raise ValueError("Unknown training phase")
    return Message(content=content, reply_to=message)


@server_app.main()
def main(grid, context):
    return run_grid(grid, context)


def run_grid(grid, context, config_override=None):
    from pathlib import Path

    supplied = (
        config_override
        if config_override is not None
        else json.loads(Path(context.run_config["config"]).read_text())
    )
    cfg = {**DEFAULTS, **supplied}
    check_output_directory(cfg)
    experiment = Experiment(cfg)
    frozen = frozen_hash(experiment.model)
    nodes = sorted(grid.get_node_ids())
    if len(nodes) != cfg["clients"]:
        raise ValueError(
            f"Config requires {cfg['clients']} clients, Flower started {len(nodes)}"
        )
    rng = np.random.default_rng(cfg["seed"])

    def exchange(chosen, rnd, phase):
        messages = []
        for cid in chosen:
            settings = {
                "config": json.dumps(cfg),
                "cid": int(cid),
                "round": rnd,
                "phase": phase,
            }
            if phase == "train":
                packet = unify(
                    {t: experiment.global_vectors[t] for t in experiment.clients[cid]}
                )
                settings["packet"] = packet.to_bytes()
            content = RecordDict(
                {
                    "config": ConfigRecord(settings),
                    "heads": head_record(experiment.heads),
                }
            )
            if phase == "train":
                experiment.download_bytes += content_bytes(content)
            else:
                experiment.probe_bytes += content_bytes(content)
            messages.append(
                Message(content=content, dst_node_id=nodes[cid], message_type="train")
            )
        replies = list(grid.send_and_receive(messages))
        validate_replies(replies, {nodes[cid] for cid in chosen})
        return replies

    for rnd in range(cfg["head_rounds"]):
        replies = exchange(range(cfg["clients"]), rnd, "probe")
        responses = [
            (
                reply.content["heads"].to_torch_state_dict(),
                json.loads(reply.content["result"]["counts"]),
            )
            for reply in replies
        ]
        experiment.probe_bytes += sum(content_bytes(reply.content) for reply in replies)
        for task, head in experiment.heads.items():
            subset = [
                (state, counts[task]) for state, counts in responses if task in counts
            ]
            total = sum(n for _, n in subset)
            head.load_state_dict(
                {
                    k: sum(state[f"{task}/{k}"] * n / total for state, n in subset)
                    for k in head.state_dict()
                }
            )
    for rnd in range(1, cfg["rounds"] + 1):
        selected = sorted(
            rng.choice(
                cfg["clients"],
                max(1, math.ceil(cfg["clients"] * cfg["participation"])),
                replace=False,
            )
        )
        replies = exchange(selected, rnd, "train")
        updates = [
            ClientUpdate(
                UnifiedVector.from_bytes(reply.content["result"]["packet"]),
                json.loads(reply.content["result"]["counts"]),
            )
            for reply in replies
        ]
        experiment.upload_bytes += sum(
            content_bytes(reply.content) for reply in replies
        )
        experiment.apply(updates, rnd)
    result = experiment.save(frozen)
    print(json.dumps(result, indent=2))


def simulate(config):
    """Run this exact Message API app with Flower's local Ray backend."""
    from flwr.simulation import run_simulation

    application = ServerApp()

    @application.main()
    def configured_main(grid, context):
        run_grid(grid, context, config)

    run_simulation(
        server_app=application,
        client_app=client_app,
        num_supernodes=config["clients"],
        backend_config={
            "init_args": {
                "num_cpus": min(2, config["clients"]),
                "include_dashboard": False,
            },
            "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
        },
    )
