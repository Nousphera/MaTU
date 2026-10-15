"""Native Flower orchestration, deployed-model evaluation and checkpoints."""

from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

from .baselines import aggregate_fedavg, aggregate_matfl
from .config import validate
from .data import manifest_path, sha256_file
from .core import ClientUpdate, UnifiedVector, aggregate, unify
from .models import set_adapter
from .protocol import encode, decode, payload_bytes
from .training import runtime, evaluate

app = ServerApp()


def wait_for_nodes(grid, count, timeout):
    deadline = time.monotonic() + timeout
    while True:
        nodes = sorted(grid.get_node_ids())
        if len(nodes) > count or len(set(nodes)) != len(nodes):
            raise ValueError("Unexpected Flower node count")
        if len(nodes) == count:
            return nodes
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Only {len(nodes)}/{count} clients registered")
        time.sleep(0.25)


def replies(grid, messages, timeout):
    expected = {m.metadata.dst_node_id: m for m in messages}
    results = {}
    for reply in grid.send_and_receive(messages, timeout=timeout):
        node = reply.metadata.src_node_id
        if reply.has_error() or node not in expected or node in results:
            raise RuntimeError(
                "Expected one successful reply from each selected client"
            )
        if reply.metadata.group_id != expected[node].metadata.group_id:
            raise ValueError("Client round mismatch")
        results[node] = reply
    if set(results) != set(expected):
        raise RuntimeError("Missing replies; refusing partial aggregation")
    return list(results.values())


def evaluate_deployed(rt, vectors, *, references=None):
    """Macro-average tasks, then clients holding each task; decode actual MaTU packets."""
    cfg = rt.cfg
    by_task = {t: [] for t in rt.tasks}
    cache = {}
    for tasks in rt.ownership:
        packet = (
            unify({t: vectors[t] for t in tasks}) if cfg["method"] == "matu" else None
        )
        for t in tasks:
            key = (t, tuple(sorted(tasks))) if packet is not None else (t,)
            if key not in cache:
                vector = packet.reconstruct(t) if packet is not None else vectors[t]
                set_adapter(rt.model, rt.origin + vector)
                reference = (
                    rt.origin + references[t] if references is not None else None
                )
                cache[key] = evaluate(
                    rt.model, rt.heads[t], rt.tasks[t].test, cfg, reference=reference
                )
            by_task[t].append(cache[key])
    return summarize_evaluation(by_task)


def summarize_evaluation(by_task):
    tasks = {
        t: {
            "accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "loss": float(np.mean([r["loss"] for r in rows])),
            "test-examples": rows[0]["count"],
            "evaluated-clients": len(rows),
        }
        for t, rows in by_task.items()
    }
    return {
        "accuracy": float(np.mean([r["accuracy"] for r in tasks.values()])),
        "loss": float(np.mean([r["loss"] for r in tasks.values()])),
        "tasks": tasks,
    }


def run_server(grid, cfg):
    cfg = validate(cfg)
    serialized = json.dumps(cfg, sort_keys=True)
    rt = runtime(serialized)
    n = cfg["num_clients"]
    nodes = wait_for_nodes(grid, n, cfg["registration_timeout"])
    completed = [0] * n
    method = cfg["method"]
    dimension = int((~rt.private).sum()) if method == "fedper" else rt.origin.size
    vectors = {t: np.zeros(dimension, dtype=np.float32) for t in rt.tasks}

    def message(node, phase, rnd, cid=None, *, final=False):
        control = {"phase": phase, "round": rnd, "final": final}
        if cid is not None:
            control["completed"] = completed[cid]
        content = RecordDict({"config": ConfigRecord(control)})
        if phase != "identify":
            tasks = rt.ownership[cid]
            content["heads"] = ArrayRecord(rt.head_state(tasks))
            if phase != "probe":
                content["model"] = encode(
                    {t: vectors[t] for t in tasks}, compressed=method == "matu"
                )
        return Message(
            content,
            dst_node_id=node,
            message_type="evaluate" if phase == "evaluate" else "train",
            group_id=f"{phase}-{rnd}",
        )

    mapping = {}
    for reply in replies(
        grid, [message(node, "identify", 0) for node in nodes], cfg["reply_timeout"]
    ):
        cid = reply.content["metrics"]["client-id"]
        if type(cid) != int or cid in mapping or not 0 <= cid < n:
            raise ValueError("Client IDs must be unique and complete")
        mapping[cid] = reply.metadata.src_node_id

    def collect(messages, selected, rnd):
        result = []
        for reply in replies(grid, messages, cfg["reply_timeout"]):
            metric = reply.content["metrics"]
            cid = metric["client-id"]
            if (
                type(cid) != int
                or cid not in selected
                or mapping[cid] != reply.metadata.src_node_id
                or metric["round"] != rnd
            ):
                raise ValueError("Invalid client identity or round")
            result.append((cid, reply.content))
        return sorted(result, key=lambda row: row[0])

    def train_metrics(cid, content):
        metric = content["metrics"]
        counts = json.loads(metric["counts"])
        losses = json.loads(metric["losses"])
        if counts != rt.counts(cid) or any(type(v) != int for v in counts.values()):
            raise ValueError("Client sample counts do not match partitions")
        if set(losses) != set(counts) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in losses.values()
        ):
            raise ValueError("Invalid task losses")
        return counts, losses

    probe_history = []
    for rnd in range(1, cfg["head_rounds"] + 1):
        messages = [message(mapping[cid], "probe", rnd, cid) for cid in range(n)]
        updates = collect(messages, list(range(n)), rnd)
        weighted = {t: [] for t in rt.tasks}
        for cid, content in updates:
            counts, _ = train_metrics(cid, content)
            state = content["heads"].to_torch_state_dict()
            rt.load_heads(state, rt.ownership[cid])
            for t in rt.ownership[cid]:
                weighted[t].append(
                    (
                        counts[t],
                        {k: state[f"{t}/{k}"] for k in rt.heads[t].state_dict()},
                    )
                )
        for t, rows in weighted.items():
            total = sum(count for count, _ in rows)
            rt.heads[t].load_state_dict(
                {
                    k: sum(count / total * state[k] for count, state in rows)
                    for k in rows[0][1]
                }
            )
        probe_history.append(
            {
                "round": rnd,
                "downstream-bytes": sum(
                    m.content["heads"].count_bytes() for m in messages
                ),
                "upstream-bytes": sum(c["heads"].count_bytes() for _, c in updates),
            }
        )

    rng = np.random.default_rng(cfg["seed"])
    history = []
    for rnd in range(1, cfg["rounds"] + 1):
        selected = sorted(
            rng.choice(
                n, max(1, math.floor(n * cfg["participation"])), replace=False
            ).tolist()
        )
        before = {t: v.copy() for t, v in vectors.items()}
        messages = [message(mapping[cid], "train", rnd, cid) for cid in selected]
        updates = collect(messages, selected, rnd)
        flat, counts_by_key, tasks_by_key = {}, {}, {}
        compressed = []
        loss_total = count_total = 0
        for cid, content in updates:
            counts, losses = train_metrics(cid, content)
            if (
                type(content["metrics"]["completed"]) != int
                or content["metrics"]["completed"] != completed[cid] + 1
            ):
                raise ValueError("Client progress mismatch")
            decoded = decode(
                content["model"],
                rt.ownership[cid],
                dimension,
                compressed=method == "matu",
            )
            if method == "matu":
                compressed.append(
                    ClientUpdate(
                        UnifiedVector.from_bytes(content["model"]["packet"]), counts
                    )
                )
            for t, value in decoded.items():
                key = f"{cid}/{t}"
                flat[key], counts_by_key[key], tasks_by_key[key] = value, counts[t], t
                loss_total += losses[t] * counts[t]
                count_total += counts[t]
        if method == "matu":
            proposed = aggregate(
                compressed,
                top_k=cfg["top_k"],
                rho=cfg["rho"],
                similarity_threshold=cfg["similarity_threshold"],
            )
        elif method == "matfl":
            deltas = aggregate_matfl(
                flat, counts_by_key, tasks_by_key, rt.layers, top_k=cfg["matfl_top_k"]
            )
            proposed = {t: before[t] + value for t, value in deltas.items()}
        elif method == "centralized":
            proposed = {tasks_by_key[key]: value for key, value in flat.items()}
        else:
            merged = aggregate_fedavg(flat, counts_by_key)
            proposed = {t: merged.copy() for t in vectors}
        for t, value in proposed.items():
            vectors[t] = (
                before[t] + cfg["server_learning_rate"] * (value - before[t])
            ).astype(np.float32)
            if not np.isfinite(vectors[t]).all():
                raise FloatingPointError("Nonfinite aggregated task vector")
        for cid in selected:
            completed[cid] += 1
        row = {
            "round": rnd,
            "clients": selected,
            "train-loss": loss_total / count_total,
            "downstream-model-bytes": sum(
                payload_bytes(m.content["model"]) for m in messages
            ),
            "upstream-model-bytes": sum(payload_bytes(c["model"]) for _, c in updates),
            "downstream-head-bytes": sum(
                m.content["heads"].count_bytes() for m in messages
            ),
        }
        if rnd % cfg["eval_every"] == 0 or rnd == cfg["rounds"]:
            if method == "fedper":
                messages_eval = [
                    message(
                        mapping[cid], "evaluate", rnd, cid, final=rnd == cfg["rounds"]
                    )
                    for cid in range(n)
                ]
                by_task = {t: [] for t in rt.tasks}
                evaluation_replies = collect(messages_eval, list(range(n)), rnd)
                for cid, content in evaluation_replies:
                    metrics = json.loads(content["metrics"]["evaluation"])
                    if set(metrics) != set(rt.ownership[cid]):
                        raise ValueError("Invalid personalized task metrics")
                    for t, value in metrics.items():
                        count = len(rt.tasks[t].test)
                        if (
                            value["count"] != count
                            or type(value["correct"]) != int
                            or not 0 <= value["correct"] <= count
                            or not math.isfinite(value["loss"])
                            or value["loss"] < 0
                            or value["accuracy"] != value["correct"] / count
                        ):
                            raise ValueError("Invalid personalized evaluation")
                        by_task[t].append(value)
                row.update(summarize_evaluation(by_task))
                row["evaluation-model-bytes"] = sum(
                    payload_bytes(m.content["model"]) for m in messages_eval
                )
                row["evaluation-head-bytes"] = sum(
                    m.content["heads"].count_bytes() for m in messages_eval
                )
            else:
                row.update(
                    evaluate_deployed(
                        rt,
                        vectors,
                        references=before if method == "ntk-fedavg" else None,
                    )
                )
        history.append(row)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k in ("round", "accuracy", "loss", "train-loss")
                }
            ),
            flush=True,
        )
    checkpoint = {
        "vectors": {t: torch.from_numpy(v) for t, v in vectors.items()},
        "origin": torch.from_numpy(rt.origin),
        "heads": rt.head_state(),
        "config": cfg,
        "model": rt.model.pretrained_manifest,
    }
    if method == "ntk-fedavg":
        checkpoint["linearization_reference"] = {
            task: torch.from_numpy(rt.origin + value) for task, value in before.items()
        }
    result = {
        "schema": 1,
        "config": cfg,
        "tasks": {
            t: {
                "classes": d.classes,
                "metadata": {k: v for k, v in d.metadata.items() if k != "samples"},
                "manifest-sha256": sha256_file(manifest_path(t, cfg["data_root"])),
            }
            for t, d in rt.tasks.items()
        },
        "ownership": rt.ownership,
        "partitions": rt.partitions,
        "probing": probe_history,
        "rounds": history,
        "model": rt.model.pretrained_manifest,
        "evaluation": "Task macro accuracy; MaTU uses deployed client reconstruction; FedPer uses private client models.",
        "communication": "Model payload bytes, including packed MaTU schema/masks/scales; heads and probing reported separately; transport framing excluded.",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "flower": version("flwr"),
        },
    }
    return checkpoint, result


@app.main()
def main(grid: Grid, context: Context):
    cfg = validate(json.loads(context.run_config["config"]))
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists() or (output / "model.pt").exists():
        raise FileExistsError("Choose a new output directory")
    checkpoint, result = run_server(grid, cfg)
    with (output / "model.pt").open("xb") as stream:
        torch.save(checkpoint, stream)
    with (output / "metrics.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
