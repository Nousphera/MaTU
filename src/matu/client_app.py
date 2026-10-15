"""Persistent multi-task Flower clients with local task heads and LoRA training."""

import json
from pathlib import Path
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from .config import validate
from .models import adapter_vector, set_adapter
from .protocol import encode, decode
from .training import (
    runtime,
    train_local,
    evaluate,
    optimizer_bytes,
    optimizer_from_bytes,
)

app = ClientApp()


def handle(msg, context):
    cfg = validate(json.loads(context.run_config["config"]))
    cid = context.node_config["partition-id"]
    if type(cid) != int or not 0 <= cid < cfg["num_clients"]:
        raise ValueError("Invalid client partition ID")
    control = msg.content["config"]
    phase, rnd = control["phase"], control["round"]
    if phase == "identify":
        return RecordDict({"metrics": ConfigRecord({"client-id": cid})})
    if phase not in ("probe", "train", "evaluate") or type(rnd) != int or rnd < 1:
        raise ValueError("Invalid phase or round")
    rt = runtime(json.dumps(cfg, sort_keys=True))
    tasks = rt.ownership[cid]
    counts = rt.counts(cid)
    heads = msg.content["heads"].to_torch_state_dict()
    rt.load_heads(heads, tasks)
    method = cfg["method"]
    private = method == "fedper" and phase != "probe"
    if private and "private-heads" in context.state:
        rt.load_heads(context.state["private-heads"].to_torch_state_dict(), tasks)
    if phase == "probe":
        if (
            rnd > cfg["head_rounds"]
            or rnd
            != context.state.get("progress-probe", ConfigRecord({"round": 0}))["round"]
            + 1
        ):
            raise ValueError("Invalid head-probing sequence")
        losses = {}
        for index, t in enumerate(tasks):
            set_adapter(rt.model, rt.origin)
            metrics = train_local(
                rt.model,
                rt.heads[t],
                rt.train_data(cid, t),
                cfg,
                seed=(cfg["seed"] + rnd * 100003 + cid * 1009 + index) % 2**32,
                probe=True,
            )
            losses[t] = metrics["loss"]
        context.state["progress-probe"] = ConfigRecord({"round": rnd})
        return RecordDict(
            {
                "heads": ArrayRecord(rt.head_state(tasks)),
                "metrics": ConfigRecord(
                    {
                        "client-id": cid,
                        "round": rnd,
                        "counts": json.dumps(counts),
                        "losses": json.dumps(losses),
                    }
                ),
            }
        )
    dimension = int((~rt.private).sum()) if private else rt.origin.size
    incoming = decode(
        msg.content["model"], tasks, dimension, compressed=method == "matu"
    )
    if private:
        personal = (
            context.state.get(
                "private", ArrayRecord({"adapter": torch.zeros(int(rt.private.sum()))})
            )
            .to_torch_state_dict()["adapter"]
            .numpy()
        )
        # Expand only the shared coordinates; personalized values never leave the client.
        for t in tasks:
            shared = incoming[t]
            full = np.zeros_like(rt.origin)
            full[~rt.private] = shared
            full[rt.private] = personal
            incoming[t] = full
    if phase == "evaluate":
        metrics = {}
        for t in tasks:
            set_adapter(rt.model, rt.origin + incoming[t])
            metrics[t] = evaluate(rt.model, rt.heads[t], rt.tasks[t].test, cfg)
        if control.get("final", False) and private:
            path = Path(cfg["output"]) / "clients" / f"client-{cid}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                torch.save(
                    {
                        "adapter": torch.from_numpy(personal.copy()),
                        "heads": rt.head_state(tasks),
                        "tasks": tasks,
                        "client": cid,
                    },
                    stream,
                )
        return RecordDict(
            {
                "metrics": ConfigRecord(
                    {"client-id": cid, "round": rnd, "evaluation": json.dumps(metrics)}
                )
            }
        )
    progress = context.state.get(
        "progress-train", ConfigRecord({"round": 0, "completed": 0})
    )
    if (
        rnd > cfg["rounds"]
        or rnd <= progress["round"]
        or control["completed"] != progress["completed"]
    ):
        raise ValueError("Duplicate, stale or inconsistent client update")
    vectors, losses = {}, {}
    for index, t in enumerate(tasks):
        if private:
            incoming[t][rt.private] = personal
        absolute = rt.origin + incoming[t]
        set_adapter(rt.model, absolute)
        key = f"optimizer-{t}"
        optimizer = (
            optimizer_from_bytes(context.state[key]["value"])
            if method == "centralized" and key in context.state
            else None
        )
        result = train_local(
            rt.model,
            rt.heads[t],
            rt.train_data(cid, t),
            cfg,
            seed=(cfg["seed"] + rnd * 100003 + cid * 1009 + index) % 2**32,
            reference=absolute,
            optimizer_state=optimizer,
        )
        trained = adapter_vector(rt.model) - rt.origin
        if private:
            personal = trained[rt.private].copy()
            vectors[t] = trained[~rt.private].copy()
        else:
            vectors[t] = trained - incoming[t] if method == "matfl" else trained
        if method == "centralized":
            context.state[key] = ConfigRecord(
                {"value": optimizer_bytes(result["optimizer"])}
            )
        losses[t] = result["loss"]
    if private:
        context.state["private"] = ArrayRecord({"adapter": torch.from_numpy(personal)})
        context.state["private-heads"] = ArrayRecord(rt.head_state(tasks))
    context.state["progress-train"] = ConfigRecord(
        {"round": rnd, "completed": progress["completed"] + 1}
    )
    return RecordDict(
        {
            "model": encode(vectors, compressed=method == "matu"),
            "metrics": ConfigRecord(
                {
                    "client-id": cid,
                    "round": rnd,
                    "completed": progress["completed"] + 1,
                    "counts": json.dumps(counts),
                    "losses": json.dumps(losses),
                }
            ),
        }
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    return Message(handle(msg, context), reply_to=msg)


@app.evaluate()
def evaluation(msg: Message, context: Context) -> Message:
    return Message(handle(msg, context), reply_to=msg)
