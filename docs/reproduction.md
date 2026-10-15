# Method and experiment settings

MaTU combines task-specific LoRA updates into one vector per client, accompanied by binary masks and scalar rescalers. The server aggregates related tasks and returns a unified vector tailored to each client's task set.

Reference: [Many-Task Federated Fine-Tuning via Unified Task Vectors](https://arxiv.org/abs/2502.06376).

## Paper-to-code map

| Component | Code | Operation |
| --- | --- | --- |
| Shared pretrained origin | `training.adapter_vector`, `set_adapter` | Adapter differences from a common deterministic initialization; frozen backbone |
| Unification and modulators, Eq. 2 | `core.unify` | Sign of vector sum, maximum aligned magnitude, positive-alignment mask, L1-ratio scale |
| Agreement, Eq. 3 | `core.aggregate` | Absolute mean signs by default; signed mode available |
| Same-task aggregation, Eq. 4 | `core.aggregate` | Sample-weighted task vectors with an agreement mask; scale applied once |
| Task similarity, Eq. 5 | `core.aggregate` | `(mean(sign(a) * sign(b)) + 1) / 2`; zero coordinates contribute 0.5 |
| Cross-task transfer, Eq. 6 | `core.aggregate` | Exclude self, select similarity above 0.5 and top-κ neighbors; unnormalized weights |
| Combination, Eq. 7 | `core.aggregate` | Sum of same-task and cross-task terms by default; mean mode available |
| Client redistribution | `training.run`, `flower.main` | Unify the tasks assigned to each selected client |

## Aggregation settings

- **Agreement:** `agreement="absolute"` uses the magnitude of mean signs, so consistent positive and negative updates both receive agreement one. `agreement="signed"` implements the signed expression printed in Eq. 3; negative agreement can reverse an update's direction.
- **Combination:** `combine="sum"` follows Eq. 7. `combine="mean"` averages the same-task and cross-task terms when related tasks exist.
- **Related tasks:** `top_k=2` selects at most two neighbors above the similarity threshold. Set it explicitly for each experiment.
- **Zero coordinates:** exact sign ties produce zero; an entirely erased task receives scale zero. Unification is a lossy representation.

## Model and task configuration

The vision backend uses a timm ViT with rank-`r` qkv LoRA adapters and unit LoRA scale. Federated linear probing first trains shared task classifiers; those heads and the backbone are then frozen while the adapters are optimized.

Client class partitions use seeded Dirichlet sampling with repair of empty partitions. Set `assignments` to declare each client's tasks. The default assigns every task to every client; use explicit task ownership for heterogeneous studies. Keep task ownership, class partitions, optimizer, augmentation and adapter settings with the dataset manifests so comparisons use the same experimental conditions.

See [dataset preparation](datasets.md) for official splits, CSV imports, task manifests and image configurations. Test examples are excluded from optimization and checkpoint selection.

## Execution and outputs

The native Flower App and sequential runner share the same training and aggregation implementation. Follow the [README](../README.md#training) for `flwr run`; use the sequential runner when debugging locally:

```bash
matu --config configs/smoke.json --output runs/sequential-01
python -m pytest
```

Flower 1.40 accepts flat federation overrides, such as `num-supernodes=3 client-resources-num-cpus=1 init-args-num-cpus=2`. Confirm `finished:completed` and the expected round count in `metrics.json`; the CLI exit status alone does not identify all worker failures.

| Artifact | Contents |
| --- | --- |
| `config.json` | Effective model, task, optimizer, seed and execution settings |
| `partitions.json` | Exact client training indices |
| `metrics.json` | Per-task accuracy, round count, adapter change, frozen-backbone hash and communication |
| `final.pt` | Adapter origin, task vectors, task heads and configuration |

Communication counts include serialized records and payloads; head-probing traffic is separate. TCP/TLS framing is outside these counters. Restore adapters and heads with the matching pretrained timm backbone. Archive its checkpoint and SHA-256 alongside the environment snapshot in [`requirements-tested-macos-py311.txt`](../requirements-tested-macos-py311.txt).

## Tests

The suite covers task modulation, aggregation conventions, payload round trips, dataset integrity, numerical optimization, frozen-backbone invariants, timm forward/backward execution and Flower transport. Synthetic and small image fixtures provide fast end-to-end checks before running prepared datasets.
