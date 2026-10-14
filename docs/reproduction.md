# Reproduction and paper correspondence

Reference: [arXiv:2502.06376v3](https://arxiv.org/pdf/2502.06376v3), 8 July 2025.

| Paper | Code | Interpretation |
| --- | --- | --- |
| Shared pretrained origin | `training.adapter_vector`, `set_adapter` | Adapter differences from a common deterministic initialization; backbone frozen |
| Eq. 2 and modulators | `core.unify` | Sign of vector sum, maximum aligned magnitude, mask task×unified>0, L1-ratio scale |
| Eq. 3 | `core.aggregate` | Default absolute mean signs; signed alternative reproduces the literal equation |
| Eq. 4 | `core.aggregate` | Reconstruct client task vector, apply sample weights and task agreement mask; scale applied once |
| Eq. 5 | `core.aggregate` | (mean(sign(a)×sign(b))+1)/2; zero coordinates contribute 0.5 |
| Eq. 6 | `core.aggregate` | Self excluded, similarity strictly >0.5, top-κ, weights not normalized |
| Eq. 7 | `core.aggregate` | Default sum; optional mean follows overview prose |
| Return unification | `training.run`, `flower.main` | Unify the tasks assigned to each selected client |

## Explicit reconstruction choices

Eq. 3 writes signed agreement without an absolute value. Applied literally, unanimous negative updates receive a negative mask and can reverse direction. The default uses agreement magnitude; `agreement="signed"` exposes the literal form for investigation. This is a disclosed correction, not evidence about the missing original implementation.

Eq. 7 adds same-task and cross-task terms, while the overview says they are averaged. Default `combine="sum"` follows the equation; `combine="mean"` averages when related tasks exist. The paper does not specify top-κ for every experiment; the supplied 2 is a runnable setting, not a recovered hyperparameter.

Exact sign ties yield zero. An entirely erased task gets scale zero, avoiding division by zero; lost information cannot be reconstructed. The representation is not lossless.

The vision reference uses a `timm` ViT qkv adapter, rank r and unit LoRA scale. Shared task classifiers are obtained with a separate federated linear-probing phase, then frozen. The paper does not fully specify classifier handling, adapter targets/scaling, optimizer, learning rate, augmentation or exact data manifests. Align these choices before claiming benchmark reproduction.

Class allocation uses seeded Dirichlet sampling with minimal repair of empty clients. Task ownership is explicit through `assignments`; the paper's ζt generator has not been recovered. Default all-task ownership is not its eight-task or 30-task experiment.

## Source provenance

Recovery examined the authors' `many_task_fl` source, research-server snapshot and archived `matu` project. They contain dataset/adapter training, TIES-style merging and later DINOv2/audio experiments. The archived `matu.py` actually performs sequential per-task fine-tuning; it does not implement federated unification or aggregation. It was not republished as a complete MaTU artifact.

This implementation is independently written from the public paper using public PyTorch, torchvision, timm and Flower APIs. No private Git history, unpublished logs, datasets or credentials are included. Author-side recovery manifests preserve original hashes.

## Validation

`python -m pytest` and the smoke configuration exercise forward/backward optimization, modulation, payload roundtrips, aggregation, evaluation and checkpoint writing. The frozen-backbone check detects unintended parameter changes and records a SHA-256 digest of parameter names and bytes. Archive the exact pretrained checkpoint as well as this digest for image experiments; the timm model name alone is not an immutable weight revision.

Test data is never used for optimization or checkpoint selection. The checkpoint records task vectors and heads; reconstructing the full pretrained model additionally requires the selected timm weights. Save configuration and dependency versions with every run.

No paper result is claimed as reproduced. Larger vision benchmarks require acquired datasets and substantial compute; record those commands and results separately.
