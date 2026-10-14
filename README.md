# MaTU

**Many-Task Federated Fine-Tuning via Unified Task Vectors**
Vasileios Tsouvalas · Tanir Ozcelebi · Nirvana Meratnia
FedGenAI @ IJCAI 2025

[Paper](https://arxiv.org/abs/2502.06376) · [Project](https://nousphera.github.io/MaTU/) · [Poster](https://nousphera.github.io/MaTU/poster.pdf)

One shared update, several tasks. MaTU combines local task updates into a **unified vector**, retains a compact **binary mask and scale per task**, and transfers knowledge between related tasks through sign-based aggregation.

This release provides a **paper-based reference implementation**, seeded training, and a Flower App. Recovered research folders contained adaptation experiments but no complete MaTU federated runner. This is therefore a documented reconstruction, **not a recovered artifact reproducing the published numbers**. See [paper correspondence](docs/reproduction.md).

## Quick start

Python 3.11 or 3.12. The default CPU example downloads no data or models.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
matu --config configs/smoke.json
python -m pytest
```

The example performs real low-rank adapter training across three clients and three tasks. Results, configuration, exact partition indices and the final checkpoint are saved under `runs/smoke/`. Synthetic accuracy is an execution check, not a research result. Existing output folders are protected; use `--output runs/another-run` when repeating a run.

## Method

| Stage | Implementation |
| --- | --- |
| Local adaptation | Frozen backbone; task-specific LoRA updates relative to a common initialization |
| Unification | Aggregate-sign election and maximum aligned magnitude; binary mask and scalar per task |
| Same-task aggregation | Sample-count weighting and coordinate-wise sign agreement |
| Cross-task transfer | Sign similarity, strict >0.5 threshold, top-κ neighbors, self excluded |
| Communication | Float32 unified vector, bit-packed masks and explicit metadata; actual encoded sizes |
| Execution | Sequential simulator and Flower Message API using the same algorithm |

The server retains task-level vectors, not a persistent model per client. Tasks absent from a round keep their previous vector.

## Image classification

Prepare disjoint train/test ImageFolder directories with identical class mappings:

```text
data/mnist/train/<class>/*.png
data/mnist/test/<class>/*.png
data/svhn/train/<class>/*.png
data/svhn/test/<class>/*.png
```

```bash
cp configs/tasks.example.json configs/tasks.json
# Edit dataset paths and the device/settings in configs/vision.json.
matu --config configs/vision.json
```

The vision runner uses ImageNet-21k-pretrained ViT-B/32 through `timm` and rank-16 adapters. A manifest can contain any number of classification datasets, including the paper's SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST and DTD. Acquire official datasets separately and preserve their official splits.

Class allocation is seeded Dirichlet. Set explicit task ownership with e.g. `"assignments": [["mnist"], ["svhn"], ["mnist", "svhn"]]` and `"clients": 3`. Without assignments, all clients hold every task; this **does not reproduce the paper's task-concentration distribution**.

Task classifiers have different output dimensions. The runner obtains shared per-task heads through initial federated linear probing and freezes them for MaTU rounds. This is a disclosed protocol choice; preparation bytes are counted separately.

## Flower App

```bash
python -m pip install -e '.[flower,test]'
matu --flower --config configs/smoke.json
```

The tested convenience command invokes the same Flower ClientApp/ServerApp through Ray. The app declarations are included for Flower bundling; the convenience command above is the verified execution path. On macOS, install the environment in a path without spaces because Ray worker launch fails with spaced virtualenv paths.

Default simulation: three clients. The Flower supernode count must match the JSON `clients` setting. Use `matu --flower --config configs/vision.json` for another configuration; its `clients` value sets the simulation size. Worker environments must have access to dataset paths. The app targets local research simulation; private dataset provisioning and production authentication are not supplied.

## Reproducibility

- **Saved:** config, seed, exact partitions, per-task accuracy, preparation traffic, upload/download bytes, frozen-backbone SHA-256 and checkpoint.
- **Checked:** numerical aggregation edge cases, serialized client updates, real CPU optimization and frozen-backbone invariants.
- **Not established:** original eight/30-task accuracy, published bitrate, seed uncertainty or GPU benchmark performance.

Payload sizes include the implemented container and metadata; Flower additionally counts its content records. Neither is TCP/TLS traffic or directly interchangeable with the paper's bitrate. See [reproduction notes](docs/reproduction.md) and the platform-specific dependency lock.

## Structure

```text
matu/core.py       unification, modulation, aggregation and payloads
matu/training.py   LoRA models, partitions, training and evaluation
matu/flower.py     Flower client/server applications
configs/          smoke and vision configurations
tests/            numerical, transport and training checks
docs/             project website and reproduction notes
```

## Cite

```bibtex
@inproceedings{tsouvalas2025matu,
  title={Many-Task Federated Fine-Tuning via Unified Task Vectors},
  author={Vasileios Tsouvalas and Tanir Ozcelebi and Nirvana Meratnia},
  booktitle={FedGenAI Workshop at IJCAI 2025},
  year={2025},
  url={https://arxiv.org/abs/2502.06376}
}
```

New implementation: MIT license. Publication figures, posters, pretrained models and datasets retain their original terms.
