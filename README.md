# MaTU

**Many-Task Federated Fine-Tuning via Unified Task Vectors**

Vasileios Tsouvalas · Tanir Ozcelebi · Nirvana Meratnia · FedGenAI @ IJCAI 2025

[<img src="docs/assets/icons/paper.svg" width="16" height="16" alt=""> Paper](https://arxiv.org/abs/2502.06376) · [<img src="docs/assets/icons/project.svg" width="16" height="16" alt=""> Project](https://nousphera.github.io/MaTU/) · [Poster](https://nousphera.github.io/assets/posters/matu-a0-landscape.pdf)

## Abstract

Federated Learning (FL) traditionally assumes homogeneous client tasks; however, in real-world scenarios, clients often specialize in diverse tasks, introducing task heterogeneity. To address this challenge, Many-Task FL (MaT-FL) has emerged, enabling clients to collaborate effectively despite task diversity. Existing MaT-FL approaches rely on client grouping or personalized layers, requiring the server to manage individual models and failing to account for clients handling multiple tasks. We propose MaTU, a MaT-FL approach that enables joint learning of task vectors across clients, eliminating the need for clustering or client-specific weight storage at the server. MaTU constructs a single “unified” task vector per client and leverages lightweight task-specific modulators — binary masks and scalar rescalers — to disentangle and adapt per-task behavior. This design ensures stateless server aggregation and naturally scales to large task sets with minimal communication overhead. Evaluated across 30 datasets, MaTU achieves superior performance over state-of-the-art MaT-FL approaches, with results comparable to per-task fine-tuning, while delivering significant communication savings.

**Implementation:** task-specific LoRA training, unified-vector exchange, related-task aggregation and a native Flower App. See the [method guide](docs/reproduction.md) for the equation-to-code map and aggregation settings.

[![MaTU client task-vector unification and server aggregation, with task-specific masks and scales.](docs/assets/readme/overview.webp)](https://nousphera.github.io/MaTU/#overview)

*One vector per client, with a binary mask and scalar for each task.*
[Method animation](https://nousphera.github.io/MaTU/#overview) · [Paper figure](https://arxiv.org/pdf/2502.06376v3#page=3)

## Requirements

Use an up-to-date **Python 3.11 or 3.12** on Linux or macOS. Keep the virtual environment in a path without spaces so Ray can start its workers.

```bash
git clone https://github.com/Nousphera/MaTU.git
cd MaTU
python3.11 -m venv "$HOME/.venvs/matu"
source "$HOME/.venvs/matu/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[flower,test]'
```

The environment pins PyTorch 2.14.1, torchvision 0.29.1, timm 1.0.30 and Flower 1.40.0. On **Linux CPU**, run this before the editable install to avoid downloading CUDA libraries:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1 torchvision==0.29.1
```

For CUDA, select compatible wheels using the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Prepare datasets

Download official splits and export verified RGB ImageFolders:

```bash
python -m matu.prepare_data --dataset mnist --download --root data
python -m matu.prepare_data --dataset svhn --download --root data
python -m matu.prepare_data --combine data/mnist/tasks.json data/svhn/tasks.json \
  --output configs/tasks.json
python -m matu.prepare_data --dataset mnist --root data --validate-only
```

MNIST, SVHN, GTSRB and DTD have automatic torchvision acquisition. DTD uses partition 1 train/test; its validation split is excluded. For SUN397, Cars, RESISC45, EuroSAT or another task, import your acquired data with existing `train/<class>/images` and `test/<class>/images` directories:

```bash
python -m matu.prepare_data --dataset eurosat --source /path/to/split-eurosat --root data
```

Alternatively, preserve published split lists in a CSV with `path,split,label` columns (`split` is `train` or `test`; paths are relative to the source root):

```bash
python -m matu.prepare_data --dataset sun397 --source /path/to/SUN397 \
  --split-file /path/to/sun397-splits.csv --root data
```

Preparation preserves raw files, checks decoding and class consistency, rejects overlapping source files, and writes **SHA-256 hashes, source IDs, split policy and a portable `tasks.json`**. It refuses to overwrite an existing dataset. Combine any number of task manifests with `--combine`. [Dataset guide](docs/datasets.md) covers all eight benchmark datasets, split provenance and the CSV format.

## Training

A three-client, three-task CPU execution check needs no dataset or model download:

```bash
flwr run . --run-config 'output="/tmp/matu-smoke"' \
  --federation-config 'num-supernodes=3 init-args-num-cpus=2 client-resources-num-cpus=1' --stream
```

Use a new output directory for every run. The app performs task-head probing, local LoRA training, unified-vector exchange, same-task aggregation and cross-task transfer.

For prepared images, edit `configs/vision.json` (device, client count and experiment size), then run:

```bash
flwr run . --run-config "config=\"$(pwd)/configs/vision.json\" output=\"$(pwd)/runs/vision-01\"" \
  --federation-config 'num-supernodes=30' --stream
```

`num-supernodes` must equal the JSON `clients` value. `manifest` resolves relative to its JSON config; dataset paths resolve relative to the task manifest. Use absolute config/output paths for worker processes. The paper-sized example uses ImageNet-21k-pretrained ViT-B/32, rank-16 adapters and CUDA; weights download through timm on first use. GPU scheduling instructions and a smaller image configuration are in [the dataset guide](docs/datasets.md).

Flower 1.40 runs a managed local SuperLink. If another project already has one running, use a separate state directory **and ports** before launching:

```bash
export FLWR_HOME="$HOME/.cache/matu-flower"
export FLWR_LOCAL_SUPERLINK_HTTP_API_PORT=39135
export FLWR_LOCAL_CONTROL_API_PORT=39137
```

See [Flower's simulation guide](https://flower.ai/docs/framework/how-to-run-simulations.html). Worker machines must have access to the configured data paths. This repository targets local research simulations.

## Pretrained models

The image configuration loads `vit_base_patch32_224.augreg_in21k` through
[timm](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k).
The backbone stays frozen while rank-16 qkv LoRA adapters are trained. Record the
weight revision and SHA-256 with each run; `final.pt` saves the adapters and task
heads to combine with that backbone. The synthetic configuration uses no download.

## Evaluation

The trainer evaluates every task on its held-out test split after each round,
reporting per-task accuracy and their unweighted mean. Test samples are excluded
from adapter training and task-head probing. Inspect a completed image run with:

```bash
python -m json.tool runs/vision-01/metrics.json
```

Use `partitions.json`, `config.json` and the prepared task manifests to compare the
same client/task allocation across methods. The [method guide](docs/reproduction.md)
defines agreement, related-task transfer and serialized-byte accounting.

## Outputs and configuration

Each output directory contains `metrics.json`, `config.json`, `partitions.json` and `final.pt`: per-task accuracy, exact partitions, seed, checkpoint, backbone hash and encoded communication counts. Preparation traffic is reported separately. Record the dataset manifests alongside results.

- **Task ownership:** explicit `assignments` can specify tasks per client. The default assigns every task to every client; use explicit assignments for heterogeneous task ownership.
- **Task heads:** initial federated linear probing creates shared heads, then freezes them.
- **Communication:** counters include serialized records and payloads, with head-probing traffic reported separately; TCP/TLS framing is excluded.
- **Tests:** aggregation, optimization, frozen-backbone invariants, dataset integrity and Flower transport.

For a sequential debugging run with the same method:

```bash
matu --config configs/smoke.json --output runs/sequential-01
python -m pytest
```

## Results

**Published eight-task benchmark:** ViT-B/32, rank-16 LoRA, 30 clients,
20% participation and multiple tasks per client (task concentration 0.5).
Accuracy is averaged across tasks; bitrate is millions of bits per task per round.

| Method | Average accuracy (%) | Bitrate (M bits/task/round) |
| --- | --- | --- |
| FedAvg | 51.63 | 20.43 |
| FedPer | 48.37 | 17.16 |
| MaT-FL | 67.60 | 20.43 |
| **MaTU** | 79.47 | 8.04 |

MaTU and FedAvg/FedPer use 100 rounds; MaT-FL uses 300. Source:
[Table 2](https://arxiv.org/pdf/2502.06376v3#page=6).
[Explore all results](https://nousphera.github.io/MaTU/#findings).

## Citation

```bibtex
@inproceedings{tsouvalas2025matu,
  title={Many-Task Federated Fine-Tuning via Unified Task Vectors},
  author={Vasileios Tsouvalas and Tanir Ozcelebi and Nirvana Meratnia},
  booktitle={FedGenAI Workshop at IJCAI 2025},
  year={2025},
  url={https://arxiv.org/abs/2502.06376}
}
```

[MIT license](LICENSE). Publication figures, posters, pretrained models and datasets retain their original terms.
