<h1 align="center">MaTU</h1>
<p align="center"><strong>Many-Task Federated Fine-Tuning via Unified Task Vectors</strong></p>
<p align="center"><a href="https://vtsouval.github.io/"><strong>Vasileios Tsouvalas</strong></a> · Tanir Özçelebi · Nirvana Meratnia</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.06376" title="Read the paper"><img src="docs/assets/icons/resource-paper.svg" width="48" height="48" alt="Paper"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://nousphera.github.io/MaTU/" title="Explore the project"><img src="docs/assets/icons/resource-project.svg" width="48" height="48" alt="Project"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://nousphera.github.io/assets/posters/matu-a0-landscape.pdf" title="View the poster"><img src="docs/assets/icons/resource-poster.svg" width="48" height="48" alt="Poster"></a>
</p>

## Abstract

**MaTU** enables federated fine-tuning when clients learn **multiple, different tasks**. Each client combines its task updates into one *unified vector*, accompanied by lightweight **binary masks and scalar rescalers** that recover task-specific behavior. The server aggregates updates within tasks and transfers information between related tasks, without maintaining a separate model for every client. The paper evaluates this approach across **30 vision datasets**, studying the balance between accuracy, task heterogeneity and communication cost.

<p align="center">
  <a href="https://nousphera.github.io/MaTU/#overview">
    <img src="docs/assets/readme/overview.webp" width="960" alt="MaTU unifies each client's task vectors, aggregates related tasks, and reconstructs task-specific updates with binary masks and scales.">
  </a>
</p>

## Setup

Create and activate the `matu` environment:

```bash
git clone https://github.com/Nousphera/MaTU.git
cd MaTU
bash scripts/setup.sh
source "$HOME/.venvs/matu/bin/activate"
```

From an existing environment, install the requirements with:

```bash
python -m pip install -r requirements.txt
```

## Data and Weights

Download and prepare **MNIST, SVHN, GTSRB, DTD and EuroSAT** from [**MaTU-30**](https://huggingface.co/datasets/vtsouval/MaTU-30):

```bash
python scripts/prepare_data.py --benchmark small --hf-repo vtsouval/MaTU-30
```

> [!NOTE]
> Downloads are pinned to their Hub revision. Dataset-specific splits and original licence conditions are listed in the [collection card](https://huggingface.co/datasets/vtsouval/MaTU-30).

<details>
<summary><strong>All 30 Datasets</strong></summary>

Prepare the complete collection:

```bash
python scripts/prepare_data.py --benchmark thirty --hf-repo vtsouval/MaTU-30
```

Use `--benchmark eight` for the eight-task benchmark, or `--dataset eurosat` for one task. Local mirrors are supported with `--hf-repo /path/to/MaTU-30`.

EMNIST is **Balanced**, Flowers is **Flowers Recognition (5 classes)**, and Fruits is the **24-class Fruits-360 original-size collection**. The [dataset card](https://huggingface.co/datasets/vtsouval/MaTU-30#configurations) lists every task, class count and split.

</details>

### Prepare Model

Download the pretrained **ImageNet-21k ViT-B/32** checkpoint:

```bash
python scripts/prepare_model.py
```

## Run Experiments

From the repository root, run a native **Flower App**:

```bash
common=(--benchmark small --nr-clients 30 --participation 0.2
        --local-epochs 1 --batch-size 32 --task-alpha 0.5
        --dirichlet-alpha 0.1 --device cuda)

# MaTU (Ours)
python -m matu.experiment "${common[@]}" --method matu

# Baselines
for method in fedavg fedprox fedper matfl ntk-fedavg centralized; do
  python -m matu.experiment "${common[@]}" --method "$method"
done
```

> [!NOTE]
> All methods train **rank-16 query/value LoRA** with a frozen backbone and probed task classifiers. FedPer keeps its final-block adapters and classifiers private. Centralized training fits each task independently.

Select `--benchmark eight`, `--benchmark thirty`, or `--datasets mnist svhn`. Without explicit overrides, `thirty` uses 300 rounds, full participation and task concentration 0.2; MaT-FL uses 300 rounds. Use `--task-alpha 0` for single-task clients, or `--tasks-per-client N` for fixed task counts. Set `--parallel-clients N` to run clients concurrently; `--device cpu` and `--device mps` are also supported.

> [!NOTE]
> Each run saves its configuration, task accuracies and adapter weights in `results/`. MaTU evaluates **reconstructed client models**; reported accuracy is the mean across tasks.

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
