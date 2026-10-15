# Dataset preparation

Use `python -m matu.prepare_data --help` for the complete CLI. The output is a portable ImageFolder task, a `tasks.json` consumed by the trainer and a `dataset.json` containing file hashes and split provenance. Raw data is never renamed or removed. Imports decode each image and export lossless RGB PNG files; allow space for both original data and prepared copies.

## Acquisition and splits

| Task | Acquire | Prepare |
| --- | --- | --- |
| MNIST | torchvision official mirrors | `--dataset mnist --download` (official train/test) |
| SVHN | [Stanford SVHN](http://ufldl.stanford.edu/housenumbers/) | `--dataset svhn --download` (train/test; extra split excluded) |
| GTSRB | [GTSRB benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html) | `--dataset gtsrb --download` (official train/test) |
| DTD | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | `--dataset dtd --download` (partition 1 train/test; validation excluded) |
| SUN397 | [SUN database](https://vision.princeton.edu/projects/2010/SUN/) | Import a split CSV derived from the official training/testing list pair; record the selected partition |
| Stanford Cars | [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | Import acquired images and their official annotation-defined train/test split; original automatic download endpoints are unavailable |
| RESISC45 | [NWPU dataset authors](https://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) | Import acquired data and explicit split assignments; there is no built-in unique train/test split |
| EuroSAT | [EuroSAT authors](https://github.com/phelber/EuroSAT) | Import acquired RGB data and explicit split assignments; do not silently invent the historical split |

Downloaded task commands accept `--root data`. All other tasks accept `--source /path/to/data` with `train/<class>/...` and `test/<class>/...`, or `--split-file splits.csv`. The script supports any additional image-classification task via these same formats. Exact original eight/30-task membership, partitions and task ownership are not recovered automatically.

The installation includes SciPy for SVHN's MATLAB files. Automatic downloads use torchvision's checksum validation and Python's safe tar data filter; use an up-to-date Python 3.11 or 3.12 patch release.

## Explicit split CSV

```csv
path,split,label
images/scene_0001.jpg,train,abbey
images/scene_0002.jpg,test,abbey
images/scene_0003.jpg,train,airport
images/scene_0004.jpg,test,airport
```

Paths are relative to `--source`. Labels must be safe class directory names (letters, numbers, `_`, `-`, `.`). Both splits must contain the same classes. Duplicate source files, cross-split symlinks/hard links, paths outside the source, corrupt images and missing files fail preparation. The source files are identified by resolved path; hashes of the prepared images record content integrity. Identical pixel values alone are not considered proof of leakage (official datasets can contain duplicate pixel arrays).

```bash
python -m matu.prepare_data --dataset custom-task --source /path/to/raw \
  --split-file splits.csv --root data
python -m matu.prepare_data --dataset custom-task --root data --validate-only
python -m matu.prepare_data --combine data/mnist/tasks.json data/custom-task/tasks.json \
  --output configs/tasks.json
```

Preparation refuses existing destinations. Validation rejects changed, missing or added images. Combining prepared tasks verifies each dataset and requires its generated task name and train/test paths to match `dataset.json`; it rejects redirected or swapped splits. Keep `dataset.json` with the run outputs. Train/test partitions are fixed during preparation; the client Dirichlet partition uses only training samples.

## Small image run

Start with two prepared datasets. Save this as `configs/vision-small.json`:

```json
{
  "backend": "imagefolder",
  "manifest": "tasks.json",
  "model": "vit_tiny_patch16_224",
  "pretrained": false,
  "clients": 2,
  "rounds": 2,
  "head_rounds": 1,
  "local_epochs": 1,
  "batch_size": 16,
  "rank": 2,
  "device": "cpu",
  "output": "runs/vision-small"
}
```

```bash
flwr run . --run-config "config=\"$(pwd)/configs/vision-small.json\" output=\"$(pwd)/runs/vision-small-01\"" \
  --federation-config 'num-supernodes=2 init-args-num-cpus=2' --stream
```

This optimizes real image data with a small randomly initialized backbone; it is a functional check, **not a paper configuration**. The original-scale `configs/vision.json` selects the pretrained ViT-B/32 and 30 clients. For CUDA, configure Ray's per-client GPU allocation in addition to `"device": "cuda"`:

```bash
flwr run . --run-config "config=\"$(pwd)/configs/vision.json\" output=\"$(pwd)/runs/vision-01\"" \
  --federation-config 'num-supernodes=30 client-resources-num-cpus=1 client-resources-num-gpus=1' --stream
```

GPU availability limits concurrent clients; the server also evaluates on the configured device. Reserve enough memory for both workloads. This CUDA command is documented but not hardware-validated in this revision. On a multi-machine deployment, provisioning shared paths and credentials is your responsibility; the supplied app is tested as a local simulation.
