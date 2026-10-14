# MaTU

**Many-Task Federated Fine-Tuning via Unified Task Vectors**

Vasileios Tsouvalas · Tanir Ozcelebi · Nirvana Meratnia  
FedGenAI @ IJCAI

Federated clients may work on several different tasks, making a single averaged model a poor fit. MaTU combines each client’s task updates into one unified vector with lightweight masks and scaling factors, then uses task similarity to share useful updates across clients while limiting interference between unrelated tasks.

## Project resources

[Project page](https://nousphera.github.io/MaTU/) · [Research overview](https://nousphera.github.io/projects/matu/) · [A0 poster](https://nousphera.github.io/MaTU/poster.pdf) · [Paper](https://arxiv.org/abs/2502.06376)

This repository hosts the paper’s project page, poster and citation. **The implementation is not included in this release.** Code will be added here after the final paper implementation has been identified and prepared for release.

## Method at a glance

- **Learn local task updates.** Fine-tune a common pre-trained model separately for each local task. Express each update as a task vector relative to the shared starting model.
- **Unify and modulate.** Merge local vectors into one representation. A binary mask and scaling factor per task recover its direction and magnitude without transmitting a separate full vector for every task.
- **Aggregate by task similarity.** Combine updates for the same task, then transfer information from the most similar tasks using sign agreement. Return a unified vector and modulators without retaining client-specific weights across rounds.

## Citation

```bibtex
@inproceedings{tsouvalas2025matu,
  title = {Many-Task Federated Fine-Tuning via Unified Task Vectors},
  author = {Vasileios Tsouvalas and Tanir Ozcelebi and Nirvana Meratnia},
  booktitle = {FedGenAI Workshop at IJCAI 2025},
  year = {2025},
  url = {https://arxiv.org/abs/2502.06376},
  eprint = {2502.06376},
  archivePrefix = {arXiv}
}
```

## Materials and licensing

Publication and linked implementation licenses remain with their respective authors and publishers. No license for unreleased code is implied by this resource repository.
