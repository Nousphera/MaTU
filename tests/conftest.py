"""Keep full paper-model checks bounded on CPU."""

import torch

torch.set_num_threads(1)
