"""Pretrained ViT-B/32 with rank-r query/value LoRA and separate task heads."""

from collections import OrderedDict
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, jvp
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import ViTModel

from .model_preparation import DEFAULT_WEIGHTS_DIR, MODEL_ID, validate_model


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=16, dropout=0.1):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError(
                "LoRA alpha must be positive and dropout must be in [0, 1)"
            )
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, values):
        return self.base(values) + self.scale * self.up(self.down(self.dropout(values)))


class VisionBackbone(nn.Module):
    """CLS features; the pretrained parameters and pooler stay frozen."""

    def __init__(self, backbone, rank=16, alpha=16, dropout=0.1):
        super().__init__()
        expected = {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "image_size": 224,
            "patch_size": 32,
        }
        if any(
            getattr(backbone.config, name) != value for name, value in expected.items()
        ):
            raise ValueError("The backbone must be the full ViT-B/32 architecture")
        self.backbone = backbone.requires_grad_(False)
        self.backbone.set_attn_implementation("eager")
        for block in self.backbone.encoder.layer:
            attention = block.attention.attention
            for name in ("query", "value"):
                setattr(
                    attention,
                    name,
                    LoRALinear(getattr(attention, name), rank, alpha, dropout),
                )
        self.num_features = 768

    def train(self, mode=True):
        # The pretrained backbone is fixed; only adapter dropout is stochastic.
        super().train(False)
        self.training = mode
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train(mode)
        return self

    def forward(self, images):
        return self.backbone(pixel_values=images, return_dict=True).last_hidden_state[
            :, 0
        ]


def build_backbone(config):
    """Load verified local assets; network access and random fallbacks are forbidden."""
    if config.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError(f"Unsupported backbone model; use {MODEL_ID}")
    directory = config.get("weights_dir", DEFAULT_WEIGHTS_DIR)
    manifest = validate_model(directory)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(config.get("seed", 42)))
        backbone, info = ViTModel.from_pretrained(
            directory,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(
            info.get(name)
            for name in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Pretrained backbone state mismatch: {info}")
        model = VisionBackbone(
            backbone,
            rank=config.get("rank", 16),
            alpha=config.get("lora_alpha", 16),
            dropout=config.get("lora_dropout", 0.1),
        )
    model.pretrained_manifest = manifest
    return model.to(config.get("device", "cpu")), model.num_features


def adapter_parameters(model):
    """Stable named factor layout, independent of temporary trainability flags."""
    return OrderedDict(
        (f"{name}.{factor}.weight", getattr(module, factor).weight)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
        for factor in ("down", "up")
    )


def adapter_vector(model):
    return (
        torch.cat(
            [p.detach().reshape(-1).cpu() for p in adapter_parameters(model).values()]
        )
        .numpy()
        .copy()
    )


def _validated_vector(model, values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    size = sum(p.numel() for p in adapter_parameters(model).values())
    if values.ndim != 1 or values.size != size:
        raise ValueError(f"Adapter vector must have shape ({size},)")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Adapter vector must contain real numeric values")
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Adapter vector contains nonfinite values")
    return values


def set_adapter(model, values):
    values = _validated_vector(model, values)
    offset = 0
    with torch.no_grad():
        for parameter in adapter_parameters(model).values():
            size = parameter.numel()
            parameter.copy_(
                torch.from_numpy(values[offset : offset + size].copy())
                .reshape(parameter.shape)
                .to(parameter)
            )
            offset += size


def personalized_mask(model):
    last = model.backbone.config.num_hidden_layers - 1
    return np.concatenate(
        [
            np.full(parameter.numel(), f".layer.{last}." in name, dtype=np.bool_)
            for name, parameter in adapter_parameters(model).items()
        ]
    )


def make_head(num_classes, seed):
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 1
    ):
        raise ValueError("The task classifier requires a positive class count")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        return nn.Linear(768, num_classes)


def frozen_hash(model):
    names = set(adapter_parameters(model))
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        if name not in names:
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


class _RGB:
    def __call__(self, image):
        return image.convert("RGB")


def model_transforms():
    transform = transforms.Compose(
        [
            _RGB(),
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform, transform


def linearized_features(model, images, reference_adapter):
    """Full-network Taylor model at a fixed supplied factor vector, with gradients."""
    values = _validated_vector(model, reference_adapter)
    named = adapter_parameters(model)
    reference = []
    offset = 0
    for parameter in named.values():
        size = parameter.numel()
        reference.append(
            torch.from_numpy(values[offset : offset + size].copy())
            .reshape(parameter.shape)
            .to(parameter)
        )
        offset += size
    reference = tuple(reference)
    delta = tuple(
        parameter - origin for parameter, origin in zip(named.values(), reference)
    )
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:

        def forward(*parameters):
            return functional_call(
                model, dict(zip(named, parameters)), (images,), strict=False
            )

        features, tangent = jvp(forward, reference, delta)
        return features + tangent
    finally:
        for module, mode in modes:
            module.training = mode
