import numpy as np
import torch
from matu import training


def test_two_rounds_train_and_preserve_backbone(tmp_path):
    cfg = dict(
        training.DEFAULTS,
        rounds=2,
        clients=2,
        tasks=2,
        samples=2,
        backend="synthetic-images",
        pretrained=False,
        batch_size=1,
        local_epochs=1,
        head_rounds=1,
        seed=17,
        output=str(tmp_path),
    )
    result = training.run(cfg)
    assert result["rounds"] == 2
    assert result["backbone_unchanged"]
    assert result["adapter_change"] > 0
    assert result["upload_bytes"] > 0
    assert len(result["history"]) == 2
    assert all(
        np.isfinite(v) for row in result["history"] for v in row["accuracy"].values()
    )
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "metrics.json").exists()


def test_partitions_cover_samples_once_and_keep_clients_nonempty():
    labels = np.repeat(np.arange(3), 15)
    parts = training.dirichlet_partition(labels, 4, 0.1, 7)
    assert all(len(p) > 0 for p in parts)
    assert sorted(np.concatenate(parts).tolist()) == list(range(45))


def test_lora_only_updates_adapters_and_task_head():
    model, dim = training.make_backbone(dict(training.DEFAULTS, pretrained=False))
    original = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }
    head = torch.nn.Linear(dim, 3)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        lr=0.1,
    )
    opt.zero_grad()
    torch.nn.functional.cross_entropy(
        head(model(torch.randn(1, 3, 224, 224))), torch.tensor([0])
    ).backward()
    opt.step()
    assert all(
        torch.equal(original[n], p)
        for n, p in model.named_parameters()
        if n in original
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "up" in n
    )


def test_vision_qkv_adapter_forward_without_downloading_weights():
    cfg = dict(
        training.DEFAULTS,
        backend="imagefolder",
        model="vit_base_patch32_224.augreg_in21k",
        rank=16,
        pretrained=False,
    )
    model, dimension = training.make_backbone(cfg)
    output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, dimension)
    output.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "up" in n
    )


def test_run_refuses_to_overwrite_existing_outputs(tmp_path):
    import pytest

    (tmp_path / "metrics.json").write_text("previous experiment")
    with pytest.raises(FileExistsError):
        training.run(dict(training.DEFAULTS, output=str(tmp_path)))
    assert (tmp_path / "metrics.json").read_text() == "previous experiment"


def test_default_backbone_uses_full_paper_vit_b32_without_weights():
    torch.set_num_threads(1)
    model, dimension = training.make_backbone(dict(training.DEFAULTS, pretrained=False))
    assert dimension == 768
    assert len(model.blocks) == 12
    assert model.patch_embed.patch_size == (32, 32)


def test_nonpaper_backbones_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="paper"):
        training.make_backbone(
            dict(training.DEFAULTS, model="vit_small_patch32_224", pretrained=False)
        )
