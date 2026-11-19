import pytest
from torch import nn

from rfdetr.training.module_model import freeze_rfdetr_base_parameters


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.base = nn.Linear(2, 2)
        self.backbone.lora_A = nn.Linear(2, 1, bias=False)
        self.transformer = nn.Linear(2, 2)
        self.class_embed = nn.Linear(2, 3)
        self.temporal_query_fusion = nn.Linear(2, 2)
        self.temporal_deformable_encoder = nn.Linear(2, 2)
        self.temporal_detection_decoder = nn.Linear(2, 2)


def test_keeps_only_temporal_and_adapter_parameters_trainable():
    model = ToyModel()
    stats = freeze_rfdetr_base_parameters(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert all(name.startswith("temporal_") or ".lora_" in name for name in trainable)
    assert "class_embed.weight" not in trainable
    assert "transformer.weight" not in trainable
    assert stats["trainable"] == stats["temporal"] + stats["adapters"]


def test_freezes_base_parameters():
    model = ToyModel()
    freeze_rfdetr_base_parameters(model)

    assert not model.backbone.base.weight.requires_grad
    assert not model.transformer.weight.requires_grad
    assert not model.class_embed.weight.requires_grad


def test_rejects_configuration_without_temporal_or_adapters():
    with pytest.raises(ValueError, match="left no trainable parameters"):
        freeze_rfdetr_base_parameters(nn.Linear(2, 2))
