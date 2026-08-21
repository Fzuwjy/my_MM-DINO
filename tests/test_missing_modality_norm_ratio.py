from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn

from scripts.missing_modality_factorial import (
    experiment_slug,
    resolve_full_probability,
    train_state,
)
from tasks.segmentation.models.MMDINO.Decoder import (
    Decoder,
    configure_decoder_normalization,
)


def test_default_run_c_trace_is_exactly_preserved():
    old_rng = random.Random(1234)
    new_rng = random.Random(1234)
    old = ["sar" if old_rng.random() < 0.5 else "full" for _ in range(100)]
    new = [train_state("C", new_rng) for _ in range(100)]
    assert new == old


def test_ratio_endpoints_and_intermediate_thresholds_are_explicit():
    class Stub:
        def __init__(self, value):
            self.value = value

        def random(self):
            return self.value

    assert train_state("A", Stub(0.9), 0.0) == "sar"
    assert train_state("B", Stub(0.1), 1.0) == "full"
    assert train_state("C", Stub(0.74), 0.25) == "sar"
    assert train_state("C", Stub(0.75), 0.25) == "full"
    with pytest.raises(ValueError, match="Run A"):
        resolve_full_probability("A", 0.25)
    with pytest.raises(ValueError, match="Run C"):
        resolve_full_probability("C", 0.0)


def test_experiment_slug_separates_normalization_and_ratio():
    assert experiment_slug("A", "batchnorm", 0.0, 42) == "run_a_bn_p000_seed42"
    assert experiment_slug("C", "groupnorm", 0.25, 42) == "run_c_gn_p025_seed42"


def test_batchnorm_policy_is_a_strict_module_and_state_dict_noop():
    torch.manual_seed(7)
    decoder = Decoder(n_classes=8, num_modalities=2, raw_logits=True)
    module_ids = {name: id(module) for name, module in decoder.named_modules()}
    state_before = {key: value.clone() for key, value in decoder.state_dict().items()}

    manifest = configure_decoder_normalization(decoder, "batchnorm", 32)

    assert manifest["source_batchnorm_count"] == 40
    assert manifest["result_batchnorm_count"] == 40
    assert manifest["result_groupnorm_count"] == 0
    assert {name: id(module) for name, module in decoder.named_modules()} == module_ids
    assert decoder.state_dict().keys() == state_before.keys()
    for key, value in decoder.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)


def test_groupnorm_replaces_only_decoder_bn_and_preserves_affine_values():
    torch.manual_seed(11)
    decoder = Decoder(n_classes=8, num_modalities=2, raw_logits=True)
    bn_affine = {
        f"{name}.weight": module.weight.detach().clone()
        for name, module in decoder.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    }
    bn_affine.update(
        {
            f"{name}.bias": module.bias.detach().clone()
            for name, module in decoder.named_modules()
            if isinstance(module, nn.BatchNorm2d)
        }
    )

    manifest = configure_decoder_normalization(decoder, "groupnorm", 32)

    assert manifest["source_batchnorm_count"] == 40
    assert manifest["result_batchnorm_count"] == 0
    assert manifest["result_groupnorm_count"] == 40
    assert set(manifest["group_counts"].values()) == {32}
    assert not any(isinstance(module, nn.BatchNorm2d) for module in decoder.modules())
    assert sum(isinstance(module, nn.GroupNorm) for module in decoder.modules()) == 40
    state = decoder.state_dict()
    for key, expected in bn_affine.items():
        torch.testing.assert_close(state[key], expected, rtol=0, atol=0)
    assert not any(
        key.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for key in state
    )
