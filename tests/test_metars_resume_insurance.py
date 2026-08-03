from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "metars_resume_insurance.py"
    spec = importlib.util.spec_from_file_location("metars_resume_insurance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _CovarianceState:
    def __init__(self, mask: torch.Tensor | None):
        self.i = torch.eye(2)
        self.mask_matrix = mask
        self.num_sensitive = torch.tensor(3.0)
        self.var_matrix = torch.full((2, 2), 0.25)
        self.count_var_cov = 7


class _Model:
    def __init__(self, mask: torch.Tensor | None):
        self.cov_matrix_layer = [_CovarianceState(mask)]


def test_resume_sidecar_round_trip_restores_mmr_state() -> None:
    module = _load_module()
    expected_mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    source = _Model(expected_mask)
    payload = module.capture_resume_state(source, global_step=1654)

    target = _Model(None)
    target.cov_matrix_layer[0].num_sensitive = 0
    target.cov_matrix_layer[0].var_matrix = None
    target.cov_matrix_layer[0].count_var_cov = 0
    module.restore_resume_state(target, payload, expected_step=1654)

    restored = target.cov_matrix_layer[0]
    assert torch.equal(restored.mask_matrix, expected_mask)
    assert restored.num_sensitive.item() == 3.0
    assert torch.equal(restored.var_matrix, torch.full((2, 2), 0.25))
    assert restored.count_var_cov == 7


def test_resume_sidecar_rejects_step_mismatch() -> None:
    module = _load_module()
    payload = module.capture_resume_state(_Model(torch.eye(2)), global_step=1654)
    with pytest.raises(RuntimeError, match="steps differ"):
        module.restore_resume_state(_Model(None), payload, expected_step=3309)
