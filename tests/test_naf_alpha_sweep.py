"""Contract tests for the NAF residual alpha-sweep state transplant."""

import unittest

import torch
from torch import nn

from scripts.evaluate_whu_naf_alpha_sweep import transplant_zero_on_e0


class _ToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.naf = nn.Conv2d(2, 2, 1, bias=False)
        self.naf_zero_conv = nn.Conv2d(2, 2, 1, bias=False)
        nn.init.zeros_(self.naf_zero_conv.weight)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.old = nn.Conv2d(2, 2, 1, bias=False)
        self.adapter = _ToyAdapter()


class NafAlphaSweepContractTest(unittest.TestCase):
    def test_transplant_copies_only_zero_conv(self):
        model = _ToyModel()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        r1_state = {key: value.clone() for key, value in before.items()}
        r1_state["old.weight"].add_(7)
        r1_state["adapter.naf_zero_conv.weight"].fill_(0.25)

        learned_zero = transplant_zero_on_e0(model, r1_state)

        after = model.state_dict()
        self.assertTrue(torch.equal(after["old.weight"], before["old.weight"]))
        self.assertTrue(
            torch.equal(after["adapter.naf.weight"], before["adapter.naf.weight"])
        )
        self.assertTrue(
            torch.equal(after["adapter.naf_zero_conv.weight"], learned_zero)
        )

    def test_transplant_rejects_changed_frozen_naf(self):
        model = _ToyModel()
        r1_state = {key: value.clone() for key, value in model.state_dict().items()}
        r1_state["adapter.naf.weight"].add_(1)

        with self.assertRaises(AssertionError):
            transplant_zero_on_e0(model, r1_state)


if __name__ == "__main__":
    unittest.main()
