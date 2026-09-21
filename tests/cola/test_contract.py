import copy
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import tifffile

from research.cola.model import Adaptation, PairedDino, CoLASegmenter, cpu_initialization, state_hash
from research.cola.diagnostics import (confusion, metrics, evaluation, crop_manifest, load_crops,
                                      save_manifest, evaluate_crops)
from research.cola.run import dataset, make_optimizer, make_loss, require_training_gpu
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.hub.backbones import dinov3_vits16

torch.set_num_threads(1)


def tiny(checkpoint=True, variant='cola'):
    with cpu_initialization(5):
        base = DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=32, depth=4, num_heads=2,
                                     n_storage_tokens=4, mask_k_bias=True, layerscale_init=0.2,
                                     pos_embed_rope_dtype='fp32', pos_embed_rope_rescale_coords=2)
        base.init_weights()
    return PairedDino(base, variant, rank=2, output_layers=(0, 1, 2, 3), checkpoint_blocks=checkpoint)


def inputs():
    generator = torch.Generator().manual_seed(22)
    return torch.randn(2, 3, 32, 32, generator=generator), torch.randn(2, 1, 32, 32, generator=generator)


def activate(model):
    with cpu_initialization(17):
        for layer in model.modules():
            if isinstance(layer, Adaptation):
                nn.init.normal_(layer.b.weight, std=0.02)
                if layer.has_cross:
                    nn.init.normal_(layer.cb.weight, std=0.02)


def feature_loss(features):
    return sum(f[..., 0].square().mean() for stream in features for f in stream)


class ModelContract(unittest.TestCase):
    def test_batch_pairing(self):
        model = tiny(False).eval()
        activate(model)
        x, s = inputs()
        baseline = model(x, s)
        permuted = model(x.flip(0), s.flip(0))
        changed = s.clone()
        changed[0] *= 3
        perturbed = model(x, changed)
        for stream in range(2):
            # Batch reordering can change float32 CPU SIMD rounding (~1e-7).
            torch.testing.assert_close(baseline[stream][-1].flip(0), permuted[stream][-1], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(baseline[stream][-1][1], perturbed[stream][-1][1], rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_checkpoint_gradients(self):
        regular = tiny(False).cuda().train()
        activate(regular)
        checked = copy.deepcopy(regular)
        checked.checkpoint_blocks = True
        batch = tuple(t.cuda() for t in inputs())
        states, losses = [], []
        for model in (regular, checked):
            torch.cuda.manual_seed_all(133)
            loss = feature_loss(model(*batch))
            loss.backward()
            losses.append(loss.detach())
            states.append(torch.cuda.get_rng_state())
        torch.testing.assert_close(*losses, rtol=0, atol=0)
        torch.testing.assert_close(*states, rtol=0, atol=0)
        for (name, a), (_, b) in zip(regular.named_parameters(), checked.named_parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7, msg=name)

    def test_zero_increment_reference_eval_and_train(self):
        model = tiny(False)
        optical, sar = inputs()
        for training in (False, True):
            model.train(training)
            table = [(model.base.rope_embed(H=2, W=2), model.base.rope_embed(H=2, W=2)) for _ in range(4)]
            actual = model(optical, sar, table)
            for m, source in enumerate((optical, sar.repeat(1, 3, 1, 1))):
                tokens, _ = model.base.prepare_tokens_with_masks(source)
                for i, block in enumerate(model.base.blocks):
                    tokens = block(tokens, table[i][m])
                    expected = model.base.norm(tokens)[:, 5:]
                    torch.testing.assert_close(actual[m][i], expected, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        self.assertTrue(all(torch.count_nonzero(b.attn.qkv.bias_mask[32:64]) == 0 for b in model.base.blocks))

    def test_pool_excludes_special_tokens(self):
        model = tiny()
        x = torch.randn(2, 9, 32)
        before = model.pool(x)
        x[:, :5] = 999
        torch.testing.assert_close(before, model.pool(x), rtol=0, atol=0)

    def test_cross_connectivity_both_directions_before_adapter(self):
        model = tiny(False).eval()
        activate(model)
        optical, sar = inputs()
        original = model(optical, sar)
        changed_sar = model(optical, sar.flip(-1))
        changed_optical = model(optical.flip(-2), sar)
        self.assertGreater(float((original[0][-1] - changed_sar[0][-1]).abs().max().detach()), 1e-7)
        self.assertGreater(float((original[1][-1] - changed_optical[1][-1]).abs().max().detach()), 1e-7)
        sar.requires_grad_()
        model(optical, sar)[0][-1][..., 0].square().mean().backward()
        self.assertGreater(float(sar.grad.abs().sum()), 0)
        for layer in model.modules():
            if isinstance(layer, Adaptation):
                layer.cross_enabled = False
        original = model(optical, sar)
        changed_sar = model(optical, sar.flip(-1))
        changed_optical = model(optical.flip(-2), sar)
        torch.testing.assert_close(original[0][-1], changed_sar[0][-1], rtol=0, atol=0)
        torch.testing.assert_close(original[1][-1], changed_optical[1][-1], rtol=0, atol=0)

    def test_gradient_startup(self):
        model = tiny()
        opt = torch.optim.AdamW(model.adapters.parameters(), lr=1e-3)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            feature_loss(model(*inputs())).backward()
            for layer in model.modules():
                if isinstance(layer, Adaptation):
                    self.assertGreater(float(layer.cb.weight.grad.abs().sum()), 0)
                    for p in (layer.ca.weight, layer.lam, layer.hyper[0].weight):
                        self.assertIsNotNone(p.grad)
                        if step == 0:
                            self.assertEqual(float(p.grad.abs().sum()), 0)
                        else:
                            self.assertGreater(float(p.grad.abs().sum()), 0)
            opt.step()
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_checkpoint_matches_outputs_rng_and_all_gradients(self):
        regular = tiny(False).train()
        activate(regular)
        checked = copy.deepcopy(regular)
        checked.checkpoint_blocks = True
        outputs, states = [], []
        for model in (regular, checked):
            torch.manual_seed(131)
            outputs.append(model(*inputs()))
            feature_loss(outputs[-1]).backward()
            states.append(torch.get_rng_state())
        for left, right in zip(outputs[0], outputs[1]):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(*states, rtol=0, atol=0)
        for (name, a), (_, b) in zip(regular.named_parameters(), checked.named_parameters()):
            if a.requires_grad:
                self.assertIsNotNone(a.grad, name)
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7, msg=name)

    def test_no_merge_save_load_and_freeze(self):
        model = tiny().train()
        base_hash = state_hash(model.base)
        opt = torch.optim.AdamW(model.adapters.parameters())
        feature_loss(model(*inputs())).backward()
        opt.step()
        for _ in range(3):
            model.eval().train()
        model.eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        other = tiny().eval()
        other.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertEqual(base_hash, state_hash(model.base))
        self.assertEqual(base_hash, state_hash(other.base))
        for a, b in zip(model(*inputs()), other(*inputs())):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_full_s_counts_common_initialization_and_backend_equivalence(self):
        from models.MMDINO.dino_segment import DINOSegmentModule
        with cpu_initialization(4):
            base = dinov3_vits16(pretrained=False)
        cola = CoLASegmenter(base)
        intra = CoLASegmenter(copy.deepcopy(base), 'intra')
        a, b = cola.provenance(), intra.provenance()
        self.assertEqual(a['adaptation_parameters'], 7641360)
        self.assertEqual(b['adaptation_parameters'], 2654208)
        for key in ('base_hash', 'intra_hash', 'adapter_hash', 'decoder_hash'):
            self.assertEqual(a[key], b[key], key)
        self.assertFalse(any(layer.has_cross for layer in intra.backbone.modules() if isinstance(layer, Adaptation)))
        make_optimizer(cola)
        make_optimizer(intra)
        del intra
        reference = DINOSegmentModule(backbone_weights=None, freeze_backbone=True, n_classes=7,
                                       backbone_type='dinov3_vits16', adapter_type='SampleAdapter', num_modalities=2)
        reference.backbone.load_state_dict(base.state_dict())
        reference.adapter.load_state_dict(cola.adapter.state_dict())
        reference.decoder.load_state_dict(cola.decoder.state_dict())
        x, s = torch.randn(2, 3, 64, 64), torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(cola.eval()(x, s), reference.eval()(x, s), rtol=0, atol=0)
            # Inject deterministic RoPE for the train-state comparison; count original shared FRM calls.
            cola.train()
            reference.train()
            cola.backbone.base.rope_embed.eval()
            reference.backbone.rope_embed.eval()
            counts = [0, 0]
            def count(index):
                def hook(*args):
                    counts[index] += 1
                return hook
            hooks = [cola.decoder.frm.register_forward_hook(count(0)), reference.decoder.frm.register_forward_hook(count(1))]
            torch.testing.assert_close(cola(x, s), reference(x, s), rtol=0, atol=0)
            for hook in hooks:
                hook.remove()
            self.assertEqual(counts, [2, 2])
            for name, value in cola.decoder.state_dict().items():
                torch.testing.assert_close(value, reference.decoder.state_dict()[name], rtol=0, atol=0)


class DiagnosticsContract(unittest.TestCase):
    def test_gpu_execution_roles(self):
        device = torch.device('cuda')
        for name in ('TITAN Xp', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 3090'):
            with patch('torch.cuda.get_device_name', return_value=name):
                if '3090' in name:
                    require_training_gpu(device)
                else:
                    with self.assertRaises(RuntimeError):
                        require_training_gpu(device)
                if 'TITAN' in name:
                    with self.assertRaises(RuntimeError):
                        require_training_gpu(device, smoke_only=True)
                else:
                    require_training_gpu(device, smoke_only=True)

    def test_eval_restores_modes_rng_bn_even_on_error(self):
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout()).train()
        model[1].eval()
        before = state_hash(model)
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        with self.assertRaisesRegex(ValueError, 'test'):
            with evaluation(model):
                self.assertFalse(model.training)
                model(torch.rand(2, 4))
                random.random()
                np.random.rand()
                raise ValueError('test')
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        self.assertEqual(before, state_hash(model))
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], np.random.rand())
        torch.testing.assert_close(expected[2], torch.rand(2), rtol=0, atol=0)

    def test_metrics_match_author_including_ignore_and_absent_class(self):
        from utils.metrics import metrics as author_metrics
        p = torch.tensor([0, 1, 1, 3, 4, 5, 5, 2, 6])
        t = torch.tensor([0, 1, 2, 3, 4, 5, 5, 7, 7])
        cm = confusion(p, t)
        actual = metrics(cm)
        expected = author_metrics(p.numpy(), t.numpy(), [str(i) for i in range(7)])
        np.testing.assert_allclose([actual[k] for k in ('miou', 'mf1', 'kappa', 'accuracy_percent')], expected)
        self.assertEqual(actual['valid_pixels'], 7)
        self.assertIsNone(actual['per_class_iou'][6])
        torch.testing.assert_close(confusion(p[:4], t[:4]) + confusion(p[4:], t[4:]), cm)

    def test_crop_manifest_preprocessing_and_repeated_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subdir in ('optical', 'sar', 'lbl'):
                (root / subdir).mkdir()
            rng = np.random.default_rng(23)
            for name in ('a.tif', 'b.tif'):
                tifffile.imwrite(root / 'optical' / name, rng.integers(0, 256, (48, 48, 4), dtype=np.uint8), photometric='rgb')
                tifffile.imwrite(root / 'sar' / name, rng.integers(0, 256, (48, 48), dtype=np.uint8))
                tifffile.imwrite(root / 'lbl' / name, rng.integers(0, 8, (48, 48), dtype=np.uint8) * 10)
            train, test = root / 'train.txt', root / 'test.txt'
            train.write_text('a.tif\n')
            test.write_text('b.tif\n')
            state = random.getstate()
            manifest = crop_manifest(root, train, test, size=32, files_per_split=1)
            self.assertEqual(state, random.getstate())
            self.assertEqual(manifest, crop_manifest(root, train, test, size=32, files_per_split=1))
            save_manifest(root / 'manifest.json', manifest)
            save_manifest(root / 'manifest.json', manifest)
            with self.assertRaises(ValueError):
                save_manifest(root / 'manifest.json', dict(manifest, seed=99))
            crops = load_crops(root, manifest)
            for split, name in [('train_source', 'a.tif'), ('test', 'b.tif')]:
                original = dataset(root, [name])[0]
                records = [r for r in manifest['crops'] if r['split'] == split]
                for (x, s, y), r in zip(crops[split], records):
                    region = np.s_[r['y']:r['y']+32, r['x']:r['x']+32]
                    torch.testing.assert_close(x, original[0][:, region[0], region[1]], rtol=0, atol=0)
                    torch.testing.assert_close(s, original[1][:, region[0], region[1]], rtol=0, atol=0)
                    np.testing.assert_array_equal(y.numpy(), original[2][region])
            class Toy(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(nn.Conv2d(4, 7, 1), nn.BatchNorm2d(7))
                def forward(self, x, s):
                    return self.net(torch.cat([x, s], 1))
            model = Toy().train()
            before = state_hash(model)
            one = evaluate_crops(model, crops, make_loss(), 'cpu')
            two = evaluate_crops(model, crops, make_loss(), 'cpu')
            self.assertEqual(one, two)
            self.assertEqual(before, state_hash(model))
            self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
