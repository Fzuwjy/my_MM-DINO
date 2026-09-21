"""Paired, synchronous DINOv3 adaptation; never merge into shared base weights."""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

TASK_ROOT = Path(__file__).resolve().parents[2] / 'tasks' / 'segmentation'
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))


@contextmanager
def cpu_initialization(seed):
    state = torch.get_rng_state()
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)


def state_hash(module, predicate=lambda name: True):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if predicate(name):
            digest.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Adaptation(nn.Module):
    def __init__(self, din, dout, context_dim, rank=16, alpha=8, cross=True, seed=42):
        super().__init__()
        if rank <= 0 or context_dim < 16:
            raise ValueError('Invalid rank/context dimension')
        self.scale = alpha / rank
        with cpu_initialization(seed):
            self.a = nn.Linear(din, rank, bias=False)
            self.b = nn.Linear(rank, dout, bias=False)
            nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.b.weight)
        self.has_cross = cross
        self.cross_enabled = cross
        if cross:
            with cpu_initialization(seed + 100000):
                self.ca = nn.Linear(din, rank, bias=False)
                self.cb = nn.Linear(rank, dout, bias=False)
                nn.init.kaiming_uniform_(self.ca.weight, a=math.sqrt(5))
                nn.init.zeros_(self.cb.weight)
                hidden = context_dim // 16
                self.hyper = nn.Sequential(
                    nn.Linear(context_dim, hidden), nn.GELU(), nn.LayerNorm(hidden, eps=1e-6),
                    nn.Linear(hidden, rank * rank), nn.Unflatten(-1, (rank, rank)),
                    nn.LayerNorm((rank, rank), eps=1e-6))
                self.lam = nn.Parameter(torch.tensor(0.1))
        self.collect = False
        self.observation = {}

    def forward(self, x, context, base_output):
        intra = self.b(self.a(x)) * self.scale
        cross = None
        if self.cross_enabled:
            phi = self.hyper(context)
            cross = self.cb(torch.bmm(self.ca(x), phi.transpose(1, 2))) * self.lam
        if self.collect:
            with torch.no_grad():
                rms = lambda t: t.float().square().mean().sqrt()
                base_rms = rms(base_output)
                self.observation = {'base_rms': base_rms, 'intra_rms': rms(intra),
                                    'context_rms': rms(context)}
                if cross is not None:
                    self.observation.update(cross_rms=rms(cross),
                                            cross_base_ratio=rms(cross) / base_rms.clamp_min(1e-12))
        result = base_output + intra
        return result if cross is None else result + cross


class PairedDino(nn.Module):
    def __init__(self, base, variant='cola', rank=16, alpha=8,
                 output_layers=(2, 5, 8, 11), checkpoint_blocks=True, seed=42):
        super().__init__()
        if variant not in ('cola', 'intra'):
            raise ValueError(variant)
        self.base = base.requires_grad_(False)
        self.variant = variant
        self.prefix = 1 + base.n_storage_tokens
        self.output_layers = tuple(output_layers)
        self.checkpoint_blocks = checkpoint_blocks
        if tuple(sorted(set(output_layers))) != self.output_layers or output_layers[-1] >= len(base.blocks):
            raise ValueError('Invalid intermediate layers')
        self.adapters = nn.ModuleList()
        d = base.embed_dim
        for i, block in enumerate(base.blocks):
            if block.sample_drop_ratio != 0 or not hasattr(block.mlp, 'fc1'):
                raise ValueError('Only the released zero-drop-path MLP backbone is supported')
            dims = [(k, d, d) for k in ('q', 'k', 'v', 'o')]
            dims += [('up', d, block.mlp.fc1.out_features), ('down', block.mlp.fc2.in_features, d)]
            self.adapters.append(nn.ModuleList([
                nn.ModuleDict({k: Adaptation(a, b, d, rank, alpha, variant == 'cola',
                                            seed + i * 100 + m * 10 + j)
                               for j, (k, a, b) in enumerate(dims)}) for m in range(2)]))

    def pool(self, tokens):
        return tokens[:, self.prefix:].mean(dim=1)

    def _block(self, block, adapters, xo, xs, ro, rs):
        inputs = (xo, xs)
        contexts = (self.pool(xs), self.pool(xo))
        attention = []
        for x, context, mods, rope in zip(inputs, contexts, adapters, (ro, rs)):
            z = block.norm1(x)
            base_qkv = block.attn.qkv(z).chunk(3, dim=-1)
            qkv = torch.cat([mods[k](z, context, v) for k, v in zip(('q', 'k', 'v'), base_qkv)], dim=-1)
            attention.append(block.attn.compute_attention(qkv, rope=rope))
        contexts = (self.pool(attention[1]), self.pool(attention[0]))
        residuals = [x + block.ls1(block.attn.proj_drop(mods['o'](u, context, block.attn.proj(u))))
                     for x, u, context, mods in zip(inputs, attention, contexts, adapters)]
        contexts = (self.pool(residuals[1]), self.pool(residuals[0]))
        result = []
        for h, context, mods in zip(residuals, contexts, adapters):
            z = block.norm2(h)
            up = mods['up'](z, context, block.mlp.fc1(z))
            up = block.mlp.drop(block.mlp.act(up))
            down = mods['down'](up, context, block.mlp.fc2(up))
            result.append(h + block.ls2(block.mlp.drop(down)))
        return tuple(result)

    def forward(self, optical, sar, rope_table=None):
        if optical.ndim != 4 or sar.ndim != 4 or optical.shape[1] != 3 or sar.shape[1] != 1:
            raise ValueError('Expected paired Bx3xHxW optical and Bx1xHxW SAR')
        if optical.shape[0] != sar.shape[0] or optical.shape[-2:] != sar.shape[-2:]:
            raise ValueError('Unpaired inputs')
        if any(s % self.base.patch_size for s in optical.shape[-2:]):
            raise ValueError('Input dimensions must be multiples of patch size')
        xo, (h, w) = self.base.prepare_tokens_with_masks(optical)
        xs, _ = self.base.prepare_tokens_with_masks(sar.repeat(1, 3, 1, 1))
        outputs = ([], [])
        for i, (block, adapters) in enumerate(zip(self.base.blocks, self.adapters)):
            ro, rs = (rope_table[i] if rope_table is not None else
                      (self.base.rope_embed(H=h, W=w), self.base.rope_embed(H=h, W=w)))
            # Bind the block: backward recomputation must not use the final loop iteration.
            def run(a, b, ro_sin, ro_cos, rs_sin, rs_cos, block=block, adapters=adapters):
                return self._block(block, adapters, a, b, (ro_sin, ro_cos), (rs_sin, rs_cos))
            args = (xo, xs, *ro, *rs)
            if self.training and torch.is_grad_enabled() and self.checkpoint_blocks:
                xo, xs = checkpoint(run, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                xo, xs = run(*args)
            if i in self.output_layers:
                for collection, tokens in zip(outputs, (xo, xs)):
                    collection.append(self.base.norm(tokens)[:, self.prefix:])
        return outputs

    def diagnostics(self, enabled):
        for module in self.modules():
            if isinstance(module, Adaptation):
                module.collect = enabled
                module.observation = {}

    def observations(self):
        result = {}
        for name, module in self.named_modules():
            if not isinstance(module, Adaptation):
                continue
            values = {k: float(v) for k, v in module.observation.items()}
            for part in ('a', 'b', 'ca', 'cb', 'hyper'):
                if hasattr(module, part):
                    grads = [p.grad.detach().float().square().sum() for p in getattr(module, part).parameters()
                             if p.grad is not None]
                    values[part + '_grad_norm'] = float(torch.stack(grads).sum().sqrt()) if grads else None
            if module.has_cross:
                values['lambda'] = float(module.lam.detach())
                values['lambda_grad'] = float(module.lam.grad) if module.lam.grad is not None else None
            result[name] = values
        return result


class CoLASegmenter(nn.Module):
    def __init__(self, base, variant='cola', rank=16, checkpoint_blocks=True, seed=42):
        super().__init__()
        from models.MMDINO.sample_adapter import SampleAdapter
        from models.MMDINO.Decoder import Decoder
        self.backbone = PairedDino(base, variant, rank, checkpoint_blocks=checkpoint_blocks, seed=seed)
        with cpu_initialization(seed + 200000):
            self.adapter = SampleAdapter(base.embed_dim, num_modalities=2)
        with cpu_initialization(seed + 300000):
            self.decoder = Decoder(n_classes=7, num_modalities=2)

    def forward(self, optical, sar):
        outputs = self.backbone(optical, sar)
        outputs = self.adapter(*outputs, patch_h=optical.shape[-2] // 16, patch_w=optical.shape[-1] // 16)
        logits = self.decoder(*outputs)
        if logits.shape[-2:] != optical.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, optical.shape[-2:], mode='bilinear', align_corners=False)
        return logits

    def provenance(self):
        return {'variant': self.backbone.variant,
                'base_hash': state_hash(self.backbone.base), 'adapter_hash': state_hash(self.adapter),
                'decoder_hash': state_hash(self.decoder),
                'intra_hash': state_hash(self.backbone.adapters, lambda n: n.endswith(('.a.weight', '.b.weight'))),
                'adaptation_parameters': sum(p.numel() for p in self.backbone.adapters.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)}


def build_model(weights, variant='cola', checkpoint_blocks=True, seed=42):
    from dinov3.hub.backbones import dinov3_vits16
    with cpu_initialization(seed):
        base = dinov3_vits16(pretrained=False)
        base.load_state_dict(torch.load(Path(weights), map_location='cpu', weights_only=True), strict=True)
    if any(not torch.isfinite(t).all() for t in base.state_dict().values()):
        raise ValueError('Non-finite backbone checkpoint')
    return CoLASegmenter(base, variant, checkpoint_blocks=checkpoint_blocks, seed=seed)
