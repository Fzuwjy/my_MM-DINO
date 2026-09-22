"""Eval-only paired cross on/off audit. No training, parameter edits or checkpoint overwrite."""
import argparse
import json
from pathlib import Path
import time

import torch

from .model import Adaptation, build_model, state_hash
from .diagnostics import evaluation, load_crops, confusion, metrics, file_hash


def rms(x):
    return float(x.double().square().mean().sqrt())


def cosine(x, y):
    x, y = x.double().reshape(-1), y.double().reshape(-1)
    denom = x.norm() * y.norm()
    return float(torch.dot(x, y) / denom) if denom > 0 else None


def ratio(a, b):
    return a / b if b > 1e-12 else None


def delta_stats(on, off):
    d = on - off
    return {'on_rms': rms(on), 'off_rms': rms(off), 'delta_rms': rms(d),
            'relative_delta': ratio(rms(d), rms(on))}


def fusion_stats(po, ps, go, pf_o, pf_s, gf, weights):
    do, ds, dg = po - pf_o, ps - pf_s, go - gf
    a, b = weights[0] * do, weights[1] * ds
    norm_sum = rms(a) + rms(b)
    incoherent = (rms(a)**2 + rms(b)**2)**0.5
    return {'optical': delta_stats(po, pf_o), 'sar': delta_stats(ps, pf_s),
            'fused': delta_stats(go, gf), 'delta_cosine': cosine(do, ds),
            'weighted_optical_delta_rms': rms(a), 'weighted_sar_delta_rms': rms(b),
            'retention_vs_weighted_norm_sum': ratio(rms(dg), norm_sum),
            'retention_vs_orthogonal_sum': ratio(rms(dg), incoherent),
            'fusion_identity_max_error': float((dg - a - b).abs().max())}


def phi_statistics(values, files):
    x = torch.stack(values).double().flatten(1)
    norms = x.norm(dim=1)
    valid = norms > 1e-12
    normalized = x / norms.clamp_min(1e-12)[:, None]
    cos = normalized @ normalized.T
    pairs = torch.triu(torch.ones(len(x), len(x), dtype=torch.bool), diagonal=1)
    pairs &= valid[:, None] & valid[None, :]
    different = torch.tensor([[a != b for b in files] for a in files]) & pairs
    centered = x - x.mean(0)
    def describe(v):
        return {'count': v.numel(), 'mean': float(v.mean()) if v.numel() else None,
                'min': float(v.min()) if v.numel() else None,
                'max': float(v.max()) if v.numel() else None}
    return {'samples': len(x), 'mean_element_population_variance': float(x.var(0, unbiased=False).mean()),
            'relative_centered_rms': ratio(rms(centered), rms(x)),
            'pair_cosine': describe(cos[pairs]), 'different_image_pair_cosine': describe(cos[different])}


class Capture:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.on = True
        self.phi = {}
        self.branches = {}
        self.projected = {}
        self.raw = None
        self.fused = None
        for i in (2, 5, 8, 11):
            for m in range(2):
                for projection, module in model.backbone.adapters[i][m].items():
                    name = f'{i}.{m}.{projection}'
                    self.handles.append(module.register_forward_hook(self.branch_hook(name)))
        for i, layer in enumerate(model.adapter.resize_layers):
            self.handles.append(layer.register_forward_hook(self.project_hook(i)))
        self.handles.append(model.backbone.register_forward_hook(self.backbone_hook))
        self.handles.append(model.adapter.register_forward_hook(self.adapter_hook))

    def branch_hook(self, name):
        def hook(module, args, output):
            if not self.on:
                return
            x, context, base = args
            # Recompute only the small adaptation math in eval; never alter the actual output.
            phi = module.hyper(context)
            intra = module.b(module.a(x)) * module.scale
            cross = module.cb(torch.bmm(module.ca(x), phi.transpose(1, 2))) * module.lam
            row = {'lambda': float(module.lam)}
            for label, region in [('patch', slice(self.model.backbone.prefix, None)), ('all', slice(None))]:
                c, l, w = cross[:, region], intra[:, region], base[:, region]
                row[label] = {'cross_rms': rms(c), 'intra_rms': rms(l), 'base_rms': rms(w),
                              'cross_intra_ratio': ratio(rms(c), rms(l)),
                              'cross_base_ratio': ratio(rms(c), rms(w)), 'cross_intra_cosine': cosine(c, l)}
            self.branches[name] = row
            self.phi[name] = phi[0].detach().cpu()
        return hook

    def project_hook(self, i):
        def hook(module, args, output):
            self.projected.setdefault(i, []).append(output.detach().cpu())
        return hook

    def backbone_hook(self, module, args, output):
        self.raw = [[x.detach().cpu() for x in stream] for stream in output]

    def adapter_hook(self, module, args, output):
        if any(not torch.equal(a, b) for a, b in zip(*output)):
            raise AssertionError('Expected released identical fused output slots')
        self.fused = [x.detach().cpu() for x in output[0]]

    def forward(self, optical, sar, enabled):
        self.on = enabled
        self.projected = {}
        self.branches = {}
        self.phi = {}
        for m in self.model.backbone.modules():
            if isinstance(m, Adaptation):
                m.cross_enabled = enabled
        logits = self.model(optical, sar).cpu()
        return {'raw': self.raw, 'projected': self.projected, 'fused': self.fused, 'logits': logits,
                'branches': self.branches, 'phi': self.phi}

    def close(self):
        for handle in self.handles:
            handle.remove()
        for m in self.model.backbone.modules():
            if isinstance(m, Adaptation):
                m.cross_enabled = m.has_cross


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/mnt/csip-113/wjy/MM-DINO'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--save-features', action='store_true')
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 32:
        parser.error('limit must be 1..32')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_model(args.root / 'weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    epoch = saved['epoch']
    del saved
    before = state_hash(model)
    model.cuda()
    manifest = json.loads((args.run / 'fixed_crops.json').read_text())
    # Interleave train/test, so a small pilot includes both groups.
    groups = [[r for r in manifest['crops'] if r['split'] == split] for split in ('train_source','test')]
    records = [r for pair in zip(*groups) for r in pair][:args.limit]
    weights = [float(torch.sigmoid(model.adapter.modality_weights[f'weight_modality_{m}'])) for m in range(2)]
    weights = [w / sum(weights) for w in weights]
    capture = Capture(model)
    rows, phi_values = [], {}
    cms = {s: {mode: torch.zeros(7,7,dtype=torch.int64) for mode in ('on','off')} for s in ('train_source','test')}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with evaluation(model):
            for index, record in enumerate(records):
                crop = load_crops(args.root / 'datasets/whu-opt-sar', {'crops':[record]})[record['split']][0]
                optical, sar, target = crop
                optical, sar = optical[None].cuda(), sar[None].cuda()
                on = capture.forward(optical, sar, True)
                off = capture.forward(optical, sar, False)
                scales = {}
                for i, block in enumerate((2,5,8,11)):
                    po, ps = on['projected'][i]
                    pfo, pfs = off['projected'][i]
                    scales[str(block)] = fusion_stats(po, ps, on['fused'][i], pfo, pfs, off['fused'][i], weights)
                    scales[str(block)]['raw_optical'] = delta_stats(on['raw'][0][i], off['raw'][0][i])
                    scales[str(block)]['raw_sar'] = delta_stats(on['raw'][1][i], off['raw'][1][i])
                logits = delta_stats(on['logits'], off['logits'])
                logits['cosine'] = cosine(on['logits'], off['logits'])
                logits['changed_prediction_fraction'] = float((on['logits'].argmax(1) != off['logits'].argmax(1)).float().mean())
                row = {'crop':record,'branches':on['branches'],'scales':scales,'output_scores':logits}
                rows.append(row)
                for name, value in on['phi'].items():
                    phi_values.setdefault(name,[]).append(value)
                for mode, value in [('on',on),('off',off)]:
                    cms[record['split']][mode] += confusion(value['logits'].argmax(1), target)
                with (args.output/'samples.jsonl').open('a') as f:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                if args.save_features:
                    torch.save({'crop':record,'weights':weights,'on':on,'off':off},args.output/f'features_{index:02d}.pt')
                print(f'crop={index+1}/{len(records)} file={record["file"]} elapsed={time.perf_counter()-started:.1f}s peak={torch.cuda.max_memory_reserved()/2**30:.3f}GiB',flush=True)
                del on,off,optical,sar
    finally:
        capture.close()
    if state_hash(model) != before:
        raise AssertionError('Audit changed model parameters or buffers')
    phi_stats = {}
    for name, values in phi_values.items():
        phi_stats[name] = {}
        for group in ('all','train_source','test'):
            indices = [i for i,r in enumerate(records) if group=='all' or r['split']==group]
            if indices:
                phi_stats[name][group] = phi_statistics([values[i] for i in indices],[records[i]['file'] for i in indices])
    torch.save(phi_values,args.output/'phi.pt')
    report = {'checkpoint':str(args.checkpoint),'checkpoint_sha256':file_hash(args.checkpoint),'epoch':epoch,
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'samples':len(records),'weights':weights,
              'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'seconds':time.perf_counter()-started,
              'state_unchanged':True,'phi':phi_stats,
              'crop_metrics':{s:{m:metrics(cm) for m,cm in modes.items()} for s,modes in cms.items()}}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('audit_complete',flush=True)


if __name__ == '__main__':
    main()
