"""Single-GPU WHU research runner. Default is a bounded deployment preflight."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .model import build_model, state_hash
from .diagnostics import (confusion, crop_manifest, evaluate_crops, evaluation, file_hash,
                          load_crops, metrics, save_manifest)
from datasets.WHU_dataset import WHU_Dataset
from losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from utils.inference import slide_inference

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = {'epochs': 50, 'seed': 42, 'batch_size': 8, 'accumulation': 1, 'crop': 512,
            'rank_intra': 16, 'rank_cross': 16, 'alpha': 8, 'lambda_initial': 0.1,
            'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 0.01, 'eta_min': 1e-7,
            'eval_every': 5, 'stride': 341, 'precision': 'fp32', 'normalization': 'common',
            'checkpoint_blocks': True, 'train_cache_per_worker': 32,
            'diagnostic_seed': 20260921, 'diagnostic_files_per_split': 8, 'diagnostic_crops_per_file': 2}


def seed_training():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loss():
    return JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=7),
                     DiceLoss(smooth=0.05, ignore_index=7), 1.0, 1.0)


def make_optimizer(model):
    groups = [{'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': 1e-4},
              {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad],
               'lr': 1e-4}]
    selected = [p for g in groups for p in g['params']]
    assert len({id(p) for p in selected}) == len(selected)
    assert {id(p) for p in selected} == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def dataset(root, names, train=False):
    return WHU_Dataset(names, str(root / 'optical' / '{}'), str(root / 'lbl' / '{}'),
                       'train' if train else 'test', window_size=(512, 512), normalize_type='common',
                       sar_dir=str(root / 'sar' / '{}'), cache_size=32 if train else 0)


def check_inputs(args):
    names = {}
    for split, expected in [('train', 80), ('test', 20)]:
        source = args.data / (split + '_list.txt')
        sealed = REPO / 'splits' / 'whu' / f'official_{split}.txt'
        if source.read_bytes() != sealed.read_bytes():
            raise ValueError(f'{split} list differs from sealed official list')
        names[split] = source.read_text().split()
        if len(set(names[split])) != expected or len(names[split]) != expected:
            raise ValueError('Invalid split length or duplicate filenames')
        for name in names[split]:
            if Path(name).name != name:
                raise ValueError('Split filenames must be basenames')
            for modality in ('optical', 'sar', 'lbl'):
                if not (args.data / modality / name).is_file():
                    raise FileNotFoundError(args.data / modality / name)
    if set(names['train']) & set(names['test']):
        raise ValueError('Overlapping train/test files')
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    return names


def resources(device):
    result = {'torch': torch.__version__, 'device': str(device)}
    if device.type == 'cuda':
        torch.cuda.synchronize()
        result.update(gpu=torch.cuda.get_device_name(), allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      total_gib=torch.cuda.get_device_properties(device).total_memory / 2**30)
    for name, path in [('cgroup_v2', '/sys/fs/cgroup/memory.max'),
                       ('cgroup_v1', '/sys/fs/cgroup/memory/memory.limit_in_bytes')]:
        if Path(path).exists():
            result[name] = Path(path).read_text().strip()
    try:
        import resource
        result['host_process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    return result


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def save_checkpoint(path, model, optimizer, scheduler, epoch, metadata):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'metadata': metadata,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}, temp)
    temp.replace(path)


def full_test(model, samples, device, crop_batch):
    cm = torch.zeros(7, 7, dtype=torch.int64)
    with evaluation(model):
        for i in range(len(samples)):
            optical, sar, target = samples[i]
            logits = slide_inference(optical[None].to(device), model, n_output_channels=7,
                                     crop_size=(512, 512), stride=(341, 341),
                                     dsm=sar[None].to(device), batch_size=crop_batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite sliding-window prediction')
            cm += confusion(logits.argmax(1), target)
            del logits
            print(f'full_test_image={i + 1}/{len(samples)}', flush=True)
    return metrics(cm)


def require_training_gpu(device, smoke_only=False):
    allowed = ('3090', '4090') if smoke_only else ('3090',)
    if device.type != 'cuda' or not any(name in torch.cuda.get_device_name(device) for name in allowed):
        raise RuntimeError('Formal training requires 3090; short smoke accepts 3090/4090. Titan Xp is deployment only.')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('This protocol is single GPU; do not launch with multi-rank torchrun')


def preflight(args, model, names, device, metadata):
    # Real TIFF decoding and a small eval forward, never a training update.
    tiny = {'crops': [{'split': 'train_source', 'file': names['train'][0],
                       'y': 0, 'x': 0, 'height': 64, 'width': 64}]}
    optical, sar, target = load_crops(args.data, tiny)['train_source'][0]
    with evaluation(model):
        output = model(optical[None].to(device), sar[None].to(device))
    if output.shape != (1, 7, 64, 64) or not torch.isfinite(output).all():
        raise AssertionError('Deployment forward failed')
    metadata.update(status='deployment_preflight_passed', input_size=64,
                    formal_512_memory_acceptance=False, resources=resources(device))
    print(json.dumps(metadata, indent=2), flush=True)


def smoke(args, model, names, device, metadata):
    require_training_gpu(device, smoke_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', metadata)
    optimizer, _ = make_optimizer(model)
    loss_fn = make_loss()
    samples = dataset(args.data, names['train'], True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(2):
        batch = [samples[i] for i in range(8)]
        optical, sar, target = [torch.stack([torch.as_tensor(v) for v in part]).to(device) for part in zip(*batch)]
        optimizer.zero_grad(set_to_none=True)
        model.backbone.diagnostics(True)
        loss = loss_fn(model(optical, sar), target.long())
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite smoke loss')
        loss.backward()
        report = {'step': step + 1, 'loss': float(loss.detach()), 'branches': model.backbone.observations()}
        append_json(args.output / 'steps.jsonl', report)
        optimizer.step()
    model.backbone.diagnostics(False)
    train_resources = resources(device)
    train_seconds = time.perf_counter() - start
    write_json(args.output / 'train_smoke.json', {'resources': train_resources, 'two_steps_seconds': train_seconds})
    del loss, optimizer, batch, optical, sar, target
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sample = dataset(args.data, names['test'])[0]
    optical, sar = [v[:, :512, :512][None].to(device) for v in sample[:2]]
    with evaluation(model):
        single = model(optical, sar)
        batched = model(optical.repeat(args.eval_batch, 1, 1, 1), sar.repeat(args.eval_batch, 1, 1, 1))
        agreement = float((batched.argmax(1) == single.argmax(1)).float().mean())
        batch_comparison = {'max_abs_error': float((batched - single).abs().max()),
                            'rms_error': float((batched - single).square().mean().sqrt()),
                            'argmax_agreement': agreement, 'rtol': 1e-3, 'atol': 1e-4,
                            'minimum_argmax_agreement': 0.999}
        write_json(args.output / 'eval_batch_comparison.json', batch_comparison)
        # Original MM-DINO also differs by ~7e-5 between B=1 and B=32 under
        # its unchanged cuDNN TF32 setting. Zero-delta same-batch tests stay exact.
        torch.testing.assert_close(batched, single.expand_as(batched), rtol=1e-3, atol=1e-4)
        if agreement < 0.999:
            raise AssertionError('Batch-size argmax agreement below 99.9%')
    report = {'train': train_resources, 'train_two_steps_seconds': train_seconds,
              'eval': resources(device), 'eval_batch': args.eval_batch,
              'eval_batch_comparison': batch_comparison,
              'frozen_base_unchanged': state_hash(model.backbone.base) == metadata['initial']['base_hash']}
    if not report['frozen_base_unchanged']:
        raise AssertionError('Frozen weights changed')
    del sample, optical, sar, single, batched
    manifest = crop_manifest(args.data, args.data / 'train_list.txt', args.data / 'test_list.txt')
    save_manifest(args.output / 'fixed_crops.json', manifest)
    crops = load_crops(args.data, manifest)
    bn_buffer = lambda n: n.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
    before = state_hash(model, bn_buffer)
    report['fixed_crop_diagnostic'] = evaluate_crops(model, crops, loss_fn, device, batch_size=2)
    report['fixed_eval_bn_unchanged'] = before == state_hash(model, bn_buffer)
    if not report['fixed_eval_bn_unchanged']:
        raise AssertionError('Fixed eval changed BN buffers')
    write_json(args.output / 'smoke.json', report)
    print(json.dumps(report, indent=2), flush=True)


def train(args, model, names, device, metadata):
    require_training_gpu(device)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', metadata)
    manifest = crop_manifest(args.data, args.data / 'train_list.txt', args.data / 'test_list.txt')
    save_manifest(args.output / 'fixed_crops.json', manifest)
    crops = load_crops(args.data, manifest)
    training = dataset(args.data, names['train'], True)
    testing = dataset(args.data, names['test'])
    sampler = DistributedSampler(training, num_replicas=1, rank=0, seed=42, shuffle=True)
    loader = DataLoader(training, batch_size=8, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(42))
    optimizer, scheduler = make_optimizer(model)
    loss_fn = make_loss()
    best = -float('inf')
    for epoch in range(1, 51):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step, (optical, sar, target) in enumerate(loader, start=1):
            optical, sar, target = optical.to(device), sar.to(device), target.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            collect = step in (1, 2, len(loader))
            model.backbone.diagnostics(collect)
            logits = model(optical, sar)
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch={epoch} step={step}; run preserved')
            loss.backward()
            if collect:
                observation = model.backbone.observations()
                input_info = {k: {'mean': float(t.mean()), 'std': float(t.std()),
                                  'sha256': __import__('hashlib').sha256(t.detach().cpu().numpy().tobytes()).hexdigest()}
                              for k, t in [('optical', optical), ('sar', sar)]}
                append_json(args.output / 'branches.jsonl', {'epoch': epoch, 'step': step,
                                                           'inputs': input_info, 'branches': observation})
            optimizer.step()
            value = float(loss.detach())
            total_loss += value
            append_json(args.output / 'steps.jsonl', {'epoch': epoch, 'step': step, 'loss': value})
            if step % 20 == 0 or step == 1:
                print(f'epoch={epoch}/50 step={step}/{len(loader)} loss={value:.6f}', flush=True)
        model.backbone.diagnostics(False)
        del logits, loss, optical, sar, target
        model.zero_grad(set_to_none=True)
        report = {'epoch': epoch, 'loss': total_loss / len(loader), 'lr': optimizer.param_groups[0]['lr'],
                  'train_seconds': time.perf_counter() - start, 'resources': resources(device)}
        scheduler.step()
        append_json(args.output / 'train.jsonl', report)
        save_checkpoint(args.output / 'latest.pt', model, optimizer, scheduler, epoch, metadata)
        if epoch % 5 == 0:
            eval_start = time.perf_counter()
            fixed = evaluate_crops(model, crops, loss_fn, device, batch_size=2)
            whole = full_test(model, testing, device, args.eval_batch)
            append_json(args.output / 'eval.jsonl', {'epoch': epoch, 'fixed': fixed, 'full_test': whole,
                                                   'eval_seconds': time.perf_counter() - eval_start})
            print(json.dumps({'epoch': epoch, 'full_test_miou': whole['miou'],
                              'fixed_train_miou': fixed['train_source']['miou'],
                              'fixed_test_miou': fixed['test']['miou']}), flush=True)
            if whole['miou'] is not None and whole['miou'] > best:
                best = whole['miou']
                save_checkpoint(args.output / 'best_test.pt', model, optimizer, scheduler, epoch, metadata)
        if epoch == 50:
            save_checkpoint(args.output / 'e50.pt', model, optimizer, scheduler, epoch, metadata)
    if state_hash(model.backbone.base) != metadata['initial']['base_hash']:
        raise AssertionError('Frozen weights changed')
    write_json(args.output / 'complete.json', {'epochs': 50, 'best_test_miou': best,
                                              'frozen_base_unchanged': True})


def main():
    root = Path(os.environ.get('MM_DINO_PERSISTENT_ROOT', '/mnt/csip-113/wjy/MM-DINO'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'train'), default='preflight')
    parser.add_argument('--variant', choices=('cola', 'intra'), required=True)
    parser.add_argument('--data', type=Path, default=root / 'datasets' / 'whu-opt-sar')
    parser.add_argument('--weights', type=Path, default=root / 'weights' / 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval-batch', type=int, default=32)
    args = parser.parse_args()
    if args.workers < 0 or args.eval_batch < 1:
        parser.error('Invalid workers/eval batch')
    args.output = args.output or root / 'outputs' / f'whu-vits-{args.variant}-{args.mode}'
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if args.mode != 'preflight':
        require_training_gpu(device, smoke_only=args.mode == 'smoke')
        if git('status', '--porcelain'):
            raise RuntimeError('Commit the code before a recorded smoke or formal run')
        if args.output.exists():
            raise FileExistsError(f'Refusing to overwrite {args.output}')
    names = check_inputs(args)
    seed_training()
    model = build_model(args.weights, args.variant)
    metadata = {'protocol': PROTOCOL, 'variant': args.variant, 'mode': args.mode,
                'commit': git('rev-parse', 'HEAD'), 'initial': model.provenance(),
                'weights_sha256': file_hash(args.weights), 'workers': args.workers,
                'eval_batch': args.eval_batch, 'diagnostic_batch': 2,
                'train_split_sha256': file_hash(args.data / 'train_list.txt'),
                'test_split_sha256': file_hash(args.data / 'test_list.txt')}
    model.to(device)
    {'preflight': preflight, 'smoke': smoke, 'train': train}[args.mode](args, model, names, device, metadata)


if __name__ == '__main__':
    main()
