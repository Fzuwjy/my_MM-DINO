"""Deterministic diagnostics isolated from training RNG, mode and BN state."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import tifffile
from skimage.io import imread
from torchvision.transforms import functional as TF


@contextmanager
def evaluation(model):
    modes = [(module, module.training) for module in model.modules()]
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def confusion(prediction, target, classes=7):
    p = torch.as_tensor(prediction).detach().to(device='cpu', dtype=torch.int64).reshape(-1)
    t = torch.as_tensor(target).detach().to(device='cpu', dtype=torch.int64).reshape(-1)
    valid = (t >= 0) & (t < classes) & (p >= 0) & (p < classes)
    return torch.bincount(t[valid] * classes + p[valid], minlength=classes ** 2).reshape(classes, classes)


def metrics(cm):
    values = cm.double()
    diag = values.diag()
    union = values.sum(0) + values.sum(1) - diag
    iou = diag / union
    f1 = 2 * diag / (values.sum(0) + values.sum(1))
    total = values.sum()
    pa = diag.sum() / total
    pe = (values.sum(0) * values.sum(1)).sum() / total.square()
    scalar = lambda value: float(value) if torch.isfinite(value) else None
    return {'miou': scalar(iou.nanmean()), 'per_class_iou': [scalar(v) for v in iou],
            'mf1': scalar(f1.nanmean()), 'accuracy_percent': scalar(100 * pa),
            'kappa': scalar((pa - pe) / (1 - pe)), 'valid_pixels': int(total), 'confusion': cm.tolist()}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def crop_manifest(data_root, train_list, test_list, size=512, files_per_split=8, crops_per_file=2, seed=20260921):
    """Select by filenames and geometry only; never select using labels or predictions."""
    root = Path(data_root)
    rng = random.Random(seed)
    result = {'version': 1, 'seed': seed, 'size': size, 'normalization': 'common',
              'train_split_sha256': file_hash(train_list), 'test_split_sha256': file_hash(test_list), 'crops': []}
    for split, source in [('train_source', train_list), ('test', test_list)]:
        names = sorted(Path(source).read_text().split())
        if len(names) < files_per_split:
            raise ValueError('Insufficient images for fixed diagnostic crops')
        for name in sorted(rng.sample(names, files_per_split)):
            with tifffile.TiffFile(root / 'optical' / name) as image:
                height, width = image.series[0].shape[:2]
            if min(height, width) < size:
                raise ValueError(f'Diagnostic image too small: {name}')
            used = set()
            if (height - size + 1) * (width - size + 1) < crops_per_file:
                raise ValueError('Insufficient distinct crop positions')
            while len(used) < crops_per_file:
                used.add((rng.randrange(height - size + 1), rng.randrange(width - size + 1)))
            for y, x in sorted(used):
                result['crops'].append({'split': split, 'file': name, 'y': y, 'x': x,
                                        'height': size, 'width': size, 'image_shape': [height, width]})
    return result


def save_manifest(path, manifest):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Existing diagnostic crop manifest differs; refusing replacement')
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def load_crops(root, manifest):
    """Retain only cropped tensors; never put whole images into the training LRU."""
    root = Path(root)
    result = {'train_source': [], 'test': []}
    records = manifest['crops']
    for name in sorted({r['file'] for r in records}):
        optical = imread(root / 'optical' / name)
        sar = imread(root / 'sar' / name)
        label = imread(root / 'lbl' / name)
        if optical.shape[:2] != sar.shape[:2] or label.shape[:2] != optical.shape[:2]:
            raise ValueError(f'Misaligned modalities: {name}')
        for r in (r for r in records if r['file'] == name):
            y, x, h, w = r['y'], r['x'], r['height'], r['width']
            region = np.s_[y:y+h, x:x+w]
            rgb = TF.normalize(TF.to_tensor(np.ascontiguousarray(optical[region][:, :, :3])),
                               (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            aux = TF.to_tensor(np.ascontiguousarray((sar[region] / 255.0).astype(np.float32)))
            target = label[region].astype(np.int32) / 10 - 1
            target[target == -1] = 7
            target = torch.from_numpy(target.astype(np.int64))
            result[r['split']].append((rgb, aux, target))
    return result


def evaluate_crops(model, crops, loss_fn, device, batch_size=2):
    result = {}
    with evaluation(model):
        for split, samples in crops.items():
            cm = torch.zeros(7, 7, dtype=torch.int64)
            total_loss, count = 0.0, 0
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                optical, sar, target = [torch.stack(part).to(device) for part in zip(*batch)]
                logits = model(optical, sar)
                loss = loss_fn(logits, target)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise FloatingPointError(f'Non-finite fixed-crop eval: {split}')
                total_loss += float(loss) * len(batch)
                count += len(batch)
                cm += confusion(logits.argmax(1), target)
            result[split] = dict(metrics(cm), loss=total_loss / count, crops=count,
                                 loss_batch_size=batch_size)
    result['miou_gap_train_minus_test'] = (
        result['train_source']['miou'] - result['test']['miou']
        if all(result[s]['miou'] is not None for s in ('train_source', 'test')) else None)
    return result
