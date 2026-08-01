"""Run the paired V4-A WHU padding screen with immutable fixed-epoch evidence.

This is a research-only runner.  It keeps the released model, optimizer,
scheduler, augmentation order, and inference protocol, while making exactly
one optional data change:

* ``official``: released shared-zero padding for mask and SAR;
* ``mask-ignore``: semantic-mask padding is class 7 (ignore), SAR remains zero.

The scheduler horizon is always 50 epochs.  ``--stop-after-epoch`` only stops
the process and never shortens the cosine schedule.  Formal use is deliberately
limited to one torchrun process because the accepted WHU baseline was produced
on one GPU and the paired data-stream contract is simplest to audit there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import dinov3.distributed as distributed  # noqa: E402
from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from scripts.spatial_diagnostics_common import (  # noqa: E402
    class_ious_from_confusion,
    mean_iou_from_confusion,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402
from utils.utils import set_seed  # noqa: E402


NUM_CLASSES = 7
IGNORE_INDEX = NUM_CLASSES
PROTOCOL_EPOCHS = 50
DEFAULT_EVALUATION_EPOCHS = (5, 10, 15)
BACKBONE_TYPE = "dinov3_vits16"
CLASS_NAMES = ("farmland", "city", "village", "water", "forest", "road", "other")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_tensor_sha256(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def update_tensor_digest(
    digest: "hashlib._Hash", name: str, tensor: torch.Tensor
) -> None:
    value = tensor.detach().cpu().contiguous().numpy()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())


def normalized_pair_label(label: torch.Tensor) -> torch.Tensor:
    """Hide the intentional padding-label difference in data-pair hashes."""

    result = label.detach().cpu().clone()
    result[result == IGNORE_INDEX] = 0
    return result


def batch_pair_fingerprint(
    optical: torch.Tensor, sar: torch.Tensor, label: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    update_tensor_digest(digest, "optical", optical)
    update_tensor_digest(digest, "sar", sar)
    update_tensor_digest(digest, "label_ignore_to_zero", normalized_pair_label(label))
    return digest.hexdigest()


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def torch_save_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def confusion_from_arrays(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    valid = (target >= 0) & (target < NUM_CLASSES)
    if np.any((prediction[valid] < 0) | (prediction[valid] >= NUM_CLASSES)):
        raise ValueError("prediction contains an invalid class")
    encoded = target[valid] * NUM_CLASSES + prediction[valid]
    return np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def confusion_summary(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    class_ious = class_ious_from_confusion(matrix)
    return {
        "confusion": matrix.tolist(),
        "pixels": int(matrix.sum()),
        "miou": float(mean_iou_from_confusion(matrix)),
        "miou_percent": float(mean_iou_from_confusion(matrix) * 100.0),
        "class_iou_percent": {
            name: (None if np.isnan(value) else float(value * 100.0))
            for name, value in zip(CLASS_NAMES, class_ious, strict=True)
        },
    }


def parse_epoch_list(value: str) -> tuple[int, ...]:
    epochs = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not epochs or any(epoch <= 0 or epoch > PROTOCOL_EPOCHS for epoch in epochs):
        raise argparse.ArgumentTypeError("evaluation epochs must be within 1..50")
    return epochs


def scientific_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "artifact_type": "whu_v4_a_padding_screen",
        "schema_version": 1,
        "variant": args.variant,
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": BACKBONE_TYPE,
        "use_lora": False,
        "seed": args.seed,
        "scheduler_horizon_epochs": PROTOCOL_EPOCHS,
        "stop_after_epoch": args.stop_after_epoch,
        "evaluation_epochs": list(args.evaluation_epochs),
        "mask_padding_ignore": args.variant == "mask-ignore",
        "mask_fill": IGNORE_INDEX if args.variant == "mask-ignore" else 0,
        "aux_fill": 0,
        "loss_change": "none",
        "soft_ce_residual_confound": (
            "ignore positions are zeroed before a mean over all pixels"
        ),
        "train_batch_size_per_gpu": 8,
        "train_workers": args.num_workers,
        "inference_batch_size": args.inference_batch_size,
        "max_train_batches": args.max_train_batches,
        "max_test_images": args.max_test_images,
        "scope": "smoke" if args.smoke else "formal-screen",
    }


def prepare_output_dir(output_dir: Path, protocol: Mapping[str, Any]) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to use non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "protocol.json", protocol)


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def require_single_process() -> None:
    if not distributed.is_enabled():
        raise RuntimeError(
            "V4-A requires torchrun, even for one GPU; use --nproc_per_node=1"
        )
    if distributed.get_world_size() != 1:
        raise RuntimeError("V4-A formal pairing is locked to exactly one process")


def build_training_state(args: argparse.Namespace):
    # This must precede get_cfg because get_cfg constructs model, optimizer, and scheduler.
    set_seed(args.seed)
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=False,
        r=3,
        backbone_type=BACKBONE_TYPE,
        use_naf=False,
    )
    if int(cfg["epochs"]) != PROTOCOL_EPOCHS:
        raise RuntimeError("released scheduler horizon is no longer 50 epochs")
    scheduler = cfg["scheduler"]
    if int(scheduler.T_max) != PROTOCOL_EPOCHS:
        raise RuntimeError("cosine scheduler T_max must remain 50")
    return cfg, cfg["model"], cfg["optimizer"], scheduler


def build_loaders(args: argparse.Namespace, cfg: Mapping[str, Any]):
    train_dataset = build_dataset(
        "WHU",
        "train",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
        mask_padding_ignore=args.variant == "mask-ignore",
    )
    test_dataset = build_dataset(
        "WHU",
        "test",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    full_test_length = len(test_dataset)
    test_names = [Path(path).stem for path in test_dataset.rgb_files]
    if args.max_test_images is not None:
        count = min(args.max_test_images, full_test_length)
        test_dataset = Subset(test_dataset, range(count))
        test_names = test_names[:count]

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader_generator = torch.Generator().manual_seed(args.seed + 104729)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader, test_names, full_test_length, loader_generator


def train_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    max_batches: int | None,
) -> dict[str, Any]:
    model.train()
    loader.sampler.set_epoch(epoch)
    pair_digest = hashlib.sha256()
    raw_label_digest = hashlib.sha256()
    first_batch_trace: list[dict[str, Any]] = []
    class_counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    loss_total = 0.0
    batch_count = 0
    started = time.perf_counter()

    iterations = tqdm(loader, desc=f"train e{epoch}/50", dynamic_ncols=True)
    for batch_index, (optical_cpu, sar_cpu, label_cpu) in enumerate(iterations, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        fingerprint = batch_pair_fingerprint(optical_cpu, sar_cpu, label_cpu)
        pair_digest.update(bytes.fromhex(fingerprint))
        update_tensor_digest(raw_label_digest, f"label_{batch_index}", label_cpu)
        counts = torch.bincount(
            label_cpu.reshape(-1).to(torch.int64), minlength=NUM_CLASSES + 1
        )[: NUM_CLASSES + 1]
        class_counts += counts.numpy()
        if batch_index <= 10:
            first_batch_trace.append(
                {
                    "batch": batch_index,
                    "pair_sha256": fingerprint,
                    "optical_sha256": tensor_sha256(optical_cpu),
                    "sar_sha256": tensor_sha256(sar_cpu),
                    "raw_label_sha256": tensor_sha256(label_cpu),
                    "normalized_label_sha256": tensor_sha256(
                        normalized_pair_label(label_cpu)
                    ),
                }
            )

        optical = optical_cpu.to(device, non_blocking=False)
        sar = sar_cpu.to(device, non_blocking=False)
        label = label_cpu.to(device, non_blocking=False)
        optimizer.zero_grad()
        logits = model(optical, sar)
        loss = loss_fn(logits, label)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch={epoch} batch={batch_index}")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        loss_total += value
        batch_count += 1
        iterations.set_postfix(loss=f"{value:.4f}")

    if batch_count == 0:
        raise RuntimeError("training epoch produced zero batches")
    valid_pixels = int(class_counts[:NUM_CLASSES].sum())
    all_pixels = int(class_counts.sum())
    return {
        "epoch": epoch,
        "average_loss": loss_total / batch_count,
        "batches": batch_count,
        "elapsed_seconds": time.perf_counter() - started,
        "paired_data_sha256": pair_digest.hexdigest(),
        "raw_label_sha256": raw_label_digest.hexdigest(),
        "first_batch_trace": first_batch_trace,
        "label_pixel_counts_0_to_ignore": class_counts.tolist(),
        "valid_pixels": valid_pixels,
        "all_pixels": all_pixels,
        "valid_fraction": valid_pixels / all_pixels,
    }


def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    sample_names: Sequence[str],
    cfg: Mapping[str, Any],
    device: torch.device,
    inference_batch_size: int,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    pooled = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    prediction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    per_image: list[dict[str, Any]] = []
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    started = time.perf_counter()
    print(f"evaluation e{epoch}: images={len(loader.dataset)} start", flush=True)

    with torch.inference_mode():
        for image_index, (optical_cpu, sar_cpu, label_cpu) in enumerate(loader):
            optical = optical_cpu.to(device)
            sar = sar_cpu.to(device)
            scores = slide_inference(
                optical,
                model,
                dsm=sar,
                n_output_channels=NUM_CLASSES,
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=inference_batch_size,
            )
            prediction = np.ascontiguousarray(
                scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
            )
            label = np.ascontiguousarray(
                label_cpu.numpy().astype(np.int64, copy=False)
            )
            confusion = confusion_from_arrays(prediction, label)
            pooled += confusion
            prediction_digest.update(prediction.tobytes())
            label_digest.update(label.tobytes())
            record = {
                "index": image_index,
                "sample_name": sample_names[image_index],
                **confusion_summary(confusion),
            }
            per_image.append(record)
            print(
                f"evaluation e{epoch}: image={image_index + 1}/{len(loader.dataset)} "
                f"name={record['sample_name']} miou={record['miou_percent']:.6f}%",
                flush=True,
            )
            del optical, sar, scores

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "epoch": epoch,
        "evaluated_images": len(per_image),
        "prediction_sha256": prediction_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "aggregate": confusion_summary(pooled),
        "per_image": per_image,
        "elapsed_seconds": time.perf_counter() - started,
    }


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    epoch: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    unwrapped = model.module if hasattr(model, "module") else model
    return {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "protocol": dict(protocol),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "loader_generator": loader_generator.get_state(),
        },
        "resume_contract": (
            "main RNG is captured; persistent worker RNG is not resumable, so formal "
            "15-to-30 extension must restart or first add epoch-addressed augmentation"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("official", "mask-ignore"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-after-epoch", type=int, default=15)
    parser.add_argument(
        "--evaluation-epochs", type=parse_epoch_list, default=DEFAULT_EVALUATION_EPOCHS
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-images", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.stop_after_epoch = 1
        args.evaluation_epochs = (1,)
        args.max_train_batches = 1
        args.max_test_images = 1
        args.num_workers = 0
    if not 1 <= args.stop_after_epoch <= PROTOCOL_EPOCHS:
        parser.error("--stop-after-epoch must be within 1..50")
    if any(epoch > args.stop_after_epoch for epoch in args.evaluation_epochs):
        parser.error("evaluation epoch exceeds --stop-after-epoch")
    if args.num_workers < 0 or args.inference_batch_size <= 0:
        parser.error("worker count must be non-negative and inference batch positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    distributed.enable(overwrite=True)
    require_single_process()
    local_rank = get_local_rank()
    if not torch.cuda.is_available():
        raise RuntimeError("V4-A requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    protocol = scientific_protocol(args)
    dirty = git_is_dirty()
    if dirty and not args.smoke:
        raise RuntimeError(
            "formal V4-A run requires a clean committed worktree; use --smoke "
            "for pre-commit diagnostics"
        )
    protocol = {
        **protocol,
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    prepare_output_dir(args.output_dir, protocol)
    print(
        f"V4-A start variant={args.variant} seed={args.seed} "
        f"stop={args.stop_after_epoch} scope={protocol['scope']}",
        flush=True,
    )
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} gpu={protocol['gpu']}",
        flush=True,
    )

    cfg, model, optimizer, scheduler = build_training_state(args)
    initial_state_sha256 = named_tensor_sha256(model.state_dict().items())
    train_loader, test_loader, test_names, full_test_length, loader_generator = (
        build_loaders(args, cfg)
    )
    protocol_update = {
        **protocol,
        "initial_model_state_sha256": initial_state_sha256,
        "train_dataset_length": len(train_loader.dataset),
        "full_test_length": full_test_length,
        "evaluated_test_length": len(test_loader.dataset),
    }
    write_json_atomic(args.output_dir / "protocol.json", protocol_update)
    print(
        f"initial_model_state_sha256={initial_state_sha256} "
        f"train_samples={len(train_loader.dataset)} test_images={len(test_loader.dataset)}",
        flush=True,
    )

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    evaluations: list[dict[str, Any]] = []
    for epoch in range(1, args.stop_after_epoch + 1):
        print(f"epoch={epoch}/{args.stop_after_epoch} train start", flush=True)
        training = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=cfg["loss_fn"],
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        scheduler.step()
        training["learning_rates_after_scheduler_step"] = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        training["variant"] = args.variant
        write_json_atomic(args.output_dir / f"train_e{epoch}.json", training)
        print(
            f"epoch={epoch} train PASS loss={training['average_loss']:.6f} "
            f"valid_fraction={training['valid_fraction']:.6f} "
            f"data_sha256={training['paired_data_sha256']}",
            flush=True,
        )

        if epoch in args.evaluation_epochs:
            evaluation = evaluate(
                model=model,
                loader=test_loader,
                sample_names=test_names,
                cfg=cfg,
                device=device,
                inference_batch_size=args.inference_batch_size,
                epoch=epoch,
            )
            evaluation["variant"] = args.variant
            write_json_atomic(
                args.output_dir / f"evaluation_e{epoch}.json", evaluation
            )
            checkpoint_path = args.output_dir / f"checkpoint_e{epoch}.pth"
            torch_save_atomic(
                checkpoint_path,
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loader_generator=loader_generator,
                    epoch=epoch,
                    protocol=protocol_update,
                ),
            )
            evaluations.append(
                {
                    "epoch": epoch,
                    "miou_percent": evaluation["aggregate"]["miou_percent"],
                    "evaluation_path": f"evaluation_e{epoch}.json",
                    "evaluation_sha256": file_sha256(
                        args.output_dir / f"evaluation_e{epoch}.json"
                    ),
                    "checkpoint_path": checkpoint_path.name,
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                }
            )
            print(
                f"epoch={epoch} evaluation PASS "
                f"mIoU={evaluation['aggregate']['miou_percent']:.6f}%",
                flush=True,
            )

    summary = {
        "status": "PASS",
        "outcome": "PENDING_PAIRED_COMPARISON",
        "artifact_type": "whu_v4_a_padding_screen_summary",
        "variant": args.variant,
        "scope": protocol["scope"],
        "git_commit": protocol["git_commit"],
        "initial_model_state_sha256": initial_state_sha256,
        "stop_after_epoch": args.stop_after_epoch,
        "evaluations": evaluations,
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(
        f"PASS summary={args.output_dir / 'summary.json'} "
        "outcome=PENDING_PAIRED_COMPARISON",
        flush=True,
    )


if __name__ == "__main__":
    main()
