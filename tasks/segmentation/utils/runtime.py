"""Runtime helpers shared by training and evaluation entry points."""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

import dinov3.distributed as distributed


def initialize_distributed(operation: str) -> bool:
    """Enable NCCL only for an explicitly launched multi-process job."""
    raw_world_size = os.environ.get(
        "WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")
    )
    try:
        world_size = int(raw_world_size)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid distributed world size: {raw_world_size!r}"
        ) from exc
    if world_size < 1:
        raise RuntimeError(
            f"Distributed world size must be positive, got {world_size}"
        )
    if world_size == 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError(f"Multi-process {operation} requires CUDA/NCCL")

    os.environ.setdefault("NCCL_TIMEOUT", "1200")
    try:
        distributed.enable(
            overwrite=False,
            nccl_async_error_handling=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize distributed {operation}. Launch with plain "
            "Python for one GPU or torchrun for multiple GPUs."
        ) from exc
    return True


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation samples across ranks without padding duplicates."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas or distributed.get_world_size()
        self.rank = distributed.get_rank() if rank is None else rank
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"Invalid rank {self.rank}/{self.num_replicas}")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device("cuda", local_rank)


def forward_batch(model, batch, device, num_modalities):
    """Move a segmentation batch to the device and normalize label dtype."""
    if num_modalities > 1:
        image, auxiliary, label = batch
        image = image.to(device, non_blocking=True)
        auxiliary = auxiliary.to(device, non_blocking=True)
        label = label.to(device, dtype=torch.long, non_blocking=True)
        logits = model(image, auxiliary)
    else:
        image, label = batch
        image = image.to(device, non_blocking=True)
        label = label.to(device, dtype=torch.long, non_blocking=True)
        logits = model(image)
    return logits, label


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    try:
        dtype = dtypes[amp_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported AMP dtype: {amp_dtype}") from exc
    return torch.autocast(device_type="cuda", dtype=dtype)


def create_grad_scaler(device: torch.device, amp_dtype: str):
    return torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and amp_dtype == "fp16",
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def atomic_torch_save(payload, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: str | Path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def git_metadata(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)

    def run_git(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "branch": run_git("branch", "--show-current"),
            "dirty": bool(run_git("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_miou, args):
    return {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_miou": float(best_miou),
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }


def load_training_checkpoint(path, model, optimizer, scheduler, scaler, device):
    # Load on CPU first so a full backbone state does not temporarily consume a
    # second copy of GPU memory during resume.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        float(checkpoint.get("best_miou", 0.0)),
        checkpoint.get("args", {}),
    )


def reduce_confusion_matrix(cm: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(cm, dtype=torch.long, device=device)
    if distributed.is_enabled():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.cpu().numpy()
