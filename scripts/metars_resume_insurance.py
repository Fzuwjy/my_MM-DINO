"""Checkpoint sidecar support for MetaRS state omitted by EVER."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


FORMAT_VERSION = 1
SIDECAR_TEMPLATE = "resume-state-{step}.pth"


def _cpu_clone(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


def capture_resume_state(model, global_step: int) -> dict:
    covariance = []
    for layer in model.cov_matrix_layer:
        covariance.append(
            {
                "mask_matrix": _cpu_clone(layer.mask_matrix),
                "num_sensitive": _cpu_clone(layer.num_sensitive),
                "var_matrix": _cpu_clone(layer.var_matrix),
                "count_var_cov": int(layer.count_var_cov),
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "global_step": int(global_step),
        "covariance": covariance,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def restore_resume_state(model, payload: dict, expected_step: int) -> None:
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("unsupported MetaRS resume sidecar version")
    if int(payload.get("global_step", -1)) != int(expected_step):
        raise RuntimeError("MetaRS checkpoint and resume sidecar steps differ")

    saved_layers = payload.get("covariance", [])
    if len(saved_layers) != len(model.cov_matrix_layer):
        raise RuntimeError("MetaRS resume sidecar has the wrong covariance layer count")
    for layer, saved in zip(model.cov_matrix_layer, saved_layers):
        device = layer.i.device
        mask = saved.get("mask_matrix")
        layer.mask_matrix = None if mask is None else mask.to(device)
        sensitive = saved.get("num_sensitive", 0)
        layer.num_sensitive = (
            sensitive.to(device) if isinstance(sensitive, torch.Tensor) else sensitive
        )
        variance = saved.get("var_matrix")
        layer.var_matrix = None if variance is None else variance.to(device)
        layer.count_var_cov = int(saved.get("count_var_cov", 0))

    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def _last_checkpoint_step(model_dir: Path) -> int | None:
    info_path = model_dir / "checkpoint_info.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["last"]["step"])


def restore_latest_sidecar(launcher, begin_mmr_iter: int) -> None:
    model_dir = Path(launcher.model_dir)
    step = _last_checkpoint_step(model_dir)
    if step is None:
        return
    sidecar = model_dir / SIDECAR_TEMPLATE.format(step=step)
    if not sidecar.is_file():
        if step >= begin_mmr_iter:
            raise RuntimeError(
                f"checkpoint {step} is not safely resumable: missing {sidecar.name}"
            )
        return
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    restore_resume_state(launcher.unwrapped_model, payload, step)
    launcher.logger.info(f"restored MetaRS resume sidecar: {sidecar.name}")


def save_sidecar_atomic(launcher) -> Path:
    step = int(launcher.checkpoint.global_step)
    destination = Path(launcher.model_dir) / SIDECAR_TEMPLATE.format(step=step)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(capture_resume_state(launcher.unwrapped_model, step), temporary)
    os.replace(temporary, destination)
    launcher.logger.info(f"saved MetaRS resume sidecar: {destination.name}")
    return destination


def register_resume_insurance(launcher, interval_epoch: int, begin_mmr_iter: int) -> None:
    from ever.interface import Callback

    restore_latest_sidecar(launcher, begin_mmr_iter)

    class MetaRSResumeSidecarCallback(Callback):
        def __init__(self):
            # Run atomically before EVER publishes its matching checkpoint.
            super().__init__(
                epoch_interval=interval_epoch,
                only_master=True,
                prior=-1,
                before_train=False,
                after_train=True,
            )

        def func(self):
            save_sidecar_atomic(self.launcher)

        def name(self):
            return "MetaRSResumeSidecar"

    launcher.register_callback(MetaRSResumeSidecarCallback())
