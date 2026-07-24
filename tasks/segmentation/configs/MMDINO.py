"""Configuration factory for MM-DINO segmentation experiments."""

from __future__ import annotations

from pathlib import Path

import torch.optim as optim

import dinov3.distributed as distributed
from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss
from models.MMDINO.dino_segment import build_model

from .common_cfg import BACKBONE_WEIGHT_FILES, WEIGHTS_ROOT, get_labels


def resolve_backbone_weights(
    backbone_type: str,
    backbone_weights: str | Path | None = None,
    weights_root: str | Path | None = None,
) -> Path:
    try:
        filename = BACKBONE_WEIGHT_FILES[backbone_type]
    except KeyError as exc:
        supported = ", ".join(BACKBONE_WEIGHT_FILES)
        raise ValueError(
            f"Unsupported backbone '{backbone_type}'. Supported backbones: {supported}"
        ) from exc

    if backbone_weights is not None:
        path = Path(backbone_weights).expanduser().resolve()
    else:
        root = Path(weights_root).expanduser().resolve() if weights_root else WEIGHTS_ROOT
        path = root / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"DINOv3 backbone weights not found: {path}. "
            "Pass --backbone-weights, set MM_DINO_WEIGHTS_ROOT, or place the "
            f"file at {WEIGHTS_ROOT / filename}."
        )
    return path


def get_cfg(model_name=None, dataset_name=None, **kwargs):
    if model_name is None:
        raise ValueError("Model name must be specified")
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    base_lr = float(kwargs.get("base_lr", 1e-4))
    batch_size = int(kwargs.get("batch_size", 8))
    epochs = int(kwargs.get("epochs", 50))
    window_size_arg = kwargs.get("window_size", 512)
    window_size = (
        (int(window_size_arg), int(window_size_arg))
        if isinstance(window_size_arg, (int, str))
        else tuple(window_size_arg)
    )
    if len(window_size) != 2 or min(window_size) <= 0:
        raise ValueError(f"Invalid window size: {window_size}")

    labels = get_labels(dataset_name)
    ignore_index = len(labels)
    loss_fn = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
        DiceLoss(smooth=0.05, ignore_index=ignore_index),
        1.0,
        1.0,
    )

    backbone_type = kwargs.get("backbone_type", "dinov3_vits16")
    if model_name == "DINOv3_ResNet50":
        backbone_weights = None
    else:
        backbone_weights = resolve_backbone_weights(
            backbone_type,
            backbone_weights=kwargs.get("backbone_weights"),
            weights_root=kwargs.get("weights_root"),
        )

    model = build_model(
        model_name=model_name,
        backbone_weights=str(backbone_weights) if backbone_weights else None,
        backbone_type=backbone_type,
        freeze_backbone=kwargs.get("freeze_backbone", True),
        n_classes=len(labels),
        use_lora=kwargs.get("use_lora", False),
        r=kwargs.get("r", 3),
        num_modalities=kwargs.get("num_modalities", 1),
    )

    # Preserve the official linear scaling rule. Batch size is per GPU.
    if kwargs.get("scale_lr", True) and distributed.is_enabled():
        base_lr *= distributed.get_world_size()

    backbone_params = []
    if hasattr(model, "backbone"):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    other_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("backbone") and param.requires_grad
    ]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})
    if not param_groups:
        raise ValueError("The model has no trainable parameters")

    weight_decay = float(kwargs.get("weight_decay", 0.01))
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(kwargs.get("eta_min", 1e-7)),
    )

    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "window_size": window_size,
        "labels": labels,
        "loss_fn": loss_fn,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "learning_rate": base_lr,
        "weight_decay": weight_decay,
        "backbone_weights": str(backbone_weights) if backbone_weights else None,
    }
