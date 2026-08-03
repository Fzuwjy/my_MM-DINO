"""Run one MetaRS training optimizer step without saving a checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from train_metars_clean import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OFFICIAL_ROOT,
    DEFAULT_TORCH_HOME,
    REPO_ROOT,
    _enter_official_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "configs" / "metars_official_reproduction.py",
    )
    parser.add_argument("--official-code-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    os.environ["EARTHMISS_ROOT"] = str(args.dataset_root.resolve())
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    os.environ["METARS_TRAIN_BATCH_PER_RANK"] = str(args.batch_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    os.environ.setdefault("LOCAL_RANK", "0")
    _enter_official_tree(args.official_code_root)

    import ever as er
    import torch
    import torch.distributed as dist
    from ever.core.builder import make_dataloader, make_model, make_optimizer
    from ever.core.config import import_config
    from ever.util import to

    if not torch.cuda.is_available():
        raise RuntimeError("training smoke requires a CUDA device")
    from data.EarthMiss import EarthM3DALoader  # noqa: F401
    from module.baseline.MetaRS import MetaRS  # noqa: F401
    from train import seed_torch

    seed_torch(2333)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cfg = import_config(str(args.config_path.resolve()))
    cfg.data.train.params.num_workers = 0

    loader = make_dataloader(cfg.data.train)
    model = make_model(cfg.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[0],
        output_device=0,
        find_unused_parameters=False,
    )
    cfg.optimizer.params.lr = cfg.learning_rate.params.base_lr
    optimizer = make_optimizer(cfg.optimizer, params=model.module.custom_param_groups())
    image, target = next(iter(loader))
    image, target = to.to_device((image, target), device)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model(image, target)
    total_loss = sum(value for key, value in losses.items() if key.endswith("loss"))
    total_loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    report = {
        "training_step": True,
        "batch_size": args.batch_size,
        "image_shape": list(image.shape),
        "loss_keys": sorted(key for key in losses if key.endswith("loss")),
        "total_loss": float(total_loss.detach().cpu()),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "device": torch.cuda.get_device_name(device),
    }
    print(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
