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
    parser.add_argument("--with-mmr", action="store_true")
    parser.add_argument("--eval-batch-size", type=int)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    os.environ["EARTHMISS_ROOT"] = str(args.dataset_root.resolve())
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    os.environ["METARS_TRAIN_BATCH_PER_RANK"] = str(args.batch_size)
    if args.eval_batch_size is not None:
        os.environ["METARS_TEST_BATCH_PER_RANK"] = str(args.eval_batch_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    os.environ.setdefault("LOCAL_RANK", "0")
    _enter_official_tree(args.official_code_root)

    import ever as er
    import torch
    import torch.distributed as dist
    from ever.core import to
    from ever.core.builder import make_dataloader, make_model, make_optimizer
    from ever.core.config import import_config

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

    if args.with_mmr:
        model.train()
        model.module.global_step.fill_(cfg.model.params.begin_mmr_iter)
        model.module.conduct_mask_matrix()
        model.train()
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model(image, target)
    total_loss = sum(value for key, value in losses.items() if key.endswith("loss"))
    total_loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    training_peak_mib = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    report = {
        "training_step": True,
        "with_mmr": args.with_mmr,
        "batch_size": args.batch_size,
        "image_shape": list(image.shape),
        "loss_keys": sorted(key for key in losses if key.endswith("loss")),
        "total_loss": float(total_loss.detach().cpu()),
        "peak_memory_mib": training_peak_mib,
        "device": torch.cuda.get_device_name(device),
    }

    if args.eval_batch_size is not None:
        optimizer.zero_grad()
        test_loader = make_dataloader(cfg.data.test)
        test_image, test_target = next(iter(test_loader))
        test_image, test_target = to.to_device((test_image, test_target), device)
        model.eval()
        report["pre_eval_allocated_mib"] = round(
            torch.cuda.memory_allocated(device) / 2**20, 1
        )
        report["pre_eval_reserved_mib"] = round(
            torch.cuda.memory_reserved(device) / 2**20, 1
        )
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            prediction = model(test_image, test_target)
        torch.cuda.synchronize(device)
        report["evaluation"] = {
            "batch_size": args.eval_batch_size,
            "input_shape": list(test_image.shape),
            "output_shape": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all().item()),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / 2**20, 1
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / 2**20, 1
            ),
        }
    print(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
