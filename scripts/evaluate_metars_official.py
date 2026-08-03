"""Inspect, load, or evaluate the official MetaRS artifact.

The released EarthMiss repository is supplied separately via
``--official-code-root`` and remains unmodified.  This launcher only patches
runtime paths and disables the redundant ImageNet download before strictly
loading the complete released checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


DEFAULT_OFFICIAL_ROOT = Path(
    "/root/autodl-tmp/mm-dino/reference/EarthMiss-edcd0374"
)
DEFAULT_CHECKPOINT = Path(
    "/root/autodl-tmp/mm-dino/checkpoints/official/metars/Best.pth"
)
DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/mm-dino/datasets/EarthMiss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("inspect", "data", "load", "eval"), default="inspect"
    )
    parser.add_argument("--official-code-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def seed_torch(seed: int = 2333) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ensure_single_process_group() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29527")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def load_released_checkpoint(path: Path) -> tuple[dict, dict]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("MetaRS checkpoint must contain a top-level 'model' mapping")
    state = payload["model"]
    if not isinstance(state, dict):
        raise TypeError("MetaRS checkpoint 'model' entry is not a state dict")
    return payload, state


def inspect_checkpoint(path: Path) -> None:
    payload, state = load_released_checkpoint(path)
    tensor_count = sum(isinstance(value, torch.Tensor) for value in state.values())
    element_count = sum(
        value.numel() for value in state.values() if isinstance(value, torch.Tensor)
    )
    report = {
        "checkpoint": str(path.resolve()),
        "top_level_keys": list(payload.keys()),
        "model_entries": len(state),
        "tensor_entries": tensor_count,
        "tensor_elements": element_count,
        "first_model_keys": list(state.keys())[:5],
        "last_model_keys": list(state.keys())[-5:],
    }
    print(json.dumps(report, indent=2))


def enter_official_tree(root: Path):
    root = root.resolve()
    if not (root / "configs" / "baseline" / "MetaRS.py").is_file():
        raise FileNotFoundError(f"invalid official EarthMiss root: {root}")
    os.chdir(root)
    sys.path.insert(0, str(root))

    import ever as er
    from ever.core.builder import make_dataloader, make_model
    from ever.core.checkpoint import remove_module_prefix
    from ever.core.config import import_config

    er.registry.register_all()
    return er, make_dataloader, make_model, remove_module_prefix, import_config


def load_official_config(args: argparse.Namespace):
    (
        er,
        make_dataloader,
        make_model,
        remove_module_prefix,
        import_config,
    ) = enter_official_tree(args.official_code_root)

    from configs.metadata.EarthMiss import test_cities

    config_path = args.official_code_root / "configs" / "baseline" / "MetaRS.py"
    cfg = import_config(str(config_path))
    cfg.model.params.encoder.pretrained = False
    cfg.data.test.params.image_dir = [
        str(args.dataset_root / city / "images") for city in test_cities
    ]
    cfg.data.test.params.mask_dir = [
        str(args.dataset_root / city / "masks") for city in test_cities
    ]
    cfg.data.test.params.distributed = False
    cfg.data.test.params.batch_size = args.batch_size
    cfg.data.test.params.num_workers = args.num_workers
    return er, make_dataloader, make_model, remove_module_prefix, cfg


def build_official_model(args: argparse.Namespace):
    er, make_dataloader, make_model, remove_module_prefix, cfg = (
        load_official_config(args)
    )

    _, released_state = load_released_checkpoint(args.checkpoint)
    model = make_model(cfg.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.load_state_dict(remove_module_prefix(released_state), strict=True)
    return er, make_dataloader, model, cfg


def data_smoke(args: argparse.Namespace) -> None:
    _, make_dataloader, _, _, cfg = load_official_config(args)
    ensure_single_process_group()
    image, target = next(iter(make_dataloader(cfg.data.test)))
    report = {
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "mask_shape": list(target["cls"].shape),
        "mask_dtype": str(target["cls"].dtype),
        "filename": list(target["fname"]),
    }
    print(json.dumps(report, indent=2))


def load_smoke(args: argparse.Namespace) -> None:
    _, _, model, cfg = build_official_model(args)
    report = {
        "strict_load": True,
        "model": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "infer_rgb": bool(model.infer_rgb),
        "test_images": list(cfg.data.test.params.image_dir),
        "test_masks": list(cfg.data.test.params.mask_dir),
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(report, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MetaRS evaluation requires a GPU; current mode has no CUDA device")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for evaluation")

    er, make_dataloader, model, cfg = build_official_model(args)
    from data.EarthMiss import COLOR_MAP
    from module.slide_test import slide_inference
    from module.viz import VisualizeSegmm
    from tqdm import tqdm

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_torch()
    device = torch.device("cuda:0")
    model.to(device).eval()
    ensure_single_process_group()
    loader = make_dataloader(cfg.data.test)
    metric = er.metric.PixelMetric(
        cfg.model.params.num_classes,
        logdir=str(args.output_dir),
        logger=logging.getLogger("metars-official"),
        class_names=list(COLOR_MAP.keys()),
    )
    visualizer = None
    if args.save_predictions:
        palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
        visualizer = VisualizeSegmm(str(args.output_dir / "predictions"), palette)

    slide = {"crop_size": (512, 512), "stride": (341, 341)}
    with torch.no_grad():
        for image, target in tqdm(loader):
            prediction = slide_inference(model, image.to(device), target, slide)
            prediction = prediction.argmax(dim=1).cpu()
            truth = target["cls"].cpu()
            valid = truth != -1
            metric.forward(truth[valid], prediction[valid])
            if visualizer is not None:
                for class_map, filename in zip(prediction, target["fname"]):
                    visualizer(
                        class_map.numpy().astype(np.uint8),
                        filename.replace("tif", "png"),
                    )

    summary = metric.summary_all()
    output = args.output_dir / "metric_summary.log"
    output.write_text(str(summary) + "\n", encoding="utf-8")
    print(summary)


def main() -> None:
    args = parse_args()
    for path in (args.official_code_root, args.checkpoint, args.dataset_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.mode == "inspect":
        inspect_checkpoint(args.checkpoint)
    elif args.mode == "data":
        data_smoke(args)
    elif args.mode == "load":
        load_smoke(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
