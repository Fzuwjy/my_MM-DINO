"""Validate a machine before starting an MM-DINO experiment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "ftfy": "ftfy",
    "iopath": "iopath",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "opencv-python-headless": "cv2",
    "pandas": "pandas",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "submitit": "submitit",
    "termcolor": "termcolor",
    "torch": "torch",
    "torchmetrics": "torchmetrics",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
}

MINIMUM_TORCH_VERSION = (2, 7, 1)
RECOMMENDED_TORCH_VERSION = (2, 7, 1)
RECOMMENDED_TORCHVISION_VERSION = (0, 22, 1)
RECOMMENDED_CUDA_VERSION = (12, 8)


def numeric_version(version):
    """Return a comparable numeric prefix for versions such as 2.7.1+cu128."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if match is None:
        return ()
    return tuple(int(part or 0) for part in match.groups())


def architecture_is_supported(arch_list, capability):
    major, minor = capability
    suffix = f"{major}{minor}"
    return f"sm_{suffix}" in arch_list or f"compute_{suffix}" in arch_list


def sanitize_thread_environment(warnings):
    """Prevent invalid cloud-image thread variables from breaking native libs."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            valid = int(value) > 0
        except ValueError:
            valid = False
        if not valid:
            warnings.append(
                f"{name}={value!r} is invalid; preflight temporarily uses 1. "
                "Set it to a positive integer before training"
            )
            os.environ[name] = "1"


def check_dependencies(errors, warnings):
    missing = [name for name, module in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]
    if missing:
        errors.append("Missing Python packages: " + ", ".join(missing))
    if sys.version_info[:2] != (3, 11):
        warnings.append(
            f"Python {sys.version.split()[0]} detected; the official code is tested with Python 3.11"
        )


def check_cuda(errors, warnings, require_cuda):
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    print(f"torch={torch.__version__}")
    torch_version = numeric_version(torch.__version__)
    if not torch_version or torch_version < MINIMUM_TORCH_VERSION:
        errors.append(
            "MM-DINO requires torch>=2.7.1; for RTX 5090 install "
            "torch==2.7.1 from the cu128 index"
        )

    if importlib.util.find_spec("torchvision") is not None:
        try:
            import torchvision
        except Exception as error:  # Binary mismatches often fail during import.
            errors.append(
                f"torchvision cannot be imported ({error}); reinstall the "
                "recommended torch/torchvision pair together"
            )
        else:
            print(f"torchvision={torchvision.__version__}")
            torchvision_version = numeric_version(torchvision.__version__)
            if (
                torch_version == RECOMMENDED_TORCH_VERSION
                and torchvision_version != RECOMMENDED_TORCHVISION_VERSION
            ):
                errors.append(
                    "torch 2.7.1 must be paired with torchvision 0.22.1 for "
                    "the recommended server environment"
                )

    print(f"cuda_available={torch.cuda.is_available()}")
    if require_cuda and not torch.cuda.is_available():
        errors.append("CUDA is required but torch.cuda.is_available() is False")
    if torch.cuda.is_available():
        print(f"cuda_runtime={torch.version.cuda}")
        arch_list = set(torch.cuda.get_arch_list())
        print(f"cuda_arch_list={','.join(sorted(arch_list))}")
        print(f"gpu_count={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            memory_gib = props.total_memory / 1024**3
            print(
                f"gpu[{index}]={props.name}, memory={memory_gib:.1f} GiB, "
                f"capability={capability[0]}.{capability[1]}"
            )
            if not architecture_is_supported(arch_list, capability):
                errors.append(
                    f"The installed PyTorch build does not contain kernels for "
                    f"{props.name} (sm_{capability[0]}{capability[1]}); install "
                    "the recommended cu128 build"
                )
            if "5090" in props.name and (
                numeric_version(torch.version.cuda) < RECOMMENDED_CUDA_VERSION
            ):
                errors.append(
                    "RTX 5090 requires the validated CUDA 12.8 PyTorch build; "
                    f"detected CUDA runtime {torch.version.cuda}"
                )
    elif not require_cuda:
        warnings.append("No CUDA GPU detected; only static checks can run efficiently")


def check_assets(args, errors):
    datasets_root = Path(args.datasets_root).expanduser().resolve()
    weights_root = Path(args.weights_root).expanduser().resolve()

    explicit_weights = (
        Path(args.backbone_weights).expanduser().resolve()
        if args.backbone_weights
        else None
    )
    if explicit_weights:
        if not explicit_weights.is_file():
            errors.append(f"Backbone weights not found: {explicit_weights}")
    else:
        matches = list(weights_root.glob(f"{args.backbone_type}_*.pth"))
        if len(matches) != 1:
            errors.append(
                f"Expected exactly one {args.backbone_type}_*.pth under {weights_root}, found {len(matches)}"
            )

    roots = {
        "WHU": datasets_root / "whu-opt-sar",
        "Potsdam": datasets_root / "ISPRS_dataset" / "Potsdam",
        "Vaihingen": datasets_root / "ISPRS_dataset" / "Vaihingen",
        "EarthMiss": datasets_root / "EarthMiss",
        "YYYJ": datasets_root / "YYYJ_dataset",
    }
    dataset_root = roots[args.dataset_name]
    if not dataset_root.is_dir():
        errors.append(f"Dataset directory not found: {dataset_root}")
        return

    if args.dataset_name == "WHU":
        required_dirs = [dataset_root / "optical", dataset_root / "lbl"]
        if args.num_modalities > 1:
            required_dirs.append(dataset_root / "sar")
        for directory in required_dirs:
            if not directory.is_dir():
                errors.append(f"Required WHU directory not found: {directory}")
        split_path = (
            Path(args.split_file).expanduser().resolve()
            if args.split_file
            else dataset_root / f"{args.split}_list.txt"
        )
        if not split_path.is_file():
            errors.append(f"Split file not found: {split_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MM-DINO machine preflight")
    parser.add_argument("--dataset-name", choices=["WHU", "Potsdam", "Vaihingen", "EarthMiss", "YYYJ"], default="WHU")
    parser.add_argument(
        "--datasets-root", default=os.environ.get("MM_DINO_DATASETS_ROOT", "data")
    )
    parser.add_argument(
        "--weights-root", default=os.environ.get("MM_DINO_WEIGHTS_ROOT", "weights")
    )
    parser.add_argument("--backbone-type", default="dinov3_vits16")
    parser.add_argument("--backbone-weights")
    parser.add_argument("--num-modalities", type=int, choices=[1, 2], default=2)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--split-file")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--allow-missing-deps", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    errors = []
    warnings = []
    sanitize_thread_environment(warnings)
    check_dependencies(errors, warnings)
    check_cuda(errors, warnings, args.require_cuda)
    if args.skip_assets:
        warnings.append("Dataset and weight checks were skipped")
    else:
        check_assets(args, errors)

    if args.allow_missing_deps:
        dependency_errors = [error for error in errors if error.startswith("Missing Python packages")]
        errors = [error for error in errors if error not in dependency_errors]
        warnings.extend(dependency_errors)

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("preflight=FAILED")
        return 1
    print("preflight=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
