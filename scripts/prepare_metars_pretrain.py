"""Download and verify the exact ImageNet ResNet-50 initialization for MetaRS."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
FILENAME = "resnet50-19c8e357.pth"
SHA256_PREFIX = "19c8e357"
DEFAULT_TORCH_HOME = Path("/root/autodl-tmp/mm-dino/cache/torch")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    args = parser.parse_args()

    target_dir = args.torch_home.resolve() / "hub" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / FILENAME
    if target.is_file() and sha256(target).startswith(SHA256_PREFIX):
        print(f"already verified: {target}")
        return

    fd, temporary_name = tempfile.mkstemp(prefix=FILENAME + ".", suffix=".part", dir=target_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(URL, temporary)
        digest = sha256(temporary)
        if not digest.startswith(SHA256_PREFIX):
            raise ValueError(f"unexpected SHA256: {digest}")
        temporary.replace(target)
        print(f"verified: {target}\nbytes: {target.stat().st_size}\nsha256: {digest}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
