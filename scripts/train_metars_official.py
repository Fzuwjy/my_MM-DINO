"""Launch the released MetaRS training behavior with portable runtime paths."""

from pathlib import Path

from train_metars_clean import REPO_ROOT, main


if __name__ == "__main__":
    main(
        default_config=Path(REPO_ROOT / "configs" / "metars_official_reproduction.py"),
        val_source="test",
        protocol=(
            "MetaRS-official-reproduction: released recipe preserving Test-backed "
            "data.val/MMR behavior"
        ),
    )
