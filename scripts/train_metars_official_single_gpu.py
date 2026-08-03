"""Single-GPU adaptation of the released MetaRS training behavior."""

from pathlib import Path

from train_metars_clean import REPO_ROOT, main


if __name__ == "__main__":
    main(
        default_config=Path(REPO_ROOT / "configs" / "metars_official_reproduction.py"),
        val_source="test",
        protocol=(
            "MetaRS-official-single-gpu: released Test-backed data.val/MMR "
            "behavior with batch 8 on one rank"
        ),
        expected_world_size=1,
        batch_per_rank=8,
        test_batch_per_rank=1,
    )
