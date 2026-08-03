"""Single-GPU adaptation of MetaRS-clean with true validation cities."""

from pathlib import Path

from train_metars_clean import REPO_ROOT, main


if __name__ == "__main__":
    main(
        default_config=Path(REPO_ROOT / "configs" / "metars_clean.py"),
        val_source="val",
        protocol=(
            "MetaRS-clean-single-gpu: true-Val MMR behavior with batch 8 on one rank"
        ),
        expected_world_size=1,
        batch_per_rank=8,
        test_batch_per_rank=4,
        save_ckpt_interval_epoch=5,
        resume_insurance=True,
    )
