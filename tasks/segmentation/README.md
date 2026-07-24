# Segmentation entry points

Use `train_multi.py` for both single-GPU and DDP training:

```bash
# Single GPU
python tasks/segmentation/train_multi.py --help

# One process per GPU
torchrun --standalone --nproc_per_node=2 \
  tasks/segmentation/train_multi.py --help
```

Use `test.py` for final validation or test evaluation:

```bash
python tasks/segmentation/test.py --help
```

Host paths must be supplied by CLI arguments or `MM_DINO_DATASETS_ROOT`,
`MM_DINO_WEIGHTS_ROOT`, and `MM_DINO_OUTPUT_ROOT`. See
[`docs/EXPERIMENT_WORKFLOW.md`](../../docs/EXPERIMENT_WORKFLOW.md) for complete
commands, split policy, checkpointing, and GPU-specific starting settings.
