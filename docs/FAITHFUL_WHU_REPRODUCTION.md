# Faithful WHU-OPT-SAR reproduction

This branch starts directly from the public MM-DINO commit and does not modify
the authors' model, adapter, decoder, dataset, augmentation, loss, optimizer,
scheduler, training loop, evaluation loop, checkpoint selection, or metrics.

The intended run is the released implementation protocol:

- all 80 names in `train_list.txt` are used for training;
- the 20-name `test_list.txt` is evaluated every five epochs;
- the best checkpoint is selected by test mIoU;
- 50 epochs, FP32, batch size 8 per GPU, no gradient accumulation;
- AdamW at `1e-4` on one GPU, cosine scheduling, and no resume;
- one-process DDP with `find_unused_parameters=True`;
- sliding-window evaluation uses `batch_size * 4 == 32` unless the memory probe
  proves this evaluation-only microbatch does not fit.

Selecting on the test set is retained solely for reproduction and must not be
used as the protocol for later research comparisons.

## Server preparation

After pulling this branch, inspect the planned paths and then create only the
missing symbolic links:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/prepare_faithful_whu.py
python scripts/prepare_faithful_whu.py --apply
```

The preparation script refuses to replace any existing non-matching path.

## Memory probes

Run the exact FP32 batch-8 training step first:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/probe_official_whu_memory.py --phase train
```

Then test the author's evaluation microbatch on one test image:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/probe_official_whu_memory.py --phase eval --inference-batch-size 32
```

If only the evaluation probe runs out of memory, retry 16, 8, and finally 4.
This changes sliding-window microbatching but not training batch size.

## Formal foreground run

Do not start this command until both probes pass and their output is recorded:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/run_seeded_official.py \
  --model-name DINOv3 \
  --dataset-name WHU \
  --num-modalities 2 \
  --backbone-type dinov3_vits16
```

Do not add `--use-lora False`: the authors used `argparse` with `type=bool`, for
which the non-empty string `False` is interpreted as true.  Omitting the option
correctly preserves its default value of false.
