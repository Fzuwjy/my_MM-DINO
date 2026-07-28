# Faithful WHU-OPT-SAR reproduction

This branch starts directly from the public MM-DINO commit and does not modify
the authors' model, adapter, decoder, dataset, augmentation, loss, optimizer,
scheduler, training loop, evaluation loop, checkpoint selection, or metrics.

The released WHU dataset returns training labels as `int32`, but its released
soft cross-entropy implementation calls `torch.gather`, which requires `int64`
class indices.  The first exact batch probe reproduced this runtime failure.
Both launchers therefore install an external compatibility shim that converts
only the training-label storage dtype from `int32` to `int64`; all class values
remain unchanged, and the authors' source files remain untouched.

The first formal run completed five training epochs and all 20 test-image
inferences, then received `SIGKILL` while materializing the official metrics.
The 90 GiB container had crossed its 86 GiB `memory.high` threshold 348,095
times cumulatively by inspection.  Four persistent workers had each populated the released capacity-100
optical/SAR/label caches.  A second external shim now caps each worker cache at
64.  The 80-image training set made the released capacity 100 effectively hold
80 images per worker; retaining 64 preserves roughly 80% of that working set
while releasing an estimated 11-12 GiB across four workers.  This changes only
image re-read frequency; the released evaluation and
metric implementation remain unchanged for the next verification run.

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
For the Table III ViT-L rows, also create the exact SAT-493M backbone link:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/prepare_faithful_whu.py --backbone-type dinov3_vitl16 --apply
```

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

### Table III ViT-L memory probes

First probe the released multi-modal ViT-L construction without LoRA at the
author's FP32 batch size 8:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/probe_official_whu_memory.py \
  --phase train \
  --backbone-type dinov3_vitl16 \
  --batch-size 8
```

If that fits, probe the paper's LoRA rank 3 row:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/probe_official_whu_memory.py \
  --phase train \
  --backbone-type dinov3_vitl16 \
  --use-lora \
  --lora-rank 3 \
  --batch-size 8
```

An out-of-memory result at batch 8 is evidence that the released training
batch does not fit this 31.4 GiB GPU.  Retry microbatch 4 with two accumulation
steps, then 2/4 and 1/8 if needed.  Each keeps a single-GPU effective batch of
8 for the diagnostic step:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/probe_official_whu_memory.py \
  --phase train \
  --backbone-type dinov3_vitl16 \
  --use-lora \
  --lora-rank 3 \
  --batch-size 4 \
  --grad-accum-steps 2
```

Gradient accumulation is not part of the released trainer.  This probe only
measures a possible compatibility fallback; a formal accumulated run requires
a separate external launcher and must remain distinguishable from the exact
batch-8 reproduction.

The RTX 5090 probes established the following Table III ViT-L memory envelope:

- no LoRA, released training batch 8: 11.270 GiB allocated / 12.299 GiB reserved;
- LoRA rank 3, released training batch 8: CUDA out of memory;
- LoRA rank 3, microbatch 4 with two accumulation steps: 16.311 / 18.090 GiB;
- LoRA rank 3, released evaluation microbatch 32: 14.437 / 16.768 GiB.

Therefore only the LoRA training DataLoader and optimizer cadence need a
compatibility adaptation.  Evaluation stays at the released microbatch 32.

## Table III ViT-L LoRA compatibility run

The external launcher fixes the scientific target to WHU, two modalities,
ViT-L SAT-493M, LoRA rank 3, FP32, seed 42, and the released 50-epoch trainer.
It uses training microbatch 4 with two gradient-accumulation steps while keeping
the config batch at 8 so evaluation remains at 32.  It also writes
`compatibility_protocol.json` into the run directory and disables the released
cleanup helper so earlier reproduction artifacts cannot be deleted.

This run is not an exact released batch-8 reproduction: BatchNorm statistics
and the loss computation see microbatches of four.  The author training source
remains unchanged.

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 \
  scripts/run_whu_vitl_lora_accumulated.py \
  --micro-batch-size 4 \
  --grad-accum-steps 2
```

## Table III ViT-L LoRA checkpoint comparison

The released standalone `tasks/segmentation/test.py` uses sliding-window
microbatch 8, which differs from the training-time evaluator's microbatch 32.
Use the external evaluator below; it loads a checkpoint and directly calls the
unmodified `train_multi.test` function.  It refuses to overwrite an existing
JSON result.

Evaluate the author-released checkpoint:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/evaluate_whu_vitl_lora.py \
  --checkpoint-path /root/autodl-tmp/mm-dino/checkpoints/official/whu/MMDINO_vitl16_lora_WHU_multi_e50_mIoU55.92.pth \
  --output-path /root/autodl-tmp/mm-dino/outputs/evaluation/whu_vitl16_lora_multi_official_train_protocol.json
```

Evaluate the epoch-45 compatibility-reproduction checkpoint:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 scripts/evaluate_whu_vitl_lora.py \
  --checkpoint-path /root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260728_021847/DINOv3_WHU_e45_mIoU55.58.pth \
  --output-path /root/autodl-tmp/mm-dino/outputs/evaluation/whu_vitl16_lora_multi_repro_e45_train_protocol.json
```

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
