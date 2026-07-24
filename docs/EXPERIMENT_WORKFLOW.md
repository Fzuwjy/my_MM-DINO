# MM-DINO experiment workflow

## Repository isolation

- `official` must stay identical to `upstream/main`.
- `main` contains reviewed and reproducible project code.
- Development happens on `exp/<topic>` branches.
- Push only to `origin` (`Fzuwjy/my_MM-DINO`). Never push to `upstream`.

The cloud server will be created after local adaptation. It must clone this fork,
not the upstream repository.

## Runtime directories

Source code contains no host-specific absolute paths. Configure the runtime with
CLI arguments or these environment variables:

```bash
export MM_DINO_DATASETS_ROOT=/path/to/datasets
export MM_DINO_WEIGHTS_ROOT=/path/to/weights
export MM_DINO_OUTPUT_ROOT=/path/to/outputs
```

Expected dataset layout keeps the official structure:

```text
datasets/
├── whu-opt-sar/
│   ├── optical/
│   ├── sar/
│   ├── lbl/
│   ├── train_list.txt
│   ├── val_list.txt
│   └── test_list.txt
└── ISPRS_dataset/
    ├── Potsdam/
    └── Vaihingen/
```

Weights are discovered from `MM_DINO_WEIGHTS_ROOT`, or can be provided explicitly
with `--backbone-weights`.

Datasets, weights, checkpoints and outputs are ignored by Git. Do not commit them.

## Validation protocol

Training defaults to `--eval-split val`. WHU uses `val_list.txt`; datasets without
an official validation split require `--eval-split-file` containing one image/tile
identifier per line.

Use `--eval-split test` only when reproducing the official training protocol. It
selects checkpoints on the test set and must not be used for formal reported
results. The final test set is evaluated once with `tasks/segmentation/test.py`.

For WHU, filenames encode map-sheet groups (for example `NH49E014`). A candidate
group-disjoint split can be generated without mixing images from the same prefix:

```bash
python scripts/make_grouped_split.py \
  --input /path/to/official_train_list.txt \
  --train-output /path/to/splits/whu_research_train.txt \
  --val-output /path/to/splits/whu_research_val.txt \
  --group-prefix-length 8 \
  --val-ratio 0.2 \
  --seed 42
```

The script writes a manifest with the source/split hashes and selected groups.
Before freezing this split for experiments, inspect class-pixel balance using the
downloaded labels; filename grouping alone does not guarantee class balance.
Keep the official dataset lists unchanged and pass the generated files through
`--train-split-file` and `--eval-split-file`.

## Preflight

Create a Python 3.11 environment. Install the CUDA-compatible `torch` and
`torchvision` builds selected for the server first, then install the remaining
packages:

```bash
pip install -r requirements-segmentation.txt
```

The exact PyTorch/CUDA command is intentionally deferred until the server GPU,
driver and image are known; do not let a generic requirements file silently
replace the CUDA build with a CPU build.

After the environment, data and weights are ready, run:

```bash
python scripts/preflight.py \
  --dataset-name WHU \
  --num-modalities 2 \
  --split val \
  --require-cuda
```

The check reports Python/PyTorch/CUDA/GPU information and fails on missing
dependencies, weights, dataset directories or split files.

## Training commands

Single 32 GB RTX 5090 starting point:

```bash
python tasks/segmentation/train_multi.py \
  --dataset-name WHU \
  --num-modalities 2 \
  --backbone-type dinov3_vits16 \
  --train-split-file /path/to/splits/whu_research_train.txt \
  --eval-split-file /path/to/splits/whu_research_val.txt \
  --batch-size 8 \
  --amp-dtype bf16 \
  --inference-batch-size 4 \
  --run-name whu_vits16_official_baseline
```

Two RTX 3090 cards use DDP. Batch size is per GPU and VRAM is not pooled:

```bash
torchrun --standalone --nproc_per_node=2 \
  tasks/segmentation/train_multi.py \
  --dataset-name WHU \
  --num-modalities 2 \
  --backbone-type dinov3_vits16 \
  --train-split-file /path/to/splits/whu_research_train.txt \
  --eval-split-file /path/to/splits/whu_research_val.txt \
  --batch-size 4 \
  --amp-dtype bf16 \
  --inference-batch-size 2 \
  --run-name whu_vits16_official_baseline
```

If memory is insufficient, reduce `--batch-size` and
`--inference-batch-size`; use `--grad-accum-steps` to recover the effective batch
size. `--cache-size` controls the number of complete WHU/EarthMiss images retained
by each DataLoader worker; its conservative default is 2 because worker caches are
not shared. Keep the exact command and generated `run_config.json` with every result.

Resume an interrupted run:

```bash
python tasks/segmentation/train_multi.py \
  [same model/data arguments] \
  --resume /path/to/run/last.pth
```

The run directory contains atomic `last.pth` and `best.pth` checkpoints, optional
periodic checkpoints, `run_config.json`, and append-only JSONL metric files.

## Final evaluation

```bash
python tasks/segmentation/test.py \
  --checkpoint-path /path/to/run/best.pth \
  --dataset-name WHU \
  --num-modalities 2 \
  --backbone-type dinov3_vits16 \
  --split test \
  --amp-dtype bf16 \
  --output-dir /path/to/outputs/final_test
```

Prediction images are disabled by default to avoid excessive disk use. Add
`--save-predictions` only when qualitative figures are needed.
