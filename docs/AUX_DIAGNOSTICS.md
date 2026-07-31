# Auxiliary-Modality Diagnostics

This document is the execution companion to the research plan.  The scripts
are read-only with respect to datasets and checkpoints, and they do not modify
the released code under `tasks/segmentation`.

## Scope

- `scripts/probe_aux_input_stats.py` audits raw files, returned tensors, file
  pairing, spatial sizes, coarse edge alignment, and small translation offsets.
- `scripts/evaluate_whu_aux_counterfactual.py` evaluates one WHU intervention
  per invocation with the released training-time test function and sliding
  inference batch 32.
- `scripts/aux_diagnostics_common.py` contains tested diagnostic wrappers and
  statistics; an auxiliary scale of 1 is numerically equivalent to the
  released SampleAdapter.

The conditions have distinct meanings:

- `aux-mean`: removes spatial Aux structure while retaining each image's mean;
- `aux-shuffle`: uses a deterministic cross-image derangement within native
  spatial-size groups, so the intervention adds no resize or crop policy;
- `rgb-only-native`: loads the multimodal checkpoint unchanged but calls
  `model(rgb)` through its released single-input path.  It does not load or
  encode SAR and it bypasses the multimodal SE-fusion path;
- `rgb-only-fusion-preserved`: also loads and encodes RGB only, but duplicates
  the projected RGB pyramid into the trained multimodal Decoder slots.  It is
  designed to be logit-equivalent to `aux-feature-off` without computing SAR;
- `aux-feature-off`: sets the Aux global fusion scale to zero and renormalizes
  RGB, rather than feeding an arbitrary zero-valued sensor image;
- `aux-weight-scale`: scans simple global Aux reweighting as a sanity control.

## Local verification

```powershell
cd D:\MM-DINO\work\my_MM-DINO
python -m unittest discover -s tests -v
python scripts\probe_aux_input_stats.py --help
python scripts\evaluate_whu_aux_counterfactual.py --help
```

## Server input audit

Run in the foreground after the server and this branch are available:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/probe_aux_input_stats.py \
  --dataset WHU \
  --splits train test \
  --samples-per-split 4 \
  --output-path /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/whu_input_audit.json \
  --artifact-dir /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/whu_alignment
```

The same command can later be repeated for `Vaihingen` and `Potsdam`, using
different output paths.  Reports refuse to overwrite existing JSON files and
inspection PNGs.

## Real-model equivalence smoke check

Before formal counterfactual evaluation, compare one-image `normal` and
scale-1 wrapper results.  They must match before any scale-zero result is used.

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
torchrun --standalone --nproc_per_node=1 scripts/evaluate_whu_aux_counterfactual.py \
  --checkpoint-path /root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260728_021847/DINOv3_WHU_e45_mIoU55.58.pth \
  --condition normal \
  --max-images 1 \
  --output-path /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/smoke_normal_1.json
```

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
torchrun --standalone --nproc_per_node=1 scripts/evaluate_whu_aux_counterfactual.py \
  --checkpoint-path /root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260728_021847/DINOv3_WHU_e45_mIoU55.58.pth \
  --condition aux-weight-scale \
  --aux-weight-scale 1 \
  --max-images 1 \
  --output-path /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/smoke_scale1_1.json
```

## Formal foreground evaluations

Omit `--max-images` for full-split results.  Run each condition separately so
the result and failure boundary remain explicit.  These are long evaluations
and must be launched by the user in the foreground.

Example for `aux-feature-off`:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
torchrun --standalone --nproc_per_node=1 scripts/evaluate_whu_aux_counterfactual.py \
  --checkpoint-path /root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260728_021847/DINOv3_WHU_e45_mIoU55.58.pth \
  --condition aux-feature-off \
  --output-path /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/whu_aux_feature_off.json
```

The direct multi-train/RGB-only-test condition uses the same command with:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
torchrun --standalone --nproc_per_node=1 scripts/evaluate_whu_aux_counterfactual.py \
  --checkpoint-path /root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260728_021847/DINOv3_WHU_e45_mIoU55.58.pth \
  --condition rgb-only-native \
  --output-path /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/whu_rgb_only_native.json
```

The same audit can reuse our faithful frozen ViT-S epoch-45 checkpoint by
changing only the locked model profile, checkpoint, and output path:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
torchrun --standalone --nproc_per_node=1 scripts/evaluate_whu_aux_counterfactual.py \
  --model-profile vits-frozen \
  --checkpoint-path /root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260727_221644/DINOv3_WHU_e45_mIoU54.15.pth \
  --condition rgb-only-native \
  --output-path /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/whu_vits_rgb_only_native.json
```

This is a missing-modality inference baseline, not an independently trained
RGB-only model.  It differs from `aux-feature-off`: the latter keeps the
multimodal Decoder/SE-fusion execution and removes SAR only at fusion, whereas
`rgb-only-native` skips the SAR backbone and exercises the released single-input
Adapter/Decoder branch.

Use `--condition rgb-only-fusion-preserved` when the scientific question is
"remove SAR while keeping the trained multimodal fusion/Decoder endpoint
unchanged."  This condition must pass the real-checkpoint logit-equivalence
smoke before a full result is interpreted.

Real-checkpoint equivalence smoke:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
cd /root/my_MM-DINO-multitrain-unimodal
python scripts/smoke_whu_rgb_only_fusion_equivalence.py \
  --model-profile vitl-lora \
  --checkpoint-path /root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260728_021847/DINOv3_WHU_e45_mIoU55.58.pth \
  --output-path /root/autodl-tmp/mm-dino/outputs/aux-diagnostics/smoke_vitl_rgb_only_fusion_equivalence.json
```

Use the same command with `--condition aux-mean` and
`--condition aux-shuffle`, changing the output filename each time.  Global
weight scales such as `0.5` and `2.0` are secondary sanity checks, not the first
formal experiment.

## Independent RGB-only control

The feature-off intervention is not an independently trained RGB baseline.
Train the RGB-only control with the same seed, ViT-L LoRA rank, 50-epoch
schedule, microbatch 4, and two-step accumulation used by the multimodal
compatibility reproduction.  The command must be launched by the user in the
foreground:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
torchrun --standalone --nproc_per_node=1 \
  scripts/run_whu_vitl_lora_accumulated.py \
  --micro-batch-size 4 \
  --grad-accum-steps 2 \
  --num-modalities 1
```

The launcher's default remains two modalities, preserving the existing
multimodal reproduction command.

## Interpretation guardrails

- Compare aligned gain and mismatch damage separately; a larger
  `normal - aux-shuffle` gap is not automatically better.
- `aux-feature-off` is a feature-path intervention, not a replacement for the
  independently trained RGB-only baseline.
- A low Aux-only score does not rule out conditional complementarity.
- No method implementation begins until the input audit and counterfactual
  evidence select H1, H2, both, or neither.
- `opt-only-feature-zero`: computes both inputs but replaces the projected SAR
  feature with exact zero immediately before the released SampleAdapter
  weighted sum.  Original weights, denominator, duplicated Decoder slots, and
  multimodal SEFusion path are unchanged;
- `sar-only-feature-zero`: the symmetric intervention, replacing projected OPT
  features with zero while preserving the same two-input execution path;
