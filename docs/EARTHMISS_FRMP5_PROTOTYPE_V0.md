# EarthMiss F2S-CSPT v0 protocol

This branch implements a falsifiable training-only mechanism pilot on top of
the released Run C deployment graph.  It must not be described as an
optical-only teacher: the privileged anchor is the canonical Full-state fused
(RGB+SAR) post-FRM P5 representation.

## Frozen method

- location: post-FRM P5, before SEFusion;
- query: canonical SAR-state P5 with eval-mode FRM BatchNorm;
- P2 anchor: detached canonical Full-state P5 using the same persistent BN
  buffers;
- P1 anchor: detached SAR query, controlling class-prototype separation;
- occupancy: exact native GT area average on the P5 grid;
- purity weight: `a^2`;
- class support: unsquared `sum(a) >= 2` P5-cell-equivalent;
- aggregation: complete homogeneous batch, equal supported-class weighting;
- loss: Prototype InfoNCE, `tau=0.1`;
- deployment: no new parameter, buffer, forward branch, or latency.

The primary SAR segmentation forward remains in train mode and is the only
auxiliary-step operation allowed to update persistent BN buffers.  SAR query
and Full anchor are recomputed through Adapter plus one FRM call each with FRM
BN in eval mode.  Their forward RNG is restored and their BN buffers are
checked exactly.

## Fixed training and selection

- P0/P1/P2 start independently from the same initialization and seed; no C-E15
  warm start and no shared E5 checkpoint claim.
- E1-E5: prototype weight zero and the P1/P2 path is not executed.
- Start of E6: 32 effective Train SAR batches calibrate the weight so the joint
  median Adapter/active-FRM-P5 gradient ratio is 10%.
- E6-E10: linear ramp; E11-E50: full calibrated weight.
- All arms run E50 without early stopping and validate every five epochs.
- E5/E10 are trajectory-only.  The unique deployment checkpoint is selected
  by pooled canonical SAR Val mIoU from E15, E20, ..., E50.  Full and per-city
  metrics are recorded from that same checkpoint.
- Test is never accessed by these scripts.

## Execution order

All commands are foreground commands.  Replace the Run C checkpoint path with
the preserved primary E15 checkpoint on server 3.

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/diagnose_earthmiss_frmp5_prototype_v0.py \
  --checkpoint /path/to/run_c_seed42/best_sar.pth \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0/run-c-e15-train-health.json
```

Before a formal run, exercise the exact P2 optimizer path:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/train_earthmiss_frmp5_prototype_v0.py \
  --arm p2 \
  --smoke-batches 2 \
  --output-root /root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0-smoke
```

The smoke JSON must show finite losses, at least one non-skipped prototype
batch, and CUDA peak allocation below the physical device limit.

Run matched P0 and P2 first.  P1 is conditional on P2 passing the frozen P0
screen.

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/train_earthmiss_frmp5_prototype_v0.py \
  --arm p0
```

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/train_earthmiss_frmp5_prototype_v0.py \
  --arm p2 \
  --health-report /root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0/run-c-e15-train-health.json \
  --confirm-health-gates-passed
```

After both E50 runs finish, apply the pre-registered Val decision without
loading either model or accessing Test:

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/analyze_earthmiss_frmp5_prototype_v0.py \
  --p0-metrics /root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0/run_p0_seed42/metrics.jsonl \
  --p2-metrics /root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0/run_p2_seed42/metrics.jsonl \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0/seed42-p2-vs-p0-val-decision.json
```

## Frozen decision

P2 proceeds to P1 only when all of the following hold on Val:

- P2 minus P0 SAR mIoU is at least `+0.50 pp`;
- P2 minus P0 Full mIoU is at least `-0.50 pp`;
- at least two of three Val cities are non-negative;
- initialized model hashes and all 50 epoch data-trace hashes match.

After P1 is run, P2 minus P1 SAR mIoU must be at least `+0.30 pp`.  Failure is
archived without sweeping temperature, weight, support threshold, scale, or
purity exponent.
