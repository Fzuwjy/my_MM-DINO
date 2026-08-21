# Missing-modality normalization x ratio protocol

## Question

Run C underperforms SAR-only Run A by about 0.47 mIoU points on both EarthMiss
and WHU.  This protocol separates two explanations before another transfer
module is introduced:

1. a fixed 50/50 schedule displaces too many SAR-only optimizer steps; and
2. shared Decoder BatchNorm mixes Full and SAR activation distributions.

It does not claim that GN or a different ratio is a new method.  It is a causal
baseline audit that decides whether the current MM-DINO training substrate is
suitable for later missing-modality transfer work.

## Frozen factors

- Decoder normalization: released BatchNorm or GroupNorm(32).
- Homogeneous-batch Full probability: 0, 0.25, or 0.50.
- `p(Full)=0` is Run A; intermediate probabilities are Run C.
- Every arm keeps the same total epochs, optimizer steps, batch size, data,
  augmentation, loss, optimizer, scheduler, frozen DINO, and canonical SAR
  deployment endpoint.
- GroupNorm replaces only the 40 trainable Decoder `BatchNorm2d` modules.  The
  frozen DINOv3 LayerNorm layers are unchanged.  GN is trained from the common
  pretrained initialization; checkpoints are never converted post hoc.
- The p=0.50 state sampler preserves the historical Run-C random trace.
- Each epoch records SAR/Full batch counts and a SHA-256 state-trace digest.

The seed-42 matrix is therefore:

| Decoder norm | p(Full)=0 | p(Full)=0.25 | p(Full)=0.50 |
|---|---:|---:|---:|
| BN | A-BN | C25-BN | C50-BN |
| GN | A-GN | C25-GN | C50-GN |

## Primary comparisons

- Ratio effect within a normalization: `C25-A` and `C50-A`.
- Normalization effect at a ratio: `GN-BN`.
- BN-conflict interaction:
  `(C_GN-A_GN) - (C_BN-A_BN)`.

If reducing p(Full) helps equally under BN and GN, lost SAR exposure is the
leading explanation.  If GN specifically improves C relative to A, shared BN
is implicated.  If neither happens, the current early-fusion/shared-decoder
method is the more likely bottleneck and normalization/ratio tuning stops.

## Checkpoints and evaluation

- Fixed E50 is the primary checkpoint for this audit on both datasets.
- EarthMiss Val is retained as a trajectory and failure diagnostic only; it
  does not select the primary checkpoint.
- Test is not used to select normalization, ratio, epoch, or seed.
- First run seed 42.  Only a stable, predeclared winning configuration is
  repeated with seeds 43/44.

## Stop rules

- Do not add another transfer module while this matrix is incomplete.
- Do not sweep GN group count, learning rate, loss weights, or class weights.
- A candidate must improve canonical SAR without causing a material Full-state
  collapse, and the direction should agree on both datasets before it becomes
  the next research substrate.
- If gains appear only on one dataset or only at one isolated ratio, report a
  dataset-specific optimization effect rather than a universal missing-modality
  mechanism.
