# EarthMiss missing-modality causal diagnostics v1

This branch does not implement a fifth transfer module. It freezes a
failure-directed decision protocol after V3 masked-KL, P5-to-P4 FAM, and
FRM-P5 Prototype InfoNCE all failed to improve canonical SAR deployment.

All MM-DINO forwards use the preserved Run C E15 checkpoint:

```text
SHA256 b038dfcfe771ca5c67da500acc2b88e74f066496cac332fc3b76d4377b73dff9
epoch  E15
seed   42
```

Test is prohibited. The diagnostics use Train and/or Val only and never update
MM-DINO parameters.

## Question 1: downstream causal sufficiency

`diagnose_earthmiss_oracle_intervention.py` captures a paired canonical Full
tensor and replaces exactly one canonical SAR scale with it. The primary screen
uses `alpha=1` at twelve single-scale locations:

- Adapter P2/P3/P4/P5;
- post-FRM P2/P3/P4/P5;
- post-SEFusion P2/P3/P4/P5.

The complete Full and SAR endpoints remain visible as ceilings/baselines. A
stage is actionable only when pooled SAR Val mIoU improves by at least `0.25 pp`
and at least two of three Val cities are non-negative. This establishes only
that the Full tensor is useful downstream; it does not establish SAR
predictability or trainable transfer.

Aggregate replacement hooks (`adapter.all`, `frm.all`, `se.all`, `prn.all`,
`head.logits`) exist as implementation controls and are covered by exact CPU
tests. They are not part of the primary screen because alpha=1 reproduces the
Full endpoint by construction.

Only after one single-scale stage passes may one response curve be run for that
preselected stage with `alpha=0.25,0.5,0.75,1.0`. The response report is not a
replacement for the complete primary screen.

### Locked post-screen response curve (2026-08-17)

The completed alpha=1 screen selected `frm.P2` before any response-curve result
was observed: it had the largest actionable pooled gain (`+8.1679 pp`) and was
positive in all three Val cities. The only authorized response curve uses
`--stages frm.P2 --alphas 0.25 0.50 0.75 1.00`. A partial intervention is
actionable under the unchanged screen rule (`>=+0.25 pp` pooled and at least two
nonnegative cities). If neither alpha 0.25 nor 0.50 is actionable, exact Full-P2
replacement is treated as insufficient evidence for a learnable small
compensation. No other stage or alpha grid may be selected from these results.

The oracle runner snapshots the repository HEAD at startup and refuses to write
the report if another task switches the shared checkout before completion.

## Question 2: SAR recoverability

> **Superseded diagnostic.** This P5 probe belongs to the original v1 decision
> tree. The completed oracle screen subsequently localized the useful causal
> signal to post-FRM P2, while post-FRM P5 replacement was approximately null.
> Therefore neither this P5 runner nor the old three-report analyzer is valid
> for deciding the next intervention and must not be rerun for that purpose.

`diagnose_earthmiss_sar_recoverability.py` freezes Run C and works on P5 cells
with at least 75% native-pixel GT purity where canonical SAR is wrong. The
target is whether the paired canonical Full state is correct. A fixed linear
probe is fit on 64 deterministic tiles per Train city and evaluated on all 277
Val tiles/cities.

Probe extraction uses non-overlapping `512/512` crops, not the official
`512/341` overlap used by endpoint evaluation. This prevents duplicated native
pixels from masquerading as independent probe examples. Pooled, city, class,
and tile-level metrics are all reported.

Controls are deliberately strong:

- GT-class-conditional Train prevalence;
- a linear probe trained after shuffling the target within each GT class.

Linear P5 recoverability is supported only if all gates pass:

- pooled AUROC at least `0.60`;
- AP exceeds the class-conditional prior by at least `0.05`;
- AP exceeds the within-class shuffle control by at least `0.05`;
- at least two of three Val cities beat the class-conditional prior.

Failure rejects linear recoverability from post-FRM SAR P5, not every possible
nonlinear predictor. It nevertheless blocks direct feature reconstruction or
teacher imitation as the next default move.

## Question 3: shared optimization conflict

`diagnose_earthmiss_gradient_conflict.py` uses eight same-city Train batches for
each of the seven Train cities. For every batch it caches frozen RGB/SAR DINO
outputs once, then measures the released segmentation-loss gradients for
canonical Full and canonical SAR under both train-BN and eval-BN semantics.
Each batch contains eight distinct tiles. Sampling follows deterministic shuffled
without-replacement cycles; if a city has fewer than 64 tiles, reuse is allowed
only across batches and the available, unique scheduled, and reused tile counts
are written to `protocol.schedule_audit`. This preserves eight equally weighted
batches per city without misreporting the samples as 64 independent tiles.

Overlapping reports cover:

- Adapter aggregate and P2/P3/P4/P5 subgroups;
- FRM aggregate and target P2/P3/P4/P5 subgraphs;
- SEFusion aggregate and per scale;
- PRN and output head.

No optimizer is constructed. Endpoint RNG and all persistent BatchNorm buffers
are restored after every forward. Shared conflict is supported when train-BN
median cosine is negative and at least half of sampled batches have negative
cosine in either aggregate Adapter or aggregate FRM.
The gradient runner also snapshots the repository HEAD before loading the model
and refuses to write a report if another task switches the shared checkout before
the diagnostic finishes.

The completed formal audit found no released-path shared-gradient conflict under
train-BN semantics: aggregate Adapter, aggregate FRM, and the FRM-P2 target
subgraph all had positive median Full/SAR gradient cosine and zero sampled
negative-cosine fraction. Eval-BN was more mixed, but the preserved V2 matched-BN
recalibration had already reduced canonical SAR Val mIoU by `7.613 pp`; simple BN
recalibration is therefore not reopened.

## Question 4: is the useful FRM-P2 correction SAR-predictable?

`diagnose_earthmiss_frmp2_correction_recoverability.py` is the current decision
experiment. It follows from two completed causal facts:

- exact Full post-FRM P2 replacement improved canonical SAR Val by `8.1679 pp`;
- partial replacement was monotonic, and `alpha=0.25` alone improved pooled Val
  by `2.808 pp`, with all seven supported classes improving.

Those interventions establish downstream utility, not deployability. The new
diagnostic asks whether a correction in the useful direction can be predicted
from SAR P2 alone.

The Run C checkpoint and all MM-DINO parameters are frozen. On Train only, the
runner selects 16 deterministic tiles from each of the seven cities, disables
Train augmentation, and uses non-overlapping `512/512` crops. P2 cells with at
least 75% native-pixel GT purity are sampled uniformly up to 64 cells per class
per crop and capped at 2048 cells per class per city. GT is used only to define
the sampling strata and shuffled control; it is not an input to the predictor.

For every leave-one-city-out fold, a fixed affine `1x1` ridge map predicts
`Full_P2 - SAR_P2` from canonical SAR post-FRM P2. Six cities fit the map and the
seventh city is completely excluded from its moments. The predicted correction
is multiplied by the already frozen `alpha=0.25`, added identically to the two
equal canonical post-FRM P2 slots, and passed through the unchanged frozen
SEFusion/PRN/head. Controls are:

- zero correction (released canonical SAR);
- the six-city global mean correction;
- an equal-capacity ridge map whose targets are shuffled within each Train city
  and GT class;
- the exact paired Full-minus-SAR correction at `alpha=0.25`, which verifies that
  the causal ceiling remains present on this held-out-Train subset.

The ridge penalty (`1e-2`), sampling, correction alpha, and seed (`20260817`) are
frozen by the CLI. A formal report requires 16 tiles per city and the cached/direct
forward equivalence check. The runner constructs no optimizer, never populates
parameter gradients, audits all BatchNorm buffers, and refuses to write if Git
HEAD changes.

Direct linear FRM-P2 correction is supported only when all gates pass:

- exact oracle is at least `+0.25 pp` over SAR and nonnegative in at least 5/7
  held-out cities;
- ridge is at least `+0.25 pp` over SAR, shuffled ridge, and global mean;
- ridge is nonnegative in at least 5/7 held-out cities against each comparator.

Failure stops direct affine P2 correction; it does not authorize a nonlinear
module. Success authorizes only a later bounded, capacity-controlled SAR-P2
residual experiment. This probe is a diagnostic fit, not MM-DINO training, and
its Train-subset mIoU is not an estimate of Val or Test performance.

## Historical v1 route selection

`analyze_earthmiss_causal_diagnostics.py` accepted only the complete formal
oracle screen, formal recoverability report, and formal train/eval-BN gradient
report from the same checkpoint. This table is retained as provenance but is
superseded by Question 4 because its recoverability unit was P5.

| Oracle useful | SAR recoverable | Shared conflict | Permitted interpretation |
|---|---|---|---|
| no | any | no | stop tested feature alignment |
| no | any | yes | protect SAR optimization; do not align Full features |
| yes | no | no | stop direct privileged transfer at tested SAR P5 |
| yes | no | yes | protect SAR anchor; do not reconstruct unpredictable Full state |
| yes | yes | yes | candidate protected SAR anchor + SAR-predictable compensation |
| yes | yes | no | change transfer predictor/unit before rewriting shared path |

## Current server execution order

Do not rerun the old P5 recoverability probe or old three-report analyzer. The
current implementation requires only the Question 4 smoke and then its frozen
formal Train-only run. Both commands are foreground commands and use distinct,
non-overwritable outputs.

### 0. Smoke the FRM-P2 path

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino

python scripts/diagnose_earthmiss_frmp2_correction_recoverability.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --tiles-per-city 1 \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/smoke-frmp2-correction-recoverability.json
```

The smoke report must have `formal=false`; it validates execution only and must
not be interpreted scientifically.

### 1. Formal leave-one-city-out diagnostic

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/diagnose_earthmiss_frmp2_correction_recoverability.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-train-frmp2-correction-recoverability.json
```

No command in this protocol reads EarthMiss Test or starts MM-DINO training.
