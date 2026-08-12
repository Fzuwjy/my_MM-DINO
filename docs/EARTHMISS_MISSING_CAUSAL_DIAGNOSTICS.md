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

## Question 2: SAR recoverability

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

Overlapping reports cover:

- Adapter aggregate and P2/P3/P4/P5 subgroups;
- FRM aggregate and target P2/P3/P4/P5 subgraphs;
- SEFusion aggregate and per scale;
- PRN and output head.

No optimizer is constructed. Endpoint RNG and all persistent BatchNorm buffers
are restored after every forward. Shared conflict is supported when train-BN
median cosine is negative and at least half of sampled batches have negative
cosine in either aggregate Adapter or aggregate FRM.

## Pre-registered route selection

`analyze_earthmiss_causal_diagnostics.py` accepts only the complete formal
oracle screen, formal recoverability report, and formal train/eval-BN gradient
report from the same checkpoint.

| Oracle useful | SAR recoverable | Shared conflict | Permitted interpretation |
|---|---|---|---|
| no | any | no | stop tested feature alignment |
| no | any | yes | protect SAR optimization; do not align Full features |
| yes | no | no | stop direct privileged transfer at tested SAR P5 |
| yes | no | yes | protect SAR anchor; do not reconstruct unpredictable Full state |
| yes | yes | yes | candidate protected SAR anchor + SAR-predictable compensation |
| yes | yes | no | change transfer predictor/unit before rewriting shared path |

## Server execution order

All commands are foreground commands. Smoke outputs and formal outputs use
different paths and are never overwritten.

### 0. Smoke all three paths

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino

python scripts/diagnose_earthmiss_oracle_intervention.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --smoke-tiles 2 \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/smoke-oracle.json

python scripts/diagnose_earthmiss_gradient_conflict.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --batches-per-city 1 \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/smoke-gradients.json

python scripts/diagnose_earthmiss_sar_recoverability.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --train-tiles-per-city 2 \
  --val-tiles 6 \
  --max-examples 5000 \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/smoke-recoverability.json
```

### 1. Formal oracle screen

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/diagnose_earthmiss_oracle_intervention.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-val-oracle-screen.json
```

### 2. Formal shared-gradient audit

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/diagnose_earthmiss_gradient_conflict.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-train-gradient-conflict.json
```

### 3. Formal SAR recoverability probe

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/diagnose_earthmiss_sar_recoverability.py \
  --checkpoint /root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe/run_c_seed42/best_sar.pth \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-sar-p5-recoverability.json
```

### 4. Apply the frozen decision tree

```bash
cd /root/my_MM-DINO
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mm-dino
python scripts/analyze_earthmiss_causal_diagnostics.py \
  --oracle /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-val-oracle-screen.json \
  --recoverability /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-sar-p5-recoverability.json \
  --gradients /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-train-gradient-conflict.json \
  --output /root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/run-c-e15-causal-decision.json
```

No command in this protocol reads EarthMiss Test or starts MM-DINO training.
