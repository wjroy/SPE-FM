# Experiment parameters

Layer indices are zero-based. Full-precision settings: [SPE-FM](spefm/hyperparameters.yaml), [baselines](baselines/configs/baseline_selected.yaml), [Figure 15](baselines/configs/figure15.yaml).

## SPE-FM selected configurations

| Experiment | Split | Horizon | Prompt length | Injection layers | Batch | Microbatch | Learning rate | Weight decay | Epoch limit | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| Main / Table 2 / Figure 15 | 40/10/50 | 1 | 8 | 0–3, 16–19 | 8 | — | 5e-4 | 0 | 24 | 42 |
| Table 5 | 60/10/30 | 1 | 4 | 0–4, 15–19 | 8 | — | 3.728e-3 | 7.143e-4 | 24 | 42, 43, 44 |
| Table 5 | 40/10/50 | 1 | 8 | 0–3, 16–19 | 8 | — | 5e-4 | 0 | 24 | 42, 43, 44 |
| Table 5 | 20/10/70 | 1 | 4 | 0–4, 15–19 | 8 | — | 3.728e-3 | 7.143e-4 | 24 | 42, 43, 44 |
| Table 6 | 40/10/50 | 3 | 16 | 0–4, 15–19 | 32 | 4 | 5e-3 | 1e-4 | 6 | 42, 43, 44 |
| Table 6 | 40/10/50 | 5 | 8 | 0–3, 16–19 | 4 | 4 | 1e-2 | 2.727e-4 | 6 | 42, 43, 44 |
| Table 6 | 40/10/50 | 8 | 4 | 0–1, 18–19 | 32 | 8 | 5e-2 | 1e-5 | 6 | 42, 43, 44 |

## Table S2

Split 40/10/50; context 8; horizon 1; seed 42; prompt length 8; batch 8; learning rate 5e-4; weight decay 0; epoch limit 24.

| Configuration | Injection layers | Selected |
|---|---|---|
| Shallow-1 | 0 | |
| Shallow-2 | 0–1 | |
| Shallow-3 | 0–2 | |
| Deep-1 | 19 | |
| Deep-2 | 18–19 | |
| Deep-3 | 17–19 | |
| Shallow-deep 3+3 | 0–2, 17–19 | |
| Shallow-deep 4+4 | 0–3, 16–19 | Yes |
| Shallow-deep 5+5 | 0–4, 15–19 | |
| All layers | 0–19 | |

## Shared model settings

| Component | Parameters |
|---|---|
| Data | Context 8; default split 40/10/50; default horizon 1 |
| Backbone | TimesFM 2.5-200M; frozen; dimension 1280; 20 layers |
| Prompt training | AdamW; mean pinball loss; patience 8 |
| Extended-horizon initialization | Import backend before seeding; deterministic algorithms enabled |
| Decoder | Width 64; batch 16; AdamW; learning rate 1e-3; weight decay 0; epoch limit 80; patience 10 |
| Decoder gate | Lower 0.2; upper 0.95; neighbors 5 |
| Point prediction | q0.5; head 5 |
| Inference | Maximum batch 8; node-sequence budget 8192; quantile crossing correction enabled |
| Interval calibration | Single-run intervals / Figure 15; enabled; target coverage 0.81; learning rate 0.25 |

## Table S1: spatial settings

| Split | Prototypes | Seed |
|---|---|---|
| 20/10/70 | 4 | 42 |
| 40/10/50 | 3 | 42 |
| 60/10/30 | 3 | 42 |

| Parameter | Value |
|---|---|
| Clustering | DTW–K-medoids; increments; absolute local cost |
| Prototype selection | Endpoint-distance elbow; candidates 1–10 |
| Restarts / maximum iterations | 30 / 1000 |
| Association | Inverse distance; path normalization enabled; epsilon 1e-12 |

## Table 4 / Figure 15

Split 40/10/50; context 8; horizon 1; seed 42.

| Method | Training | Learning rate | Weight decay | Batch | Epoch limit | Patience | Calibration |
|---|---|---|---|---|---|---|---|
| SPE-FM | Spatial prompts + gated decoder | 5e-4 / 1e-3 | 0 / 0 | 8 / 16 | 24 / 80 | 8 / 10 | Enabled |
| IO | Input/output modules; mean pinball | 1e-3 | 1e-5 | 8 | 24 | 8 | Disabled |
| LoRA | Rank 6; alpha 8; dropout 0; mean pinball | 1e-3 | 1e-5 | 8 | 24 | 8 | Disabled |
| FROZEN | Frozen backbone | — | — | 8 | — | — | Disabled |

## Risk settings

| Parameter | Value |
|---|---|
| Perturbation replicas / detrending window / seed | 300 / 8 / 42 |
| Risk bounds, mm | 0, 10, 20, 30, 40, 50 |
| Entropy divisor / hyperentropy ratio | 3 / 0.1 |
| Grade scores | 0, 0.25, 0.5, 0.75, 1 |
| Cloud drops / risk seed | 3000 / 42 |
| Terminal shoulders / sampled standard entropy | Enabled / enabled |
| Weighting | Membership entropy × risk severity; normalized |

## Figure 16

| Variant | Perturbation | Node weights |
|---|---|---|
| Proposed | Enabled | Normalized information × severity |
| Without perturbation | Disabled | Normalized information × severity |
| Entropy only | Enabled | Normalized information |
| Severity only | Enabled | Normalized severity |

## Tables S3–S4

| Experiment | Nodes | Perturbation replicas | Warm-up runs | Timed runs | Seed |
|---|---|---|---|---|---|
| Table S3 | 84 | 300 | 10 | 30 | 42 |
| Table S4 | 21, 42, 84 | 50, 100, 200, 300 | 10 | 30 | 42 |
