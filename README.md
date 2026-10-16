# SPE-FM

DTW spatial prototypes, frozen TimesFM 2.5 prompt adaptation, quantile forecasting and cloud-based risk synthesis.

## Installation

Python 3.11+ and PyTorch 2.5:

```bash
python -m pip install -r requirements.txt
```

Place the official [TimesFM 2.5-200M checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/1d952420fba87f3c6dee4f240de0f1a0fbc790e3) at `weights/model.safetensors`, or set `model.backbone_weights` in `config.yaml`.

## Configurations

| File | Conditions |
|---|---|
| [spefm/hyperparameters.yaml](spefm/hyperparameters.yaml) | SPE-FM: seed 42; Table 2/S2; Table 5 splits 60/10/30, 40/10/50, 20/10/70; Table 6 horizons 1, 3, 5, 8; Figure 15 |
| [config.yaml](config.yaml) | Data, weights and device |
| [baselines/configs/baseline_selected.yaml](baselines/configs/baseline_selected.yaml) | Selected baseline parameters |
| [baselines/configs/baseline_search.yaml](baselines/configs/baseline_search.yaml) | Baseline search spaces and seeds |
| [baselines/configs/figure15.yaml](baselines/configs/figure15.yaml) | Figure 15 comparison and plotting settings |

Data: `data/settlement.csv`, 340 observations × 84 nodes, millimetres. The default split is 40/10/50, context 8, horizon 1. Injection-layer indices in YAML are zero-based.

## SPE-FM

```bash
python run.py spatial --output outputs/run40 --split 40/10/50
python run.py train   --output outputs/run40 --split 40/10/50
python run.py predict --output outputs/run40 --split 40/10/50
python run.py warning --output outputs/run40 --split 40/10/50
```

Use `--horizon 3`, `--horizon 5` or `--horizon 8` for Table 6 and a separate output directory per condition. The warning stage supports horizon 1. Point metrics use q0.5; one-step intervals use causal calibration.

## Baselines

```bash
python run_baselines.py --mode selected --models Persistence "Moving average" ARIMA --split 40/10/50 --horizon 1 --output outputs/statistical40
python run_baselines.py --mode selected --models Att-BiLSTM --split 40/10/50 --horizon 1 --output outputs/attbilstm40
python run_baselines.py --mode selected --models PatchTST --split 40/10/50 --horizon 3 --output outputs/patchtst40_h3
```

Modes: `plan`, `selected`, `search`. Selection uses validation data. Neural baselines report the arithmetic mean of per-seed, node-averaged metrics for seeds 42, 43 and 44; evaluation seed 42. Statistical baselines are deterministic.

## Figure 15

```bash
python baselines/figure15.py run --output outputs/figure15
```

Single-seed comparison (42): SPE-FM uses calibrated intervals; IO, LoRA and FROZEN use native intervals.

## License

Original code: [MIT](LICENSE). Dependencies: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. Monitoring-data redistribution requires the owner's permission.
