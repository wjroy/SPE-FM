# SPE-FM

SPE-FM combines DTW–K-medoids spatial prototypes, prompt adaptation of a frozen TimesFM 2.5 backbone, quantile forecasting and cloud-based risk synthesis.

## Structure

```text
SPE-FM/
├── spefm/
├── baselines/
│   ├── configs/
│   └── vendor/
├── data/
│   └── settlement.csv
├── licenses/
├── config.yaml
├── run.py
├── run_baselines.py
├── requirements.txt
├── LICENSE
├── THIRD_PARTY_NOTICES.md
└── README.md
```

SPE-FM settings: `config.yaml`. Baseline settings: `baselines/configs/baseline_search.yaml` and `baselines/configs/baseline_selected.yaml`.

## Installation

Use Python 3.11+ and a suitable PyTorch 2.5 CPU/CUDA build:

```bash
python -m pip install -r requirements.txt
```

TimesFM is installed as a code dependency. Supply the official [TimesFM 2.5-200M checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/1d952420fba87f3c6dee4f240de0f1a0fbc790e3) locally and set `model.backbone_weights` to its file path. The default is `weights/model.safetensors`; relative paths resolve from the configuration directory. Weights are loaded locally and excluded from this repository. Baselines do not require TimesFM weights.

## Data and settings

`data/settlement.csv`: 340 observations × 84 nodes. Input requires finite, preprocessed measurements, a unique, increasing `date` column and node columns prefixed `DB`. Values are in millimetres, positive for settlement.

The default train/validation/test split is 40/10/50, with eight input observations and one forecast step. Supported splits also include 20/10/70 and 60/10/30. Lengths count valid observations. Splitting precedes window construction; spatial priors and normalization use training data.

## SPE-FM

Run the four stages using the same configuration, split and output directory:

```bash
python run.py spatial --config config.yaml --output outputs/run40 --split 40/10/50
python run.py train   --config config.yaml --output outputs/run40 --split 40/10/50
python run.py predict --config config.yaml --output outputs/run40 --split 40/10/50
python run.py warning --config config.yaml --output outputs/run40 --split 40/10/50
```

Use a separate output directory for each experiment. Point metrics use q0.5. The warning stage requires one-step forecasts. Existing outputs are protected from overwriting.

## Selected SPE-FM hyperparameters

All SPE-FM conditions use seed 42, an input length of 8, a frozen TimesFM backbone, AdamW, mean pinball loss and prompt early-stopping patience 8. Layer indices below are one-based; `config.yaml` uses zero-based indices. Epoch values are maximum training budgets.

### Table 5: chronological splits, horizon 1

| Train/validation/test | Batch size | Prompt length R | Learning rate | Weight decay | Injection layers | Epochs |
|---|---:|---:|---:|---:|---|---:|
| 60/10/30 | 8 | 4 | 0.003727593720314938 | 0.0007142857142857144 | 1–5, 16–20 | 24 |
| 40/10/50 | 8 | 8 | 0.0005 | 0 | 1–4, 17–20 | 24 |
| 20/10/70 | 8 | 4 | 0.003727593720314938 | 0.0007142857142857144 | 1–5, 16–20 | 24 |

The default 40/10/50 one-step configuration uses shallow–deep 4+4 prompting. Prompt micro-batch size is 8 for these three conditions.

### Table 6: extended horizons, split 40/10/50

| Horizon T | Batch size | Micro-batch size | Prompt length R | Learning rate | Weight decay | Injection layers | Epochs |
|---|---:|---:|---:|---:|---:|---|---:|
| 3 | 32 | 4 | 16 | 0.005 | 0.0001 | 1–5, 16–20 | 6 |
| 5 | 4 | 4 | 8 | 0.01 | 0.000272727 | 1–4, 17–20 | 6 |
| 8 | 32 | 8 | 4 | 0.05 | 0.00001 | 1–2, 19–20 | 6 |

These three horizon profiles set `initialize_after_backend_import: true`. The split profile is applied before the horizon profile. Select conditions with `--split` and `--horizon`.

### Shared SPE-FM settings

| Component | Settings |
|---|---|
| Spatial prior | K candidates 1–10; 30 restarts; maximum 1000 iterations; seed 42; inverse-distance soft association; path-normalized DTW |
| Gated decoder | Enabled; width 64; batch size 16; learning rate 0.001; weight decay 0; gate bounds 0.20–0.95; 5 neighbours; maximum 80 epochs; patience 10 |
| One-step interval calibration | Enabled; target coverage 0.81; update learning rate 0.25 |
| Inference | Point head 5 (q0.5); quantile-crossing correction enabled; maximum batch size 8; node-sequence budget 8192 |

## Baselines

Models: Persistence, Moving average, ARIMA, Att-BiLSTM, Transformer, Informer, PatchTST and A3TGCN. ARIMA defaults to an eight-observation fitting window. Third-party model sources are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

```bash
python run_baselines.py --mode plan --output outputs/baseline_plan
python run_baselines.py --mode selected --models Persistence "Moving average" ARIMA --horizon 1 --output outputs/statistical_h1
python run_baselines.py --mode selected --models Persistence "Moving average" ARIMA --split 40/10/50 --horizon 3 --output outputs/statistical_h3
python run_baselines.py --mode selected --models Att-BiLSTM --split 40/10/50 --horizon 1 --output outputs/attbilstm40
python run_baselines.py --mode selected --models PatchTST --split 40/10/50 --horizon 3 --output outputs/patchtst40_h3
python run_baselines.py --mode search --models ARIMA --split 40/10/50 --horizon 1 --output outputs/arima40_search
```

`plan` exports configurations; `selected` runs fixed configurations; `search` selects configurations on validation data. Statistical models use grid search. Neural search evaluates the explicit candidate lists in the configuration: 16 sampled/anchored candidates per split, successive halving with 6 and 3 survivors, and a protected candidate. Training states continue across stages.

Neural baselines run three independent training seeds (42, 43 and 44). Final MAE/RMSE are the arithmetic mean of the per-seed metrics, after node averaging; predictions are not ensembled. Checkpoints and configurations are selected using validation data with evaluation seed 42. Persistence, Moving average and ARIMA are deterministic and do not use training seeds.

PatchTST uses normalized MSE with RevIN disabled; the other neural models use RMSE. Temporal models use epoch budgets; A3TGCN uses optimizer-update budgets. Settings are in `baselines/configs/baseline_search.yaml`.

Runs save settings, checkpoints, predictions, diagnostics, and per-seed and aggregated MAE/RMSE.

For longer horizons, use `--horizon 5` or `--horizon 8` with separate output directories. In selected mode, extended-horizon PatchTST reuses the 40/10/50 one-step hyperparameters and trains a separate output head for each horizon.

## License

Original code: [MIT License](LICENSE). Third-party licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. The code license does not cover monitoring data; redistribution requires the data owner's permission.

