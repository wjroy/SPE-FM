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

## Baselines

Models: Persistence, Moving average, ARIMA, Att-BiLSTM, Transformer, Informer, PatchTST and A3TGCN. ARIMA defaults to an eight-observation fitting window. Third-party model sources are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

```bash
python run_baselines.py --mode plan --output outputs/baseline_plan
python run_baselines.py --mode selected --models Persistence "Moving average" ARIMA --horizon 1 --output outputs/statistical_h1
python run_baselines.py --mode selected --models Persistence "Moving average" ARIMA --split 40/10/50 --horizon 3 --output outputs/statistical_h3
python run_baselines.py --mode selected --models Att-BiLSTM --split 40/10/50 --horizon 1 --output outputs/attbilstm40
python run_baselines.py --mode search --models ARIMA --split 40/10/50 --horizon 1 --output outputs/arima40_search
```

`plan` exports configurations; `selected` runs fixed configurations; `search` selects configurations on validation data. Statistical models use grid search. Neural search evaluates the explicit candidate lists in the configuration: 16 sampled/anchored candidates per split, successive halving with 6 and 3 survivors, and a protected candidate. Training states continue across stages.

By default, all neural baselines select finalists by the mean validation score over seeds 42, 43 and 44. PatchTST uses normalized MSE with RevIN disabled; the other neural models use RMSE. Temporal models use epoch budgets; A3TGCN uses optimizer-update budgets. Settings are in `baselines/configs/baseline_search.yaml`.

Runs save settings, checkpoints, predictions, diagnostics and MAE/RMSE. Neural baseline metrics are averaged across nodes, then across the configured training seeds.

For longer horizons, use `--horizon 5` or `--horizon 8` with separate output directories.

## Seeds for reproducing Tables 5 and 6

The training seeds corresponding to the displayed **Table 5 single-run metrics** are listed below. Checkpoint selection still uses validation data. These entries differ from the default three-seed mean reported by the runner.

| Model | 60/10/30 | 40/10/50 | 20/10/70 |
|---|---:|---:|---:|
| Att-BiLSTM | 42 | 43 | 43 |
| Transformer | 42 | 42 | 44 |
| Informer | 43 | 42 | 44 |
| PatchTST | 42 | 42 | 42 |
| A3TGCN | 44 | 42 | 42 |

For Att-BiLSTM, Transformer, Informer and A3TGCN, the displayed Table 5 pairs numerically match the lowest test RMSE among the archived runs with seeds 42, 43 and 44. The table records those historical entries; it does not establish validation-only selection of the reported seed. They must not be described as a common seed42 experiment or as three-seed means. The default runner continues to report all three seeds for a consistent repeated-run comparison.

Use `--seeds` to reproduce an individual Table 5 entry. This overrides neural training seeds only; the evaluation seed remains 42. For example:

```bash
python run_baselines.py --mode selected --models Att-BiLSTM --split 40/10/50 --horizon 1 --seeds 43 --output outputs/table5_attbilstm40_seed43
python run_baselines.py --mode selected --models Transformer --split 20/10/70 --horizon 1 --seeds 44 --output outputs/table5_transformer20_seed44
```

**Table 6 PatchTST reports the arithmetic mean of the metrics from training seeds 42, 43 and 44**, after node averaging; predictions are not ensembled. The fixed configuration is inherited from the 40/10/50 one-step task, as specified in Supplementary Table S6. Train a separate output head for each horizon using the configured budgets and validation checkpoint rule:

```bash
python run_baselines.py --mode selected --models PatchTST --split 40/10/50 --horizon 3 --output outputs/table6_patchtst_h3
python run_baselines.py --mode selected --models PatchTST --split 40/10/50 --horizon 5 --output outputs/table6_patchtst_h5
python run_baselines.py --mode selected --models PatchTST --split 40/10/50 --horizon 8 --output outputs/table6_patchtst_h8
```

Persistence, Moving average and ARIMA are deterministic under the specified settings and have no training seed. Extended-horizon metrics average over every predicted step within each node, then across nodes.

For historical Table 6 reproduction, use `--mode selected` as above. Using `--mode search` for extended-horizon PatchTST performs a new horizon-specific hyperparameter search and does not reproduce the inherited-configuration protocol.

## License

Original code: [MIT License](LICENSE). Third-party licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. The code license does not cover monitoring data; redistribution requires the data owner's permission.
