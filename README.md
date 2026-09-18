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

All neural baselines select finalists by the mean validation score over seeds 42, 43 and 44. PatchTST uses normalized MSE with RevIN disabled; the other neural models use RMSE. Temporal models use epoch budgets; A3TGCN uses optimizer-update budgets. Settings are in `baselines/configs/baseline_search.yaml`.

Runs save settings, checkpoints, predictions, diagnostics and MAE/RMSE. Neural baseline metrics are averaged across nodes, then across the three seeds.

For longer horizons, use `--horizon 5` or `--horizon 8` with separate output directories.

## License

Original code: [MIT License](LICENSE). Third-party licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. The code license does not cover monitoring data; redistribution requires the data owner's permission.
