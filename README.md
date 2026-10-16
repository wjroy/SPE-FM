# SPE-FM

Settlement forecasting and risk synthesis with DTW spatial prototypes and a frozen TimesFM 2.5 backbone.

## Files

| Path | Contents |
|---|---|
| [spefm/](spefm/) | Model, decoder, data, spatial priors, training, prediction, metrics, calibration and risk modules |
| [spefm/hyperparameters.yaml](spefm/hyperparameters.yaml) | SPE-FM parameters, split/horizon settings and seeds |
| [baselines/](baselines/) | Statistical, temporal and graph baselines; four-method adaptation comparison |
| [baselines/configs/](baselines/configs/) | `baseline_selected.yaml`: fixed parameters; `baseline_search.yaml`: search spaces and seeds; `figure15.yaml`: adaptation comparison |
| [baselines/vendor/](baselines/vendor/) | Informer, PatchTST and A3TGCN sources; `UPSTREAM.json`: source revisions and licenses |
| [data/settlement.csv](data/settlement.csv) | Monitoring data: 340 observations × 84 nodes, millimetres |
| [config.yaml](config.yaml) | Data path, chronological split, input/output lengths, weights and device |
| [PARAMETERS.md](PARAMETERS.md) | Experiment parameter tables and seeds |
| [reproduce.py](reproduce.py) | Multi-seed SPE-FM point forecasts and RMSE/MAE summaries |
| [run.py](run.py) / [calibrate.py](calibrate.py) | Individual model stages / interval calibration |
| [run_baselines.py](run_baselines.py) | Baseline experiment entry point |
| [requirements.txt](requirements.txt) | Python dependencies |
| [licenses/](licenses/) / [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Third-party licenses and attribution |

## Setup

Python 3.11+ and PyTorch 2.5. Run commands from the repository root:

```bash
git clone https://github.com/wjroy/SPE-FM.git
cd SPE-FM
python -m pip install -r requirements.txt
```

Place the official [TimesFM 2.5-200M checkpoint](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/1d952420fba87f3c6dee4f240de0f1a0fbc790e3) at `weights/model.safetensors`, or set `model.backbone_weights` in `config.yaml`. Baselines do not require this checkpoint.

The supplied CSV is ready to use. Custom data need unique increasing `date` values and finite `DB*` node columns in millimetres. Set paths and device in `config.yaml`; defaults are split 40/10/50, context 8 and horizon 1. Full-precision model settings are in YAML; layer indices are zero-based.

## SPE-FM point forecasts

Table 5 and Table 6:

```bash
python reproduce.py --split 60/10/30 40/10/50 20/10/70 --horizon 1 --output outputs/table5
python reproduce.py --split 40/10/50 --horizon 3 5 8 --output outputs/table6
```

Each condition trains seeds **42, 43, 44**; spatial seed 42. `summary.csv` contains the mean and sample standard deviation (ddof=1) of per-seed, node-averaged RMSE/MAE; `per_seed_metrics.csv` contains individual results. Point forecasts use q0.5 without interval calibration.

Use an empty output directory. Add `--jobs 2 --threads 4` for two concurrent runs; `--help` lists all options.

## Intervals and risk

Default single-seed run: 42, split 40/10/50, horizon 1. Keep the same configuration and output directory for all stages:

```bash
python run.py spatial --output outputs/run40
python run.py train --output outputs/run40
python run.py predict --output outputs/run40
python run.py warning --output outputs/run40
```

`predict` saves calibrated intervals and `metrics.json`; `warning` saves `warning.npz` and supports horizon 1.

## Baselines

```bash
python run_baselines.py --mode selected --models Persistence "Moving average" ARIMA --split 40/10/50 --horizon 1 --output outputs/statistical40
python run_baselines.py --mode selected --models Att-BiLSTM --split 40/10/50 --horizon 1 --output outputs/attbilstm40
python run_baselines.py --mode selected --models PatchTST --split 40/10/50 --horizon 3 --output outputs/patchtst40_h3
```

`selected` uses fixed parameters; `search` selects on validation data; `plan` exports configurations. Neural baselines average seeds 42, 43, 44 with evaluation seed 42. Statistical baselines are deterministic. Results: `summary.csv` and model subfolders.

## Figure 15

```bash
python baselines/figure15.py run --output outputs/figure15
```

Single seed 42. SPE-FM uses calibrated intervals; IO, LoRA and FROZEN use native intervals. Outputs: `results.csv`, `results.json`, `parameters.csv` and `source_data.csv`.

## License

Original code: [MIT](LICENSE). Third-party licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Monitoring-data redistribution requires the owner's permission.
