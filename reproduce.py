import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "6")

import pandas as pd
import yaml

from run import load_config
from spefm.data import SPLITS
from spefm.weights import resolve_backbone_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--split", choices=list(SPLITS), nargs="+", default=list(reversed(SPLITS)))
    parser.add_argument("--backbone-weights", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Select an empty output directory.")
    source = args.config.resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    rows, completed = [], {}
    for split in dict.fromkeys(args.split):
        for setting, selected in (("default", False), ("selected", True)):
            cfg = load_config(args.config, split, selected=selected)
            data = Path(cfg["data"]["csv"])
            cfg["data"]["csv"] = str((source/data).resolve() if not data.is_absolute() else data)
            cfg["model"]["backbone_weights"] = str(resolve_backbone_weights(
                args.backbone_weights or cfg["model"]["backbone_weights"], config_directory=source))
            key = json.dumps(cfg, sort_keys=True)
            if key not in completed:
                directory = output/split.replace("/", "_")/setting
                directory.mkdir(parents=True)
                config = directory/"config.yaml"
                config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                run = directory/"model"
                for stage in ("spatial", "train", "predict"):
                    subprocess.run([sys.executable, "-X", "utf8", "-s", "-u",
                                    str(Path(__file__).with_name("run.py")), stage,
                                    "--config", str(config), "--output", str(run)], check=True)
                completed[key] = run
            run = completed[key]
            metrics = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
            rows.append(dict(split=split, setting=setting, seed=cfg["training"]["seed"],
                             **{name: metrics[name] for name in ("RMSE", "MAE", "PICP", "PINAW", "CWC")}))
            pd.DataFrame(rows).to_csv(output/"comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
