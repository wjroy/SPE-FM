import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import pandas as pd
import yaml

from run import load_config
from spefm.data import SPLITS
from spefm.weights import resolve_backbone_weights


def summarize(rows, seeds):
    records = []
    conditions = dict.fromkeys((row["split"], row["horizon"]) for row in rows)
    for split, horizon in conditions:
        group = [row for row in rows if (row["split"], row["horizon"]) == (split, horizon)]
        if {row["seed"] for row in group} != set(seeds) or len(group) != len(seeds):
            continue
        record = dict(model="SPE-FM", split=split, horizon=horizon,
                      seeds=",".join(map(str, seeds)), n_seeds=len(seeds))
        for metric in ("RMSE", "MAE"):
            values = [row[metric] for row in group]
            record[metric + "_mean"] = statistics.mean(values)
            record[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        records.append(record)
    return pd.DataFrame(records)


def run_seed(split, horizon, seed, cfg, output, environment):
    directory = output/split.replace("/", "_")/f"h{horizon}"/f"seed{seed}"
    directory.mkdir(parents=True)
    cfg = deepcopy(cfg)
    cfg["training"].pop("seeds", None)
    cfg["training"]["seed"] = seed
    config = directory/"config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run = directory/"model"
    for stage in ("spatial", "train", "predict"):
        command = [sys.executable, "-X", "utf8", "-s", "-u",
                   str(Path(__file__).with_name("run.py")), stage,
                   "--config", str(config), "--output", str(run)]
        if stage == "predict":
            command.append("--point-only")
        with (directory/f"{stage}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    metrics = json.loads((run/"point_metrics.json").read_text(encoding="utf-8"))
    if any(not math.isfinite(metrics[name]) for name in ("RMSE", "MAE")):
        raise ValueError(f"Nonfinite metrics in {run}")
    return dict(model="SPE-FM", split=split, horizon=horizon, seed=seed,
                RMSE=metrics["RMSE"], MAE=metrics["MAE"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--split", choices=list(SPLITS), nargs="+", default=list(reversed(SPLITS)))
    parser.add_argument("--horizon", type=int, nargs="+", default=[1])
    parser.add_argument("--seeds", type=int, nargs="+", help="Defaults to training.seeds in the parameter file.")
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent independent runs.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per run.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    if (not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        parser.error("Seeds must be distinct integers in [0, 2**32).")
    if min(args.horizon) < 1 or args.jobs < 1 or args.threads < 1:
        parser.error("Horizons, jobs and threads must be positive.")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Select an empty output directory.")
    source = args.config.resolve().parent
    tasks = []
    for split in dict.fromkeys(args.split):
        for horizon in dict.fromkeys(args.horizon):
            cfg = load_config(args.config, split, horizon=horizon)
            data = Path(cfg["data"]["csv"])
            cfg["data"]["csv"] = str((source/data).resolve() if not data.is_absolute() else data)
            cfg["model"]["backbone_weights"] = str(resolve_backbone_weights(
                args.backbone_weights or cfg["model"]["backbone_weights"], config_directory=source))
            for seed in seeds:
                tasks.append((split, horizon, seed, cfg))
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[variable] = str(args.threads)
    (output/"protocol.json").write_text(json.dumps(dict(
        seeds=seeds, jobs=args.jobs, threads_per_run=args.threads,
        aggregation="arithmetic mean of per-seed, node-averaged test metrics",
        standard_deviation="sample (ddof=1); undefined for a single seed",
        prediction="q0.5; no prediction ensemble"), indent=2), encoding="utf-8")
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {executor.submit(run_seed, *task, output, environment): task[:3] for task in tasks}
        for future in as_completed(pending):
            try:
                row = future.result()
            except Exception as exc:
                split, horizon, seed = pending[future]
                for other in pending:
                    other.cancel()
                raise RuntimeError(f"Failed: split={split}, horizon={horizon}, seed={seed}; "
                                   f"see {output/split.replace('/', '_')/f'h{horizon}'/f'seed{seed}'}") from exc
            rows.append(row)
            rows.sort(key=lambda item: (args.split.index(item["split"]),
                                       args.horizon.index(item["horizon"]), seeds.index(item["seed"])))
            pd.DataFrame(rows).to_csv(output/"per_seed_metrics.csv", index=False)
            summary = summarize(rows, seeds)
            if not summary.empty:
                summary.to_csv(output/"summary.csv", index=False)
            print(f"Completed {len(rows)}/{len(tasks)}: {row}", flush=True)
    print(summarize(rows, seeds).to_string(index=False))


if __name__ == "__main__":
    main()
