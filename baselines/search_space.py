import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path

import yaml

from spefm.data import SPLITS


def candidates(spec, split=None):
    if spec['family'] == 'neural':
        if split not in spec['candidates']:
            raise ValueError('No neural candidates for this split.')
        for row in spec['candidates'][split]:
            if set(row) & set(spec['fixed']):
                raise ValueError('A parameter cannot be both searched and fixed.')
            yield {**spec['fixed'], **row}
        return
    grid = spec["grid"]
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Each search axis must be a nonempty list.")
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        parameters = dict(zip(keys, values))
        fixed = dict(spec.get("fixed", {}))
        if set(parameters) & set(fixed):
            raise ValueError("A parameter cannot be both searched and fixed.")
        if fixed.pop("stride_rule", None) == "half_patch_length":
            fixed["stride"] = parameters["patch_length"] // 2
        if "d_ff_multiplier" in fixed:
            fixed["d_ff"] = fixed.pop("d_ff_multiplier") * parameters["d_model"]
        yield {**fixed, **parameters}


def planned_trials(config):
    for experiment, conditions in config["experiments"].items():
        for split, horizon in itertools.product(conditions["splits"], conditions["horizons"]):
            if split not in SPLITS or not isinstance(horizon, int) or horizon < 1:
                raise ValueError("Invalid split or forecast horizon.")
            for model, spec in config["models"].items():
                if (spec["family"] not in conditions["families"]
                        or model not in conditions.get("models", config["models"])):
                    continue
                for index, params in enumerate(candidates(spec, split), 1):
                    trial = dict(status="planned_not_run", experiment=experiment,
                                 split=split, horizon=horizon, model=model, candidate=index,
                                 context_length=config["context_length"], parameters=params,
                                 selection=dict(config["selection"]))
                    if spec["family"] == "neural":
                        trial["training"] = deepcopy({**config["neural_training"], **spec['training']})
                    yield trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=Path(__file__).resolve().parent/"configs/baseline_search.yaml")
    parser.add_argument("--output", type=Path, help="Destination JSONL file for planned candidates; requires an unused path.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    counts = Counter()
    for trial in planned_trials(config):
        counts[trial["model"]] += 1
    print(json.dumps({"status": "planned_not_run", "counts": counts,
                      "total": sum(counts.values())}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            for trial in planned_trials(config):
                handle.write(json.dumps(trial)+"\n")


if __name__ == "__main__":
    main()
