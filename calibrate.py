import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from run import digest
from spefm.calibration import OnlineInterval, fit_width_scale, replay
from spefm.data import read_monitoring, split_bounds, windows
from spefm.metrics import evaluate
from spefm.prediction import Predictor


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    staged.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    staged.replace(path)


def signature(directory):
    info = read_json(directory/"run.json")
    names = ["run.json", "prompt.pt", "spatial.npz"]
    if info["config"].get("decoder", {}).get("enabled", False):
        names.append("decoder.pt")
    return {name: digest(directory/name) for name in names}


def monitoring(info, config_directory, path=None):
    cfg = info["config"]["data"]
    path = Path(path or cfg["csv"])
    if not path.is_absolute():
        path = config_directory/path
    values, dates, nodes = read_monitoring(path, cfg["date_column"], cfg["node_prefix"])
    bounds = split_bounds(len(values), cfg["split"])
    if (digest(path) != info["data_sha256"] or nodes != info["nodes"]
            or {key: list(value) for key, value in bounds.items()} != info["split_bounds"]):
        raise ValueError("Monitoring data do not match the model run.")
    return values, dates, nodes, bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("fit", "predict", "forecast", "observe"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--coverage", type=float, default=.81)
    parser.add_argument("--learning-rate", type=float, default=.25)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--actual", type=Path)
    parser.add_argument("--origin", type=int)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    info = read_json(args.run/"run.json")
    if info["config"]["data"]["horizon"] != 1:
        parser.error("Online calibration requires one-step forecasts.")
    if info["config"]["inference"]["point_head"] not in (0, 5):
        parser.error("Calibration requires point head 0 or 5.")
    output = args.output.resolve()
    calibration_path = output/"calibration.json"
    state_path = args.state or output/"state.json"
    if args.stage == "fit":
        if calibration_path.exists():
            raise FileExistsError(calibration_path)
        if not 0 < args.coverage < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("Invalid coverage or learning rate.")
        predictor = Predictor(args.run, args.config_directory, args.backbone_weights)
        values, dates, nodes, bounds = monitoring(predictor.info, args.config_directory, args.data)
        x, y, origins = windows(values[:bounds["val"][1]], bounds["val"], predictor.config["data"]["context"], 1)
        q = predictor.predict(x)
        parameters = fit_width_scale(q, y, args.coverage)
        policy = dict(version="causal_interval_width_feedback_v1", target_coverage=args.coverage,
                      learning_rate=args.learning_rate)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/"validation.npz", predictions=q, actual=y, origins=origins,
                            dates=dates[origins], nodes=np.asarray(nodes))
        write_json(calibration_path, dict(parameters=parameters, policy=policy, source=signature(args.run),
                   backbone_sha256=predictor.backbone_sha256, data_sha256=predictor.info["data_sha256"], nodes=nodes))
        print(json.dumps(parameters))
        return
    saved = read_json(calibration_path)
    if saved["source"] != signature(args.run):
        raise ValueError("Calibration does not match the model run.")
    if args.stage == "observe":
        if args.actual is None or args.origin is None:
            parser.error("Missing observation or origin.")
        controller = OnlineInterval.from_dict(read_json(state_path))
        if controller.parameters != saved["parameters"] or controller.policy != saved["policy"]:
            raise ValueError("Calibration state mismatch.")
        record = controller.observe(np.load(args.actual, allow_pickle=False), args.origin)
        write_json(output/f"observation_{args.origin}.json", record)
        write_json(state_path, controller.to_dict(), overwrite=True)
        print(json.dumps(record))
        return
    predictor = Predictor(args.run, args.config_directory, args.backbone_weights)
    if predictor.backbone_sha256 != saved["backbone_sha256"] or predictor.info["nodes"] != saved["nodes"]:
        raise ValueError("Calibration model mismatch.")
    if args.stage == "forecast":
        if args.contexts is None or args.origin is None:
            parser.error("Missing contexts or origin.")
        target = output/f"forecast_{args.origin}.npz"
        if target.exists():
            raise FileExistsError(target)
        controller = OnlineInterval.from_dict(read_json(state_path)) if state_path.exists() else OnlineInterval(saved["parameters"], saved["policy"])
        if controller.parameters != saved["parameters"] or controller.policy != saved["policy"]:
            raise ValueError("Calibration state mismatch.")
        q = predictor.predict(np.load(args.contexts, allow_pickle=False))
        calibrated = controller.forecast(q, args.origin)
        np.savez_compressed(target, predictions=calibrated, original_predictions=q,
                            origin=args.origin, nodes=np.asarray(saved["nodes"]))
        write_json(state_path, controller.to_dict(), overwrite=True)
        print(str(target))
        return
    if (output/"predictions.npz").exists():
        raise FileExistsError(output/"predictions.npz")
    values, dates, nodes, bounds = monitoring(predictor.info, args.config_directory, args.data)
    x, y, origins = windows(values, bounds["test"], predictor.config["data"]["context"], 1)
    q = predictor.predict(x)
    calibrated, records, state = replay(q, y, origins, saved["parameters"], saved["policy"])
    per_node, summary = evaluate(y, calibrated, point_head=predictor.config["inference"]["point_head"])
    np.savez_compressed(output/"predictions.npz", predictions=calibrated, original_predictions=q,
                        actual=y, origins=origins, dates=dates[origins], nodes=np.asarray(nodes))
    pd.DataFrame({"node": nodes, **per_node}).to_csv(output/"metrics.csv", index=False)
    pd.DataFrame(records).to_csv(output/"online_trace.csv", index=False)
    write_json(output/"metrics.json", summary)
    write_json(output/"final_state.json", state)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
