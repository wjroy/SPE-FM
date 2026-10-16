import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from uuid import uuid4

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import yaml

from spefm.data import read_monitoring, split_bounds, windows, SPLITS
from spefm.spatial import fit_spatial_prior
from spefm.weights import resolve_backbone_weights


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path, split=None, selected=True, horizon=None):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if split:
        cfg["data"]["split"] = SPLITS[split].copy()
    if horizon is not None:
        cfg["data"]["horizon"] = horizon
    overrides = cfg.pop("split_overrides", {})
    if selected:
        for name, sections in overrides.items():
            if cfg["data"]["split"] == SPLITS[name]:
                for section, settings in sections.items():
                    cfg[section].update(settings)
    for section, settings in cfg.pop("horizon_overrides", {}).get(cfg["data"]["horizon"], {}).items():
        cfg[section].update(settings)
    return cfg


def prediction_stage(stage, cfg, output, config_directory, values, dates, nodes, bounds):
    from spefm.prediction import Predictor
    from spefm.metrics import evaluate
    from spefm.uncertainty import increment_scale, perturb, event_cloud
    from spefm.risk import warning
    from spefm.calibration import OnlineInterval, apply_width_scale

    context, horizon = cfg["data"]["context"], cfg["data"]["horizon"]
    predictor = Predictor(output, config_directory)
    x, actual, origins = windows(values, bounds["test"], context, horizon)
    if stage == "predict":
        pred = predictor.predict(x)
        per_node, summary = evaluate(actual, pred, point_head=cfg["inference"]["point_head"])
        np.savez_compressed(output/"predictions.npz", predictions=pred, actual=actual,
                            origins=origins, dates=dates[origins], nodes=np.asarray(nodes))
        pd.DataFrame({"node": nodes, **per_node}).to_csv(output/"metrics.csv", index=False)
        (output/"metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary)
        return
    if horizon != 1:
        raise ValueError("The warning entry point supports a one-step forecast.")
    controller = None
    if cfg.get("calibration", {}).get("enabled", False):
        from calibrate import signature
        saved = json.loads((output/"calibration.json").read_text(encoding="utf-8"))
        if (saved["source"] != signature(output) or saved["backbone_sha256"] != predictor.backbone_sha256
                or saved["nodes"] != nodes or saved["data_sha256"] != predictor.info["data_sha256"]):
            raise ValueError("Calibration does not match the warning model.")
        controller = OnlineInterval(saved["parameters"], saved["policy"])
    spatial, uc = predictor.spatial, cfg["uncertainty"]
    normalized = (values-spatial["mean"])/spatial["std"]
    sigma = increment_scale(normalized[:bounds["train"][1]], detrend_window=uc["detrend_window"])
    collected, scales = [], []
    for index, origin in enumerate(origins):
        replicas = perturb((x[index]-spatial["mean"])/spatial["std"], sigma,
                           replicas=uc["replicas"], seed=uc["seed"]+index)
        pred = predictor.predict(replicas*spatial["std"]+spatial["mean"])
        if controller is not None:
            controller.forecast(predictor.predict(x[index:index+1]), origin)
            scale = controller.pending["scale"]
            pred = apply_width_scale(pred, dict(controller.parameters, scale=scale))
        else:
            scale = 1.
        cloud = event_cloud(pred[:, 0, :, 1:], np.arange(1, 10)/10)
        collected.append(warning(cloud, cfg["risk"], cfg["risk"]["seed"]+2*index))
        scales.append(scale)
        if controller is not None:
            controller.observe(actual[index:index+1], origin)
    np.savez_compressed(output/"warning.npz",
                        **{key: np.stack([r[key] for r in collected]) for key in collected[0]},
                        sigma_normalized=sigma, interval_scale=np.asarray(scales),
                        origins=origins, dates=dates[origins], nodes=np.asarray(nodes))
    print(f"Saved {len(origins)} warning updates.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["spatial", "train", "predict", "warning"])
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--split", choices=list(SPLITS), help="Override the chronological split; use a separate output directory.")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--output", type=Path, required=True, help="Run directory with overwrite protection for completed stages.")
    args = ap.parse_args()
    cfg = load_config(args.config, args.split, horizon=args.horizon)
    path = Path(cfg["data"]["csv"])
    if not path.is_absolute():
        path = args.config.resolve().parent/path
    if not path.is_file():
        ap.error(f"Monitoring data have not been supplied: {path}")
    values, dates, nodes = read_monitoring(path, cfg["data"]["date_column"], cfg["data"]["node_prefix"])
    bounds = split_bounds(len(values), cfg["data"]["split"])
    c, h = cfg["data"]["context"], cfg["data"]["horizon"]
    output = args.output.resolve()
    manifest = output/"run.json"
    metadata = dict(protocol_version="spefm-gated-v1" if cfg.get("decoder", {}).get("enabled", False) else "spefm-v1", config=cfg,
                    data_sha256=digest(path), nodes=nodes, split_bounds=bounds,
                    split_dates={name: {"first": dates[a], "last": dates[b-1]}
                                 for name, (a, b) in bounds.items()})
    if args.stage == "spatial":
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            ap.error("Select an empty output directory for the spatial stage.")
        arrays, costs, details = fit_spatial_prior(values, cfg["data"]["split"], cfg["spatial"])
        np.savez_compressed(output/"spatial.npz", **arrays)
        pd.DataFrame(costs, columns=["K", "J", "mean_cost", "std_cost"]).to_csv(output/"elbow.csv", index=False)
        metadata.update(details, medoids=[nodes[i] for i in arrays["medoids"]])
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Elbow K={details['elbow_selected_k']}; configured/used K={details['used_k']}; medoids={metadata['medoids']}")
        return
    if not manifest.is_file():
        ap.error("Run the spatial stage first.")
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    if any(previous.get(key) != json.loads(json.dumps(metadata[key])) for key in metadata):
        ap.error("Data or configuration differ from this run; use a new run directory.")
    stage_file = output/{"train": "decoder.pt" if cfg.get("decoder", {}).get("enabled", False) else "prompt.pt",
                         "predict": "predictions.npz", "warning": "warning.npz"}[args.stage]
    if stage_file.exists():
        ap.error("This stage already has results; use a new run directory.")
    if args.stage == "predict" and h == 1 and cfg.get("calibration", {}).get("enabled", False):
        settings = cfg["calibration"]
        for stage, marker in (("fit", "calibration.json"), ("predict", "predictions.npz")):
            if not (output/marker).exists():
                subprocess.run([sys.executable, "-X", "utf8", "-s", "-u",
                                str(Path(__file__).with_name("calibrate.py")), stage,
                                "--run", str(output), "--output", str(output),
                                "--config-directory", str(args.config.resolve().parent),
                                "--coverage", str(settings["coverage"]),
                                "--learning-rate", str(settings["learning_rate"])], check=True)
        return
    if args.stage == "warning" and h == 1 and cfg.get("calibration", {}).get("enabled", False) and not (output/"calibration.json").exists():
        settings = cfg["calibration"]
        subprocess.run([sys.executable, "-X", "utf8", "-s", "-u",
                        str(Path(__file__).with_name("calibrate.py")), "fit",
                        "--run", str(output), "--output", str(output),
                        "--config-directory", str(args.config.resolve().parent),
                        "--coverage", str(settings["coverage"]),
                        "--learning-rate", str(settings["learning_rate"])], check=True)
    if args.stage in ("predict", "warning"):
        prediction_stage(args.stage, cfg, output, args.config.resolve().parent,
                         values, dates, nodes, bounds)
        return
    try:
        weights_path = resolve_backbone_weights(
            cfg["model"].get("backbone_weights"), config_directory=args.config.resolve().parent)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        ap.error(str(exc))
    print(f"Backbone checkpoint: {weights_path}", flush=True)
    import torch
    from spefm.model import SPEFM, load_backbone
    from spefm.training import fit
    if cfg["training"].get("initialize_after_backend_import", False):
        import importlib
        importlib.import_module("timesfm.timesfm_2p5.timesfm_2p5_torch")
        torch.use_deterministic_algorithms(True)
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = cfg["model"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    spatial = np.load(output/"spatial.npz", allow_pickle=False)
    normalized = (values-spatial["mean"])/spatial["std"]
    model = SPEFM(load_backbone(weights_path), previous["used_k"], cfg["model"]["prompt_length"], cfg["model"]["injection_layers"]).to(device)
    backbone_hash = digest(weights_path)
    prompt_file = output/"prompt.pt"
    if prompt_file.exists():
        saved = torch.load(prompt_file, map_location="cpu", weights_only=True)
        if saved["backbone_sha256"] != backbone_hash or saved["prompts"].shape != model.prompts.shape:
            ap.error("Backbone hash or prompt shape does not match the selected model.")
        with torch.no_grad():
            model.prompts.copy_(saved["prompts"].to(device))
    else:
        train = windows(normalized, bounds["train"], c, h)[:2]
        val = windows(normalized, bounds["val"], c, h)[:2]
        history = fit(model, train, val, spatial["weights"], cfg["training"])
        torch.save({"prompts": model.prompts.detach().cpu(), "backbone_sha256": backbone_hash}, prompt_file)
        pd.DataFrame(history).to_csv(output/"training.csv", index=False)
    if not cfg.get("decoder", {}).get("enabled", False):
        return
    if cfg["inference"]["point_head"] != 5:
        ap.error("The gated decoder requires median point head 5.")
    from spefm.decoder import fit_decoder

    model.eval().requires_grad_(False)
    node_weights = torch.as_tensor(spatial["weights"], dtype=torch.float32, device=device)
    micro_batch = max(1, min(cfg["inference"].get("max_batch_size", 8),
                              cfg["inference"]["node_sequence_budget"]//len(nodes)))

    @torch.inference_mode()
    def native_points(contexts):
        parts = []
        for start in range(0, len(contexts), micro_batch):
            normalized_context = (contexts[start:start+micro_batch]-spatial["mean"])/spatial["std"]
            tensor = torch.as_tensor(normalized_context, dtype=torch.float32, device=device)
            pred = model(tensor, node_weights, h)
            parts.append(pred[..., 5].cpu().numpy()*spatial["std"]+spatial["mean"])
        return np.concatenate(parts)

    train = windows(values[:bounds["train"][1]], bounds["train"], c, h)[:2]
    val = windows(values[:bounds["val"][1]], bounds["val"], c, h)[:2]
    decoder, history = fit_decoder(train, val, native_points(train[0]), native_points(val[0]),
                                  values[:bounds["train"][1]], spatial["weights"], cfg["decoder"], seed)
    staged = output/f".decoder-{uuid4().hex}.pt"
    torch.save(decoder.checkpoint(digest(prompt_file), digest(output/"spatial.npz")), staged)
    pd.DataFrame(history).to_csv(output/"decoder_training.csv", index=False)
    staged.replace(output/"decoder.pt")
    print(f"Decoder validation RMSE={min(row['validation_RMSE'] for row in history):.6f}")


if __name__ == "__main__":
    main()
