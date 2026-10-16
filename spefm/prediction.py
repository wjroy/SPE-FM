import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .model import SPEFM, load_backbone
from .weights import resolve_backbone_weights


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


class Predictor:
    def __init__(self, directory, config_directory, backbone_weights=None, native_only=False):
        directory = Path(directory)
        self.info = json.loads((directory/"run.json").read_text(encoding="utf-8"))
        self.config = self.info["config"]
        self.horizon = self.config["data"]["horizon"]
        seed = self.config["training"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = self.config["model"]["device"]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = resolve_backbone_weights(backbone_weights or self.config["model"]["backbone_weights"],
                                        config_directory=config_directory)
        self.backbone_sha256 = file_digest(path)
        with np.load(directory/"spatial.npz", allow_pickle=False) as spatial:
            self.spatial = {key: spatial[key].copy() for key in spatial.files}
        self.model = SPEFM(load_backbone(path), self.info["used_k"],
                           self.config["model"]["prompt_length"],
                           self.config["model"]["injection_layers"]).to(self.device)
        saved = torch.load(directory/"prompt.pt", map_location="cpu", weights_only=True)
        if saved["backbone_sha256"] != self.backbone_sha256 or saved["prompts"].shape != self.model.prompts.shape:
            raise ValueError("Model checkpoint mismatch.")
        with torch.no_grad():
            self.model.prompts.copy_(saved["prompts"].to(self.device))
        self.model.eval().requires_grad_(False)
        self.node_weights = torch.as_tensor(self.spatial["weights"], dtype=torch.float32, device=self.device)
        self.decoder = None
        settings = self.config.get("decoder", {})
        if settings.get("enabled", False) and not native_only:
            from .decoder import GatedDecoder
            if self.config["inference"]["point_head"] != 5:
                raise ValueError("The gated decoder requires median point head 5.")
            saved = torch.load(directory/"decoder.pt", map_location="cpu", weights_only=True)
            if (saved["prompt_sha256"] != file_digest(directory/"prompt.pt")
                    or saved["spatial_sha256"] != file_digest(directory/"spatial.npz")):
                raise ValueError("Decoder source checkpoint mismatch.")
            self.decoder = GatedDecoder.restore(saved, self.config["data"]["context"],
                                                 self.horizon, settings).to(self.device)

    def predict(self, contexts):
        raw = np.asarray(contexts, dtype=np.float32)
        shape = (self.config["data"]["context"], len(self.info["nodes"]))
        if raw.ndim != 3 or raw.shape[1:] != shape or not len(raw) or not np.isfinite(raw).all():
            raise ValueError("Invalid monitoring contexts.")
        size = max(1, min(self.config["inference"].get("max_batch_size", 8),
                          self.config["inference"]["node_sequence_budget"]//shape[1]))
        parts = []
        with torch.inference_mode():
            mean = torch.as_tensor(self.spatial["mean"], device=self.device)
            std = torch.as_tensor(self.spatial["std"], device=self.device)
            for start in range(0, len(raw), size):
                context = torch.as_tensor(raw[start:start+size], device=self.device)
                pred = self.model((context-mean)/std, self.node_weights, self.horizon,
                                  correct_crossing=self.config["inference"]["quantile_crossing_correction"])
                pred = pred*std[None, None, :, None]+mean[None, None, :, None]
                if self.decoder is not None:
                    pred = self.decoder(context, pred)
                parts.append(pred.cpu().numpy())
        result = np.concatenate(parts)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite forecasts.")
        return result
