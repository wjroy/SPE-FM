from copy import deepcopy

import numpy as np
import torch
from torch import nn


class GatedReadout(nn.Module):
    def __init__(self, dimension, width, horizon, zero_feature, gate_lower, gate_upper):
        super().__init__()
        if not 0 < gate_lower < gate_upper <= 1:
            raise ValueError("Invalid gate bounds.")
        self.gate_lower, self.gate_upper = gate_lower, gate_upper
        self.body = nn.Sequential(nn.Linear(dimension, width), nn.Tanh(),
                                  nn.Linear(width, width), nn.Tanh())
        self.delta = nn.Linear(width, horizon, bias=False)
        self.gate = nn.Linear(width, horizon)
        self.register_buffer("zero_feature", torch.as_tensor(zero_feature, dtype=torch.float32))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, features, base, last, scale):
        hidden = self.body(features)
        delta = self.delta(hidden)-self.delta(self.body(self.zero_feature))
        gate = self.gate_lower+(self.gate_upper-self.gate_lower)*torch.sigmoid(self.gate(hidden))
        direct = last[..., None]+scale[None, :, None]*delta
        return gate*base+(1-gate)*direct


def fit_operators(training, weights, neighbors):
    training = np.asarray(training, dtype=np.float64)
    if training.ndim != 2 or len(training) < 3 or not np.isfinite(training).all():
        raise ValueError("Invalid decoder training prefix.")
    increments = np.diff(training, axis=0)
    scale = np.maximum(increments.std(0), .01).astype(np.float32)
    nodes = training.shape[1]
    if nodes == 1:
        graph = np.ones((1, 1), dtype=np.float32)
    else:
        centered = increments-increments.mean(0)
        norms = np.linalg.norm(centered, axis=0)
        correlation = centered.T@centered / np.maximum(norms[:, None]*norms[None, :], 1e-30)
        np.fill_diagonal(correlation, -np.inf)
        count = min(int(neighbors), nodes-1)
        if count < 1:
            raise ValueError("Neighbor count must be positive.")
        indices = np.argsort(correlation, axis=1)[:, -count:]
        selected = np.take_along_axis(correlation, indices, axis=1)
        probability = np.exp(selected-selected.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        graph = np.zeros((nodes, nodes))
        np.put_along_axis(graph, indices, probability, axis=1)
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.ndim != 2 or weights.shape[0] != nodes or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(weights.sum(1) <= 0)):
        raise ValueError("Invalid spatial associations.")
    weights = weights/weights.sum(1, keepdims=True)
    prototype = weights@(weights/np.maximum(weights.sum(0, keepdims=True), 1e-30)).T
    return dict(scale=scale, graph=graph.astype(np.float32), prototype=prototype.astype(np.float32))


def features(contexts, operators):
    own = (np.diff(np.asarray(contexts, dtype=np.float32), axis=1)
           /operators["scale"][None, None, :]).transpose(0, 2, 1)[:, :, ::-1].copy()
    neighbor = np.einsum("ij,bjl->bil", operators["graph"], own)
    prototype = np.einsum("ij,bjl->bil", operators["prototype"], own)
    return np.concatenate((own, neighbor-own, prototype-own), axis=-1)


class GatedDecoder(nn.Module):
    def __init__(self, context, horizon, settings, operators, feature_mean, feature_std):
        super().__init__()
        if context < 2 or horizon < 1:
            raise ValueError("Decoder requires at least two observations and one lead.")
        self.context, self.horizon = int(context), int(horizon)
        self.settings = dict(settings)
        for name, value in operators.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        self.readout = GatedReadout(len(feature_mean), settings["width"], horizon,
                                   -self.feature_mean/self.feature_std,
                                   settings["gate_lower"], settings["gate_upper"])

    def forward(self, contexts, quantiles):
        expected = (len(contexts), self.horizon, len(self.scale), 10)
        if (contexts.ndim != 3 or contexts.shape[1:] != (self.context, len(self.scale))
                or quantiles.shape != expected):
            raise ValueError("Decoder context or output dimensions do not match.")
        own = ((contexts[:, 1:]-contexts[:, :-1])/self.scale).permute(0, 2, 1).flip(-1)
        neighbor = torch.einsum("ij,bjl->bil", self.graph, own)
        prototype = torch.einsum("ij,bjl->bil", self.prototype, own)
        value = torch.cat((own, neighbor-own, prototype-own), dim=-1)
        value = (value-self.feature_mean)/self.feature_std
        point = self.readout(value, quantiles[..., 5].permute(0, 2, 1),
                             contexts[:, -1], self.scale).permute(0, 2, 1)
        return point[..., None]+(quantiles-quantiles[..., 5:6])

    def checkpoint(self, prompt_sha256, spatial_sha256):
        return dict(version="gated_decoder_v1", context=self.context, horizon=self.horizon,
                    settings=self.settings,
                    state_dict={name: value.detach().cpu().clone() for name, value in self.state_dict().items()},
                    prompt_sha256=prompt_sha256, spatial_sha256=spatial_sha256)

    @classmethod
    def restore(cls, saved, context, horizon, settings):
        if (saved["version"] != "gated_decoder_v1" or saved["context"] != context
                or saved["horizon"] != horizon or saved["settings"] != settings):
            raise ValueError("Decoder configuration does not match checkpoint.")
        state = saved["state_dict"]
        operators = {name: state[name] for name in ("scale", "graph", "prototype")}
        result = cls(context, horizon, settings, operators, state["feature_mean"], state["feature_std"])
        result.load_state_dict(state, strict=True)
        return result.eval()


def fit_decoder(train_pair, validation_pair, train_base, validation_base,
                training_prefix, weights, settings, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y = train_pair
    vx, vy = validation_pair
    operators = fit_operators(training_prefix, weights, settings["neighbors"])
    tx = features(x, operators)
    mean = tx.mean((0, 1))
    std = np.maximum(tx.std((0, 1)), .1)
    decoder = GatedDecoder(x.shape[1], y.shape[1], settings, operators, mean, std)
    head = decoder.readout
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])
    inputs = torch.from_numpy(((tx-mean)/std).astype(np.float32))
    base = torch.as_tensor(train_base.transpose(0, 2, 1), dtype=torch.float32)
    last = torch.as_tensor(x[:, -1], dtype=torch.float32)
    targets = torch.as_tensor(y.transpose(0, 2, 1), dtype=torch.float32)
    val_inputs = torch.from_numpy(((features(vx, operators)-mean)/std).astype(np.float32))
    val_base = torch.as_tensor(validation_base.transpose(0, 2, 1), dtype=torch.float32)
    val_last = torch.as_tensor(vx[:, -1], dtype=torch.float32)
    size = int(settings["batch_size"])
    if size < 1 or settings["epochs"] < 1 or settings["patience"] < 1:
        raise ValueError("Invalid decoder training settings.")
    history, best_state, best, stale = [], None, float("inf"), 0
    for epoch in range(settings["epochs"]+1):
        losses = []
        if epoch:
            head.train()
            for indices in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/size))):
                optimizer.zero_grad(set_to_none=True)
                point = head(inputs[indices], base[indices], last[indices], decoder.scale)
                loss = (point-targets[indices]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite decoder training loss.")
                loss.backward()
                optimizer.step()
                losses.append((float(loss.detach()), len(indices)))
        head.eval()
        with torch.inference_mode():
            points = [head(val_inputs[i:i+size], val_base[i:i+size], val_last[i:i+size], decoder.scale)
                      for i in range(0, len(vx), size)]
            prediction = torch.cat(points).numpy().transpose(0, 2, 1)
        error = prediction.astype(np.float64)-vy.astype(np.float64)
        metric = float(np.sqrt(np.square(error).mean((0, 1))).mean())
        if not np.isfinite(metric):
            raise FloatingPointError("Nonfinite decoder validation metric.")
        row = dict(epoch=epoch, validation_RMSE=metric)
        if losses:
            row["train_MSE"] = sum(value*n for value, n in losses)/sum(n for _, n in losses)
        history.append(row)
        if metric < best:
            best, stale, best_state = metric, 0, deepcopy(head.state_dict())
        elif epoch:
            stale += 1
        if stale >= settings["patience"]:
            break
    head.load_state_dict(best_state)
    return decoder.eval(), history
