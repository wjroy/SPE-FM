import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from spefm.data import read_monitoring, split_bounds, windows
from spefm.model import SPEFM, load_backbone
from spefm.training import forecast_loss


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)


class NoPrompt(SPEFM):
    def __init__(self, backbone, prototypes, settings):
        super().__init__(backbone, prototypes, settings['prompt_length'], settings['injection_layers'])
        self.prompts = nn.Parameter(torch.empty(prototypes, 0, backbone.md), requires_grad=False)
        self.injection_layers = frozenset()


def build(method, config, settings, prototypes):
    random.seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    torch.manual_seed(config['training']['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = NoPrompt(load_backbone(Path(config['model']['backbone_weights'])), prototypes, config['model'])
    if method == 'IO':
        model.backbone.tokenizer.requires_grad_(True)
        model.backbone.output_projection_point.requires_grad_(True)
    elif method == 'LORA':
        lora = settings['lora']
        if lora['dropout'] != 0:
            raise ValueError('Supported LoRA dropout: 0.')
        for layer in model.backbone.stacked_xf:
            layer.ff0 = LoRALinear(layer.ff0, lora['rank'], lora['alpha'])
            layer.ff1 = LoRALinear(layer.ff1, lora['rank'], lora['alpha'])
    elif method != 'FROZEN':
        raise ValueError(method)
    device = config['model']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval()


def fit(model, pairs, weights, settings, seed, output):
    if settings['loss'] != 'mean_pinball':
        raise ValueError('Supported loss: mean_pinball.')
    selected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(selected.values(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    train, val = pairs['train'][:2], pairs['val'][:2]
    size = settings['batch_size']
    best, stale, history = float('inf'), 0, []
    for epoch in range(settings['epochs']):
        model.train()
        training = []
        for ids in np.array_split(rng.permutation(len(train[0])), math.ceil(len(train[0]) / size)):
            x, y = (torch.as_tensor(a[ids], device=device) for a in train)
            optimizer.zero_grad(set_to_none=True)
            loss = forecast_loss(model(x, weights, y.shape[1]), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            optimizer.step()
            training.append((float(loss.detach()), len(ids)))
        model.eval()
        validation = []
        with torch.inference_mode():
            for start in range(0, len(val[0]), size):
                x, y = (torch.as_tensor(a[start:start + size], device=device) for a in val)
                validation.append((float(forecast_loss(model(x, weights, y.shape[1]), y)), len(x)))
        train_loss = sum(a * n for a, n in training) / sum(n for _, n in training)
        val_loss = sum(a * n for a, n in validation) / sum(n for _, n in validation)
        if not np.isfinite(val_loss):
            raise FloatingPointError('Nonfinite validation loss.')
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss))
        if val_loss < best:
            best, stale = val_loss, 0
            saved = dict(seed=seed, epoch=epoch + 1, val_loss=val_loss,
                         trainable={name: p.detach().cpu().clone() for name, p in selected.items()})
            staged = output / 'best.tmp'
            torch.save(saved, staged)
            staged.replace(output / 'best.pt')
        else:
            stale += 1
        pd.DataFrame(history).to_csv(output / 'training.csv', index=False)
        if stale >= settings['patience']:
            break
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        for name, parameter in selected.items():
            parameter.copy_(saved['trainable'][name].to(device))
    return dict(epochs=len(history), best_epoch=saved['epoch'], best_val_loss=best)


@torch.inference_mode()
def predict(model, contexts, weights, spatial, batch_size, horizon):
    device = next(model.parameters()).device
    mean = torch.as_tensor(spatial['mean'], device=device)
    std = torch.as_tensor(spatial['std'], device=device)
    outputs = []
    model.eval()
    for start in range(0, len(contexts), batch_size):
        raw = torch.as_tensor(contexts[start:start + batch_size], device=device)
        q = model((raw - mean) / std, weights, horizon, correct_crossing=True)
        outputs.append((q * std[None, None, :, None] + mean[None, None, :, None]).cpu().numpy())
    return np.concatenate(outputs)


def run(method, config, settings, spatial_path, output):
    if settings['calibration_enabled']:
        raise ValueError('Baseline calibration must be disabled.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    values, dates, nodes = read_monitoring(config['data']['csv'], config['data']['date_column'], config['data']['node_prefix'])
    bounds = split_bounds(len(values), config['data']['split'])
    with np.load(spatial_path, allow_pickle=False) as z:
        spatial = {key: z[key].copy() for key in z.files}
    if not np.array_equal(values[:bounds['train'][1]].mean(0).astype(np.float32), spatial['mean']):
        raise ValueError('Spatial normalization does not match the training prefix.')
    model = build(method, config, settings, spatial['weights'].shape[1])
    device = next(model.parameters()).device
    weights = torch.as_tensor(spatial['weights'], device=device)
    normalized = (values - spatial['mean']) / spatial['std']
    context, horizon = config['data']['context'], config['data']['horizon']
    pairs = {name: windows(normalized, bound, context, horizon) for name, bound in bounds.items()}
    result = fit(model, pairs, weights, settings, config['training']['seed'], output) if method != 'FROZEN' else dict(epochs=0, best_epoch=None)
    parameters = {name: dict(shape=list(p.shape), numel=p.numel()) for name, p in model.named_parameters() if p.requires_grad}
    count = sum(p['numel'] for p in parameters.values())
    for subset, filename in [('val', 'validation.npz'), ('test', 'native_predictions.npz')]:
        x, actual, origins = windows(values, bounds[subset], context, horizon)
        predictions = predict(model, x, weights, spatial, settings['batch_size'], horizon)
        np.savez_compressed(output / filename, predictions=predictions, actual=actual, origins=origins,
                            dates=dates[origins], nodes=np.asarray(nodes))
    effective = dict(data=config['data'], model=dict(config['model'], prompt_length=0, injection_layers=[]),
                     decoder=dict(enabled=False), calibration=dict(enabled=False),
                     training=dict(settings, seed=config['training']['seed']) if method != 'FROZEN' else dict(enabled=False, epochs=0))
    if method == 'LORA':
        effective['lora'] = settings['lora']
    for filename, value in [('run.json', dict(method=method, config=effective, trainable_parameters=count)),
                            ('trainable_manifest.json', dict(count=count, parameters=parameters)),
                            ('training_result.json', dict(result, trainable_parameters=count))]:
        (output / filename).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['IO', 'LORA', 'FROZEN'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--spatial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.settings.read_text(encoding='utf-8'))
    torch.set_num_threads(settings['threads'])
    run(args.method, yaml.safe_load(args.config.read_text(encoding='utf-8')), settings['baselines'], args.spatial, args.output)


if __name__ == '__main__':
    main()
