import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from run import load_config
from spefm.data import read_monitoring, split_bounds, windows
from spefm.metrics import evaluate

METHODS = ['SPEFM', 'IO', 'LORA', 'FROZEN']
LABELS = dict(SPEFM='SPE-FM (spatial prompts + gated decoder)', IO='Input- and output-module fine-tuning',
              LORA='LoRA fine-tuning', FROZEN='Frozen pretrained backbone')


def configuration(path, backbone_weights=None, device=None):
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    if settings['methods'] != METHODS or settings['baselines']['calibration_enabled']:
        raise ValueError('Figure 15 requires four methods and uncalibrated baselines.')
    base = (path.resolve().parent / settings['base_config']).resolve()
    config = load_config(base, settings['split'], horizon=settings['horizon'])
    settings.setdefault('seed', config['training']['seed'])
    for section, values in settings.get('SPEFM', {}).items():
        config.setdefault(section, {}).update(deepcopy(values))
    config['data'].update(context=settings['context'], horizon=settings['horizon'])
    config['training']['seed'] = config['spatial']['seed'] = settings['seed']
    for section, name in [('data', 'csv'), ('model', 'backbone_weights')]:
        value = Path(config[section][name]).expanduser()
        config[section][name] = str((base.parent / value).resolve() if not value.is_absolute() else value.resolve())
    if backbone_weights is not None:
        config['model']['backbone_weights'] = str(backbone_weights.resolve())
    if device is not None:
        config['model']['device'] = device
    if (settings['horizon'] != 1 or config['inference']['point_head'] != 5
            or not config['calibration']['enabled'] or not config['decoder']['enabled']):
        raise ValueError('Figure 15 requires one-step gated SPE-FM with median head 5 and calibration enabled.')
    return settings, config


def parameter_counts(directory, method, info):
    if method != 'SPEFM':
        count = json.loads((directory / 'trainable_manifest.json').read_text(encoding='utf-8'))['count']
        return count, 0, 0
    import torch
    from spefm.decoder import GatedDecoder
    prompt = torch.load(directory / 'prompt.pt', map_location='cpu', weights_only=True)['prompts'].numel()
    saved = torch.load(directory / 'decoder.pt', map_location='cpu', weights_only=True)
    cfg = info['config']
    decoder = GatedDecoder.restore(saved, cfg['data']['context'], cfg['data']['horizon'], cfg['decoder'])
    readout = sum(p.numel() for p in decoder.parameters())
    return prompt + readout, prompt, readout


def summarize(runs, output, config):
    rows, source, parameters, reference = [], [], [], None
    values, dates, nodes = read_monitoring(config['data']['csv'], config['data']['date_column'], config['data']['node_prefix'])
    bounds = split_bounds(len(values), config['data']['split'])
    _, expected, origins = windows(values, bounds['test'], config['data']['context'], 1)
    for method in METHODS:
        directory = runs / method
        path = directory / ('predictions.npz' if method == 'SPEFM' else 'native_predictions.npz')
        with np.load(path, allow_pickle=False) as z:
            actual, predictions = z['actual'], z['predictions']
            if (not np.array_equal(actual, expected) or not np.array_equal(z['origins'], origins)
                    or not np.array_equal(z['nodes'], np.asarray(nodes)) or not np.array_equal(z['dates'], dates[origins])):
                raise ValueError(f'Target or origin mismatch: {method}')
            if reference is None:
                reference = predictions[..., 5].shape
            if predictions[..., 5].shape != reference:
                raise ValueError('Prediction shape mismatch.')
            per_node, summary = evaluate(actual.astype(np.float64), predictions.astype(np.float64), point_head=5)
        info = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        total, prompt, decoder = parameter_counts(directory, method, info)
        row = dict(method=method, label=LABELS[method], calibration_enabled=method == 'SPEFM',
                   **summary, trainable_parameters=total)
        rows.append(row)
        for index, node in enumerate(nodes):
            source.append(dict(method=method, node=node, **{name: float(array[index]) for name, array in per_node.items()}))
        active = info['config']
        parameters.append(dict(method=method, trainable_parameters=total, prompt_parameters=prompt,
                               decoder_parameters=decoder, calibration_enabled=method == 'SPEFM',
                               settings=json.dumps(active, separators=(',', ':'))))
    pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
    pd.DataFrame(source).to_csv(output / 'source_data.csv', index=False)
    pd.DataFrame(parameters).to_csv(output / 'parameters.csv', index=False)
    (output / 'results.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n', encoding='utf-8')


def execute(settings, config, settings_path, output):
    runtime = output / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        env[name] = str(settings['threads'])
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    prefix = [sys.executable, '-X', 'utf8', '-s']
    logs = output / 'logs'
    logs.mkdir()
    for stage in ['spatial', 'train', 'predict']:
        with (logs / f'SPEFM_{stage}.log').open('w', encoding='utf-8') as handle:
            subprocess.run(prefix + [str(ROOT / 'run.py'), stage, '--config', str(runtime),
                           '--output', str(output / 'SPEFM')], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    for method in METHODS[1:]:
        with (logs / f'{method}.log').open('w', encoding='utf-8') as handle:
            subprocess.run(prefix + [str(ROOT / 'baselines/adaptation.py'), '--method', method,
                           '--config', str(runtime), '--settings', str(settings_path.resolve()),
                           '--spatial', str(output / 'SPEFM/spatial.npz'), '--output', str(output / method)],
                           cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['run', 'evaluate'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('configs') / 'figure15.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--backbone-weights', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    if args.stage == 'run' and args.runs is not None:
        parser.error('--runs is only valid for evaluate.')
    if args.stage == 'evaluate' and args.runs is None:
        parser.error('evaluate requires --runs.')
    settings, config = configuration(args.config, args.backbone_weights, args.device)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'experiment.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    if args.stage == 'run':
        execute(settings, config, args.config, output)
    summarize(output if args.stage == 'run' else args.runs.resolve(), output, config)


if __name__ == '__main__':
    main()
