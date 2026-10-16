import argparse
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from spefm.data import SPLITS, read_monitoring, split_bounds, windows
from spefm.metrics import evaluate_points
from .search_space import planned_trials
from .configurations import selected_trials, validate_protocol
from .neural_search import run_neural_case
from .statistical import forecast, arima_history_mode


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def environment():
    packages = {}
    for name in ('numpy', 'pandas', 'torch', 'torch-geometric', 'statsmodels', 'scipy', 'PyYAML'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def group_trials(trials):
    groups = OrderedDict()
    for trial in trials:
        key = (trial['model'], trial['split'], trial['horizon'])
        groups.setdefault(key, []).append(trial)
    return groups


def select_candidate(trials, train_values, train_pair, val_pair, fit_settings, device, log=None, *,
                     history_values=None, val_origins=None, arima_history=None):
    best, chosen, chosen_model, chosen_info = float('inf'), None, None, None
    records = []
    for number, trial in enumerate(trials, 1):
        model = None
        predictions, diagnostics = forecast(trial['model'], val_pair[0], trial['horizon'],
                                            trial['parameters'], fit_settings,
                                            history_values=history_values, forecast_origins=val_origins,
                                            arima_history=arima_history)
        score = float(np.sqrt(np.square(predictions - val_pair[1]).mean()))
        info = dict(validation_score=score, diagnostics=diagnostics)
        score = info['validation_score']
        if not np.isfinite(score):
            raise FloatingPointError('A candidate returned a nonfinite validation score.')
        record = dict(candidate=trial['candidate'], parameters=trial['parameters'],
                      validation_score=score, best_epoch=info.get('best_epoch'),
                      epochs_run=info.get('epochs_run'),
                      validation_warning_fits=len(info.get('diagnostics', [])))
        if trial['model'] == 'ARIMA':
            record['arima_history_mode'] = arima_history_mode(arima_history)
        records.append(record)
        if log is not None:
            log(trial, info, record)
        if score < best:
            best, chosen, chosen_model, chosen_info = score, trial, model, info
        print(f"{trial['model']} {trial['split']} H={trial['horizon']} candidate {number}/{len(trials)} validation={score:.8f}", flush=True)
    if chosen is None:
        raise RuntimeError('No candidate was evaluated.')
    return chosen, chosen_model, chosen_info, records


def run_case(trials, values, dates, nodes, search, device, output, *, arima_history=None):
    first = trials[0]
    if 'training' in first:
        return run_neural_case(trials, values, dates, nodes, device, output)
    history_mode = arima_history_mode(arima_history) if first['model'] == 'ARIMA' else None
    bounds = split_bounds(len(values), SPLITS[first['split']])
    train_values = values[slice(*bounds['train'])]
    working = values
    context, horizon = first['context_length'], first['horizon']
    train_pair = windows(working, bounds['train'], context, horizon)[:2]
    val_x, val_y, val_origins = windows(working, bounds['val'], context, horizon)
    val_pair = (val_x, val_y)
    output.mkdir(parents=True, exist_ok=False)

    def log(trial, info, record):
        folder = output / 'validation_trials'
        folder.mkdir(exist_ok=True)
        trial_id = f"candidate_{trial['candidate']:04d}"
        write_json(folder / (trial_id + '.json'), record)
        if 'history' in info:
            pd.DataFrame(info['history']).to_csv(folder / (trial_id + '.csv'), index=False)
        if info.get('diagnostics'):
            write_json(folder / (trial_id + '_diagnostics.json'), info['diagnostics'])

    selected, _, info, records = select_candidate(
        trials, train_values, train_pair, val_pair, search['statistical_fit'], device, log,
        history_values=working, val_origins=val_origins, arima_history=history_mode)
    selected = deepcopy(selected)
    if selected['status'] == 'planned_not_run':
        selected['status'] = 'validation_selected'
    elif history_mode == 'expanding':
        selected['status'] = 'configured_order'
    if history_mode is not None:
        selected['arima_history_mode'] = history_mode
    write_json(output / 'selected.json', dict(configuration=selected,
                                             validation_score=info['validation_score'],
                                             best_epoch=info.get('best_epoch'),
                                             selection_scope='configured_candidates_validation_only',
                                             evaluated_candidates=len(records)))
    test_x, _, origins = windows(working, bounds['test'], context, horizon)
    actual = windows(values, bounds['test'], context, horizon)[1]
    diagnostics = []
    predictions, diagnostics = forecast(selected['model'], test_x, horizon,
                                              selected['parameters'], search['statistical_fit'],
                                              history_values=working, forecast_origins=origins,
                                              arima_history=history_mode)
    per_node, metrics = evaluate_points(actual, predictions)
    target_dates = dates[origins[:, None] + np.arange(horizon)[None, :]]
    np.savez_compressed(output / 'predictions.npz', predictions=predictions, actual=actual,
                        origins=origins, target_dates=target_dates, nodes=np.asarray(nodes))
    pd.DataFrame(dict(node=nodes, **per_node)).to_csv(output / 'metrics.csv', index=False)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'test_diagnostics.json', diagnostics)
    protocol = dict(split_bounds=bounds,
                    split_dates={key: dict(first=dates[a], last=dates[b-1]) for key, (a, b) in bounds.items()},
                    window_counts={key: len(windows(values, value, context, horizon)[0]) for key, value in bounds.items()},
                    metric_aggregation='node_macro_average_over_all_forecast_steps',
                    graph_scope='training_only' if selected['model'] == 'A3TGCN' else None,
                    graph_storage='raw_symmetric_correlation_without_self_loops' if selected['model'] == 'A3TGCN' else None,
                    graph_normalization='GCNConv_self_loops_and_symmetric_degree' if selected['model'] == 'A3TGCN' else None)
    if history_mode is not None:
        protocol['arima'] = dict(history_mode=history_mode, refit_each_origin=True,
                                forecast_origin_definition='absolute index of the first target; excluded from fit',
                                history_scope='available_observations_before_origin' if history_mode == 'expanding' else 'latest_context_observations',
                                validation_history_lengths=[int(t) if history_mode == 'expanding' else context for t in val_origins],
                                test_history_lengths=[int(t) if history_mode == 'expanding' else context for t in origins])
    write_json(output / 'protocol.json', protocol)
    return dict(model=selected['model'], split=selected['split'], horizon=horizon,
                configuration_status=selected['status'], validation_score=info['validation_score'],
                best_epoch=info.get('best_epoch'), evaluated_candidates=len(records),
                output_directory=output.name, test_warning_fits=len(diagnostics),
                arima_history_mode=history_mode, **metrics)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('plan', 'selected', 'search'), default='plan')
    parser.add_argument('--data-config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--search-config', type=Path, default=ROOT / 'baselines/configs/baseline_search.yaml')
    parser.add_argument('--selected-config', type=Path, default=ROOT / 'baselines/configs/baseline_selected.yaml')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--split', choices=list(SPLITS))
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Override neural training seeds; evaluation_seed is unchanged.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    search = yaml.safe_load(args.search_config.read_text(encoding='utf-8'))
    selected = yaml.safe_load(args.selected_config.read_text(encoding='utf-8'))
    data_cfg = yaml.safe_load(args.data_config.read_text(encoding='utf-8'))['data']
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
            parser.error('Training seeds must be distinct integers in [0, 2**32).')
        for spec in search['models'].values():
            if spec['family'] == 'neural':
                spec['training']['seeds'] = args.seeds.copy()
    validate_protocol(search)
    if args.threads < 1 or data_cfg['context'] != search['context_length']:
        parser.error('Use a positive thread count and matching data/search context lengths.')
    if args.models and set(args.models) - search['models'].keys():
        parser.error('A requested model is absent from the search configuration.')
    trials = list(planned_trials(search) if args.mode == 'search' else selected_trials(search, selected))
    trials = [row for row in trials if (not args.models or row['model'] in args.models)
              and (not args.split or row['split'] == args.split)
              and (args.horizon is None or row['horizon'] == args.horizon)]
    if not trials:
        parser.error('No configured experiment matches the requested conditions.')
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Use a new or empty output directory.')
    data_path = Path(data_cfg['csv'])
    if not data_path.is_absolute():
        data_path = args.data_config.resolve().parent / data_path
    if args.mode != 'plan' and not data_path.is_file():
        parser.error('Monitoring data file is required for execution.')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode != 'plan' and device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    torch.set_num_threads(args.threads)
    output.mkdir(parents=True, exist_ok=True)
    files = [ROOT/'run_baselines.py', args.data_config, args.search_config, args.selected_config,
             *sorted((ROOT/'baselines').rglob('*.py')),
             *sorted((ROOT/'baselines/vendor').rglob('UPSTREAM.json')),
             *sorted((ROOT/'licenses').glob('*.txt')), ROOT/'spefm/data.py',
             ROOT/'spefm/metrics.py']
    metadata = dict(mode=args.mode, created_utc=datetime.now(timezone.utc).isoformat(),
                    arima_history_mode=arima_history_mode() if any(t['model'] == 'ARIMA' for t in trials) else None,
                    environment=environment(), device=device, threads=args.threads,
                    data_sha256=sha256(data_path) if data_path.is_file() else None,
                    file_sha256={str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name: sha256(path)
                                 for path in files}, search_configuration=search,
                    data_configuration=data_cfg, trial_count=len(trials),
                    condition_count=len(group_trials(trials)),
                    effective_conditions=[dict(model=model,split=split,horizon=horizon)
                                          for model,split,horizon in group_trials(trials)])
    write_json(output/'manifest.json', metadata)
    write_json(output/'configurations.json', trials)
    if args.mode == 'plan':
        print(f'Exported {len(trials)} fixed configurations; execution=not_run.')
        return
    try:
        values, dates, nodes = read_monitoring(data_path, data_cfg['date_column'], data_cfg['node_prefix'])
        summary = []
        for (model, split, horizon), group in group_trials(trials).items():
            label = model.replace(' ', '_') + '_train' + split.split('/')[0] + f'_h{horizon}'
            row = run_case(group, values, dates, nodes, search, device, output/label)
            summary.append(row)
            pd.DataFrame(summary).to_csv(output/'summary.csv', index=False)
        write_json(output/'completion.json', dict(status='completed', conditions=len(summary)))
    except Exception as exc:
        write_json(output/'failure.json', dict(status='failed', type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
