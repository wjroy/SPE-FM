from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch

from spefm.data import SPLITS, normalization, split_bounds, windows
from spefm.metrics import evaluate_points
from .neural import build_model, evaluate_model, fit_rung, seed_everything


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def promoted_candidates(records, keep, protected):
    ranked = sorted(records, key=lambda row: (row['validation_score'], row['candidate']))
    selected = [row['candidate'] for row in ranked[:keep]]
    if protected not in selected:
        selected[-1] = protected
    return selected


def run_neural_case(trials, values, dates, nodes, device, output):
    first = trials[0]
    settings = first['training']
    context, horizon = first['context_length'], first['horizon']
    bounds = split_bounds(len(values), SPLITS[first['split']])
    training = values[slice(*bounds['train'])]
    mean, std = normalization(training)
    working = (values[:bounds['val'][1]]-mean)/std
    data = dict(training=training, mean=mean, std=std,
                train=windows(working, bounds['train'], context, horizon)[:2],
                val=windows(working, bounds['val'], context, horizon)[:2],
                val_actual=windows(values, bounds['val'], context, horizon)[1])
    output.mkdir(parents=True, exist_ok=False)
    indexed = {trial['candidate']: trial for trial in trials}
    records = {}

    def train(candidate, seed, budget):
        key = (candidate, seed, budget)
        if key not in records:
            folder = output/f'candidate_{candidate:03d}'/f'seed{seed}'
            result = fit_rung(indexed[candidate], data, seed, budget, folder, device)
            pd.DataFrame(result.pop('history')).to_csv(folder/f'history_{budget}.csv', index=False)
            write_json(folder/f'budget_{budget}.json', result)
            records[key] = result
            print(f"{first['model']} {first['split']} candidate={candidate} seed={seed} budget={budget} validation={result['validation_score']:.8f}", flush=True)
        return records[key]

    active = list(indexed)
    searching = first['status'] == 'planned_not_run'
    if searching:
        for stage, budget in enumerate(settings['budgets']):
            rows = [train(candidate, settings['seeds'][0], budget) for candidate in active]
            if stage < len(settings['keep']):
                active = promoted_candidates(rows, settings['keep'][stage], settings['protected_candidate'])
            write_json(output/f'stage_{stage+1}.json', dict(budget=budget, evaluated=rows, promoted=active))
    final_budget = settings['budgets'][-1]
    finalists = []
    for candidate in active:
        runs = [train(candidate, seed, final_budget) for seed in settings['seeds']]
        finalists.append(dict(candidate=candidate, validation_score=float(np.mean([row['validation_score'] for row in runs])), runs=runs))
    winner = min(finalists, key=lambda row: (row['validation_score'], row['candidate']))
    candidate, chosen_runs = winner['candidate'], winner['runs']
    write_json(output/'finalists.json', finalists)
    selected = deepcopy(indexed[candidate])
    selected['status'] = 'validation_selected' if searching else 'fixed_configuration'
    write_json(output/'selected.json', dict(configuration=selected, validation_score=winner['validation_score'],
                                           seeds=settings['seeds'], runs=chosen_runs, evaluated_candidates=len(trials)))
    np.savez_compressed(output/'normalization.npz', mean=mean, std=std)
    x, _, origins = windows((values-mean)/std, bounds['test'], context, horizon)
    actual = windows(values, bounds['test'], context, horizon)[1]
    seed_metrics = []
    for record in chosen_runs:
        seed_everything(record['seed'])
        model = build_model(first['model'], selected['parameters'], context, horizon, training).to(device)
        folder = output/f'candidate_{candidate:03d}'/f"seed{record['seed']}"
        state = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state['state_dict'], strict=True)
        metrics, predictions = evaluate_model(model, x, actual, mean, std, settings, device)
        target = actual.astype(np.float64) if settings['metric'] == 'RMSE' else actual
        per_node = evaluate_points(target, predictions)[0]
        seed_dir = output/f"seed{record['seed']}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir/'predictions.npz', actual=actual, predictions=predictions,
                            origins=origins, target_dates=dates[origins[:,None]+np.arange(horizon)[None,:]], nodes=np.asarray(nodes))
        pd.DataFrame(dict(node=nodes, **per_node)).to_csv(seed_dir/'metrics.csv', index=False)
        write_json(seed_dir/'metrics.json', metrics)
        seed_metrics.append(dict(seed=record['seed'], **metrics))
        if first['model'] == 'A3TGCN' and len(seed_metrics) == 1:
            np.save(output/'adjacency.npy', model.adjacency.cpu().numpy())
    summary = {}
    for metric in ('RMSE', 'MAE'):
        numbers = [row[metric] for row in seed_metrics]
        summary[metric] = float(np.mean(numbers))
        summary[metric+'_sd'] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
    write_json(output/'metrics.json', dict(**summary, n_seeds=len(seed_metrics), seeds=seed_metrics,
                                          aggregation='arithmetic_mean_of_seed_metrics'))
    write_json(output/'protocol.json', dict(split_bounds=bounds, context=context, horizon=horizon,
                                            training=settings, normalization='training_node_mean_std',
                                            metric_aggregation='node_macro_average_then_seed_mean',
                                            graph_scope='training_only' if first['model']=='A3TGCN' else None))
    return dict(model=first['model'], split=first['split'], horizon=horizon, candidate=candidate,
                configuration_status=selected['status'], validation_score=winner['validation_score'],
                n_seeds=len(seed_metrics), evaluated_candidates=len(trials), output_directory=output.name, **summary)
