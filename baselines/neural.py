import random
from pathlib import Path

import numpy as np
import torch

from .graph import A3TGCN, correlation_graph
from .temporal import AttBiLSTM, TemporalTransformer, InformerBaseline, PatchTSTBaseline
from spefm.metrics import evaluate_points


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_model(name, parameters, context, horizon, train_values):
    p = parameters
    if name == 'Att-BiLSTM':
        if p['attention'] != 'additive_temporal':
            raise ValueError('Att-BiLSTM uses additive temporal attention.')
        model = AttBiLSTM(horizon, p['hidden_size'], p['layers'], p['dropout'])
    elif name == 'Informer':
        model = InformerBaseline(context, horizon, p)
    elif name == 'PatchTST':
        model = PatchTSTBaseline(context, horizon, train_values.shape[1], p)
    elif name == 'Transformer':
        if p['pooling'] != 'additive':
            raise ValueError('Transformer uses additive pooling.')
        model = TemporalTransformer(context, horizon, p['d_model'], p['layers'], p['nhead'],
                                   p['dropout'], p['d_ff'], name)
    elif name == 'A3TGCN':
        if (p['graph'], p['graph_source'], p['recurrence'], p['attention']) != (
                'absolute_correlation_knn', 'training_only', 'TGCN', 'temporal'):
            raise ValueError('Unsupported graph recurrent formulation.')
        model = A3TGCN(context, horizon, p['hidden_size'], correlation_graph(train_values, p['graph_k']), p['batch_size'])
        model.core.improved = model.core._base_tgcn.improved = p['improved']
        for key in ('conv_z', 'conv_r', 'conv_h'):
            getattr(model.core._base_tgcn, key).improved = p['improved']
    else:
        raise ValueError(f'Unknown neural model: {name}')
    initialization = p.get('initialization', 'default')
    with torch.no_grad():
        if initialization == 'forget_bias_one' and name == 'Att-BiLSTM':
            for key, parameter in model.rnn.named_parameters():
                if key.startswith('bias_'):
                    parameter[model.rnn.hidden_size:2*model.rnn.hidden_size].fill_(1. if key.startswith('bias_ih') else 0.)
        elif initialization == 'independent_encoder' and name == 'Transformer':
            for layer in model.encoder.layers:
                for key, parameter in layer.named_parameters():
                    if parameter.ndim > 1:
                        torch.nn.init.xavier_uniform_(parameter)
                    elif 'norm' in key and key.endswith('weight'):
                        parameter.fill_(1.)
                    else:
                        parameter.zero_()
        elif initialization != 'default':
            raise ValueError('Unsupported model initialization.')
    return model


def evaluate_loss(model, pair, batch_size, device, seed):
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    model.eval()
    total, count = 0., 0
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(seed)
        for start in range(0, len(pair[0]), batch_size):
            x, y = [torch.as_tensor(a[start:start+batch_size], dtype=torch.float32, device=device)
                    for a in pair]
            error = model(x) - y
            total += float(error.square().sum())
            count += error.numel()
    return total / count


def state_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def evaluate_model(model, x, actual, mean, std, settings, device):
    device = torch.device(device)
    devices = [device.index or 0] if device.type == 'cuda' else []
    was_training = model.training
    model.eval()
    outputs = []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(settings['evaluation_seed'])
        for start in range(0, len(x), settings['evaluation_batch_size']):
            batch = torch.as_tensor(x[start:start+settings['evaluation_batch_size']], dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    z = np.concatenate(outputs)
    if settings['metric'] == 'RMSE':
        predictions = z.astype(np.float64)*std.astype(np.float64)+mean.astype(np.float64)
        actual = actual.astype(np.float64)
    else:
        predictions = z*std+mean
    metrics = evaluate_points(actual, predictions)[1]
    model.train(was_training)
    return metrics, predictions


def fit_rung(trial, data, seed, budget, folder, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    p, settings = trial['parameters'], trial['training']
    seed_everything(seed)
    model = build_model(trial['model'], p, trial['context_length'], trial['horizon'], data['training']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **settings['scheduler'])
    rng = np.random.default_rng(seed)
    history, order = [], np.array([], dtype=int)
    cursor = epoch = updates = examples = stale = count = 0
    total, best = 0., float('inf')
    best_state, best_row, stopped = None, None, False
    graph = settings['budget_unit'] == 'updates'
    state_path = folder/'continuation.pt'
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        if state['trial'] != trial or state['seed'] != seed:
            raise ValueError('Continuation settings differ from the current trial.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['cpu_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        history, order = state['history'], np.asarray(state['order'], dtype=int)
        cursor, epoch, updates, examples, stale, count = [state[key] for key in ('cursor','epoch','updates','examples','stale','count')]
        total, best, best_state, best_row, stopped = [state[key] for key in ('total','best','best_state','best_row','stopped')]
    while not stopped and (updates < budget if graph else epoch < budget or cursor < len(order)):
        if cursor >= len(order):
            epoch += 1
            order = rng.permutation(len(data['train'][0]))
            cursor = 0
        model.train()
        ids = order[cursor:cursor+p['batch_size']]
        cursor += len(ids)
        x, y = [torch.as_tensor(values[ids], dtype=torch.float32, device=device) for values in data['train']]
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        if output.shape != y.shape:
            raise ValueError('Model output differs from the forecast target shape.')
        loss = (output-y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        examples += len(ids)
        total += float(loss.detach())*y.numel()
        count += y.numel()
        check = updates % settings['evaluation_interval'] == 0 if graph else cursor == len(order)
        if check:
            if settings['metric'] == 'normalized_mse':
                score = evaluate_loss(model, data['val'], settings['evaluation_batch_size'], device, settings['evaluation_seed'])
            else:
                metrics, _ = evaluate_model(model, data['val'][0], data['val_actual'], data['mean'], data['std'], settings, device)
                score = metrics['RMSE']
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite validation score.')
            row = dict(epoch=epoch, updates=updates, examples=examples, training_mse=total/count,
                       learning_rate=optimizer.param_groups[0]['lr'], validation_score=score)
            history.append(row)
            if score < best:
                best, stale, best_row, best_state = score, 0, row.copy(), state_cpu(model)
            else:
                stale += 1
            scheduler.step(score)
            stopped = stale >= settings['patience_checks'] and updates >= settings['min_updates']
            total, count = 0., 0
    if best_state is None:
        raise RuntimeError('Training did not produce a finite validation checkpoint.')
    state = dict(trial=trial, seed=seed, model=state_cpu(model), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), numpy_rng=rng.bit_generator.state, cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                 history=history, order=order.tolist(), cursor=cursor, epoch=epoch, updates=updates,
                 examples=examples, stale=stale, total=total, count=count, best=best,
                 best_row=best_row, best_state=best_state, stopped=stopped)
    torch.save(state, state_path)
    torch.save(dict(state_dict=best_state, trial=trial, seed=seed, best=best_row), folder/'best.pt')
    return dict(candidate=trial['candidate'], seed=seed, budget=budget, validation_score=best,
                best_epoch=best_row['epoch'], best_updates=best_row['updates'],
                epochs_run=epoch, updates=updates, stopped=stopped, history=history)


def predict(model, contexts, batch_size, device, seed):
    model.to(device).eval()
    seed_everything(seed)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            x = torch.as_tensor(contexts[start:start+batch_size], dtype=torch.float32, device=device)
            outputs.append(model(x).cpu().numpy())
    return np.concatenate(outputs)
