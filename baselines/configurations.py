from copy import deepcopy

from .search_space import candidates, planned_trials


def selected_trials(search, selected):
    if search['context_length'] != selected['context_length']:
        raise ValueError('Search and selected configurations have different input lengths.')
    specifications = []
    for split, models in selected['split_comparison']['conditions'].items():
        for model, parameters in models.items():
            specifications.append(('split_comparison', split,
                                   selected['split_comparison']['horizon'], model, parameters))
    extended = selected['horizon_comparison']
    for horizon in extended['horizons']:
        for model, parameters in extended['models'].items():
            specifications.append(('horizon_comparison', extended['split'], horizon, model, parameters))
    allowed = {(row['experiment'], row['split'], row['horizon'], row['model'])
               for row in planned_trials(search)}
    for experiment, split, horizon, model, parameters in specifications:
        spec = search['models'][model]
        if (experiment, split, horizon, model) not in allowed:
            raise ValueError('Selected condition is outside the configured experiments.')
        matching = [(index, row) for index, row in enumerate(candidates(spec, split), 1)
                    if row == {**spec['fixed'], **parameters}]
        if len(matching) != 1:
            raise ValueError(f'Selected configuration for {model} is outside its search space.')
        index, full = matching[0]
        trial = dict(status='fixed_configuration', experiment=experiment,
                     split=split, horizon=horizon, model=model, candidate=index,
                     context_length=selected['context_length'], parameters=full,
                     selection=deepcopy(search['selection']))
        if spec['family'] == 'neural':
            trial['training'] = deepcopy({**search['neural_training'], **spec['training']})
        yield trial


def validate_protocol(search):
    selection = search['selection']
    if selection['subset'] != 'validation' or selection['test_usage'] != 'selected_configuration_only':
        raise ValueError('Selection uses validation data; testing uses the selected configuration.')
    if selection['statistical_metric'] != 'pooled_rmse_mm':
        raise ValueError('Unsupported validation metric.')
    if selection['tie_break'] != 'candidate_order':
        raise ValueError('Unsupported tie-breaking rule.')
    settings = search['neural_training']
    if settings['optimizer'] != 'AdamW' or settings['loss'] != 'MSE' or settings['augmentation'] != 'none':
        raise ValueError('Neural fitting uses AdamW and MSE on the observed training windows.')
    if settings['clip_norm'] <= 0 or settings['evaluation_batch_size'] < 1:
        raise ValueError('Invalid neural training settings.')
    for name, spec in search['models'].items():
        if spec['family'] != 'neural':
            continue
        training = spec['training']
        budgets, keep = training['budgets'], training['keep']
        if not budgets or any(b < 1 for b in budgets) or sorted(set(budgets)) != budgets or len(keep) != len(budgets)-1:
            raise ValueError('Invalid successive-halving budgets.')
        if training['metric'] not in ('RMSE', 'normalized_mse') or training['budget_unit'] not in ('epochs', 'updates'):
            raise ValueError('Unsupported neural selection metric or budget unit.')
        if training['final_selection'] != 'finalist_seed_mean':
            raise ValueError('Unsupported final selection rule.')
        if training['patience_checks'] < 1 or training['min_updates'] < 0:
            raise ValueError('Invalid stopping settings.')
        if training['budget_unit'] == 'updates' and (training['evaluation_interval'] < 1 or any(b % training['evaluation_interval'] for b in budgets)):
            raise ValueError('Update budgets must end at validation checks.')
        if not training['seeds'] or len(set(training['seeds'])) != len(training['seeds']):
            raise ValueError('Specify distinct training seeds.')
        for split in spec['candidates']:
            rows = list(candidates(spec, split))
            if len(rows) < 1 or not 1 <= training['protected_candidate'] <= len(rows):
                raise ValueError('Invalid protected candidate.')
            if any(not 1 <= k <= len(rows) for k in keep) or keep != sorted(keep, reverse=True):
                raise ValueError('Invalid candidate retention counts.')
    fit = search['statistical_fit']
    if fit['fit_error'] != 'raise' or fit['nonconvergence'] not in ('record', 'error') or fit['maxiter'] < 1:
        raise ValueError('Invalid statistical fitting policy.')
    expected = dict(temporal_node_handling='shared_nodewise', normalization='training_node_mean_std',
                    transformer_activation='gelu', transformer_norm_first=True,
                    informer_formulation='official_informer_encoder_decoder',
                    patchtst_formulation='official_supervised_backbone', patchtst_pooling='flatten',
                    a3tgcn_formulation='pytorch_geometric_temporal_A3TGCN2',
                    graph_symmetrization='maximum', graph_self_loops=True,
                    graph_normalization='symmetric_degree_in_gcn')
    if search['implementation'] != expected:
        raise ValueError('Implementation settings differ from the supported baseline definitions.')
