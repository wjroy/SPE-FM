import warnings

import numpy as np


ARIMA_HISTORY_MODE = 'window'


class StatisticalFitError(RuntimeError):
    pass


def arima_history_mode(mode=None):
    mode = ARIMA_HISTORY_MODE if mode is None else mode
    if mode not in ('window', 'expanding'):
        raise ValueError('ARIMA history mode must be window or expanding.')
    return mode


def forecast(name, contexts, horizon, parameters, fit_settings, *,
             history_values=None, forecast_origins=None, arima_history=None):
    x = np.asarray(contexts, dtype=float)
    if x.ndim != 3 or not np.isfinite(x).all() or min(x.shape) < 1 or horizon < 1:
        raise ValueError('Expected finite contexts [origins,time,nodes] and a positive horizon.')
    if x.shape[1] != parameters['input_length']:
        raise ValueError('Statistical input length differs from the configured length.')
    if name == 'Persistence':
        return np.repeat(x[:, -1:, :], horizon, axis=1), []
    if name == 'Moving average':
        window = parameters['window']
        if not 1 <= window <= x.shape[1]:
            raise ValueError('Moving-average window exceeds the available context.')
        return np.repeat(x[:, -window:, :].mean(axis=1, keepdims=True), horizon, axis=1), []
    if name != 'ARIMA':
        raise ValueError(f'Unknown statistical model: {name}')
    from statsmodels.tsa.arima.model import ARIMA

    mode = arima_history_mode(arima_history)
    history, cutoffs = None, None
    if mode == 'expanding':
        if history_values is None or forecast_origins is None:
            raise ValueError('Expanding ARIMA requires history_values and absolute forecast_origins.')
        history = np.asarray(history_values, dtype=float)
        cutoffs = np.asarray(forecast_origins)
        if history.ndim != 2 or history.shape[1] != x.shape[2]:
            raise ValueError('History must have shape [observations,nodes] matching the contexts.')
        if cutoffs.shape != (len(x),) or not np.issubdtype(cutoffs.dtype, np.integer):
            raise ValueError('Forecast origins must be one absolute integer cutoff per context.')
        if np.any(cutoffs < x.shape[1]) or np.any(cutoffs > len(history)):
            raise ValueError('Forecast origins are outside the available history.')
        if not np.isfinite(history[:int(cutoffs.max())]).all():
            raise ValueError('Observed history must be finite.')
        for index, cutoff in enumerate(cutoffs):
            if not np.array_equal(history[cutoff-x.shape[1]:cutoff], x[index]):
                raise ValueError('Context does not match the history immediately before its forecast origin.')

    result = np.empty((len(x), horizon, x.shape[2]))
    diagnostics = []
    for origin in range(len(x)):
        observations = x[origin] if mode == 'window' else history[:cutoffs[origin]]
        for node in range(x.shape[2]):
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    fitted = ARIMA(observations[:, node], order=tuple(parameters['order']),
                                   trend=parameters['trend'],
                                   enforce_stationarity=parameters['enforce_stationarity'],
                                   enforce_invertibility=parameters['enforce_invertibility']).fit(
                                       method='statespace', method_kwargs={
                                           'maxiter': fit_settings['maxiter'], 'disp': 0})
                    predicted = np.asarray(fitted.forecast(horizon), dtype=float)
                converged = bool(fitted.mle_retvals.get('converged', True))
                if predicted.shape != (horizon,) or not np.isfinite(predicted).all():
                    raise ValueError('Nonfinite or incorrectly shaped forecast.')
                if not converged and fit_settings['nonconvergence'] == 'error':
                    raise ValueError('ARIMA optimization did not converge.')
                if captured or not converged:
                    diagnostics.append({'origin_index': origin, 'node_index': node,
                                        'history_mode': mode, 'history_length': len(observations),
                                        'forecast_origin': int(cutoffs[origin]) if cutoffs is not None else None,
                                        'converged': converged,
                                        'warnings': [str(item.message) for item in captured]})
                result[origin, :, node] = predicted
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                raise StatisticalFitError(f'ARIMA fit failed at origin {origin}, node {node}: {exc}') from exc
    return result, diagnostics
