import numpy as np
import pandas as pd


def increment_scale(train_values, *, detrend_window=8):
    increments = np.diff(np.asarray(train_values, dtype=float), axis=0)
    if increments.ndim != 2 or len(increments) < 2 or not np.isfinite(increments).all():
        raise ValueError("Provide at least three finite training observations per node.")
    if not isinstance(detrend_window, int) or detrend_window < 1:
        raise ValueError("The trailing median-filter width must be a positive integer.")
    trend = pd.DataFrame(increments).rolling(detrend_window, min_periods=1).median().to_numpy()
    increments = increments-trend
    median = np.median(increments, axis=0)
    sigma = 1.4826*np.median(np.abs(increments-median), axis=0)
    return sigma.astype(np.float32)


def perturb(context, sigma, *, replicas, seed):
    x = np.asarray(context, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2 or sigma.shape != (x.shape[1],):
        raise ValueError("Expected context [observations,nodes] and sigma [nodes].")
    if replicas < 1 or np.any(sigma < 0) or not np.isfinite(x).all() or not np.isfinite(sigma).all():
        raise ValueError("Invalid perturbation parameters.")
    noise = np.random.default_rng(seed).normal(size=(replicas, len(x)-1, x.shape[1])).astype(np.float32)
    increments = np.diff(x, axis=0)[None]+noise*sigma
    result = np.empty((replicas, *x.shape), dtype=np.float32)
    result[:, 0] = x[0]
    result[:, 1:] = x[0]+np.cumsum(increments, axis=1)
    return result


def event_cloud(quantiles, probabilities):
    q, p = np.asarray(quantiles, dtype=float), np.asarray(probabilities, dtype=float)
    if q.ndim != 3 or q.shape[-1] != len(p) or len(p) < 2 or q.shape[0] < 1:
        raise ValueError("Invalid quantile array or probability grid.")
    if not np.isfinite(q).all() or not np.isfinite(p).all() or np.any(np.diff(p) <= 0) or not 0 <= p[0] < p[-1] <= 1:
        raise ValueError("Quantile inputs must be finite and probability levels increasing.")
    if np.any(np.diff(q, axis=-1) < 0):
        raise ValueError("Correct quantile crossing before cloud inversion.")
    w = np.empty_like(p)
    w[0], w[-1] = .5*(p[1]-p[0]), .5*(p[-1]-p[-2])
    w[1:-1] = .5*(p[2:]-p[:-2])
    w = w/w.sum()
    ex = (q*w).sum(-1).mean(0)
    en = np.sqrt(np.pi/2)*(np.abs(q-ex[None, :, None])*w).sum(-1).mean(0)
    variance = (np.square(q-ex[None, :, None])*w).sum(-1).mean(0)
    he = np.sqrt(np.maximum(variance-en**2, 0))
    return np.stack((ex, en, he), axis=-1)
