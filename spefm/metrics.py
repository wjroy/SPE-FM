import numpy as np


def evaluate_points(actual, predictions):
    y, p = np.asarray(actual), np.asarray(predictions)
    if y.ndim != 3 or p.shape != y.shape or min(y.shape) < 1:
        raise ValueError('Expected matching targets and predictions [origins,horizon,nodes].')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Targets and predictions must be finite.')
    error = p-y
    per_node = dict(MAE=np.abs(error).mean(axis=(0, 1)),
                    RMSE=np.sqrt(np.square(error).mean(axis=(0, 1))))
    return per_node, {key: float(value.mean()) for key, value in per_node.items()}


def evaluate(actual, outputs, *, point_head, coverage=.8, beta=1., offset=.1, penalty_rate=1.):
    y, q = np.asarray(actual), np.asarray(outputs)
    if y.ndim != 3 or q.shape != (*y.shape, 10) or not 0 <= point_head <= 9:
        raise ValueError("Invalid target/output shapes or point head.")
    if not np.isfinite(y).all() or not np.isfinite(q).all() or np.any(np.diff(q[..., 1:], axis=-1) < 0):
        raise ValueError("Provide finite targets and monotonic quantiles.")
    axis = (0, 1)
    point_metrics, _ = evaluate_points(y, q[..., point_head])
    low, high = q[..., 1], q[..., 9]
    picp = ((low <= y) & (y <= high)).mean(axis=axis)
    ranges = np.maximum(y.max(axis=axis)-y.min(axis=axis), 1e-12)
    pinaw = (high-low).mean(axis=axis)/ranges
    cwc = np.where(picp >= coverage, beta*pinaw,
                   (offset+beta*pinaw)*(1+np.exp(-penalty_rate*(picp-coverage))))
    per_node = dict(**point_metrics, PICP=picp, PINAW=pinaw, CWC=cwc)
    return per_node, {k: float(v.mean()) for k, v in per_node.items()}
