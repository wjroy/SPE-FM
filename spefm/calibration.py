import math

import numpy as np


VERSION = "median_width_scale_v1"
MIN_WIDTH_MM = 1e-6


def _check(predictions):
    q = np.asarray(predictions)
    if q.ndim != 4 or q.shape[-1] != 10 or q.dtype != np.float32 or min(q.shape) < 1:
        raise ValueError("Expected float32 outputs [origins,horizon,nodes,10].")
    if not np.isfinite(q).all() or np.any(np.diff(q[..., 1:], axis=-1) < 0):
        raise ValueError("Outputs must be finite and ordered.")
    return q


def fit_width_scale(predictions, actual, coverage=.81):
    q = _check(predictions)
    y = np.asarray(actual)
    if y.shape != q.shape[:-1] or not np.isfinite(y).all() or not 0 < coverage < 1:
        raise ValueError("Invalid calibration targets or coverage.")
    median = q[..., 5].astype(np.float64)
    left = np.maximum(median-q[..., 1].astype(np.float64), MIN_WIDTH_MM)
    right = np.maximum(q[..., 9].astype(np.float64)-median, MIN_WIDTH_MM)
    scores = np.where(y < median, (median-y)/left, (y-median)/right).ravel()
    rank = min(len(scores), math.ceil((len(scores)+1)*coverage))
    exact = float(np.partition(scores, rank-1)[rank-1])
    scale = float(np.nextafter(np.float32(exact), np.float32(np.inf)))
    return dict(version=VERSION, target_coverage=float(coverage), scale=scale,
                exact_order_statistic=exact, rank_one_based=rank,
                calibration_cases=len(scores), calibration_origins=q.shape[0],
                calibration_nodes=q.shape[2], minimum_width_mm=MIN_WIDTH_MM,
                fitting_scope="validation_only", point_head=5,
                exchangeable_coverage_guarantee=False)


def apply_width_scale(predictions, parameters):
    q = _check(predictions)
    if parameters["version"] != VERSION or parameters["minimum_width_mm"] != MIN_WIDTH_MM:
        raise ValueError("Invalid calibration parameters.")
    scale = float(parameters["scale"])
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Invalid calibration scale.")
    median = q[..., 5:6].astype(np.float64)
    lower = np.maximum(median-q[..., 1:5].astype(np.float64),
                       MIN_WIDTH_MM*np.array([1., .75, .5, .25]))
    upper = np.maximum(q[..., 6:10].astype(np.float64)-median,
                       MIN_WIDTH_MM*np.array([.25, .5, .75, 1.]))
    result = q.copy()
    result[..., 1:5] = median-scale*lower
    result[..., 6:10] = median+scale*upper
    _check(result)
    return result


class OnlineInterval:
    def __init__(self, parameters, policy=None):
        self.parameters = dict(parameters)
        self.policy = dict(policy) if policy is not None else dict(
            version="causal_interval_width_feedback_v1",
            target_coverage=parameters["target_coverage"], learning_rate=.25)
        coverage = self.policy["target_coverage"]
        rate = self.policy["learning_rate"]
        scale = float(parameters["scale"])
        if (not 0 < coverage < 1 or coverage != parameters["target_coverage"]
                or not math.isfinite(rate) or rate <= 0
                or not math.isfinite(scale) or scale <= 0):
            raise ValueError("Invalid interval policy.")
        self.log_scale = math.log(scale)
        self.initial_log_scale = self.log_scale
        self.steps = 0
        self.last_origin = None
        self.pending = None

    def forecast(self, predictions, origin):
        if self.pending is not None:
            raise RuntimeError("A forecast is awaiting its observation.")
        if not isinstance(origin, (int, np.integer)) or isinstance(origin, bool):
            raise ValueError("Origin must be an integer.")
        origin = int(origin)
        if self.last_origin is not None and origin != self.last_origin+1:
            raise ValueError("Origins must be consecutive.")
        q = _check(predictions)
        if q.shape[:2] != (1, 1) or q.shape[2] != self.parameters["calibration_nodes"]:
            raise ValueError("Expected one origin, one horizon and matching nodes.")
        scale = math.exp(self.log_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid interval state.")
        result = apply_width_scale(q, dict(self.parameters, scale=scale))
        self.pending = dict(origin=origin, log_scale=self.log_scale, scale=scale,
                            lower=result[..., 1].tolist(), upper=result[..., 9].tolist())
        return result

    def observe(self, actual, origin):
        if not isinstance(origin, (int, np.integer)) or isinstance(origin, bool):
            raise ValueError("Origin must be an integer.")
        if self.pending is None or int(origin) != self.pending["origin"]:
            raise RuntimeError("Observation does not match a pending forecast.")
        lower = np.asarray(self.pending["lower"], dtype=np.float32)
        upper = np.asarray(self.pending["upper"], dtype=np.float32)
        y = np.asarray(actual)
        if y.shape != lower.shape or not np.isfinite(y).all():
            raise ValueError("Invalid observation.")
        covered = int(np.count_nonzero((lower <= y) & (y <= upper)))
        coverage = covered/y.size
        before = self.log_scale
        self.log_scale += self.policy["learning_rate"]*(self.policy["target_coverage"]-coverage)
        record = dict(origin=int(origin), scale_before=self.pending["scale"],
                      log_scale_before=before, covered=covered, cases=y.size,
                      coverage=coverage, log_scale_after=self.log_scale,
                      scale_after=math.exp(self.log_scale))
        self.steps += 1
        self.last_origin = int(origin)
        self.pending = None
        return record

    def to_dict(self):
        return dict(version="causal_interval_width_feedback_v1", parameters=self.parameters,
                    policy=self.policy, log_scale=self.log_scale,
                    initial_log_scale=self.initial_log_scale, steps=self.steps,
                    last_origin=self.last_origin, pending=self.pending)

    @classmethod
    def from_dict(cls, state):
        if state["version"] != "causal_interval_width_feedback_v1":
            raise ValueError("Invalid interval state version.")
        result = cls(state["parameters"], state["policy"])
        if (not math.isfinite(state["log_scale"])
                or state["initial_log_scale"] != result.initial_log_scale
                or not isinstance(state["steps"], int) or state["steps"] < 0):
            raise ValueError("Invalid interval state.")
        for name in ("log_scale", "steps", "last_origin", "pending"):
            setattr(result, name, state[name])
        return result


def replay(predictions, actual, origins, parameters, policy=None):
    q = _check(predictions)
    y, origins = np.asarray(actual), np.asarray(origins)
    if y.shape != q.shape[:-1] or origins.shape != (len(q),) or not np.isfinite(y).all():
        raise ValueError("Invalid replay arrays.")
    controller = OnlineInterval(parameters, policy)
    outputs, records = [], []
    for index, origin in enumerate(origins):
        outputs.append(controller.forecast(q[index:index+1], origin))
        records.append(controller.observe(y[index:index+1], origin))
    return np.concatenate(outputs), records, controller.to_dict()
