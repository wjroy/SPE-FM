import numpy as np

from .data import normalization, split_bounds


def dtw(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or not x.size or not y.size:
        raise ValueError("DTW expects two nonempty vectors.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("DTW inputs must be finite.")
    previous = np.full(len(y)+1, np.inf)
    lengths = np.zeros(len(y)+1, dtype=int)
    previous[0] = 0.
    for a in x:
        current = np.full_like(previous, np.inf)
        current_lengths = np.zeros_like(lengths)
        for j, b in enumerate(y, 1):
            costs = (previous[j-1], previous[j], current[j-1])
            sizes = (lengths[j-1], lengths[j], current_lengths[j-1])
            k = int(np.argmin(costs))
            current[j] = abs(a-b) + costs[k]
            current_lengths[j] = 1 + sizes[k]
        previous, lengths = current, current_lengths
    return float(previous[-1]), int(lengths[-1])


def distance_matrices(increments):
    n = increments.shape[1]
    distances = np.zeros((n, n))
    lengths = np.full((n, n), len(increments), dtype=int)
    for i in range(n):
        for j in range(i):
            cost, length = dtw(increments[:, i], increments[:, j])
            distances[i, j] = distances[j, i] = cost
            lengths[i, j] = lengths[j, i] = length
    return distances, lengths


def _best_medoid_swap(distances, medoids):
    assigned = distances[:, medoids]
    labels = assigned.argmin(axis=1)
    nearest = assigned[np.arange(len(distances)), labels]
    second = (np.partition(assigned, 1, axis=1)[:, 1] if len(medoids) > 1
              else np.full(len(distances), np.inf))
    cost = float(nearest.sum())
    candidates = np.setdiff1d(np.arange(len(distances)), medoids)
    best, best_cost = None, cost
    tolerance = 64 * np.finfo(float).eps * max(abs(cost), np.finfo(float).tiny)
    for slot in range(len(medoids)):
        if not len(candidates):
            break
        without = np.where(labels == slot, second, nearest)
        costs = np.minimum(without[:, None], distances[:, candidates]).sum(axis=0)
        index = int(costs.argmin())
        if costs[index] < best_cost - tolerance:
            best = slot, int(candidates[index])
            best_cost = float(costs[index])
    return best, best_cost


def kmedoids(distances, k, restarts=30, seed=42, max_iter=1000):
    d = np.asarray(distances, dtype=float)
    n = len(d)
    if d.shape != (n, n) or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Expected a finite, nonnegative square distance matrix.")
    if not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0):
        raise ValueError("Expected symmetric distances with zero diagonal.")
    if not 1 <= k <= n or restarts < 1 or max_iter < 1:
        raise ValueError("Invalid K or optimization budget.")
    best = None
    all_costs = []
    for run_seed in np.random.RandomState(seed).randint(0, 10**9, size=restarts):
        rng = np.random.RandomState(run_seed)
        medoids = np.sort(rng.choice(n, size=k, replace=False))
        for _ in range(max_iter):
            swap, _ = _best_medoid_swap(d, medoids)
            if swap is None:
                break
            slot, candidate = swap
            medoids[slot] = candidate
            medoids.sort()
        else:
            if _best_medoid_swap(d, medoids)[0] is not None:
                raise RuntimeError("K-medoids did not converge; increase max_iter.")
        labels = d[:, medoids].argmin(axis=1)
        labels[medoids] = np.arange(k)
        cost = float(d[:, medoids].min(axis=1).sum())
        all_costs.append(cost)
        if best is None or cost < best[0]:
            best = cost, medoids.copy(), labels.copy()
    return best[1], best[2], best[0], np.asarray(all_costs)


def elbow(distances, candidates, restarts=30, seed=42, max_iter=1000):
    ks = list(candidates)
    if len(ks) < 3 or ks != sorted(set(ks)):
        raise ValueError("Elbow selection needs at least three increasing distinct K values.")
    solutions, rows = {}, []
    for k in ks:
        medoids, labels, cost, costs = kmedoids(
            distances, k, restarts, seed + 1000*k, max_iter)
        solutions[k] = (medoids, labels)
        rows.append((k, cost, float(costs.mean()), float(costs.std())))
    points = np.asarray(rows)[:, :2]
    direction = points[-1] - points[0]
    offset = points - points[0]
    perpendicular = np.abs(direction[0]*offset[:, 1] - direction[1]*offset[:, 0])
    selected = ks[1 + int(perpendicular[1:-1].argmax())]
    return selected, solutions, np.asarray(rows)


def association_weights(distances, *, method, temperature=1., epsilon=1e-12):
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or not np.isfinite(d).all() or np.any(d < 0) or epsilon <= 0:
        raise ValueError("Expected finite nonnegative node-to-prototype distances.")
    if method == "softmax":
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        score = -d / temperature
        weights = np.exp(score-score.max(axis=1, keepdims=True))
    elif method == "inverse":
        weights = 1 / (d+epsilon)
    else:
        raise ValueError("Association method must be softmax or inverse.")
    return (weights/weights.sum(axis=1, keepdims=True)).astype(np.float32)


def fit_spatial_prior(values, ratios, settings):
    start, stop = split_bounds(len(values), ratios)["train"]
    train = np.asarray(values)[start:stop]
    if train.ndim != 2 or len(train) < 2 or not train.shape[1]:
        raise ValueError("Provide at least two training observations and node columns.")
    if not np.isfinite(train).all():
        raise ValueError("Training observations must be finite.")
    order = np.lexsort(train[::-1])
    inverse = np.argsort(order)
    distances, lengths = distance_matrices(np.diff(train[:, order], axis=0))
    selected, solutions, costs = elbow(distances, settings["candidates"],
        settings["restarts"], settings["seed"], settings["max_iter"])
    used = settings["prototypes"] if settings["prototypes"] is not None else selected
    if used not in solutions:
        raise ValueError("The requested prototype count must belong to the candidate grid.")
    medoids, labels = solutions[used]
    medoids, labels = order[medoids], labels[inverse]
    distances = distances[np.ix_(inverse, inverse)]
    lengths = lengths[np.ix_(inverse, inverse)]
    associated = distances[:, medoids].copy()
    if settings["path_normalization"]:
        associated /= lengths[:, medoids]
    weights = association_weights(associated, method=settings["association"],
        temperature=settings.get("temperature", 1.), epsilon=settings.get("epsilon", 1e-12))
    mean, std = normalization(train)
    arrays = dict(distances=distances, path_lengths=lengths, medoids=medoids,
                  labels=labels, weights=weights, mean=mean, std=std)
    details = dict(elbow_selected_k=int(selected), used_k=int(used),
                   training_observations=len(train), increment_length=len(train)-1,
                   spatial_prior_scope="training_only", clustering_solver="pam_swap",
                   initialization_order="training_series_lexicographic")
    return arrays, costs, details
