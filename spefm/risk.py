import numpy as np


def normalize(weights):
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be finite and nonnegative.")
    total = w.sum(axis=-1, keepdims=True)
    return np.divide(w, total, out=np.full_like(w, 1/w.shape[-1]), where=total > 1e-12)


def standard_clouds(bounds, entropy_divisor, hyperentropy_ratio):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 1 or len(bounds) < 3 or not np.isfinite(bounds).all() or np.any(np.diff(bounds) <= 0):
        raise ValueError("Provide increasing grade interval bounds.")
    if entropy_divisor <= 0 or hyperentropy_ratio < 0:
        raise ValueError("Invalid standard cloud parameters.")
    ex = (bounds[:-1]+bounds[1:])/2
    en = np.diff(bounds)/entropy_divisor
    return np.stack((ex, en, en*hyperentropy_ratio), axis=-1)


def memberships(clouds, standards, *, drops, seed, shoulders, sample_standard_entropy):
    c, s = np.asarray(clouds, dtype=float), np.asarray(standards, dtype=float)
    if c.ndim != 2 or s.ndim != 2 or c.shape[1] != 3 or s.shape[1] != 3 or len(s) < 2:
        raise ValueError("Cloud parameters must have three columns: Ex, En, He.")
    if not np.isfinite(c).all() or not np.isfinite(s).all() or np.any(c[:, 1:] < 0) or np.any(s[:, 1:] < 0):
        raise ValueError("Cloud parameters must be finite; En and He nonnegative.")
    if drops < 1:
        raise ValueError("Number of cloud drops must be positive.")
    rng = np.random.default_rng(seed)
    mu = np.empty((len(c), len(s)))
    for i, (ex, en, he) in enumerate(c):
        entropy = np.maximum(np.abs(rng.normal(en, he, drops)), 1e-6)
        samples = rng.normal(ex, entropy)
        if sample_standard_entropy:
            grade_entropy = rng.normal(s[:, 1, None], s[:, 2, None], (len(s), drops))
        else:
            grade_entropy = s[:, 1, None]
        grade_entropy = np.maximum(np.abs(grade_entropy), 1e-6)
        certainty = np.exp(-.5*((samples[None]-s[:, 0, None])/grade_entropy)**2)
        if shoulders:
            certainty[0, samples <= s[0, 0]] = 1
            certainty[-1, samples >= s[-1, 0]] = 1
        mu[i] = certainty.mean(axis=1)
    return mu, normalize(mu)


def contribution_weights(mu, gamma, scores, *, entropy_mode="membership"):
    scores = np.asarray(scores, dtype=float)
    if mu.shape != gamma.shape or scores.shape != (gamma.shape[1],) or np.any(scores < 0):
        raise ValueError("Memberships and nonnegative grade scores must align.")
    if entropy_mode != "membership":
        raise ValueError("Supported entropy mode: membership (normalized grade memberships).")
    p = normalize(gamma)
    entropy = -(p*np.log(np.maximum(p, 1e-12))).sum(-1)/np.log(p.shape[-1])
    info = normalize(np.maximum(1-entropy, 0))
    severity = gamma@scores
    return {"information": info, "severity": severity,
            "contribution": normalize(info*severity)}


def fuse_clouds(clouds, weights):
    c, w = np.asarray(clouds, dtype=float), normalize(weights)
    if c.shape != (len(w), 3) or not np.isfinite(c).all() or np.any(c[:, 1:] < 0):
        raise ValueError("Cloud parameters and node weights must align.")
    spread_weights = w*c[:, 1]
    if spread_weights.sum() <= 1e-12:
        return np.array([w@c[:, 0], 0., w@c[:, 2]])
    return np.array([spread_weights@c[:, 0]/spread_weights.sum(),
                     spread_weights.sum(), spread_weights@c[:, 2]/spread_weights.sum()])


def warning(clouds, settings, seed):
    standards = standard_clouds(settings["bounds_mm"], settings["entropy_divisor"], settings["hyperentropy_ratio"])
    args = dict(drops=settings["cloud_drops"], shoulders=settings["terminal_shoulders"],
                sample_standard_entropy=settings["sample_standard_entropy"])
    mu, gamma = memberships(clouds, standards, seed=seed, **args)
    weights = contribution_weights(mu, gamma, settings["grade_scores"], entropy_mode=settings["entropy_mode"])
    project = fuse_clouds(clouds, weights["contribution"])
    _, project_gamma = memberships(project[None], standards, seed=seed+1, **args)
    return {"node_clouds": clouds, "node_gamma": gamma, **weights,
            "project_cloud": project, "project_gamma": project_gamma[0]}
