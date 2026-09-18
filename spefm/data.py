import csv
from collections import Counter
from pathlib import Path
from decimal import Decimal
import re

import numpy as np
import pandas as pd

SPLITS = {"20/10/70": [0.2, 0.1, 0.7],
          "40/10/50": [0.4, 0.1, 0.5],
          "60/10/30": [0.6, 0.1, 0.3]}


def _node_sort_key(node):
    parts = tuple((1, int(part)) if part.isdigit() else (0, part)
                  for part in re.split(r"(\d+)", node))
    return parts, node


def read_monitoring(path, date_column="date", node_prefix="DB"):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate CSV column names: {', '.join(duplicates)}")
    frame = pd.read_csv(path)
    nodes = sorted((c for c in frame.columns if c.startswith(node_prefix)), key=_node_sort_key)
    if date_column not in frame or not nodes:
        raise ValueError("Expected a date column and monitoring-node columns.")
    dates = pd.to_datetime(frame[date_column], errors="raise")
    values = frame[nodes].to_numpy(dtype=np.float32)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be valid, unique and increasing.")
    if not np.isfinite(values).all():
        raise ValueError("Expected finite preprocessed monitoring values.")
    return values, dates.dt.strftime("%Y-%m-%d").to_numpy(dtype=str), nodes


def split_bounds(length, ratios):
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Three positive split ratios must sum to one.")
    a, b = Decimal(str(ratios[0])), Decimal(str(ratios[1]))
    first = int(length * a)
    second = int(length * (a + b))
    return {"train": (0, first), "val": (first, second), "test": (second, length)}


def windows(values, bounds, context, horizon):
    start, stop = bounds
    if context < 1 or horizon < 1 or not 0 <= start < stop <= len(values):
        raise ValueError("Invalid window lengths or subset bounds.")
    origins = np.arange(start + context, stop - horizon + 1)
    if not len(origins):
        raise ValueError("Subset is too short for the requested context and horizon.")
    x = np.stack([values[t-context:t] for t in origins])
    y = np.stack([values[t:t+horizon] for t in origins])
    return x, y, origins


def normalization(train_values):
    mean = np.mean(train_values, axis=0).astype(np.float32)
    std = np.std(train_values, axis=0).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-6))
