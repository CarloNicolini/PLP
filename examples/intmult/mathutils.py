from __future__ import annotations

from math import inf, isfinite

import numpy as np
from scipy.special import logsumexp, softmax


def softmean_value(values: np.ndarray | list[float], temperature: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return -inf
    return float(
        temperature * (logsumexp(arr / temperature) - np.log(arr.size)),
    )


def softmax_value(values: np.ndarray | list[float], temperature: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return -inf
    return float(temperature * logsumexp(arr / temperature))


def softmax_probabilities(
    mode_values: dict[str, float], temperature: float
) -> dict[str, float]:
    probabilities = {mode: 0.0 for mode in mode_values}
    finite_modes = [m for m, v in mode_values.items() if isfinite(v)]
    if not finite_modes:
        return probabilities
    logits = np.array([mode_values[m] for m in finite_modes], dtype=np.float64)
    probs = softmax(logits / temperature)
    for mode, p in zip(finite_modes, probs, strict=True):
        probabilities[mode] = float(p)
    return probabilities
