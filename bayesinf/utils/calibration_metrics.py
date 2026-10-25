"""Marginal residual diagnostics, distinct from posterior calibration tests."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def residual_normality_diagnostics(truth, mean, std, *, bins=80, clip=5.0):
    """Histogram KL to N(0,1), retaining both tails as explicit bins.

    Mean/std residual normality is not a general calibration criterion for a
    non-Gaussian posterior. Use interval coverage and posterior ranks as well.
    Infinite residuals from finite inputs remain in their appropriate tail.
    """
    truth, mean, std = np.broadcast_arrays(truth, mean, std)
    if (bins < 2 or clip <= 0 or not np.isfinite(clip)
            or not all(np.isfinite(value).all() for value in (truth, mean, std))
            or np.any(std < 0)):
        raise ValueError("Residual diagnostics require finite inputs, nonnegative std, bins >= 2 and clip > 0")
    if truth.size == 0:
        return {"kl": float("nan"), "tail_fraction": float("nan"), "count": 0}
    with np.errstate(over="ignore"):
        z = ((truth - mean) / np.maximum(std, 1e-12)).ravel()
    edges = np.concatenate(([-np.inf], np.linspace(-clip, clip, bins + 1), [np.inf]))
    counts, _ = np.histogram(z, bins=edges)
    p = counts.astype(float) / z.size
    q = np.diff(ndtr(edges))
    occupied = p > 0
    q = np.maximum(q, np.finfo(float).tiny)
    kl = np.sum(p[occupied] * np.log(p[occupied] / q[occupied]))
    return {"kl": float(kl), "tail_fraction": float(np.mean(np.abs(z) > clip)),
            "count": int(z.size)}


def calibration_kl_divergence(truth, mean, std, bins=80, clip=5.0):
    """Compatibility name for residual-normality KL; not general calibration."""
    return residual_normality_diagnostics(truth, mean, std, bins=bins, clip=clip)["kl"]
