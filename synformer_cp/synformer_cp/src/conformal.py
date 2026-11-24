"""Split-conformal prediction intervals on top of any (mu, sigma) regressor. Per-DoF, normalised scores."""
import numpy as np


def conformal_quantile(scores, alpha):
    n = scores.shape[0]
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(scores, axis=0)[k - 1]


def calibrate(y, mu, sigma, alpha=0.1, eps=1e-3):
    return conformal_quantile(np.abs(y - mu) / (sigma + eps), alpha)      # (D,)


def intervals(mu, sigma, q, eps=1e-3):
    half = q * (sigma + eps)
    return mu - half, mu + half


def coverage(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))
