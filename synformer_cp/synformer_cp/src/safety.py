"""Uncertainty-gated controller: when the conformal interval is wide the command moves slowly (holds),
otherwise it follows the prediction with no added lag."""
import numpy as np


def gate_reference(calib_width):
    return np.median(calib_width, 0), np.percentile(calib_width, 95, 0)


def gate(mu, width, w_lo, w_hi, a_min=0.15):
    out = np.empty_like(mu)
    prev = mu[0].copy()
    span = np.maximum(w_hi - w_lo, 1e-8)
    for t in range(len(mu)):
        conf = 1.0 - np.clip((width[t] - w_lo) / span, 0.0, 1.0)
        alpha = a_min + (1 - a_min) * conf
        prev = prev + alpha * (mu[t] - prev)
        out[t] = prev
    return out
