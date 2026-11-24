"""Real-time streaming simulator: causal filtering with persistent state, ring buffer, per-hop inference,
conformal intervals, uncertainty gate and latency accounting. NumPy only (the predictor is injected)."""
import time

import numpy as np
from scipy.signal import sosfilt

from .preprocess import emg_sos
from .safety import gate


class StreamPreprocessor:
    def __init__(self, cfg, sd):
        self.sos = emg_sos(cfg.fs, notch=cfg.notch_hz)
        self.zi = np.zeros((self.sos.shape[0], 2, cfg.n_ch))
        self.sd = sd
        self.win = int(cfg.fs * cfg.win_ms / 1000)
        self.buf = np.zeros((0, cfg.n_ch))

    def push(self, chunk):
        y, self.zi = sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        self.buf = np.concatenate([self.buf, y / self.sd])[-self.win:]

    def window(self):
        return self.buf.T[None].astype(np.float32) if len(self.buf) == self.win else None


def run_stream(emg, predict_fn, cfg, sd, ymu, ysd, q, hop_ms=25, w_lo=None, w_hi=None, warm_ms=500):
    """predict_fn(x (1,C,T)) -> (mu, logvar) in normalised units. Returns arrays in physical units + latency."""
    hop = int(cfg.fs * hop_ms / 1000)
    pre = StreamPreprocessor(cfg, sd)
    mus, sigs, lat, ts = [], [], [], []
    for i in range(0, len(emg) - hop + 1, hop):
        pre.push(emg[i:i + hop])
        w = pre.window()
        if w is None or i < cfg.fs * warm_ms / 1000:
            continue
        t0 = time.perf_counter()
        mu, lv = predict_fn(w)
        lat.append((time.perf_counter() - t0) * 1000)
        mus.append(mu[0]); sigs.append(np.exp(0.5 * np.clip(lv[0], -6, 3))); ts.append(i + hop - 1)
    mu, sig = np.array(mus), np.array(sigs)
    half = q * (sig + 1e-3)
    out = {"t": np.array(ts), "mu": mu * ysd + ymu, "lo": (mu - half) * ysd + ymu,
           "hi": (mu + half) * ysd + ymu, "latency_ms": np.array(lat)}
    if w_lo is not None:
        out["gated"] = gate(mu, 2 * half, w_lo, w_hi) * ysd + ymu
    out["latency_p50"], out["latency_p95"] = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    return out
