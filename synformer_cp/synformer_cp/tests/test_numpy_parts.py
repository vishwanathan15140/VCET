"""Torch-free tests. Run:  python tests/test_numpy_parts.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import Cfg
from src.datasets import build_all, merge
from src.metrics import summarize
from src.baselines import fit_ridge
from src.conformal import calibrate, intervals, coverage
from src.safety import gate, gate_reference


def main():
    cfg = Cfg(n_syn_subjects=5)
    data = build_all(cfg, log=lambda *_: None)
    lags = [w.lag * 1000 / cfg.fs for w in data.values()]
    print("estimated lags (true 60 ms):", [round(l) for l in lags])
    assert all(30 <= l <= 100 for l in lags), "lag estimation off"
    w = data[0]
    assert w.X.shape[1:] == (12, 300) and w.X.dtype == np.float16 and (w.nxt >= np.arange(len(w.nxt))).all()
    print("windows:", w.X.shape, w.Y.shape)

    # leave-one-subject-out ridge baseline
    scores = []
    for ts in data:
        tr = [s for s in data if s != ts]
        Xtr, Ytr, _, _ = merge(data, tr)
        mu_, sd_ = Ytr.mean(0), Ytr.std(0)
        f = fit_ridge(Xtr, (Ytr - mu_) / sd_)
        p = f(data[ts].X); y = (data[ts].Y - mu_) / sd_
        scores.append(summarize(y, p)["r2"])
    print("ridge LOSO R2 per subject:", np.round(scores, 3))
    assert np.mean(scores) > 0.2, "ridge baseline should learn something"

    # conformal coverage on synthetic gaussian noise
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4000, 3)); mu = y + rng.standard_normal((4000, 3)) * 0.5; sg = np.full_like(y, 0.5)
    q = calibrate(y[:1000], mu[:1000], sg[:1000], 0.1)
    lo, hi = intervals(mu[1000:], sg[1000:], q)
    cov = coverage(y[1000:], lo, hi)
    print("conformal coverage (target 0.90):", round(cov, 3))
    assert 0.86 < cov < 0.94

    # safety gate: wide intervals -> slower response
    mu_s = np.concatenate([np.zeros((50, 1)), np.ones((50, 1))])
    narrow = np.full((100, 1), 0.1); wide = np.full((100, 1), 1.0)
    lo_, hi_ = gate_reference(np.full((100, 1), 0.1) + np.linspace(0, 0.2, 100)[:, None])
    g1, g2 = gate(mu_s, narrow, lo_, hi_), gate(mu_s, wide, lo_, hi_)
    print("gate step response @ +5 samples: narrow %.2f, wide %.2f" % (g1[55, 0], g2[55, 0]))
    assert g1[55, 0] > g2[55, 0]
    print("ALL NUMPY TESTS PASSED")


if __name__ == "__main__":
    main()
