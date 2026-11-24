"""Leave-one-subject-out experiment: train -> zero-shot test -> few-shot adapt -> conformal -> safety gate."""
import json
import os
import time

import numpy as np

from .baselines import fit_ridge
from .conformal import calibrate, coverage, intervals
from .datasets import build_all, merge
from .metrics import r2, summarize
from .safety import gate, gate_reference


def split_target(w, cfg):
    """Chronological slices with a gap of one window between them (windows overlap) so nothing leaks:
    [adapt (up to the longest calibration length)] gap [conformal calibration] gap [test]."""
    spw = 1000.0 / cfg.stride_ms
    na = int(cfg.adapt_s * spw)
    nmax = int(max([cfg.adapt_s] + list(cfg.adapt_curve)) * spw)
    nc = int(cfg.conf_s * spw)
    gap = int(np.ceil(cfg.win_ms / cfg.stride_ms))
    a = slice(0, na)
    c = slice(nmax + gap, nmax + gap + nc)
    t = slice(nmax + gap + nc + gap, None)
    assert len(w.Y[t]) >= 100, "not enough data left for testing: reduce adapt_s / conf_s / adapt_curve"
    return a, c, t


def run_fold(cfg, data, ts, dev=None, log=print):
    subjects = sorted(data)
    train_s = [s for s in subjects if s != ts]
    n_val = min(cfg.n_val, max(len(train_s) - 2, 0))
    val_s, tr_s = (train_s[-n_val:], train_s[:-n_val]) if n_val else (train_s[-1:], train_s)
    Xtr, Ytr_raw, Str, NXT = merge(data, tr_s)
    ymu, ysd = Ytr_raw.mean(0), Ytr_raw.std(0) + 1e-6
    Ytr = (Ytr_raw - ymu) / ysd
    Xva, Yva_raw, _, _ = merge(data, val_s)
    Yva = (Yva_raw - ymu) / ysd
    w = data[ts]
    a, c, t = split_target(w, cfg)
    Xa, Ya = w.X[a], (w.Y[a] - ymu) / ysd
    Xc, Yc = w.X[c], (w.Y[c] - ymu) / ysd
    Xt, Yt = w.X[t], (w.Y[t] - ymu) / ysd
    res = {}

    if cfg.model == "ridge":
        f = fit_ridge(Xtr, Ytr)
        res["zero_shot"] = summarize(Yt, f(Xt), ysd)
        return res

    from .engine import adapt, get_device, predict, train_model
    dev = dev or get_device()
    model = train_model(cfg, Xtr, Ytr, Str, NXT, Xva, Yva, len(tr_s), dev, log=log)
    mu0, _ = predict(model, Xt, dev, cfg)
    res["zero_shot"] = summarize(Yt, mu0, ysd)

    if cfg.n_ring > 0 and cfg.max_shift > 0:                   # robustness to armband rotation
        res["shift_r2"] = {}
        for s in range(-cfg.max_shift, cfg.max_shift + 1):
            mu_s, _ = predict(model, Xt, dev, cfg, shift=s)
            res["shift_r2"][str(s)] = float(np.mean(r2(Yt, mu_s)))

    res["adapt_curve"] = {}
    for sec in cfg.adapt_curve:                                 # accuracy vs calibration time
        n = int(sec * 1000 / cfg.stride_ms)
        m_s = adapt(model, w.X[:n], (w.Y[:n] - ymu) / ysd, cfg, dev)
        res["adapt_curve"][str(sec)] = float(np.mean(r2(Yt, predict(m_s, Xt, dev, cfg)[0])))

    m_a = adapt(model, Xa, Ya, cfg, dev)
    mu_c, sg_c = predict(m_a, Xc, dev, cfg, mc=cfg.mc if cfg.model == "synformer" else 0)
    mu_t, sg_t = predict(m_a, Xt, dev, cfg, mc=cfg.mc if cfg.model == "synformer" else 0)
    res["adapted"] = summarize(Yt, mu_t, ysd)

    q = calibrate(Yc, mu_c, sg_c, cfg.alpha)                    # conformal on a disjoint calibration slice
    lo, hi = intervals(mu_t, sg_t, q)
    res["conformal"] = {"target": 1 - cfg.alpha, "coverage": coverage(Yt, lo, hi),
                        "mean_width": float(np.mean(hi - lo)),
                        "corr_sigma_err": float(np.corrcoef(sg_t.ravel(), np.abs(Yt - mu_t).ravel())[0, 1])}
    lo_c, hi_c = intervals(mu_c, sg_c, q)
    w_lo, w_hi = gate_reference(hi_c - lo_c)
    g = gate(mu_t, hi - lo, w_lo, w_hi)
    res["safety_gate"] = {"r2_ungated": res["adapted"]["r2"], "r2_gated": float(np.mean(r2(Yt, g))),
                          "rmse_ungated": float(np.sqrt(np.mean((Yt - mu_t) ** 2))),
                          "rmse_gated": float(np.sqrt(np.mean((Yt - g) ** 2)))}
    res["_ckpt"] = (model, ymu, ysd, q)
    return res


def run_loso(cfg, log=print):
    np.random.seed(cfg.seed)
    data = build_all(cfg, log)
    subjects = sorted(data)
    folds = cfg.test_subjects or subjects
    dev = None
    if cfg.model != "ridge":
        from .engine import get_device
        dev = get_device()
    out_dir = os.path.join(cfg.out_dir, cfg.tag)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for ts in folds:
        t0 = time.time()
        log(f"=== fold: test subject {ts} ({len(subjects) - 1} train subjects) ===")
        r = run_fold(cfg, data, ts, dev, log)
        r.pop("_ckpt", None)
        results[str(ts)] = r
        key = "adapted" if "adapted" in r else "zero_shot"
        log(f"subject {ts}: zero-shot R2 {r['zero_shot']['r2']:.3f} | {key} R2 {r[key]['r2']:.3f} | {time.time() - t0:.0f}s")
        with open(os.path.join(out_dir, "loso.json"), "w") as f:
            json.dump(results, f, indent=1)
    for k in ("zero_shot", "adapted"):
        v = [r[k]["r2"] for r in results.values() if k in r]
        if v:
            log(f"MEAN {k} R2 = {np.mean(v):.3f} +/- {np.std(v):.3f} over {len(v)} subjects")
    return results
