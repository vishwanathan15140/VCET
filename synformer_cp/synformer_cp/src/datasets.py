"""Dataset loaders (synthetic / NinaPro DB2 / Hyser) and the per-subject processing pipeline. NumPy only.

Every loader returns  {subject_id: [(emg (T,C), y (T,D), fs_raw), ...]}  (list of recording segments).
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from .preprocess import (bandpass_notch, envelope, estimate_lag, lowpass,
                         make_windows)


@dataclass
class Windows:
    X: np.ndarray      # (N, C, T) float16
    Y: np.ndarray      # (N, D) float32  (physical units, normalised later with train-fold stats)
    seg: np.ndarray    # (N,) segment id
    nxt: np.ndarray    # (N,) index of the next window in the same segment (self at segment end)
    lag: int = 0       # EMG->label lag in samples that was compensated


# --------------------------------------------------------------------------- synthetic
def _smooth_traj(rng, T, fs, k_dim, f_cut=2.0):
    from .preprocess import lowpass as lp
    k = int(T / fs * 8) + 4
    c = rng.standard_normal((k, k_dim))
    t_lo, t_hi = np.linspace(0, 1, k), np.linspace(0, 1, T)
    s = np.stack([np.interp(t_hi, t_lo, c[:, i]) for i in range(k_dim)], 1)
    return lp(s, fs, f_cut)


def load_synthetic(cfg, n_dof=10, n_ch=12, n_latent=4, dur_s=160, n_seg=4, lag_ms=60, seed=0):
    """Hand kinematics live in a low-dim synergy space; each subject has its own muscle->electrode mixing,
    gain, and ring rotation. Lets you verify the whole pipeline (incl. shift & subject shift) without data."""
    rng = np.random.default_rng(seed)
    fs = cfg.fs
    W = rng.standard_normal((n_dof, n_latent))
    A0 = rng.standard_normal((n_ch, n_dof))
    out = {}
    for s in range(cfg.n_syn_subjects):
        A = A0 + 0.4 * rng.standard_normal(A0.shape)
        if cfg.n_ring > 0:
            sh = int(rng.integers(-cfg.max_shift, cfg.max_shift + 1))
            A[: cfg.n_ring] = np.roll(A[: cfg.n_ring], sh, axis=0)
        gain = rng.uniform(0.5, 2.0)
        segs = []
        for _ in range(n_seg):
            T = int(dur_s / n_seg * fs)
            y = _smooth_traj(rng, T, fs, n_latent) @ W.T
            y = (y - y.mean(0)) / (y.std(0) + 1e-8)
            exc = lowpass(np.maximum(y @ A.T, 0) + 0.05, fs, 8.0)
            emg = gain * np.maximum(exc, 0) * rng.standard_normal((T, n_ch))
            emg += 0.03 * np.sin(2 * np.pi * cfg.notch_hz * np.arange(T) / fs)[:, None]
            L = int(lag_ms * fs / 1000)
            y_lab = np.concatenate([np.repeat(y[:1], L, 0), y[:-L]])   # label lags EMG
            segs.append((emg, y_lab, fs))
        out[s] = segs
    return out


# --------------------------------------------------------------------------- NinaPro DB2
def load_ninapro_db2(cfg):
    """Expects the official .mat files (S{n}_E{e}_A1.mat) anywhere below cfg.data_root.
    Keys used: 'emg' (T,12) and 'glove' (T,22), both at 2 kHz. Check Ninapro docs for the sensor map and
    pass --dof_idx to choose the glove channels (some sensors are dead for some subjects)."""
    import scipy.io as sio
    out = {}
    for s in range(1, 41):
        if cfg.max_subjects and len(out) >= cfg.max_subjects:
            break
        files = glob.glob(os.path.join(cfg.data_root, "**", f"S{s}_E{cfg.exercise}_A1.mat"), recursive=True)
        if not files:
            continue
        m = sio.loadmat(files[0])
        emg, gl = m["emg"].astype(np.float64), m["glove"].astype(np.float64)
        if cfg.dof_idx:
            gl = gl[:, cfg.dof_idx]
        out[s] = [(emg, gl, 2000)]
    if not out:
        raise FileNotFoundError(f"No NinaPro DB2 files found under {cfg.data_root}")
    return out


# --------------------------------------------------------------------------- Hyser
def load_hyser(cfg):
    """PhysioNet 'hd-semg' (Hyser). Layout: <root>/<task>_dataset/subjectXX_sessionY/<task>_{preprocess|force}_*.{hea,dat}
    EMG 256 ch @2048 Hz, force 5 fingers @100 Hz (up-sampled here). Verify scaling with the dataset README."""
    import wfdb
    out = {}
    for s in range(1, 21):
        if cfg.max_subjects and len(out) >= cfg.max_subjects:
            break
        d = os.path.join(cfg.data_root, f"{cfg.hyser_task}_dataset", f"subject{s:02d}_session{cfg.hyser_session}")
        segs = []
        for h in sorted(glob.glob(os.path.join(d, f"{cfg.hyser_task}_preprocess_*.hea"))):
            base = h[:-4]
            fbase = base.replace("_preprocess_", "_force_")
            if not os.path.exists(fbase + ".hea"):
                continue
            emg = wfdb.rdrecord(base).p_signal[:, :: cfg.ch_sub]
            frc = wfdb.rdrecord(fbase).p_signal
            t_e, t_f = np.arange(emg.shape[0]) / 2048.0, np.arange(frc.shape[0]) / 100.0
            frc_up = np.stack([np.interp(t_e, t_f, frc[:, j]) for j in range(frc.shape[1])], 1)
            segs.append((emg.astype(np.float64), frc_up, 2048))
        if segs:
            out[s] = segs
    if not out:
        raise FileNotFoundError(f"No Hyser records found under {cfg.data_root}")
    return out


def load_raw(cfg):
    return {"synthetic": load_synthetic, "ninapro_db2": load_ninapro_db2, "hyser": load_hyser}[cfg.dataset](cfg)


# --------------------------------------------------------------------------- per-subject pipeline
def process_subject(segs, cfg):
    win = int(cfg.fs * cfg.win_ms / 1000)
    stride = max(1, int(cfg.fs * cfg.stride_ms / 1000))
    proc = []
    for emg, y, fs0 in segs:
        if int(fs0) != int(cfg.fs):
            g = int(np.gcd(int(fs0), int(cfg.fs)))
            up, down = cfg.fs // g, int(fs0) // g
            emg, y = resample_poly(emg, up, down, axis=0), resample_poly(y, up, down, axis=0)
        emg = bandpass_notch(emg, cfg.fs, notch=cfg.notch_hz, causal=cfg.causal)
        y = lowpass(y, cfg.fs, cfg.label_lp_hz)
        proc.append((emg, y))
    lag = int(np.median([estimate_lag(envelope(e, cfg.fs), y, cfg.fs) for e, y in proc]))
    sd = np.concatenate([e for e, _ in proc]).std(0) + 1e-8          # label-free per-subject normalisation
    Xs, Ys, Ss, Ns, off = [], [], [], [], 0
    for i, (e, y) in enumerate(proc):
        if lag > 0:
            e, y = e[:-lag], y[lag:]
        X, Y = make_windows(e / sd, y, win, stride)
        n = len(Y)
        nxt = np.minimum(np.arange(n) + 1, n - 1) + off
        Xs.append(X.astype(np.float16)); Ys.append(Y); Ss.append(np.full(n, i)); Ns.append(nxt)
        off += n
    return Windows(np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ss),
                   np.concatenate(Ns).astype(np.int64), lag)


def build_all(cfg, log=print):
    raw = load_raw(cfg)
    data = {}
    for s, segs in raw.items():
        data[s] = process_subject(segs, cfg)
        log(f"subject {s}: {len(data[s].Y)} windows, lag {data[s].lag * 1000 / cfg.fs:.0f} ms")
    s0 = next(iter(data))
    cfg.n_ch, cfg.n_dof = data[s0].X.shape[1], data[s0].Y.shape[1]
    return data


def merge(data, subs):
    X, Y, S, N, off = [], [], [], [], 0
    for k, s in enumerate(subs):
        w = data[s]
        X.append(w.X); Y.append(w.Y); S.append(np.full(len(w.Y), k)); N.append(w.nxt + off)
        off += len(w.Y)
    return (np.concatenate(X), np.concatenate(Y),
            np.concatenate(S).astype(np.int64), np.concatenate(N).astype(np.int64))
