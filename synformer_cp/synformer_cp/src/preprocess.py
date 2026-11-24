"""Signal processing: filtering, lag compensation, windowing. NumPy/SciPy only."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos


def emg_sos(fs, lo=20.0, hi=450.0, notch=50.0, order=4):
    hi = min(hi, 0.45 * fs)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    if notch and notch < 0.5 * fs:
        b, a = iirnotch(notch, 30.0, fs=fs)
        sos = np.vstack([sos, tf2sos(b, a)])
    return sos


def bandpass_notch(x, fs, lo=20.0, hi=450.0, notch=50.0, causal=True):
    """x: (T, C). Causal by default so training matches what a real-time device can do."""
    f = sosfilt if causal else sosfiltfilt
    return f(emg_sos(fs, lo, hi, notch), x, axis=0)


def lowpass(x, fs, fc, order=4, causal=False):
    if fc >= 0.5 * fs:
        return x
    sos = butter(order, fc, btype="low", fs=fs, output="sos")
    return (sosfilt if causal else sosfiltfilt)(sos, x, axis=0)


def envelope(emg, fs, fc=5.0):
    return lowpass(np.abs(emg).mean(1), fs, fc)


def estimate_lag(env, y, fs, max_ms=300, step_ms=5):
    """Samples by which the label lags the EMG (EMG leads movement by the electromechanical delay).
    Pair EMG at time t with label at t+lag."""
    yy = np.abs(y - y.mean(0)).sum(1)
    env = (env - env.mean()) / (env.std() + 1e-8)
    yy = (yy - yy.mean()) / (yy.std() + 1e-8)
    step = max(1, int(fs * step_ms / 1000))
    best, best_l = -np.inf, 0
    for L in range(0, int(fs * max_ms / 1000) + 1, step):
        a = env[: len(env) - L] if L else env
        c = float(np.mean(a * yy[L:]))
        if c > best:
            best, best_l = c, L
    return best_l


def make_windows(emg, y, win, stride):
    """emg (T,C), y (T,D) -> X (N,C,win), Y (N,D); target = label at the window's last sample."""
    sw = sliding_window_view(emg, win, axis=0)          # (T-win+1, C, win)
    X = sw[::stride]
    Y = y[win - 1:][::stride]
    return np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(Y, dtype=np.float32)
