"""Classical baseline: hand-crafted time-domain features + ridge regression. NumPy/sklearn only."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def hand_features(X, chunk=2000):
    out = []
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].astype(np.float32)
        d = np.diff(x, axis=-1)
        mav, rms = np.abs(x).mean(-1), np.sqrt((x ** 2).mean(-1))
        wl = np.abs(d).sum(-1)
        zc = ((x[..., :-1] * x[..., 1:]) < 0).sum(-1)
        ssc = ((d[..., :-1] * d[..., 1:]) < 0).sum(-1)
        out.append(np.concatenate([np.log1p(mav), np.log1p(rms), np.log1p(wl), zc, ssc], 1))
    return np.concatenate(out)


def fit_ridge(Xtr, Ytr, alpha=10.0):
    F = hand_features(Xtr)
    sc = StandardScaler().fit(F)
    m = Ridge(alpha=alpha).fit(sc.transform(F), Ytr)
    return lambda X: m.predict(sc.transform(hand_features(X)))
