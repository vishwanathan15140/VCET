import numpy as np


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)


def rmse(y, p):
    return np.sqrt(((y - p) ** 2).mean(0))


def nrmse(y, p):
    return rmse(y, p) / (y.max(0) - y.min(0) + 1e-12)


def ccc(y, p):
    my, mp = y.mean(0), p.mean(0)
    cov = ((y - my) * (p - mp)).mean(0)
    return 2 * cov / (y.var(0) + p.var(0) + (my - mp) ** 2 + 1e-12)


def pearson(y, p):
    yc, pc = y - y.mean(0), p - p.mean(0)
    return (yc * pc).sum(0) / (np.sqrt((yc ** 2).sum(0) * (pc ** 2).sum(0)) + 1e-12)


def summarize(y, p, ysd=None):
    d = {"r2": r2(y, p), "nrmse": nrmse(y, p), "ccc": ccc(y, p), "pearson": pearson(y, p), "rmse": rmse(y, p)}
    out = {k: float(np.mean(v)) for k, v in d.items()}
    out["r2_per_dof"] = d["r2"].tolist()
    out["nrmse_per_dof"] = d["nrmse"].tolist()
    if ysd is not None:
        out["rmse_phys"] = float(np.mean(d["rmse"] * ysd))
    return out
