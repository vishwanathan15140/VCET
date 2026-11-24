import torch
import torch.nn.functional as F


def ccc(y, p, eps=1e-8):
    my, mp = y.mean(0), p.mean(0)
    cov = ((y - my) * (p - mp)).mean(0)
    return (2 * cov / (y.var(0, unbiased=False) + p.var(0, unbiased=False) + (my - mp) ** 2 + eps)).mean()


def beta_nll(mu, lv, y, beta=0.5):
    """beta-NLL (Seitzer et al. 2022): heteroscedastic Gaussian NLL that does not hurt the mean fit."""
    lv = lv.clamp(-6, 3)
    var = lv.exp()
    return (var.detach() ** beta * 0.5 * (lv + (y - mu) ** 2 / var)).mean()


def total_loss(out, y, cfg, dom_t=None, shift_t=None, teacher=None):
    mu = out["mu"]
    parts = {"mse": F.mse_loss(mu, y), "ccc": cfg.lam_ccc * (1 - ccc(y, mu))}
    mu_a, mu_b = mu.chunk(2)
    y_a, y_b = y.chunk(2)
    parts["delta"] = cfg.lam_delta * F.smooth_l1_loss(mu_b - mu_a, y_b - y_a)     # match temporal dynamics
    if cfg.use_nll and cfg.model == "synformer":
        parts["nll"] = cfg.lam_nll * beta_nll(mu, out["logvar"], y)
    if out.get("res") is not None:
        parts["res"] = cfg.lam_res * out["res"].pow(2).mean()                       # keep output near synergy space
    if out.get("dom") is not None and dom_t is not None:
        parts["dom"] = 1.0 * F.cross_entropy(out["dom"], dom_t)                     # GRL already scales gradient
    if out.get("shift") is not None and shift_t is not None:
        parts["shift"] = cfg.lam_shift * F.cross_entropy(out["shift"], shift_t)
    if teacher is not None:
        parts["kd"] = cfg.kd_w * (F.mse_loss(mu, teacher["mu"]) + 0.1 * F.mse_loss(out["logvar"], teacher["logvar"]))
    return sum(parts.values()), parts
