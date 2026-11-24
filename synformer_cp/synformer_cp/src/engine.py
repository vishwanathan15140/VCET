"""Training, prediction (with MC-dropout), few-shot adaptation."""
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .losses import ccc, total_loss
from .metrics import r2
from .models import build_model, ring_shift, set_mc_dropout


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pca_basis(Y, k, n=50000):
    Y = Y[np.random.default_rng(0).permutation(len(Y))[:n]]
    k = min(k, Y.shape[1])
    _, _, Vt = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
    return Vt[:k].T.astype(np.float32)                       # (D, k)


def augment(xx, cfg, B):
    """xx = [window_t ; window_t+1] (2B,C,T). Gain/dropout/shift are shared inside a pair so that the
    delta-loss compares like with like."""
    dev, C = xx.device, xx.shape[1]
    xx = xx * torch.empty(B, C, 1, device=dev).uniform_(0.7, 1.3).repeat(2, 1, 1)
    if cfg.ch_drop > 0:
        xx = xx * (torch.rand(B, C, 1, device=dev) > cfg.ch_drop).float().repeat(2, 1, 1)
    xx = xx + torch.empty(B, 1, 1, device=dev).uniform_(0, cfg.noise_max).repeat(2, 1, 1) * torch.randn_like(xx)
    s = None
    if cfg.shift_aug and cfg.n_ring > 0 and cfg.max_shift > 0:
        s = torch.randint(-cfg.max_shift, cfg.max_shift + 1, (B,), device=dev).repeat(2)
        xx = ring_shift(xx, s, cfg.n_ring)
    return xx, s


@torch.no_grad()
def predict(model, X, dev, cfg, mc=0, shift=0, bs=512):
    """Returns mu (N,D) and total predictive std sigma (N,D) = sqrt(aleatoric + epistemic)."""
    model.eval()
    mus, sigs = [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(dev)
        if shift and cfg.n_ring > 0:
            x = ring_shift(x, torch.full((len(x),), shift, device=dev, dtype=torch.long), cfg.n_ring)
        if mc > 0:
            set_mc_dropout(model, True)
            outs = [model(x) for _ in range(mc)]
            set_mc_dropout(model, False)
            mk, lk = torch.stack([o["mu"] for o in outs]), torch.stack([o["logvar"].clamp(-6, 3) for o in outs])
            mu, var = mk.mean(0), mk.var(0, unbiased=False) + lk.exp().mean(0)
        else:
            o = model(x)
            mu, var = o["mu"], o["logvar"].clamp(-6, 3).exp()
        mus.append(mu.cpu().numpy()); sigs.append(var.sqrt().cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigs)


def train_model(cfg, Xtr, Ytr, Str, NXT, Xva, Yva, n_dom, dev, teacher=None, log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_model(cfg, n_dom).to(dev)
    if cfg.model == "synformer" and cfg.use_syn:
        V = torch.from_numpy(pca_basis(Ytr, cfg.n_syn))
        with torch.no_grad():
            model.syn_dec.weight[:, : V.shape[1]] = V.to(dev)
            model.syn_dec.bias.zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    N = len(Xtr)
    steps = max(N // cfg.bs, 1)
    total = cfg.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * total + 1)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    best, best_state, bad, step = -1e9, None, 0, 0
    vsel = rng.permutation(len(Xva))[:4000]
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(N)
        agg = {}
        for b in range(steps):
            idx = perm[b * cfg.bs:(b + 1) * cfg.bs]
            B = len(idx)
            x = torch.from_numpy(np.concatenate([Xtr[idx], Xtr[NXT[idx]]]).astype(np.float32)).to(dev)
            y = torch.from_numpy(np.concatenate([Ytr[idx], Ytr[NXT[idx]]])).to(dev)
            sj = torch.from_numpy(Str[idx]).to(dev)
            x, s_aug = augment(x, cfg, B)
            sj_in = torch.where(torch.rand(B, device=dev) < cfg.film_dropout, torch.full_like(sj, n_dom), sj).repeat(2)
            lam = cfg.lam_dann * (2 / (1 + math.exp(-10 * step / total)) - 1)
            out = model(x, subj=sj_in, grl_lam=lam)
            t_out = None
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
            shift_t = (-s_aug + cfg.max_shift) if s_aug is not None else None
            loss, parts = total_loss(out, y, cfg, dom_t=sj.repeat(2), shift_t=shift_t, teacher=t_out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        mu, _ = predict(model, Xva[vsel], dev, cfg)
        v = float(np.mean(r2(Yva[vsel], mu)))
        log(f"  ep {ep + 1:02d} val R2 {v:.3f} | " + " ".join(f"{k} {a / steps:.3f}" for k, a in agg.items()))
        if v > best:
            best, bad = v, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def adapt(model, Xa, Ya, cfg, dev, seed=0):
    """Few-shot calibration: only the FiLM vector + synergy-coefficient head (<2 % of weights) are updated,
    with an L2 pull towards the generic solution. The base model is never modified (works on a copy)."""
    m = copy.deepcopy(model).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if hasattr(m, "film"):
        m.film_override = nn.Parameter(m.film.data[-1].clone())
        params = [m.film_override] + list(m.syn_head.parameters())
        if not cfg.use_syn:
            params += list(m.res_head.parameters())
    else:
        params = list(m.head.parameters())
    for p in params:
        p.requires_grad_(True)
    p0 = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=cfg.adapt_lr)
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(Xa.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(Ya).to(dev)
    for _ in range(cfg.adapt_steps):
        idx = torch.from_numpy(rng.choice(len(Xt), min(256, len(Xt)), replace=False)).to(dev)
        mu = m(Xt[idx])["mu"]
        reg = sum(((p - q) ** 2).sum() for p, q in zip(params, p0))
        loss = ((mu - Yt[idx]) ** 2).mean() + 0.5 * (1 - ccc(Yt[idx], mu)) + cfg.adapt_l2 * reg
        opt.zero_grad(); loss.backward(); opt.step()
    for p in params:
        p.requires_grad_(False)
    return m.eval()
