"""SynFormer-CP: synergy-constrained, shift-robust, subject-adaptive, uncertainty-aware EMG regressor."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grl(x, lam):
    return _GRL.apply(x, lam)


def sinusoid(n, d):
    pos = torch.arange(n).float()[:, None]
    div = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2).float() / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe[None]


def set_mc_dropout(model, on):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(on)


def ring_shift(x, s, n_ring):
    """Rotate the first n_ring channels by s[b] positions (per sample). roll semantics: out[c] = in[(c-s) % n]."""
    B, C, T = x.shape
    idx = (torch.arange(n_ring, device=x.device)[None, :] - s[:, None]) % n_ring
    ring = x[:, :n_ring].gather(1, idx[:, :, None].expand(-1, -1, T))
    return torch.cat([ring, x[:, n_ring:]], 1)


# ----------------------------------------------------------------------------- C1: activation dynamics
class ActivationDynamics(nn.Module):
    """Learnable first-order muscle activation dynamics (rectified EMG -> exponential-kernel integrator),
    one learnable time constant per channel. Physiologically motivated envelope extractor."""

    def __init__(self, n_ch, fs_eff, k=48, tau_ms=(15.0, 80.0)):
        super().__init__()
        taus = torch.exp(torch.linspace(math.log(tau_ms[0]), math.log(tau_ms[1]), n_ch)) / 1000.0 * fs_eff
        self.log_tau = nn.Parameter(torch.log(taus))
        self.k = k

    def forward(self, x):                                   # x (B,C,T) already rectified
        tau = torch.exp(self.log_tau).clamp(min=0.5)         # (C,) in samples
        t = torch.arange(self.k, device=x.device, dtype=x.dtype)
        ker = torch.exp(-t[None, :] / tau[:, None])          # lag 0..k-1
        ker = (ker / ker.sum(-1, keepdim=True)).flip(-1).unsqueeze(1)   # oldest..newest for cross-correlation
        return F.conv1d(F.pad(x, (self.k - 1, 0)), ker, groups=x.shape[1])


# ----------------------------------------------------------------------------- C3: electrode-shift aligner
class ShiftAligner(nn.Module):
    """Spatial-transformer for ring electrode arrays: predicts a soft rotation from the channel-energy profile
    and re-aligns the ring. Supervised by the shift-augmentation that is applied during training."""

    def __init__(self, n_ring, max_shift):
        super().__init__()
        self.n_ring, self.max_shift = n_ring, max_shift
        self.shifts = list(range(-max_shift, max_shift + 1))
        self.net = nn.Sequential(nn.Linear(n_ring, 32), nn.GELU(), nn.Linear(32, len(self.shifts)))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()
            self.net[-1].bias[max_shift] = 2.0               # start near identity

    def forward(self, x):
        ring, rest = x[:, : self.n_ring], x[:, self.n_ring:]
        e = ring.abs().mean(-1)
        e = e / (e.mean(-1, keepdim=True) + 1e-6)
        logits = self.net(e)
        w = F.softmax(logits, -1)
        rolled = torch.stack([torch.roll(ring, s, dims=1) for s in self.shifts], 1)   # (B,S,R,T)
        aligned = (w[:, :, None, None] * rolled).sum(1)
        return torch.cat([aligned, rest], 1), logits


class ChannelSE(nn.Module):
    def __init__(self, C, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(C, max(C // r, 4)), nn.GELU(), nn.Linear(max(C // r, 4), C), nn.Sigmoid())

    def forward(self, x):
        return x * (0.5 + self.fc(x.abs().mean(-1)))[..., None]


class TCNBlock(nn.Module):
    def __init__(self, d, k, dil, drop):
        super().__init__()
        p = (k - 1) * dil // 2
        self.c1, self.b1 = nn.Conv1d(d, d, k, padding=p, dilation=dil), nn.BatchNorm1d(d)
        self.c2, self.b2 = nn.Conv1d(d, d, k, padding=p, dilation=dil), nn.BatchNorm1d(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.drop(F.gelu(self.b1(self.c1(x))))
        return F.gelu(x + self.b2(self.c2(h)))


class Block(nn.Module):
    """Pre-norm transformer block with explicit nn.Dropout modules (so MC-dropout really works)."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ff(self.n2(x)))


# ----------------------------------------------------------------------------- the model
class SynFormer(nn.Module):
    def __init__(self, cfg, n_dom):
        super().__init__()
        d, C, D = cfg.d_model, cfg.n_ch, cfg.n_dof
        assert d % 8 == 0
        self.cfg, self.n_dom = cfg, n_dom
        self.aligner = ShiftAligner(cfg.n_ring, cfg.max_shift) if (cfg.use_shift and cfg.n_ring > 0) else None
        self.se = ChannelSE(C)
        ks = (5, 9, 17, 33)                                    # multi-scale temporal kernels
        per = d // 2 // len(ks)
        self.raw = nn.ModuleList([nn.Sequential(nn.Conv1d(C, per, k, stride=4, padding=k // 2),
                                                nn.BatchNorm1d(per), nn.GELU()) for k in ks])
        env_out = d - per * len(ks)
        self.dyn = ActivationDynamics(C, fs_eff=cfg.fs / 4) if cfg.use_dyn else None
        self.env = nn.Sequential(nn.Conv1d(C, env_out, 1), nn.BatchNorm1d(env_out), nn.GELU())
        self.tcn = nn.Sequential(*[TCNBlock(d, 3, dl, cfg.drop) for dl in (1, 2, 4)])
        self.down = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.register_buffer("pos", sinusoid(512, d), persistent=False)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.drop) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.pool = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.drop))
        # C4: per-subject FiLM table (last row = generic / unseen subject) + override for new users
        self.film = nn.Parameter(torch.zeros(n_dom + 1, 2 * d))
        self.register_parameter("film_override", None)
        # C2: synergy bottleneck + small residual; C5: heteroscedastic log-variance
        self.syn_head = nn.Linear(d, cfg.n_syn)
        self.syn_dec = nn.Linear(cfg.n_syn, D)
        self.res_head = nn.Linear(d, D)
        nn.init.normal_(self.res_head.weight, std=0.01); nn.init.zeros_(self.res_head.bias)
        self.lv_head = nn.Linear(d, D)
        nn.init.zeros_(self.lv_head.weight); nn.init.constant_(self.lv_head.bias, -1.0)
        self.dom = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, max(n_dom, 1)))

    def forward(self, x, subj=None, grl_lam=0.0):
        cfg = self.cfg
        sh_logits = None
        if self.aligner is not None:
            x, sh_logits = self.aligner(x)
        x = self.se(x)
        r = torch.cat([m(x) for m in self.raw], 1)
        e = F.avg_pool1d(x.abs(), 4, 4, ceil_mode=True)
        if self.dyn is not None:
            e = self.dyn(e)
        h = torch.cat([r, self.env(e)], 1)
        h = self.down(self.tcn(h)).transpose(1, 2)
        h = h + self.pos[:, : h.shape[1]]
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        z = self.pool(torch.cat([h[:, -1], h.mean(1)], -1))        # subject-invariant embedding
        if cfg.use_film:
            B = z.shape[0]
            if self.film_override is not None:
                gb = self.film_override[None].expand(B, -1)
            elif subj is not None:
                gb = self.film[subj]
            else:
                gb = self.film[-1][None].expand(B, -1)
            g, b_ = gb.chunk(2, -1)
            zf = z * (1 + g) + b_
        else:
            zf = z
        res = self.res_head(zf)
        mu = (self.syn_dec(self.syn_head(zf)) + res) if cfg.use_syn else res
        dom = self.dom(grl(z, grl_lam)) if (self.training and cfg.use_dann and self.n_dom > 0) else None
        return dict(mu=mu, logvar=self.lv_head(zf), res=res if cfg.use_syn else None, dom=dom, shift=sh_logits)


class LSTMBaseline(nn.Module):
    def __init__(self, cfg, n_dom=0):
        super().__init__()
        self.lstm = nn.LSTM(cfg.n_ch, 64, 2, batch_first=True, dropout=cfg.drop)
        self.head = nn.Linear(64, cfg.n_dof)

    def forward(self, x, subj=None, grl_lam=0.0):
        e = F.avg_pool1d(x.abs(), 8, 8, ceil_mode=True).transpose(1, 2)
        mu = self.head(self.lstm(e)[0][:, -1])
        return dict(mu=mu, logvar=torch.zeros_like(mu), res=None, dom=None, shift=None)


def build_model(cfg, n_dom):
    return SynFormer(cfg, n_dom) if cfg.model == "synformer" else LSTMBaseline(cfg, n_dom)
