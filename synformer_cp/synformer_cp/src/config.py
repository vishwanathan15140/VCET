"""Single source of truth for every hyper-parameter. All fields become CLI flags automatically."""
import argparse
import dataclasses
from dataclasses import dataclass, field


@dataclass
class Cfg:
    # ---------------- data ----------------
    dataset: str = "synthetic"        # synthetic | ninapro_db2 | hyser
    data_root: str = ""
    exercise: int = 1                 # NinaPro DB2 exercise (1,2,3)
    dof_idx: list = field(default_factory=list)   # glove sensor subset (empty = all)
    hyser_task: str = "ndof"          # 1dof | ndof | random
    hyser_session: int = 1
    ch_sub: int = 4                   # Hyser: keep every k-th of the 256 channels
    max_subjects: int = 0             # 0 = all
    n_syn_subjects: int = 8           # synthetic only
    # ---------------- preprocessing ----------------
    fs: int = 1000                    # working sampling rate (Hz)
    win_ms: int = 300
    stride_ms: int = 50
    notch_hz: float = 50.0            # India = 50 Hz mains
    causal: bool = True               # causal filters (what a real prosthesis can do)
    label_lp_hz: float = 10.0
    n_ring: int = 8                   # first n_ring electrodes form an armband ring (0 = disable shift modules)
    max_shift: int = 2                # max electrode rotation (in electrode positions)
    # ---------------- model ----------------
    model: str = "synformer"          # synformer | lstm | ridge
    d_model: int = 64                 # must be divisible by 8
    n_heads: int = 4
    n_layers: int = 2
    n_syn: int = 5                    # hand-synergy bottleneck size
    drop: float = 0.2
    use_dyn: bool = True              # C1 learnable activation dynamics
    use_shift: bool = True            # C3 electrode-shift aligner
    use_syn: bool = True              # C2 synergy bottleneck head
    use_film: bool = True             # C4 subject FiLM / few-shot adaptation
    use_dann: bool = True             # domain-adversarial invariance
    use_nll: bool = True              # C5 heteroscedastic head
    # ---------------- training ----------------
    epochs: int = 30
    bs: int = 64                      # pairs per batch (effective batch = 2*bs)
    lr: float = 1e-3
    wd: float = 1e-2
    patience: int = 6
    n_val: int = 2                    # held-out *training* subjects for early stopping
    lam_ccc: float = 0.5
    lam_nll: float = 0.2
    lam_delta: float = 0.5
    lam_res: float = 0.05
    lam_dann: float = 0.1
    lam_shift: float = 0.2
    film_dropout: float = 0.3
    ch_drop: float = 0.05
    noise_max: float = 0.15
    shift_aug: bool = True
    kd_w: float = 1.0                 # distillation weight (student only)
    # ---------------- few-shot calibration / uncertainty ----------------
    adapt_s: float = 30.0             # seconds of labelled data from the new user
    conf_s: float = 20.0              # seconds used for conformal calibration
    adapt_steps: int = 150
    adapt_lr: float = 1e-2
    adapt_l2: float = 1e-2
    adapt_curve: list = field(default_factory=lambda: [10, 20, 60])
    mc: int = 10                      # MC-dropout passes for epistemic uncertainty
    alpha: float = 0.1                # conformal miscoverage (0.1 -> 90 % intervals)
    # ---------------- run ----------------
    seed: int = 0
    out_dir: str = "results"
    tag: str = "run"
    test_subjects: list = field(default_factory=list)   # empty = all (LOSO)
    # runtime-filled
    n_ch: int = 0
    n_dof: int = 0


def parse(argv=None) -> Cfg:
    p = argparse.ArgumentParser(description="SynFormer-CP")
    for f in dataclasses.fields(Cfg):
        v = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        name = "--" + f.name
        if isinstance(v, bool):
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=v)
        elif isinstance(v, list):
            p.add_argument(name, nargs="*", type=int, default=v)
        else:
            p.add_argument(name, type=type(v), default=v)
    return Cfg(**vars(p.parse_args(argv)))
