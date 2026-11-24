"""Train teacher on one LOSO fold, distil a small student, export ONNX (+INT8), benchmark latency.
python scripts/export_edge.py --dataset synthetic --test_subjects 0 --tag edge"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
from src.config import parse
from src.datasets import build_all, merge
from src.engine import adapt, get_device, predict, train_model
from src.export import benchmark, export_onnx
from src.metrics import summarize
from src.pipeline import run_fold, split_target

cfg = parse()
data = build_all(cfg)
ts = (cfg.test_subjects or sorted(data))[0]
dev = get_device()
res = run_fold(cfg, data, ts, dev)
teacher, ymu, ysd, q = res["_ckpt"]
print("teacher zero-shot R2:", round(res["zero_shot"]["r2"], 3))

subs = [s for s in sorted(data) if s != ts]
val_s, tr_s = subs[-cfg.n_val:], subs[:-cfg.n_val]
Xtr, Ytr, Str, NXT = merge(data, tr_s); Xva, Yva, _, _ = merge(data, val_s)
Ytr, Yva = (Ytr - ymu) / ysd, (Yva - ymu) / ysd
scfg = replace(cfg, d_model=32, n_heads=2, n_layers=1, mc=0)
student = train_model(scfg, Xtr, Ytr, Str, NXT, Xva, Yva, len(tr_s), dev, teacher=teacher)
a, c, t = split_target(data[ts], cfg)
Xt, Yt = data[ts].X[t], (data[ts].Y[t] - ymu) / ysd
m_s = adapt(student, data[ts].X[a], (data[ts].Y[a] - ymu) / ysd, scfg, dev)
summ = summarize(Yt, predict(m_s, Xt, dev, scfg)[0], ysd)
print("student adapted R2:", round(summ["r2"], 3))

out = os.path.join(cfg.out_dir, cfg.tag); os.makedirs(out, exist_ok=True)
paths = export_onnx(m_s, scfg, os.path.join(out, "student.onnx"))
bench = {k: benchmark(v, scfg) for k, v in paths.items()}
print(json.dumps(bench, indent=1))
json.dump({"teacher_r2": res["zero_shot"]["r2"], "student_adapted": summ, "bench": bench,
           "params_teacher": sum(p.numel() for p in teacher.parameters()),
           "params_student": sum(p.numel() for p in student.parameters())},
          open(os.path.join(out, "edge.json"), "w"), indent=1)
