"""Summarise runs and test significance (paired Wilcoxon over subjects) against a reference run."""
import argparse, json, os
import numpy as np
from scipy.stats import wilcoxon

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results"); p.add_argument("--ref", required=True)
p.add_argument("--tags", nargs="+", required=True); a = p.parse_args()


def load(tag):
    return json.load(open(os.path.join(a.dir, tag, "loso.json")))


def vec(res, split, key):
    k = split if split in next(iter(res.values())) else "zero_shot"
    return np.array([res[s][k][key] for s in sorted(res)])


ref = load(a.ref)
print(f"{'run':24s} {'split':9s} {'R2':>13s} {'NRMSE':>13s} {'CCC':>13s}   p(vs ref, R2)")
for tag in [a.ref] + a.tags:
    try:
        r = load(tag)
    except FileNotFoundError:
        continue
    for split in ("zero_shot", "adapted"):
        if split not in next(iter(r.values())):
            continue
        r2, nr, cc = (vec(r, split, k) for k in ("r2", "nrmse", "ccc"))
        pv = ""
        if tag != a.ref and split in next(iter(ref.values())) and len(r2) == len(vec(ref, split, "r2")) > 5:
            try:
                pv = f"{wilcoxon(vec(ref, split, 'r2'), r2).pvalue:.4f}"
            except ValueError:
                pv = "n/a"
        print(f"{tag:24s} {split:9s} {r2.mean():6.3f}+-{r2.std():5.3f} {nr.mean():6.3f}+-{nr.std():5.3f} "
              f"{cc.mean():6.3f}+-{cc.std():5.3f}   {pv}")
