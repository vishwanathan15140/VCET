"""Run this FIRST on your machine (needs torch). ~2-5 min on CPU. Verifies the whole stack end-to-end
on synthetic data: train -> zero-shot -> adapt -> conformal -> gate."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import parse
from src.pipeline import run_loso

if __name__ == "__main__":
    cfg = parse(["--n_syn_subjects", "5", "--epochs", "3", "--d_model", "32", "--n_heads", "2",
                 "--n_layers", "1", "--test_subjects", "0", "--adapt_steps", "30", "--mc", "3",
                 "--adapt_curve", "10", "--tag", "smoke"] + sys.argv[1:])
    r = run_loso(cfg)
    print("SMOKE TEST OK" if "adapted" in r["0"] else "SMOKE TEST FAILED")
