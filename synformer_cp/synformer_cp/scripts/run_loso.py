"""python scripts/run_loso.py --dataset synthetic --tag syn_full
   python scripts/run_loso.py --dataset ninapro_db2 --data_root /data/ninapro/DB2 --tag db2_full
   python scripts/run_loso.py --dataset hyser --data_root /data/hyser --n_ring 0 --n_syn 3 --tag hyser_full"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import parse
from src.pipeline import run_loso

if __name__ == "__main__":
    run_loso(parse())
