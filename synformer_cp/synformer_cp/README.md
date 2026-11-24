# SynFormer-CP
**Synergy-constrained, shift-robust, subject-adaptive and conformally-calibrated Transformer for continuous finger kinematics / force estimation from sEMG (prosthetic control)**

Datasets: NinaPro DB2 (joint angles), Hyser (finger forces), plus a built-in synthetic generator for testing the pipeline with no data.

---
## 1. What is novel (contributions)

Each ingredient exists in the ML literature; the contribution is adapting them to EMG regression and combining them into one deployable system (verify against your own literature search before claiming "first").

| # | Contribution | Where | Why it matters |
|---|---|---|---|
| C1 | **Learnable muscle-activation dynamics layer**: rectify -> per-channel learnable exponential integrator (time constants initialised 15-80 ms) | `models.ActivationDynamics` | Physiologically grounded envelope extractor, parallel to a multi-scale raw-signal branch |
| C2 | **Synergy-bottleneck output head**: joints = PCA-initialised synergy decoder(latent) + small penalised residual | `models.SynFormer`, `losses` | Regularises towards coordinated hand motion, helps unseen subjects, 5-6 numbers instead of 22 |
| C3 | **Electrode-shift aligner**: spatial-transformer predicts armband rotation from channel energy profile, supervised by shift augmentation; plus a rotation-robustness test | `models.ShiftAligner`, `engine.augment` | Targets the #1 real-world failure (re-donning the band) |
| C4 | **Subject FiLM + 30 s few-shot calibration**: only a FiLM vector + synergy head (<2% of weights) updated with an L2 pull to the generic solution; DANN for invariance | `engine.adapt` | Fast, safe personalisation; accuracy-vs-calibration-time curve |
| C5 | **Calibrated uncertainty**: heteroscedastic beta-NLL head + MC-dropout, then **split-conformal intervals** with coverage guarantee (per DoF) | `conformal.py` | Honest "how sure am I" for a device on the body |
| C6 | **Uncertainty-gated controller**: command slows/holds when the interval is wide, no added lag otherwise | `safety.py`, `stream.py` | Turns uncertainty into a safety behaviour; measurable error reduction |
| C7 | **Temporal-delta loss** (match frame-to-frame change, not smoothing) + MSE + CCC | `losses.py` | Smooth *and* responsive output |
| C8 | **Distillation + INT8 ONNX** with measured latency; **causal streaming** pipeline | `export.py`, `stream.py` | Real-time, edge-ready |

## 2. Architecture
```
EMG (B,C,T) -> [ShiftAligner (ring electrodes)] -> ChannelSE
   |-> multi-scale strided conv (k=5,9,17,33) --------------------\
   |-> |x| -> avgpool -> ActivationDynamics -> 1x1 conv -----------+-> concat -> dilated TCN x3
                                                                              -> stride-2 conv -> Transformer x L
                                                                              -> [last token ; mean] -> z (subject-invariant, DANN)
z -> FiLM(subject / new-user vector) -> synergy head -> synergy decoder (PCA init) (+ residual) -> mu (joint angles / forces)
                                     -> log-variance head -> sigma
(mu, sigma) -> MC-dropout -> conformal intervals -> uncertainty gate -> prosthesis command
```

## 3. Quick start
```bash
pip install -r requirements.txt
python tests/test_numpy_parts.py          # torch-free sanity checks (passes)
python scripts/smoke_test.py              # FIRST run on your machine: tiny end-to-end run, needs torch
python scripts/run_loso.py --dataset synthetic --tag syn_full
python scripts/run_loso.py --dataset ninapro_db2 --data_root /data/DB2 --exercise 1 --tag db2_full
python scripts/run_loso.py --dataset hyser --data_root /data/hyser --n_ring 0 --n_syn 3 --tag hyser_full
sh scripts/run_ablation.sh "--dataset ninapro_db2 --data_root /data/DB2" db2     # baselines + 6 ablations + stats
python scripts/export_edge.py --dataset synthetic --test_subjects 0 --tag edge   # distil + ONNX + INT8 + latency
```
Dataset notes
* **NinaPro DB2**: `emg` (12 ch) and `glove` (22 sensors) at 2 kHz. Choose sensors with `--dof_idx` (check the Ninapro sensor map; some sensors are dead for some subjects). `--n_ring 8` assumes electrodes 0-7 form the forearm ring: confirm from the DB2 documentation.
* **Hyser**: needs `wfdb`. Loader keeps every 4th of 256 channels (`--ch_sub`). Force is 5-DoF and individuated, so use `--n_ring 0` and a small `--n_syn` (or `--no-use_syn`). Verify scaling and file naming against the PhysioNet page; normalise by MVC if you need absolute forces.
* Outputs: `results/<tag>/loso.json` per subject (zero-shot, adapted, shift robustness, adaptation curve, conformal coverage/width, gate effect).

## 4. Experiments to report (maps to thesis chapters)
| ID | Experiment | Output |
|---|---|---|
| E1 | LOSO zero-shot vs Ridge / LSTM / (published models you re-implement) | R2, NRMSE, CCC, Wilcoxon p |
| E2 | Few-shot calibration curve (10/20/30/60 s) | R2 vs seconds (`adapt_curve`) |
| E3 | Electrode-rotation robustness (`shift_r2`) | R2 vs shift, full vs `--no-use_shift` |
| E4 | Ablations: dyn / syn / shift / film / dann / nll | table (`run_ablation.sh`) |
| E5 | Conformal validity | coverage vs target 90%, width, corr(sigma, error) |
| E6 | Safety gate | RMSE gated vs ungated (`safety_gate`) |
| E7 | Edge | params, size, p50/p95 latency fp32 vs INT8 (`export_edge.py`); target < 50 ms |
| E8 | Cross-dataset / cross-session (Hyser session 1 -> 2) | generalisation |

**Report honestly:** subject-wise splits only; cross-subject numbers are the headline; state that conformal coverage is approximate because EMG windows are temporally dependent (calibration slice is disjoint and chronologically before the test slice).

## 5. Status of this code (read this)
| Part | Status |
|---|---|
| Preprocessing, lag estimation, windowing, synthetic data, ridge baseline, LOSO pipeline (ridge), metrics, conformal, safety gate, streaming filter | **Executed and tested** in the authoring sandbox (see `tests/`) |
| PyTorch model, losses, training, adaptation, MC-dropout, ONNX/INT8 export | **Written and syntax-checked, NOT executed** (no PyTorch in the authoring sandbox). Run `scripts/smoke_test.py` first and expect to fix small shape/API issues |
| NinaPro / Hyser loaders | Written from the dataset documentation, **not run on real files**; verify keys, sampling rates and sensor maps |
| Any accuracy number | **None claimed.** Results must come from your runs |

## 6. Suggested hyper-parameter starting points
`win_ms 300, stride_ms 50 (25 for final), d_model 64, n_layers 2, epochs 30, lr 1e-3, adapt_s 30, alpha 0.1`. For DB2 consider `--dof_idx` with the ~10 most informative finger joints first, then all 22.

## 7. Layout
```
src/  config preprocess datasets metrics conformal safety baselines | models losses engine pipeline export stream
scripts/ run_loso run_ablation.sh compare export_edge smoke_test
tests/ test_numpy_parts.py        results/ (outputs)
```
