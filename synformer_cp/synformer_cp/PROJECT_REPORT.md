# Project Report Template: SynFormer-CP

## Title
Cross-Subject Continuous Finger Kinematics and Force Estimation from Surface EMG Using a Synergy-Constrained, Shift-Robust Transformer with Conformal Uncertainty for Prosthetic Control

## Abstract
Commercial myoelectric prostheses mostly rely on discrete gesture classification, which prevents simultaneous, proportional finger control. Deep regression models can estimate continuous joint angles and forces from surface EMG, but are usually evaluated within a single subject; accuracy drops for new users, after electrode re-placement, and models rarely indicate when their output is unreliable. This work proposes SynFormer-CP, a causal temporal-convolutional Transformer that (i) extracts envelopes with a learnable muscle-activation-dynamics layer alongside a multi-scale raw branch, (ii) constrains outputs through a PCA-initialised hand-synergy bottleneck, (iii) compensates armband rotation with a supervised electrode-shift aligner, (iv) personalises to a new user from about 30 s of data by updating only a FiLM vector and synergy head, and (v) outputs calibrated uncertainty using a heteroscedastic head, MC-dropout and split-conformal intervals that drive an uncertainty-gated controller. The model is evaluated with leave-one-subject-out protocols on NinaPro DB2 (joint angles) and Hyser (finger forces) against feature-based ridge regression, LSTM and published Transformer estimators, using R2, NRMSE and CCC with paired significance tests, ablations, rotation-robustness and calibration-time studies. A distilled, INT8-quantised student is deployed through ONNX with a causal streaming pipeline and a latency target below 50 ms. Results: [to be filled from experiments].

## Objectives
1. Continuous multi-finger angle/force regression from sEMG.
2. Cross-subject and cross-session generalisation under honest subject-wise evaluation.
3. Robustness to electrode rotation and 30-second personalisation.
4. Calibrated uncertainty with a safety behaviour.
5. Real-time edge deployment (< 50 ms per prediction).

## Research gap (verify with your own review)
* Most Transformer/CNN regressors report within-subject results; cross-subject work is limited.
* Uncertainty is rarely evaluated for EMG regression, and conformal coverage even less.
* Rotation robustness, personalisation cost and edge latency are usually studied separately.

Anchor papers to read and reproduce: Lin et al., IEEE TNSRE 2025 (efficient Transformer, DB2); An et al., SCTNet, IEEE Sensors J. 2024; Guo & Yang, sMSDCTrans, ICCAR 2025; Jiang et al., Hyser dataset, IEEE TNSRE 2021. Method references: DANN (Ganin 2016), FiLM (Perez 2018), beta-NLL (Seitzer 2022), MC-dropout (Gal 2016), conformal prediction (Vovk; Angelopoulos & Bates tutorial).

## Chapters
1 Introduction | 2 Literature review | 3 Datasets and preprocessing | 4 Proposed method (C1-C8) | 5 Experimental protocol | 6 Results (E1-E8) | 7 Discussion and limitations | 8 Conclusion and future work (amputee data, closed-loop user study)

## Timeline (6 months)
M1 literature + data + baselines | M2 model on DB2, debug | M3 modules + ablations | M4 Hyser, cross-session, stats | M5 distillation, edge, streaming demo | M6 writing, paper, viva

## Likely viva questions
* Why subject-wise splits? (random window splits leak data and inflate R2)
* Why a synergy bottleneck if fingers are individuated? (soft constraint + penalised residual; ablation shows effect)
* What does conformal coverage guarantee and what breaks it? (exchangeability; temporal dependence)
* Why delta loss instead of smoothing? (keeps responsiveness)
* How was EMG-to-label delay handled? (estimated per subject, compensated)
* Limitations: intact-subject data, glove noise, no closed-loop user study.
