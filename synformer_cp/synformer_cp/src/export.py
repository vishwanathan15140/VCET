"""Edge deployment: ONNX export, INT8 dynamic quantisation, latency benchmark."""
import os
import time

import numpy as np
import torch
import torch.nn as nn


class _Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return o["mu"], o["logvar"]


def export_onnx(model, cfg, path, int8=True):
    model = model.eval().cpu()
    T = int(cfg.fs * cfg.win_ms / 1000)
    dummy = torch.randn(1, cfg.n_ch, T)
    kw = dict(input_names=["emg"], output_names=["mu", "logvar"], opset_version=17,
              dynamic_axes={"emg": {0: "batch"}, "mu": {0: "batch"}, "logvar": {0: "batch"}})
    try:
        torch.onnx.export(_Wrap(model), dummy, path, dynamo=False, **kw)
    except TypeError:                                   # older torch without the dynamo kwarg
        torch.onnx.export(_Wrap(model), dummy, path, **kw)
    paths = {"fp32": path}
    if int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            q = path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(path, q, weight_type=QuantType.QInt8)
            paths["int8"] = q
        except Exception as e:                          # keep going: fp32 model is still valid
            print("INT8 quantisation skipped:", e)
    return paths


def benchmark(path, cfg, n=300):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, cfg.n_ch, int(cfg.fs * cfg.win_ms / 1000)).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"emg": x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"emg": x})
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "size_kb": os.path.getsize(path) / 1024}


def ort_predict_fn(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {"emg": x.astype(np.float32)})
