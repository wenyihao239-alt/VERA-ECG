#!/usr/bin/env python3
import numpy as np
from scipy.signal import butter, sosfiltfilt


def stats_hp(x, fs=100, cutoff=0.5):
    sos = butter(4, cutoff / (fs / 2.0), btype="highpass", output="sos")
    out = []
    for rec in x:
        rec = np.asarray(rec, dtype=np.float32)
        valid = np.any(np.abs(rec) > 1e-8, axis=0)
        if valid.any():
            first = int(np.argmax(valid)); last = len(valid) - int(np.argmax(valid[::-1]))
            rec = rec[:, first:last]
        try:
            rec = sosfiltfilt(sos, rec, axis=1).astype(np.float32)
        except Exception:
            rec = rec.astype(np.float32)
        std = rec.std(axis=1)
        p95 = np.percentile(rec, 95, axis=1)
        p5 = np.percentile(rec, 5, axis=1)
        out.append(np.concatenate([std, p95, p5], axis=0))
    return np.stack(out, axis=0).astype(np.float32)
