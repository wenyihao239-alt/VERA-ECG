#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit

from vera_coral_calib import coral, em_prior, metrics, predict, sqrtm, train_head


def l2_normalize(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)


def damped_coral(Xs, Xt, lam=0.1, reg=1e-3):
    mu_s, mu_t = Xs.mean(0, keepdims=True), Xt.mean(0, keepdims=True)
    Cs = np.cov(Xs, rowvar=False) + reg * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + reg * np.eye(Xt.shape[1])
    Cd = (1.0 - lam) * Cs + lam * Ct
    Xt_c = Xt - mu_t
    Xt_a = Xt_c @ sqrtm(Cd, inv=True) @ sqrtm(Cs)
    return (Xt_a + mu_s).astype(np.float32)


def fit_temperature(logits, y):
    def nll(t):
        T = np.exp(t)
        p = expit(logits / T)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    r = minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(r.x))


def apply_temp(p, T):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return expit(logit / T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambdas", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--l2", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ztr = np.load(os.path.join(args.feature_dir, "signal_train.npy")).astype(np.float32)
    ytr = np.load(os.path.join(args.feature_dir, "y_train_cpsc.npy")).astype(np.float32)
    zva = np.load(os.path.join(args.feature_dir, "signal_val.npy")).astype(np.float32)
    yva = np.load(os.path.join(args.feature_dir, "y_val_cpsc.npy")).astype(np.float32)
    if args.l2:
        ztr, zva = l2_normalize(ztr), l2_normalize(zva)
    mu, sd = ztr.mean(0, keepdims=True), ztr.std(0, keepdims=True) + 1e-6
    ztr = (ztr - mu) / sd
    zva = (zva - mu) / sd
    np.random.seed(0); torch.manual_seed(0)
    head = train_head(ztr, ytr, device=device)
    src_prior = np.bincount(ytr.argmax(1), minlength=9) / len(ytr)
    val_p = predict(head, zva, device)
    T = fit_temperature(np.log(np.clip(val_p, 1e-6, 1 - 1e-6) / np.clip(1 - val_p, 1e-6, 1)), yva)
    rows = []
    for lam in [float(x) for x in args.lambdas.split(",")]:
        fold_rows = []
        for f in range(5):
            zt = np.load(os.path.join(args.feature_dir, f"cpsc_fold{f}.npy")).astype(np.float32)
            yt = np.load(os.path.join(args.feature_dir, f"cpsc_fold{f}_label.npy")).astype(np.float32)
            if args.l2:
                zt = l2_normalize(zt)
            zt = (zt - mu) / sd
            p0 = predict(head, zt, device)
            p_c = predict(head, damped_coral(ztr, zt, lam=lam), device)
            pi = em_prior(p_c, src_prior)
            p_cp = p_c * (pi / np.maximum(src_prior, 1e-8))
            p_cp = p_cp / np.maximum(p_cp.sum(1, keepdims=True), 1e-8)
            fold_rows.append({"fold": f, "n": int(len(yt)),
                              "zero": metrics(yt, apply_temp(p0, T)),
                              "coral": metrics(yt, apply_temp(p_c, T)),
                              "coral_prior": metrics(yt, apply_temp(p_cp, T))})
        mean = {}
        for k in ["zero", "coral", "coral_prior"]:
            mean[k] = {m: float(np.mean([r[k][m] for r in fold_rows])) for m in ["macro_auc", "macro_ap", "macro_f1", "ece", "aurc"]}
            mean[k]["std_auc"] = float(np.std([r[k]["macro_auc"] for r in fold_rows]))
        rows.append({"lambda": lam, "temperature": T, "folds": fold_rows, "mean": mean})
        print("lambda", lam, "T", round(T, 3), "zero", round(mean["zero"]["macro_auc"], 4),
              "coral", round(mean["coral"]["macro_auc"], 4), "prior", round(mean["coral_prior"]["macro_auc"], 4), flush=True)
    out = {"l2": bool(args.l2), "lambdas": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
