#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_fusion import FusionNet
from train_fusion_corrupted import load_all
from train_signal import ece


def get_logits(model, sig, img, cs, ci, device, batch_size=512):
    zs, qs = sig[cs]
    zi, qi = img[ci]
    zs_t = torch.from_numpy(zs).to(device)
    zi_t = torch.from_numpy(zi).to(device)
    qs_t = torch.from_numpy(qs).to(device)
    qi_t = torch.from_numpy(qi).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(zs_t), batch_size):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, _ = model(zs_t[i:i + batch_size], zi_t[i:i + batch_size], qs_t[i:i + batch_size], qi_t[i:i + batch_size])
            out.append(logits.float().cpu())
    return torch.cat(out, dim=0)


def fit_temperature(logits, y):
    best_t, best_loss = 1.0, float("inf")
    loss_fn = nn.BCEWithLogitsLoss()
    for t in np.linspace(0.5, 5.0, 91):
        loss = float(loss_fn(logits / float(t), y))
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def risk_coverage(logits, y):
    p = torch.sigmoid(logits).numpy()
    y = y.numpy()
    ent = -(p * np.log(np.clip(p, 1e-6, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-6, 1)))
    conf = 1.0 - ent.mean(axis=1) / np.log(2)
    err = 1.0 - ((p >= 0.5) == y).mean(axis=1)
    order = np.argsort(-conf)
    grid = np.linspace(0.05, 1.0, 20)
    risks = []
    for c in grid:
        k = max(1, int(c * len(y)))
        risks.append(float(err[order[:k]].mean()))
    return {
        "aurc": float(np.mean(risks)),
        "risk_at_80": risks[int(np.argmin(np.abs(grid - 0.8)))],
        "risk_at_90": risks[int(np.argmin(np.abs(grid - 0.9)))],
        "risk_at_100": risks[-1],
        "conf_mean": float(conf.mean()),
    }


def metrics(logits, y, T=1.0):
    z = logits / float(T)
    p = torch.sigmoid(z)
    yy = y.numpy()
    pp = p.numpy()
    return {
        "temperature": float(T),
        "ece": float(ece(yy, pp)),
        "brier": float(np.mean((pp - yy) ** 2)),
        "nll": float(F.binary_cross_entropy_with_logits(z, y).item()),
        **risk_coverage(z, y),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    q_dim = np.load(os.path.join(args.feature_root, "signal", "clean", "q_train.npy")).shape[1]
    sig_va, img_va = load_all(args.feature_root, "val", q_dim)
    sig_te, img_te = load_all(args.feature_root, "test", q_dim)
    y_va = torch.from_numpy(np.load(os.path.join(args.data_dir, f"y_val_{args.task}.npy")).astype(np.float32))
    y_te = torch.from_numpy(np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy")).astype(np.float32))

    pairs = [("clean", "clean"), ("noise10", "clean"), ("baseline03", "clean"), ("leaddrop3", "clean"),
             ("clean", "blur2"), ("clean", "crop10"), ("clean", "jpeg30"),
             ("noise10", "blur2"), ("baseline03", "crop10"), ("leaddrop3", "jpeg30")]
    out = []
    for run in args.runs:
        ck = torch.load(os.path.join(run, "best.pt"), map_location=device)
        mode = ck["args"]["mode"]
        model = FusionNet(mode, sig_te["clean"][0].shape[1], img_te["clean"][0].shape[1], q_dim, y_te.shape[1]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        logits_va = get_logits(model, sig_va, img_va, "clean", "clean", device)
        T = fit_temperature(logits_va, y_va)
        for cs, ci in pairs:
            logits = get_logits(model, sig_te, img_te, cs, ci, device)
            m0 = metrics(logits, y_te, 1.0)
            mT = metrics(logits, y_te, T)
            row = {"run": os.path.basename(run), "mode": mode, "sig": cs, "img": ci,
                   "ece_raw": m0["ece"], "brier_raw": m0["brier"], "nll_raw": m0["nll"],
                   "aurc_raw": m0["aurc"], "risk80_raw": m0["risk_at_80"], "risk100_raw": m0["risk_at_100"],
                   "T": T, "ece_T": mT["ece"], "brier_T": mT["brier"], "nll_T": mT["nll"],
                   "aurc_T": mT["aurc"], "risk80_T": mT["risk_at_80"], "risk100_T": mT["risk_at_100"]}
            out.append(row)
            print(row, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
