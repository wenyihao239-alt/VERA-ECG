#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch

from train_fusion import FusionNet
from train_fusion_corrupted import load_all
from train_signal import macro_ap, macro_auc, macro_f1


def eval_pair(model, sig, img, y, cs, ci, device, mask_s=1.0, mask_i=1.0, force_gate=None):
    model.force_gate = force_gate
    zs, qs = sig[cs]
    zi, qi = img[ci]
    zs_t = torch.from_numpy(zs).to(device)
    zi_t = torch.from_numpy(zi).to(device)
    qs_t = torch.from_numpy(qs).to(device)
    qi_t = torch.from_numpy(qi).to(device)
    ms = torch.full((len(zs_t), 1), mask_s, device=device)
    mi = torch.full((len(zi_t), 1), mask_i, device=device)
    ps, gates = [], []
    with torch.no_grad():
        for i in range(0, len(zs_t), 512):
            logits, aux = model(zs_t[i:i + 512], zi_t[i:i + 512], qs_t[i:i + 512], qi_t[i:i + 512], ms[i:i + 512], mi[i:i + 512])
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            if "gate" in aux:
                gates.append(aux["gate"].float().cpu().numpy())
    p = np.concatenate(ps, axis=0)
    g = np.concatenate(gates, axis=0)
    return {"auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "gate_mean": float(g.mean())}


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
    sig, img = load_all(args.feature_root, "test", q_dim)
    y = np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy"))
    pairs = [
        ("clean", "clean"),
        ("noise10", "clean"), ("baseline03", "clean"), ("leaddrop3", "clean"),
        ("clean", "blur2"), ("clean", "crop10"), ("clean", "jpeg30"),
        ("noise10", "blur2"), ("baseline03", "crop10"), ("leaddrop3", "jpeg30"),
    ]
    out = []
    for run in args.runs:
        ck = torch.load(os.path.join(run, "best.pt"), map_location=device)
        mode = ck["args"]["mode"]
        model = FusionNet(mode, sig["clean"][0].shape[1], img["clean"][0].shape[1], q_dim, y.shape[1]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        for cs, ci in pairs:
            a = eval_pair(model, sig, img, y, cs, ci, device)
            b = eval_pair(model, sig, img, y, cs, ci, device, force_gate=0.5)
            row = {"run": os.path.basename(run), "mode": mode, "sig": cs, "img": ci,
                   "auc": a["auc"], "auc_const_gate": b["auc"], "gate_mean": a["gate_mean"]}
            out.append(row)
            print(row, flush=True)
        for cond, ms, mi in [("signal_only", 1.0, 0.0), ("image_only", 0.0, 1.0)]:
            a = eval_pair(model, sig, img, y, "clean", "clean", device, mask_s=ms, mask_i=mi)
            row = {"run": os.path.basename(run), "mode": mode, "sig": cond, "img": "clean", "auc": a["auc"], "auc_const_gate": a["auc"], "gate_mean": a["gate_mean"]}
            out.append(row)
            print(row, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
