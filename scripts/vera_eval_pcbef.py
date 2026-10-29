#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch

from train_signal import ece, macro_ap, macro_auc, macro_f1
from vera_train_pcbef import PCBEFNet, load_all


def eval_cond(model, sig, img, y, cs, ci, device, q_perm=None):
    zs, qs = sig[cs]
    zi, qi = img[ci]
    if q_perm is not None:
        qs, qi = qs[q_perm], qi[q_perm]
    zs_t = torch.from_numpy(zs).to(device)
    zi_t = torch.from_numpy(zi).to(device)
    qs_t = torch.from_numpy(qs).to(device)
    qi_t = torch.from_numpy(qi).to(device)
    ps, gs, gi, us = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(zs_t), 512):
            p, aux = model(zs_t[i:i + 512], zi_t[i:i + 512], qs_t[i:i + 512], qi_t[i:i + 512])
            ps.append(p.float().cpu().numpy())
            gs.append(aux["gamma_s"].float().cpu().numpy())
            gi.append(aux["gamma_i"].float().cpu().numpy())
            us.append(aux["uncertainty"].float().cpu().numpy())
    p = np.concatenate(ps, axis=0)
    return {
        "auc": macro_auc(y, p), "ap": macro_ap(y, p), "f1": macro_f1(y, p, 0.5), "ece": ece(y, p),
        "gamma_s": float(np.concatenate(gs).mean()), "gamma_i": float(np.concatenate(gi).mean()),
        "uncertainty": float(np.concatenate(us).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    q_dim = np.load(os.path.join(args.feature_root, "signal", "clean", "q_train.npy")).shape[1]
    sig, img = load_all(args.feature_root, "test", q_dim)
    y = np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy"))
    ck = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    model = PCBEFNet(sig["clean"][0].shape[1], img["clean"][0].shape[1], q_dim, y.shape[1], mode=ck["args"]["mode"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    pairs = [("clean", "clean"), ("noise10", "clean"), ("baseline03", "clean"), ("leaddrop3", "clean"),
             ("clean", "blur2"), ("clean", "crop10"), ("clean", "jpeg30"),
             ("noise10", "blur2"), ("baseline03", "crop10"), ("leaddrop3", "jpeg30")]
    rng = np.random.default_rng(0)
    rows = []
    for cs, ci in pairs:
        base = eval_cond(model, sig, img, y, cs, ci, device)
        aucs = []
        for _ in range(10):
            perm = rng.permutation(len(y))
            aucs.append(eval_cond(model, sig, img, y, cs, ci, device, perm)["auc"])
        row = {"sig": cs, "img": ci, **base, "q_perm_mean": float(np.mean(aucs)),
               "q_perm_std": float(np.std(aucs)), "q_perm_delta": float(base["auc"] - np.mean(aucs))}
        rows.append(row)
        print(row, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
