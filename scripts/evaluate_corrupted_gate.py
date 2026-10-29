#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch

from train_fusion import FusionNet
from train_fusion_corrupted import load_all
from train_signal import macro_ap, macro_auc, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="super")
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    q_dim = np.load(os.path.join(args.feature_root, "signal", "clean", "q_train.npy")).shape[1]
    sig, img = load_all(args.feature_root, "test", q_dim)
    y = np.load(os.path.join(args.data_dir, f"y_test_{args.task}.npy"))
    ck = torch.load(os.path.join(args.run, "best.pt"), map_location=device)
    mode = ck["args"]["mode"]
    model = FusionNet(mode, sig["clean"][0].shape[1], img["clean"][0].shape[1], q_dim, y.shape[1]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    def evaluate(cs, ci, perm=None):
        zs, qs = sig[cs]
        zi, qi = img[ci]
        if perm is not None:
            qs = qs[perm]
            qi = qi[perm]
        zs_t = torch.from_numpy(zs).to(device)
        zi_t = torch.from_numpy(zi).to(device)
        qs_t = torch.from_numpy(qs).to(device)
        qi_t = torch.from_numpy(qi).to(device)
        ps = []
        gates = []
        with torch.no_grad():
            for i in range(0, len(zs_t), 512):
                logits, aux = model(zs_t[i:i + 512], zi_t[i:i + 512], qs_t[i:i + 512], qi_t[i:i + 512])
                ps.append(torch.sigmoid(logits).float().cpu().numpy())
                if "gate" in aux:
                    gates.append(aux["gate"].float().cpu().numpy())
        p = np.concatenate(ps, axis=0)
        g = np.concatenate(gates, axis=0) if gates else None
        return macro_auc(y, p), macro_ap(y, p), macro_f1(y, p, 0.5), float(g.mean()) if g is not None else float("nan")

    pairs = [("clean", "clean"), ("noise10", "clean"), ("baseline03", "clean"), ("leaddrop3", "clean"), ("clean", "blur2"), ("clean", "crop10"), ("clean", "jpeg30")]
    rng = np.random.default_rng(0)
    out = []
    for cs, ci in pairs:
        base = evaluate(cs, ci)
        perms = []
        for _ in range(args.n_perm):
            perm = rng.permutation(len(y))
            perms.append(evaluate(cs, ci, perm))
        aucs = [x[0] for x in perms]
        row = {
            "sig": cs,
            "img": ci,
            "clean_auc": base[0],
            "perm_auc_mean": float(np.mean(aucs)),
            "perm_auc_std": float(np.std(aucs)),
            "delta_auc": float(base[0] - np.mean(aucs)),
            "gate_mean": base[3],
        }
        out.append(row)
        print(row, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
